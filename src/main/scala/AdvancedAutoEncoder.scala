import java.io.File
import java.net.URL
import java.util
import java.util.concurrent.TimeUnit

import org.apache.commons.io.FileUtils
import org.apache.spark.api.java.JavaRDD
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import org.nd4j.util.ArchiveUtils
import org.datavec.spark.transform.misc.StringToWritablesFunction
import org.apache.spark.sql.{DataFrame, SparkSession, functions}
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.transform.{ReduceOp, TransformProcess}
import org.datavec.api.transform.condition.{BooleanCondition, ConditionOp}
import org.datavec.api.transform.condition.column.DoubleColumnCondition
import org.datavec.api.transform.metadata.{ColumnMetaData, StringMetaData}
import org.datavec.api.transform.ops.IAggregableReduceOp
import org.datavec.api.transform.reduce.{AggregableColumnReduction, Reducer}
import org.datavec.api.transform.schema.Schema
import org.datavec.api.transform.sequence.comparator.NumericalColumnComparator
import org.datavec.api.transform.sequence.window.{ReduceSequenceByWindowTransform, TimeWindowFunction}
import org.datavec.api.transform.transform.doubletransform.MinMaxNormalizer
import org.datavec.api.transform.transform.string.ConcatenateStringColumns
import org.datavec.api.transform.transform.time.StringToTimeTransform
import org.datavec.api.writable.{DoubleWritable, Text, Writable}
import org.datavec.spark.storage.SparkStorageUtils
import org.datavec.spark.transform.SparkTransformExecutor
import org.joda.time.DateTimeZone
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.{MultiDataSet, MultiDataSetPreProcessor}
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.{BooleanIndexing, INDArrayIndex}
import org.nd4j.linalg.indexing.conditions.Conditions


object AdvancedAutoEncoder {

  case class Stats(numPositions : Int, minTime : Long, maxTime : Long, totalTime : Long)

  // Class to filter and transform Anomalous data
  object Reductions extends Serializable {

    class GeoAveragingReduction(val colName : String = "AveragedLatLon", val delim : String = ",") extends AggregableColumnReduction {

      override def reduceOp(): IAggregableReduceOp[Writable, util.List[Writable]] = new AverageCoordinateReduceOp(delim)

      override def getColumnsOutputName(columnInputName: String): List[String] = List(colName)

      override def getColumnOutputMetaData(newColumnName: util.List[String], columnInputMeta: ColumnMetaData): List[StringMetaData] =
        List(new StringMetaData(colName))

      override def transform(inputSchema: Schema): Schema = inputSchema

      override def outputColumnName(): String = null

      override def outputColumnNames(): Array[String] = new Array[String](0)

      override def columnName(): String = null

      override def columnNames(): Array[String] = new Array[String](0)

      override def getInputSchema(): Schema = ???

      override def setInputSchema(inputSchema: Schema): Unit = ???
    }

    class AverageCoordinateReduceOp(val delim : String) extends IAggregableReduceOp[Writable, util.List[Writable]] {
      final val PI_180 = Math.PI / 180.0
      var sumX : Double = 0.0
      var sumY : Double = 0.0
      var sumZ : Double = 0.0
      var count : Int = 0

      override def combine[W <: IAggregableReduceOp[Writable, util.List[Writable]]](accu: W): Unit = {
        if(accu.isInstanceOf[AverageCoordinateReduceOp]) {
          sumX += accu.asInstanceOf[AverageCoordinateReduceOp].sumX
          sumY += accu.asInstanceOf[AverageCoordinateReduceOp].sumY
          sumZ += accu.asInstanceOf[AverageCoordinateReduceOp].sumZ
          count += accu.asInstanceOf[AverageCoordinateReduceOp].count
        }
        else {
          throw new IllegalStateException("Cannot combine type of class: " + accu.getClass)
        }
      }

      override def accept(t: Writable): Unit = {
        val str : String = t.toString
        val split : Array[String] = str.split(delim)
        if(split.length != 2) {
          throw new IllegalStateException("Could not parse lat/long string: \"" + str + "\"")
        }

        // Converting Degress to Radians
        val latRadian : Double = split(0).asInstanceOf[Double] * PI_180
        val longRadian : Double = split(1).asInstanceOf[Double] * PI_180

        // X,Y and Z coordinates on Earth Surface
        sumX += Math.cos(latRadian) * Math.cos(longRadian)
        sumY += Math.cos(latRadian) * Math.sin(longRadian)
        sumZ += Math.sin(latRadian)
        count += 1
      }

      override def get(): List[Writable] = {
        val x : Double = sumX / count
        val y : Double = sumY / count
        val z : Double = sumZ / count

        val latDegree : Double = Math.atan2(z, Math.sqrt(Math.pow(x,2) + Math.pow(y,2))) / PI_180
        val longDegree : Double = Math.atan2(y, x) / PI_180

        List(new Text(f"$latDegree%f$delim%s$longDegree%f"))
      }
    }
  }

  object Preprocessor extends Serializable {

    class SeqToSeqAutoencoderPreprocessor extends MultiDataSetPreProcessor {
      override def preProcess(multiDataSet: MultiDataSet): Unit = {
        val input: INDArray = multiDataSet.getFeatures(0)
        val features: Array[INDArray] = Array.ofDim[INDArray](2)
        val labels: Array[INDArray] = Array.ofDim[INDArray](1)

        features(0) = input
        val inputSize : Long = input.size(0)
        val numClass : Long = input.size(1)
        val maxTimeSeqLength : Long = input.size(2)
        val goStopTokens : Long = numClass

        // New time step for New class for GO/STOP.
        val newShape : Array[Int] = Array(inputSize, numClass + 1, maxTimeSeqLength + 1)
        features(1) = Nd4j.create(newShape:_*)
        labels(0) = Nd4j.create(newShape:_*)

        // Create features. Append existing at time 1 to end. Put GO token at time 0
        features(1).put(Array[INDArrayIndex](all(), interval(0, input.size(1)), interval(1, newShape(2))), input)

        //Set GO token
        features(1).get(all(), point(goStopTokens), all()).assign(1)

        //Create labels. Append existing at time 0 to end-1. Put STOP token at last time step - Accounting for variable length / masks
        labels(0).put(Array[INDArrayIndex](all(), interval(0, input.size(1)), interval(0, newShape(2) - 1)), input)

        var lastTimeStepPos: Array[Int] = null

        if (multiDataSet.getFeaturesMaskArray(0) == null) {
          // No masks
          lastTimeStepPos = Array.ofDim[Int](input.size(0).asInstanceOf[Int])
          for (i <- 0 until lastTimeStepPos.length) {
            lastTimeStepPos(i) = (input.size(2) - 1).asInstanceOf[Int]
          }
        } else {
          val fm: INDArray = multiDataSet.getFeaturesMaskArray(0)
          val lastIdx: INDArray = BooleanIndexing.lastIndex(fm, Conditions.notEquals(0), 1)
          lastTimeStepPos = lastIdx.data().asInt()
        }
        for (i <- 0 until lastTimeStepPos.length) {
          labels(0).putScalar(i, goStopTokens, lastTimeStepPos(i), 1.0)
        }
        // In practice: Just need to append an extra 1 at the start (as all existing time series are now 1 step longer)
        var featureMasks: Array[INDArray] = null
        var labelsMasks: Array[INDArray] = null

        if (multiDataSet.getFeaturesMaskArray(0) != null) {//Masks are present - variable length
          featureMasks = Array.ofDim[INDArray](2)
          featureMasks(0) = multiDataSet.getFeaturesMaskArray(0)
          labelsMasks = Array.ofDim[INDArray](1)
          val newMask: INDArray = Nd4j.hstack(Nd4j.ones(inputSize, 1), multiDataSet.getFeaturesMaskArray(0))
          featureMasks(1) = newMask
          labelsMasks(0) = newMask
        } else {
          // All same length
          featureMasks = null
          labelsMasks = null
        }

        // Same for labels
        multiDataSet.setFeatures(features)
        multiDataSet.setLabels(labels)
        multiDataSet.setFeaturesMaskArrays(featureMasks)
        multiDataSet.setLabelsMaskArray(labelsMasks)
      }
    }
  }

  var dataParams : Map[String, Any] = Map(
    "url" -> "https://dl4jdata.blob.core.windows.net/datasets/aisdk_20171001.csv.zip",
    "imageHeight" -> 28,
    "imageWidth" -> 28,
    "numClass" -> 10
  )

  val hyperParams : Map[String, Any] = Map(
    "batchSize" -> 300,
    "rngSeed" -> 123,
    "numEpochs" -> 5,
    "trainSplit" -> 0.80,
    "recordThreshold" -> 7 // Used for filtering low number of sequences for an MMSI
  )

  def downloadData(sourceURL : String) : String = {
    val fileCache : File = new File(System.getProperty("user.home", "/.deeplearning4j"))
    val downloadDir : File = new File(fileCache, "/aisdk_20171001.csv")

    if(!downloadDir.exists()) {
      val tempZip : File = new File(fileCache, "aisdk_20171001.csv.zip")

      // Download data
      FileUtils.copyURLToFile(new URL(sourceURL), tempZip)

      // Unzip data
      ArchiveUtils.unzipFileTo(tempZip.getAbsolutePath, fileCache.getAbsolutePath)
      tempZip.delete()
    }
    else {
      println("File already exists!")
    }

    downloadDir.getAbsolutePath
  }


  def examineDataSequence(sourcePath : String) : Unit = {
    val sqlContext : SparkSession = SparkSession.builder()
      .master("local[*]") // launch application with #threads = #cores
      .getOrCreate()

    val rawData : DataFrame = sqlContext.read
      .format("com.databricks.spark.csv")
      .options(Map(
        "header" -> "true", // Use first line of a file as header
        "inferSchema"-> "true" // Automatically infer schema
      ))
      .load(sourcePath)

    val positions : DataFrame = rawData.withColumn("Timestamp", functions.unix_timestamp(rawData("# Timestamp"), "dd/MM/YYYY HH:mm:ss"))
      .select("Timestamp", "MMSI", "Latitude", "Longitude")

    val sequences : RDD[(Int, Seq[(Long,(Double, Double))])] = positions.rdd
      .map(row => (row.getInt(1), (row.getLong(0), (row.getDouble(3), row.getDouble(2))))) // Tuple of Ship Id to Time and Coordinates
      .groupBy(_._1) // Grouping all entries of the Ship
      .map(group => (group._1, group._2.map(pos => pos._2).toSeq.sortBy(_._1)))

    val stats = sequences
      .map { seq =>
        val timestamps : Array[Long] = seq._2.map(_._1).toArray
        Stats(seq._2.size, timestamps.min, timestamps.max, timestamps.max - timestamps.min)
      }
  }


  def prepareSequenceData(filePath : String) : Tuple2[JavaRDD[Any], JavaRDD[Any]] = {
    // NOTE : While building schema it is important to keep parsing in order of type of columns to which the schema
    // is to be applied. Column name could be anything.
    val schema : Schema = new Schema.Builder()
      .addColumnsString("Timestamp")
      .addColumnCategorical("VesselType")
      .addColumnsString("MMSI")
      .addColumnsDouble("Lat", "Lon")
      .addColumnCategorical("Status")
      .addColumnsDouble("ROT", "SOG", "COG")
      .addColumnInteger("Heading")
      .addColumnsString("IMO", "Callsign", "Name")
      .addColumnCategorical("Shiptype", "Cargotype")
      .addColumnsInteger("Width", "Length")
      .addColumnCategorical("FixingDevice")
      .addColumnDouble("Draught")
      .addColumnsString("Destination", "ETA")
      .addColumnCategorical("SourceType")
      .addColumnString("end")
      .build()

    val transform : TransformProcess = new TransformProcess.Builder(schema)
      .removeAllColumnsExceptFor("Timestamp", "MMSI", "Lat", "Lon")
      .filter(BooleanCondition.OR(
        new DoubleColumnCondition("Lat", ConditionOp.GreaterThan, 90.0),
        new DoubleColumnCondition("Lat", ConditionOp.LessThan, -90.0)
      )) // Filter out(Remove) Latitude Outliers
      .filter(BooleanCondition.OR(
        new DoubleColumnCondition("Lon", ConditionOp.GreaterThan, 180.0),
        new DoubleColumnCondition("Lon", ConditionOp.LessThan, -180.0)
      )) // Remove Longitude Outliers
      .transform(new MinMaxNormalizer("Lat",-90.0, 90.0, 0.0,1.0))
      .transform(new MinMaxNormalizer("Lon",-180.0,180.0,0.0,1.0))
      .convertToString("Lat")
      .convertToString("Lon")
      .transform(new StringToTimeTransform("Timestamp", "dd/MM/YYYY HH:mm:ss", DateTimeZone.UTC))
      .transform(new ConcatenateStringColumns("LatLon", ",", "Lat","Lon"))
      .convertToSequence("MMSI", new NumericalColumnComparator("Timestamp", true))
      .transform(
        // Removing sequence data within span of 1 hour starting from time as per Timestamp for each MMSI.
        // Thus MMSI without any sequence will be removed.
        new ReduceSequenceByWindowTransform(
          new Reducer.Builder(ReduceOp.Count)
            .keyColumns("MMSI")
            .countColumns("Timestamp")
            .customReduction("LatLon", new Reductions.GeoAveragingReduction("LatLon"))
            .takeFirstColumns("Timestamp")
            .build(),
          new TimeWindowFunction.Builder()
            .timeColumn("Timestamp")
            .windowSize(1L, TimeUnit.HOURS)
            .excludeEmptyWindows(true)
            .build()
        )
      )
      .removeAllColumnsExceptFor("LatLon")
      .build()

    val sparkContext : SparkContext = new SparkContext(new SparkConf().setMaster("local[*]"))

    val rawData : JavaRDD[util.List[Writable]] = sparkContext.textFile(filePath)
      .filter(row => !row.startsWith("# Timestamp"))
      .toJavaRDD()
      .map(new StringToWritablesFunction(new CSVRecordReader()))

    val dataRecord = SparkTransformExecutor
      .executeToSequence(rawData, transform)
      .rdd
      .filter(seq => seq.size() > hyperParams("recordThreshold").asInstanceOf[Int])
      .map{ row : util.List[util.List[Writable]] =>
        row.map{seq =>
          seq.map(_.toString)
            .map(
              _.split(",").toList.map(
                coord => new DoubleWritable(coord.toDouble).asInstanceOf[Writable]
              )
            )
            .flatten
        }
      }
      .map(_.toList.map(_.asJava).asJava)
      .toJavaRDD()

    val split = dataRecord.randomSplit(Array[Double](
      hyperParams("trainSplit").asInstanceOf[Double], 1.0-hyperParams("trainSplit").asInstanceOf[Double])
    )

    Tuple2(split(0), split(1))
  }

  // Save pre-processed data as Hadoop Map file
  def saveProcessedData(cache : File, trainSeq : JavaRDD[Any], testSeq : JavaRDD[Any]) : Unit = {
    val trainFile : File = new File(cache, "/ais_trajectories_train/")
    val testFile : File = new File(cache, "/ais_trajectories_test/")

    if(!trainFile.exists()) SparkStorageUtils.saveMapFileSequences(
      trainFile.getAbsolutePath,
      trainSeq.asInstanceOf[JavaRDD[util.List[util.List[Writable]]]]
    )
    if(!testFile.exists()) SparkStorageUtils.saveMapFileSequences(
      testFile.getAbsolutePath,
      testSeq.asInstanceOf[JavaRDD[util.List[util.List[Writable]]]]
    )
  }

  def prepareData(trainSplit : Double, batchSize : Int, numSamples : Int) : Unit = {

  }

  def prepareModelLayers(imageHeight : Int, imageWidth : Int) : Unit = {

  }

  def prepareModel(seedValue : Int, layersNN : Map[String, Any]) : Unit = {

  }


  def evaluateModel(model : MultiLayerNetwork, validationFeatures : util.ArrayList[INDArray],
                    validationLabels : util.ArrayList[INDArray]) : Unit = {

  }

  def main(args: Array[String]): Unit = {

    // Download AIS zip data
    downloadData(sourceURL = dataParams("url").toString

    // Extract Tar data
    val extractPath : String = extractData(rootPath = dataPath, sourcePath = downloadDir)


    // Prepare data ,Model layers and Multilayer Model

    // Initialise UI Server

    // Train model

    // Evaluate model

  }

}
