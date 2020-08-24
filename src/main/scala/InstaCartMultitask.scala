import java.net.URL
import java.io.{BufferedInputStream, BufferedOutputStream, File, FileInputStream, FileOutputStream}

import org.apache.commons.compress.archivers.ArchiveEntry
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream
import org.apache.commons.io.{FileUtils, FilenameUtils}
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader
import org.datavec.api.split.NumberedFileInputSplit
import org.deeplearning4j.api.storage.StatsStorage
import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{GradientNormalization, NeuralNetConfiguration}
import org.deeplearning4j.nn.conf.layers.{LSTM, RnnOutputLayer}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
import org.nd4j.evaluation.classification.ROC
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction

object InstaCartMultitask {
  val dataParams : Map[String, Any] = Map(
    "url" -> "https://dl4jdata.blob.core.windows.net/training/tutorials/instacart.tar.gz",
    "numFeatures" -> 134,
    "trainSamples" -> 4000,
    "testSamples" -> 1000
  )
  val hyperParams : Map[String, Int] = Map(
    "miniBatchSize" -> 32,
    "rngSeed" -> 123,
    "numEpochs" -> 8
  )

  def downloadData(sourceURL : String, destinationPath : String) : (String, Boolean) = {
    val directory : File = new File(destinationPath)
    var exists : Boolean = false

    // Create new if directory does not exists
    if(!directory.exists()) {
      directory.mkdir()
      val tarFile : File = new File(destinationPath + "instacart.tar.gz")

      // Begin Download
      FileUtils.copyURLToFile(new URL(sourceURL), tarFile)
    }
    else {
      print("File already downloaded!")
      exists = true
    }
    (destinationPath + "instacart.tar.gz", exists)
  }

  def extractData(rootPath : String, sourcePath : (String, Boolean)) : String = {
    var numFiles : Int = 0
    var numDirs : Int = 0
    val bufferSize : Int = 4096

    if(!sourcePath._2) {
      // Open tar file to Stream contents
      val tarInputStream : TarArchiveInputStream = new TarArchiveInputStream(
        new GzipCompressorInputStream(new BufferedInputStream(new FileInputStream(sourcePath._1)))
      )

      // Begin Streaming
      var tarEntry : ArchiveEntry = tarInputStream.getNextEntry
      while(tarEntry != null) {

        if(tarEntry.isDirectory) {
          val newTarDir : File = new File(rootPath + tarEntry.getName)
          newTarDir.mkdirs()
          numDirs += 1
          numFiles = 0
        }

        else {
          val fileData = new Array[scala.Byte](4 * bufferSize)

          // Open Output Stream
          val outputStream = new BufferedOutputStream(new FileOutputStream(rootPath + tarEntry.getName), bufferSize)
          var fileSize = tarInputStream.read(fileData, 0, bufferSize)

          // Read File from Input Stream to Array[Byte]
          while(fileSize != -1) {
            outputStream.write(fileData, 0, fileSize)
            fileSize = tarInputStream.read(fileData, 0, bufferSize)
          }

          // Close Output Stream
          outputStream.close()
          numFiles += 1
        }

        tarEntry = tarInputStream.getNextEntry
      }
    }

    FilenameUtils.concat(rootPath, "instacart")
  }

  def prepareData(basePath: String, numTrainSamples: Int, numValidSamples: Int, batchSize : Int) : Map[String, RecordReaderMultiDataSetIterator] = {

    // Set Base directories
    val featuresBaseDir : String = FilenameUtils.concat(basePath, "features")
    val targetABaseDir : String = FilenameUtils.concat(basePath, "breakfast")
    val targetBBaseDir : String = FilenameUtils.concat(basePath, "dairy")

    // Training data
    val trainFeatures : CSVSequenceRecordReader = new CSVSequenceRecordReader(1,",")
    trainFeatures.initialize(new NumberedFileInputSplit(featuresBaseDir + "/%d.csv", 1, numTrainSamples))
    val trainTargetA : CSVSequenceRecordReader = new CSVSequenceRecordReader(1,",")
    trainTargetA.initialize(new NumberedFileInputSplit(targetABaseDir + "/%d.csv", 1, numTrainSamples))
    val trainTargetB : CSVSequenceRecordReader = new CSVSequenceRecordReader(1,",")
    trainTargetB.initialize(new NumberedFileInputSplit(targetBBaseDir + "/%d.csv", 1, numTrainSamples))

    // Validation Data
    val validFeatures : CSVSequenceRecordReader = new CSVSequenceRecordReader(1,",")
    validFeatures.initialize(new NumberedFileInputSplit(featuresBaseDir + "/%d.csv", numTrainSamples + 1, numTrainSamples + numValidSamples))
    val validTargetA : CSVSequenceRecordReader = new CSVSequenceRecordReader(1,",")
    validTargetA.initialize(new NumberedFileInputSplit(targetABaseDir + "/%d.csv", numTrainSamples + 1, numTrainSamples + numValidSamples))
    val validTargetB : CSVSequenceRecordReader = new CSVSequenceRecordReader(1, ",")
    validTargetB.initialize(new NumberedFileInputSplit(targetBBaseDir + "/%d.csv", numTrainSamples + 1, numTrainSamples + numValidSamples))

    // Prepare data
    Map(
      "train" -> new RecordReaderMultiDataSetIterator.Builder(batchSize)
          .addSequenceReader("reader1", trainFeatures)
          .addInput("reader1")
          .addSequenceReader("reader2", trainTargetA)
          .addOutput("reader2")
          .addSequenceReader("reader3", trainTargetB)
          .addOutput("reader3")
          .sequenceAlignmentMode(RecordReaderMultiDataSetIterator.AlignmentMode.ALIGN_END)
          .build(),
      "validation" -> new RecordReaderMultiDataSetIterator.Builder(batchSize)
          .addSequenceReader("reader1", validFeatures)
          .addInput("reader1")
          .addSequenceReader("reader2", validTargetA)
          .addOutput("reader2")
          .addSequenceReader("reader3", validTargetB)
          .addOutput("reader3")
          .sequenceAlignmentMode(RecordReaderMultiDataSetIterator.AlignmentMode.ALIGN_END)
          .build()
    )
  }

  def prepareModel(seedValue : Int, numFeatures : Int) : ComputationGraph = {
    // Model Configuration
    val lstmGraphConf = new NeuralNetConfiguration.Builder()
      .seed(seedValue) // seed value is used for random initialization of learnable parameters between runs
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(new Adam())
      .weightInit(WeightInit.XAVIER)
      .dropOut(0.1)
      .graphBuilder()
      .addInputs("input")
      .addLayer("hidden", new LSTM.Builder()
        .nIn(numFeatures)
        .nOut(150)
        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
        .gradientNormalizationThreshold(10)
        .activation(Activation.TANH)
        .build(),"input")
      .addLayer("out1", new RnnOutputLayer.Builder(LossFunction.XENT)
        .nIn(150)
        .nOut(1)
        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
        .gradientNormalizationThreshold(10)
        .activation(Activation.SIGMOID)
        .build(), "hidden")
      .addLayer("out2", new RnnOutputLayer.Builder(LossFunction.XENT)
        .nIn(150)
        .nOut(1)
        .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
        .gradientNormalizationThreshold(10)
        .activation(Activation.SIGMOID)
        .build(), "hidden")
      .setOutputs("out1", "out2")
      .build()

    // Build Raw Neural Net Model
    val lstmGraph : ComputationGraph = new ComputationGraph(lstmGraphConf)
    lstmGraph.init()

    lstmGraph
  }

  def main(args: Array[String]): Unit = {

    // Download Tar data
    val dataPath : String = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "instacart/")
    val downloadDir : (String, Boolean) = downloadData(sourceURL = dataParams("url").toString, destinationPath = dataPath)

    // Extract Tar data
    val extractPath : String = extractData(rootPath = dataPath, sourcePath = downloadDir)

    // Prepare data ,Model layers and Multilayer Model
    val dataSets : Map[String, RecordReaderMultiDataSetIterator] = prepareData(basePath = extractPath,
      numTrainSamples = dataParams("trainSamples").asInstanceOf[Int], numValidSamples = dataParams("testSamples").asInstanceOf[Int],
      batchSize = hyperParams("miniBatchSize"))

    val lstmGraph : ComputationGraph = prepareModel(seedValue = hyperParams("rngSeed"), numFeatures = dataParams("numFeatures").asInstanceOf[Int])

    // Initialise UI Server
    val uiServer : UIServer = UIServer.getInstance()
    val statsStorage : StatsStorage = new InMemoryStatsStorage()
    uiServer.attach(statsStorage)

    // Train model
    lstmGraph.addListeners(new StatsListener(statsStorage), new ScoreIterationListener(20)) // returns avg value of loss function after 20 iterations
    lstmGraph.fit(dataSets("train"), hyperParams("numEpochs"))

    // Evaluate model
    val roc = new ROC()
    while(dataSets("validation").hasNext) {
      val validData = dataSets("validation").next()
      val pred = lstmGraph.output(validData.getFeatures(0))
      roc.evalTimeSeries(validData.getLabels(0), pred(0))
    }
    println(roc.calculateAUC())
  }
}
