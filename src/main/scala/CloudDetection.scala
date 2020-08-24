import java.io.{BufferedInputStream, BufferedOutputStream, File, FileInputStream, FileOutputStream}
import java.net.URL

import org.apache.commons.compress.archivers.ArchiveEntry
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream
import org.apache.commons.io.{FileUtils, FilenameUtils}
import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.deeplearning4j.api.storage.StatsStorage
import org.deeplearning4j.datasets.datavec.RecordReaderMultiDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.graph.MergeVertex
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
import org.nd4j.evaluation.classification.ROC
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.lossfunctions.LossFunctions

object CloudDetection {

  val dataParams : Map[String, Any] = Map(
    "url" -> "https://dl4jdata.blob.core.windows.net/training/tutorials/Cloud.tar.gz",
    "numFeatures" -> Map(
      "input1" -> 3,
      "input2" -> 3,
      "input3" -> 3,
      "input4" -> 3,
      "input5" -> 3
    ),
    "trainSamples" -> 4000,
    "testSamples" -> 1000
  )
  val hyperParams : Map[String, Int] = Map(
    "miniBatchSize" -> 32,
    "rngSeed" -> 123,
    "numEpochs" -> 5
  )

  def downloadData(sourceURL : String, destinationPath : String) : (String, Boolean) = {
    val directory : File = new File(destinationPath)
    var exists : Boolean = false

    // Create new if directory does not exists
    if(!directory.exists()) {
      directory.mkdir()
      val tarFile : File = new File(destinationPath + "Cloud.tar.gz")

      // Begin Download
      FileUtils.copyURLToFile(new URL(sourceURL), tarFile)
    }
    else {
      print("File already downloaded!")
      exists = true
    }
    (destinationPath + "Cloud.tar.gz", exists)
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

    FilenameUtils.concat(rootPath, "Cloud")
  }

  def prepareData(basePath: String, numTrainSamples: Int, numValidSamples: Int, batchSize : Int) : Map[String, RecordReaderMultiDataSetIterator] = {

    // Set Base directories
    val trainBaseDir1 : String = FilenameUtils.concat(basePath, "train/n1/train.csv")
    val trainBaseDir2 : String = FilenameUtils.concat(basePath, "train/n2/train.csv")
    val trainBaseDir3 : String = FilenameUtils.concat(basePath, "train/n3/train.csv")
    val trainBaseDir4 : String = FilenameUtils.concat(basePath, "train/n4/train.csv")
    val trainBaseDir5 : String = FilenameUtils.concat(basePath, "train/n5/train.csv")

    val testBaseDir1 : String = FilenameUtils.concat(basePath, "test/n1/test.csv")
    val testBaseDir2 : String = FilenameUtils.concat(basePath, "test/n2/test.csv")
    val testBaseDir3 : String = FilenameUtils.concat(basePath, "test/n3/test.csv")
    val testBaseDir4 : String = FilenameUtils.concat(basePath, "test/n4/test.csv")
    val testBaseDir5 : String = FilenameUtils.concat(basePath, "test/n5/test.csv")

    // Training data
    val rrTrain1 : CSVRecordReader = new CSVRecordReader(1)
    rrTrain1.initialize(new FileSplit(new File(trainBaseDir1)))
    val rrTrain2 : CSVRecordReader = new CSVRecordReader(1)
    rrTrain2.initialize(new FileSplit(new File(trainBaseDir2)))
    val rrTrain3 : CSVRecordReader = new CSVRecordReader(1)
    rrTrain3.initialize(new FileSplit(new File(trainBaseDir3)))
    val rrTrain4 : CSVRecordReader = new CSVRecordReader(1)
    rrTrain4.initialize(new FileSplit(new File(trainBaseDir4)))
    val rrTrain5 : CSVRecordReader = new CSVRecordReader(1)
    rrTrain5.initialize(new FileSplit(new File(trainBaseDir5)))

    // Test data
    val rrTest1 : CSVRecordReader = new CSVRecordReader(1)
    rrTest1.initialize(new FileSplit(new File(testBaseDir1)))
    val rrTest2 : CSVRecordReader = new CSVRecordReader(1)
    rrTest2.initialize(new FileSplit(new File(testBaseDir2)))
    val rrTest3 : CSVRecordReader = new CSVRecordReader(1)
    rrTest3.initialize(new FileSplit(new File(testBaseDir3)))
    val rrTest4 : CSVRecordReader = new CSVRecordReader(1)
    rrTest4.initialize(new FileSplit(new File(testBaseDir4)))
    val rrTest5 : CSVRecordReader = new CSVRecordReader(1)
    rrTest5.initialize(new FileSplit(new File(testBaseDir5)))


    Map(
      "train" -> new RecordReaderMultiDataSetIterator.Builder(batchSize)
        .addReader("rr1", rrTrain1)
        .addReader("rr2", rrTrain2)
        .addReader("rr3", rrTrain3)
        .addReader("rr4", rrTrain4)
        .addReader("rr5", rrTrain5)
        .addInput("rr1",1,3)
        .addInput("rr2",0,2)
        .addInput("rr3",0,2)
        .addInput("rr4",0,2)
        .addInput("rr5",0,2)
        .addOutputOneHot("rr1",0,2)
        .build(),
      "test" -> new RecordReaderMultiDataSetIterator.Builder(batchSize)
        .addReader("rr1", rrTest1)
        .addReader("rr2", rrTest2)
        .addReader("rr3", rrTest3)
        .addReader("rr4", rrTest4)
        .addReader("rr5", rrTest5)
        .addInput("rr1",1,3)
        .addInput("rr2",0,2)
        .addInput("rr3",0,2)
        .addInput("rr4",0,2)
        .addInput("rr5",0,2)
        .addOutputOneHot("rr1",0,2)
        .build()
    )
  }

  def prepareModel(seedValue : Int, numFeatures : Map[String, Int]) : ComputationGraph = {
    // Model Configuration
    val denseGraphConf = new NeuralNetConfiguration.Builder()
      .seed(seedValue) // seed value is used for random initialization of learnable parameters between runs
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(new Adam())
      .graphBuilder()
      .addInputs("input1", "input2", "input3", "input4", "input5")
      .addLayer("hiddenL1", new DenseLayer.Builder()
        .nIn(numFeatures("input1"))
        .nOut(50)
        .activation(Activation.RELU)
        .weightInit(WeightInit.XAVIER)
        .build(),"input1")
      .addLayer("hiddenL2", new DenseLayer.Builder()
        .nIn(numFeatures("input2"))
        .nOut(50)
        .activation(Activation.RELU)
        .weightInit(WeightInit.XAVIER)
        .build(), "input2")
      .addLayer("hiddenL3", new DenseLayer.Builder()
        .nIn(numFeatures("input3"))
        .nOut(50)
        .activation(Activation.RELU)
        .weightInit(WeightInit.XAVIER)
        .build(), "input3")
      .addLayer("hiddenL4", new DenseLayer.Builder()
        .nIn(numFeatures("input4"))
        .nOut(50)
        .activation(Activation.RELU)
        .weightInit(WeightInit.XAVIER)
        .build(), "input4")
      .addLayer("hiddenL5", new DenseLayer.Builder()
        .nIn(numFeatures("input5"))
        .nOut(50)
        .activation(Activation.RELU)
        .weightInit(WeightInit.XAVIER)
        .build(), "input5")
      .addVertex("mergeHidden", new MergeVertex(), "hiddenL1", "hiddenL2", "hiddenL3", "hiddenL4", "hiddenL5")
      .addLayer("hiddenL6", new DenseLayer.Builder()
        .nIn(250)
        .nOut(125)
        .activation(Activation.RELU)
        .weightInit(WeightInit.XAVIER)
        .build(), "mergeHidden")
      .addLayer("output", new OutputLayer.Builder()
        .nIn(125)
        .nOut(2)
        .activation(Activation.SOFTMAX)
        .weightInit(WeightInit.XAVIER)
        .lossFunction(LossFunctions.LossFunction.MCXENT)
        .build(), "hiddenL6")
      .setOutputs("output")
      .build()

    // Build Raw Neural Net Model
    val denseGraph : ComputationGraph = new ComputationGraph(denseGraphConf)
    denseGraph.init()

    denseGraph
  }

  def main(args: Array[String]): Unit = {

    // Download Tar data
    val dataPath : String = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "Cloud/")
    val downloadDir : (String, Boolean) = downloadData(sourceURL = dataParams("url").toString, destinationPath = dataPath)

    // Extract Tar data
    val extractPath : String = extractData(rootPath = dataPath, sourcePath = downloadDir)

    // Prepare data ,Model layers and Multilayer Model
    val dataSets : Map[String, RecordReaderMultiDataSetIterator] = prepareData(basePath = extractPath,
      numTrainSamples = dataParams("trainSamples").asInstanceOf[Int], numValidSamples = dataParams("testSamples").asInstanceOf[Int],
      batchSize = hyperParams("miniBatchSize"))

    val denseGraph : ComputationGraph = prepareModel(seedValue = hyperParams("rngSeed"), numFeatures = dataParams("numFeatures").asInstanceOf[Map[String, Int]])

    // Initialise UI Server
    val uiServer : UIServer = UIServer.getInstance()
    val statsStorage : StatsStorage = new InMemoryStatsStorage()
    uiServer.attach(statsStorage)

    // Train model
    denseGraph.addListeners(new StatsListener(statsStorage), new ScoreIterationListener(20)) // returns avg value of loss function after 20 iterations
    denseGraph.fit(dataSets("train"), hyperParams("numEpochs"))

    // Evaluate model
    val roc = denseGraph.evaluateROC(dataSets("test"),100).asInstanceOf[ROC]
    println(f"TEST AUC : ${roc.calculateAUC()}%2.2f")
  }
}
