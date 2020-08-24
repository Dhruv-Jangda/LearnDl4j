import java.net.URL
import java.io.{BufferedInputStream, BufferedOutputStream, File, FileInputStream, FileOutputStream}
import org.apache.commons.compress.archivers.ArchiveEntry
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader
import org.datavec.api.split.NumberedFileInputSplit
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{LSTM, RnnOutputLayer}
import org.deeplearning4j.nn.graph.ComputationGraph
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.nn.weights.WeightInit
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.learning.config.Adam
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
import org.apache.commons.io.{FileUtils, FilenameUtils}
import org.deeplearning4j.api.storage.StatsStorage
import org.nd4j.evaluation.classification.ROC
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet


object RecurrentNeuralNetwork {

  val dataParams : Map[String, Any] = Map(
    "url" -> "https://dl4jdata.blob.core.windows.net/training/physionet2012/physionet2012.tar.gz",
    "numFeatures" -> 86,
    "trainSamples" -> 3200,
    "testSamples" -> 800,
    "numClass" -> 2
  )
  val hyperParams : Map[String, Int] = Map(
    "miniBatchSize" -> 32,
    "rngSeed" -> 123,
    "numEpochs" -> 5
  )

  def downloadData(sourceURL : String, destinationPath : String) : String = {
    val directory : File = new File(destinationPath)

    // Create new if directory does not exists
    if(!directory.exists()) {
      directory.mkdir()
      val tarFile : File = new File(destinationPath + "physionet2012.tar.gz")

      // Begin Download
      FileUtils.copyURLToFile(new URL(sourceURL), tarFile)
    }
    else {
      print("File already downloaded!")
    }
    destinationPath + "physionet2012.tar.gz"
  }

  def extractData(rootPath : String, sourcePath : String) : String = {
    var numFiles : Int = 0
    var numDirs : Int = 0
    val bufferSize : Int = 4096

    // Open tar file to Stream contents
    val tarInputStream : TarArchiveInputStream = new TarArchiveInputStream(
      new GzipCompressorInputStream(new BufferedInputStream(new FileInputStream(sourcePath)))
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
    FilenameUtils.concat(rootPath, "physionet2012")
  }


  def prepareData(basePath: String, numTrainSamples: Int, numValidationSamples: Int, miniBatchSize : Int, numLabels : Int) : Map[String, DataSetIterator] = {

    // Set Base directories
    val samplesBaseDir : String = FilenameUtils.concat(basePath, "sequence")
    val labelBaseDir : String = FilenameUtils.concat(basePath, "mortality")

    // Load data
    val trainSamples : CSVSequenceRecordReader = new CSVSequenceRecordReader(1,",")
    trainSamples.initialize(new NumberedFileInputSplit(samplesBaseDir + "/%d.csv", 0, numTrainSamples - 1))
    val trainLabels : CSVSequenceRecordReader = new CSVSequenceRecordReader()
    trainLabels.initialize(new NumberedFileInputSplit(labelBaseDir + "/%d.csv", 0, numTrainSamples - 1))
    val validationSamples : CSVSequenceRecordReader = new CSVSequenceRecordReader(1,",")
    validationSamples.initialize(new NumberedFileInputSplit(samplesBaseDir + "/%d.csv", numTrainSamples, numTrainSamples + numValidationSamples - 1))
    val validationLabels : CSVSequenceRecordReader = new CSVSequenceRecordReader()
    validationLabels.initialize(new NumberedFileInputSplit(labelBaseDir + "/%d.csv", numTrainSamples, numTrainSamples + numValidationSamples - 1))

    // Prepare data
    Map(
      "train" -> new SequenceRecordReaderDataSetIterator(
        trainSamples, trainLabels, miniBatchSize, numLabels,false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END
      ),
      "validation" -> new SequenceRecordReaderDataSetIterator(
        validationSamples, validationLabels, miniBatchSize, numLabels,false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END
      )
    )
  }

  def prepareModelLayers(numFeatures : Int, numClass : Int) : Map[String, Any] = {
    val modelLayers : Map[String, Map[String, Int]] = Map(
      "hidden" -> Map(
        "nodesInput" -> numFeatures, // #inputs to a neuron in the layer
        "nodesOutput" -> 200 // #neurons in the layer
      ),
      "output" -> Map(
        "nodesInput" -> 200,
        "nodesOutput" -> numClass
      )
    )
    val layersNN : Map[String, Any]= Map(
      // NOTE -
      // 1. Input layer has no activation i.e. why first layer is Hidden
      // 2. Each layer considers
      //    a. #weights = #inputs
      //    b. #activations = #outputs
      "hidden" -> new LSTM.Builder()
          .nIn(modelLayers("hidden")("nodesInput"))
          .nOut(modelLayers("hidden")("nodesOutput"))
          .activation(Activation.TANH) // Activation Function
          .weightInit(WeightInit.XAVIER)
          .build(),
      // NOTE - Only Output layer has the Loss function
      "output" -> new RnnOutputLayer.Builder(LossFunctions.LossFunction.XENT)
          .nIn(modelLayers("output")("nodesInput"))
          .nOut(modelLayers("output")("nodesOutput"))
          .activation(Activation.SIGMOID)
          .weightInit(WeightInit.XAVIER)
          .build()
      )

    layersNN
  }

  def prepareModel(seedValue : Int, layersNN : Map[String, Any]) : ComputationGraph = {
    // Model Configuration
    val configNN = new NeuralNetConfiguration.Builder()
      .seed(seedValue) // seed value is used for random initialization of learnable paramters between runs
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(new Adam(0.0005))
      .l2(1e-4) // Regularization L2, lambda(l2 coefficient) = 0.0001
      .dropOut(0.15)
      .graphBuilder()
      .addInputs("trainFeatures")
      .addLayer("hidden", layersNN("hidden").asInstanceOf[LSTM], "trainFeatures")
      .addLayer("output", layersNN("output").asInstanceOf[RnnOutputLayer], "hidden")
      .setOutputs("output")
      .build()

    // Build Raw Neural Net Model
    val neuralNetModel = new ComputationGraph(configNN)
    neuralNetModel.init() // Initializes all learnable parameters

    neuralNetModel
  }

  def main(args: Array[String]): Unit = {

    // Download Tar data
    val dataPath : String = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_physionet/")
    val downloadDir : String = downloadData(sourceURL = dataParams("url").toString, destinationPath = dataPath)

    // Extract Tar data
    val extractPath : String = extractData(rootPath = dataPath, sourcePath = downloadDir)

    // Prepare data ,Model layers and Multilayer Model
    val dataSets : Map[String, DataSetIterator] = prepareData(basePath = extractPath,
      numTrainSamples = dataParams("trainSamples").asInstanceOf[Int], numValidationSamples = dataParams("testSamples").asInstanceOf[Int],
      miniBatchSize = hyperParams("miniBatchSize"), numLabels = dataParams("numClass").asInstanceOf[Int])

    val modelLayersNN : Map[String, Any] = prepareModelLayers(
      numFeatures = dataParams("numFeatures").asInstanceOf[Int],
      numClass = dataParams("numClass").asInstanceOf[Int]
    )
    val multiLayerNN : ComputationGraph = prepareModel(seedValue = hyperParams("rngSeed"), layersNN = modelLayersNN)

    // Initialise UI Server
    val uiServer : UIServer = UIServer.getInstance()
    val statsStorage : StatsStorage = new InMemoryStatsStorage()
    uiServer.attach(statsStorage)

    // Train model
    multiLayerNN.addListeners(new StatsListener(statsStorage), new ScoreIterationListener(20)) // returns avg value of loss function after 100 iterations
    multiLayerNN.fit(dataSets("train").asInstanceOf[SequenceRecordReaderDataSetIterator], hyperParams("numEpochs"))

    // Evaluate model
    var numSample : Int = 0
    val rocScore : ROC = new ROC(100)
    while(dataSets("validation").hasNext) {
      numSample += 1
      val batch : DataSet = dataSets("validation").next()
      val output : Array[INDArray] = multiLayerNN.output(batch.getFeatures)
      rocScore.evalTimeSeries(batch.getLabels, output(0))
      print(f"Validation Sample $numSample%d : AUC = ${rocScore.calculateAUC()}%2.2f\n")
    }
  }

}
