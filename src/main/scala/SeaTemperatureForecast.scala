import java.net.URL
import org.apache.commons.io.{FileUtils, FilenameUtils}
import org.apache.commons.compress.archivers.ArchiveEntry
import org.apache.commons.compress.archivers.tar.TarArchiveInputStream
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream
import java.io.{BufferedInputStream, BufferedOutputStream, File, FileInputStream, FileOutputStream}
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader
import org.datavec.api.split.NumberedFileInputSplit
import org.deeplearning4j.api.storage.StatsStorage
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.{GradientNormalization, NeuralNetConfiguration}
import org.deeplearning4j.nn.conf.layers.{ConvolutionLayer, LSTM, RnnOutputLayer, SubsamplingLayer}
import org.deeplearning4j.nn.conf.preprocessor.{CnnToRnnPreProcessor, RnnToCnnPreProcessor}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
import org.nd4j.evaluation.regression.RegressionEvaluation
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.api.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.NDArrayIndex
import org.nd4j.linalg.learning.config.AdaGrad
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction


object SeaTemperatureForecast {

  /*
    Data is sequential in nature with each example of 50 time steps and flattened image of 52 pixels.
    Idea :
    Image analysis implies Convolutional Layer, hence 52 pixels = 13 x 4 pixels and flattened => 1 channel
    Time series analysis implies Recurrent layer.

    NOTE - Due to CNN & RNN involved, CnnToRnn & RnnToCnn pre-processors will be used.The RnnToCnnPreProcessor reshapes
    3d o/p(as i/p to CNN) of RNN [batch size, #channels x height x width of grid, time series length] to 4d [#examples x
    time series length ,#channels, width, height] which is suitable as input CNN layer. CnnToRnnPreProcessor convert this 4d back to 3d.
  */

  val dataParams : Map[String, Any] = Map(
    "url" -> "https://dl4jdata.blob.core.windows.net/training/seatemp/sea_temp.tar.gz",
    "imageHt" -> 13,
    "imageWt" -> 4,
    "trainSamples" -> 1600,
    "testSamples" -> 136
  )
  val hyperParams : Map[String, Int] = Map(
    "miniBatchSize" -> 32,
    "rngSeed" -> 123,
    "numEpochs" -> 15,
    "kernelSize" -> 2,
    "numChannels" -> 1
  )

  def downloadData(sourceURL : String, destinationPath : String) : (String, Boolean) = {
    val directory : File = new File(destinationPath)
    var exists : Boolean = false

    // Create new if directory does not exists
    if(!directory.exists()) {
      directory.mkdir()
      val tarFile : File = new File(destinationPath + "sea_temp.tar.gz")

      // Begin Download
      FileUtils.copyURLToFile(new URL(sourceURL), tarFile)
    }
    else {
      print("File already downloaded!")
      exists = true
    }
    (destinationPath + "sea_temp.tar.gz", exists)
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

    FilenameUtils.concat(rootPath, "sea_temp")
  }

  def prepareData(basePath: String, numTrainSamples: Int, numValidationSamples: Int, miniBatchSize : Int) : Map[String, DataSetIterator] = {

    // Set Base directories
    val featuresBaseDir : String = FilenameUtils.concat(basePath, "features")
    val targetsBaseDir : String = FilenameUtils.concat(basePath, "targets")
    val futuresBaseDir : String = FilenameUtils.concat(basePath, "futures")

    // Training data
    val trainFeatures : CSVSequenceRecordReader = new CSVSequenceRecordReader(1,",")
    trainFeatures.initialize(new NumberedFileInputSplit(featuresBaseDir + "/%d.csv", 1, numTrainSamples))
    val trainTargets : CSVSequenceRecordReader = new CSVSequenceRecordReader(1,",")
    trainTargets.initialize(new NumberedFileInputSplit(targetsBaseDir + "/%d.csv", 1, numTrainSamples))

    // Validation Data
    val validationFeatures : CSVSequenceRecordReader = new CSVSequenceRecordReader(1,",")
    validationFeatures.initialize(new NumberedFileInputSplit(featuresBaseDir + "/%d.csv", numTrainSamples + 1, numTrainSamples + numValidationSamples))
    val validationTargets : CSVSequenceRecordReader = new CSVSequenceRecordReader(1,",")
    validationTargets.initialize(new NumberedFileInputSplit(targetsBaseDir + "/%d.csv", numTrainSamples + 1, numTrainSamples + numValidationSamples))

    // Future Data
    val futureFeatures : CSVSequenceRecordReader = new CSVSequenceRecordReader(1,",")
    futureFeatures.initialize(new NumberedFileInputSplit(futuresBaseDir + "/%d.csv", numTrainSamples + 1, numTrainSamples + numValidationSamples))
    val futureLabels : CSVSequenceRecordReader = new CSVSequenceRecordReader(1,",")
    futureLabels.initialize(new NumberedFileInputSplit(futuresBaseDir + "/%d.csv", numTrainSamples + 1, numTrainSamples + numValidationSamples))

    // Prepare data
    Map(
      "train" -> new SequenceRecordReaderDataSetIterator(
        trainFeatures, trainTargets, miniBatchSize, 10,true, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END
      ), // Due to regression, numPossibleLabels can be any number
      "validation" -> new SequenceRecordReaderDataSetIterator(
        validationFeatures, validationTargets, miniBatchSize, 10,true, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END
      ),
      "future" -> new SequenceRecordReaderDataSetIterator(
        futureFeatures, futureLabels, miniBatchSize,10,true, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END
      )
    )
  }


  def prepareModel(seedValue : Int) : MultiLayerNetwork = {
    // Model Configuration
    val convLSTM = new NeuralNetConfiguration.Builder()
      .seed(seedValue) // seed value is used for random initialization of learnable paramters between runs
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(new AdaGrad(0.0005))
      .weightInit(WeightInit.XAVIER)
      .list()
      .layer(0, new ConvolutionLayer.Builder(hyperParams("kernelSize"), hyperParams("kernelSize"))
          .nIn(hyperParams("numChannels")) // IN : image size [13,4,1]
          .nOut(7)
          .stride(2,2)
          .activation(Activation.RELU)
          .build()) // 1 + (#rows + 2p - f)/s => OUT : image size [6,2,7]
      .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
          .kernelSize(hyperParams("kernelSize"), hyperParams("kernelSize")) // IN : image size [6,2,7]
          .stride(2,2)
          .build()) // OUT : image size [3,1,7]
      .layer(2, new LSTM.Builder()
          .nIn(21) // Flattened out pixels = 3 x 1 x 7
          .nOut(100)
          .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
          .gradientNormalizationThreshold(10)
          .build())
      .layer(3, new RnnOutputLayer.Builder(LossFunction.MSE)
          .activation(Activation.IDENTITY)
          .nIn(100)
          .nOut(52)
          .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
          .gradientNormalizationThreshold(10)
          .build())
      // RnnToCnnreProcessor serves a purpose of de-flattening while CnnToRnnPreprocessor serves flattening
      .inputPreProcessor(0, new RnnToCnnPreProcessor(dataParams("imageHt").asInstanceOf[Int],
        dataParams("imageWt").asInstanceOf[Int], hyperParams("numChannels")))
      .inputPreProcessor(2, new CnnToRnnPreProcessor(3,1,7))
      .build()

    // Build Raw Neural Net Model
    val neuralNetModel = new MultiLayerNetwork(convLSTM)
    neuralNetModel.init() // Initializes all learnable parameters

    neuralNetModel
  }

  def main(args: Array[String]): Unit = {

    // Download Tar data
    val dataPath : String = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "dl4j_seas/")
    val downloadDir : (String, Boolean) = downloadData(sourceURL = dataParams("url").toString, destinationPath = dataPath)

    // Extract Tar data
    val extractPath : String = extractData(rootPath = dataPath, sourcePath = downloadDir)

    // Prepare data ,Model layers and Multilayer Model
    val dataSets : Map[String, DataSetIterator] = prepareData(basePath = extractPath,
      numTrainSamples = dataParams("trainSamples").asInstanceOf[Int], numValidationSamples = dataParams("testSamples").asInstanceOf[Int],
      miniBatchSize = hyperParams("miniBatchSize"))

    val convLSTMNetwork : MultiLayerNetwork = prepareModel(seedValue = hyperParams("rngSeed"))

    // Initialise UI Server
    val uiServer : UIServer = UIServer.getInstance()
    val statsStorage : StatsStorage = new InMemoryStatsStorage()
    uiServer.attach(statsStorage)

    // Train model
    // For an example : features[t] when fed to model outputs preds[t] equivalent to features[t+1] or ytrue[t] and so compared with ytrue[t]
    convLSTMNetwork.addListeners(new StatsListener(statsStorage), new ScoreIterationListener(20)) // returns avg value of loss function after 20 iterations
    convLSTMNetwork.fit(dataSets("train").asInstanceOf[SequenceRecordReaderDataSetIterator], hyperParams("numEpochs"))

    // Evaluate model
    val regressEval : RegressionEvaluation = new RegressionEvaluation()
    while(dataSets("validation").hasNext) {
      val nextValidation : DataSet = dataSets("validation").next()
      val validationFeatures : INDArray = nextValidation.getFeatures

      var pred : INDArray = Nd4j.zeros(1L,2L) // Initialize to anything
      (0 to 49).foreach {i =>
        pred = convLSTMNetwork.rnnTimeStep(validationFeatures.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(i,i+1)))
        // pred.shape() = Array[Long](32,52,1) : #rows = 1, #columns = 52, batchSize = 32 => Each row is taken as time step.
      }

      val correctTemp : DataSet = dataSets("future").next()
      val correctFeatures : INDArray = correctTemp.getFeatures

      (0 to 9).foreach {j =>
        // pred[j-1] will be used as feature[j], which outputs pred[j] & compared with correctFeatures[j]
        pred = convLSTMNetwork.rnnTimeStep(pred)
        regressEval.evalTimeSeries(pred, correctFeatures.get(NDArrayIndex.all(), NDArrayIndex.all(), NDArrayIndex.interval(j, j+1)))
      }

      convLSTMNetwork.rnnClearPreviousState()
    }

    println(regressEval.stats())
  }
}
