import org.deeplearning4j.api.storage.StatsStorage
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.nd4j.linalg.learning.config.Adam
import org.nd4j.linalg.activations.Activation
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.deeplearning4j.datasets.iterator.impl.EmnistDataSetIterator
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
import org.nd4j.evaluation.classification.Evaluation
import org.nd4j.evaluation.classification.ROCMultiClass

object LogisticRegression {

  // EMNIST data (handwritten non-RGB images of digits 0-9, each of size 28 x 28 pixels):
  // 1. Flattened i.e. X dimensions [samples,features] where
  //    a. samples = 70000
  //    b. features = 28 x 28 = 784
  // 2. No. of Classes = 10(OH encoded), balanced across all samples.
  var dataParams : Map[String, Int] = Map(
    "imageHeight" -> 28,
    "imageWidth" -> 28,
    "numClass" -> 10
  )
  val hyperParams : Map[String, Int] = Map(
    "batchSize" -> 300,
    "rngSeed" -> 123,
    "numEpochs" -> 5
  )

  def prepareData(batchSize : Int) : Map[String, Any] = {
    val emnistSet = EmnistDataSetIterator.Set.MNIST
    val dataSets : Map[String, Any] = Map(
      "train" -> new EmnistDataSetIterator(emnistSet, batchSize, true),
      "validation" -> new EmnistDataSetIterator(emnistSet, batchSize, false)
    )
    dataSets
  }

  def prepareModelLayers(imageHeight : Int, imageWidth : Int, numClass : Int, numChannels : Int) : Map[String, Any] = {
    val modelLayers : Map[String, Map[String, Int]] = Map(
      "hidden" -> Map(
        "nodesInput" -> imageHeight * imageWidth, // #inputs to a neuron in the layer
        "nodesOutput" -> 500 // #neurons in the layer
      ),
      "output" -> Map(
        "nodesInput" -> 500,
        "nodesOutput" -> numClass
      )
    )
    val layersNN : Map[String, Any]= Map(
      // NOTE -
      // 1. Input layer has no activation i.e. why first layer is Hidden
      // 2. Each layer considers
      //    a. #weights = #inputs
      //    b. #activations = #outputs
      "hidden" -> new DenseLayer.Builder()
        .nIn(modelLayers("hidden")("nodesInput"))
        .nOut(modelLayers("hidden")("nodesOutput"))
        .activation(Activation.RELU) // Activation Function
        .weightInit(WeightInit.XAVIER)
        .build(),
      // NOTE - Only Output layer has the Loss function
      "output" -> new OutputLayer.Builder()
        .nIn(modelLayers("output")("nodesInput"))
        .nOut(modelLayers("output")("nodesOutput"))
        .lossFunction(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
        .activation(Activation.SOFTMAX)
        .weightInit(WeightInit.XAVIER)
        .build()
    )
    layersNN
  }

  def prepareModel(seedValue : Int, layersNN : Map[String, Any]) : MultiLayerNetwork = {
    // Model Configuration
    val configNN = new NeuralNetConfiguration.Builder()
      .seed(seedValue) // seed value is used for random initialization of learnable paramters between runs
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .updater(new Adam())
      .l2(1e-4) // Regularization L2, lambda(l2 coefficient) = 0.0001
      .list()
      .layer(layersNN("hidden").asInstanceOf[DenseLayer])
      .layer(layersNN("output").asInstanceOf[OutputLayer])
      .build()

    // Build Raw Neural Net Model
    val neuralNetModel = new MultiLayerNetwork(configNN)
    neuralNetModel.init() // Initializes all learnable parameters

    neuralNetModel
  }

  def main(args: Array[String]): Unit = {
    // Prepare data ,Model layers and Multilayer Model
    val dataSets : Map[String, Any] = prepareData(batchSize = hyperParams("batchSize"))
    val modelLayersNN : Map[String, Any] = prepareModelLayers(imageHeight = dataParams("imageHeight"),
      imageWidth = dataParams("imageWidth"), numClass = dataParams("numClass"))
    val multiLayerNN : MultiLayerNetwork = prepareModel(seedValue = hyperParams("rngSeed"), layersNN = modelLayersNN)

    // Initialise UI Server
    val uiServer : UIServer = UIServer.getInstance()
    val statsStorage : StatsStorage = new InMemoryStatsStorage()
    uiServer.attach(statsStorage)

    // Train model
    multiLayerNN.addListeners(new StatsListener(statsStorage), new ScoreIterationListener(50)) // returns avg value of loss function after 100 iterations
    multiLayerNN.fit(dataSets("train").asInstanceOf[EmnistDataSetIterator], hyperParams("numEpochs"))

    // Evaluate model
    // 1. Basic Evaluation
    val evalBasic = multiLayerNN.evaluate(
      dataSets("validation").asInstanceOf[EmnistDataSetIterator]).asInstanceOf[Evaluation]
    println(s"1. Accuracy - ${evalBasic.accuracy()*100} %")
    println(s"2. Precision - ${evalBasic.precision()*100} %")
    println(s"3. Recall - ${evalBasic.recall()*100} %")
    // 2. Other Evaluation
    val evalROC = multiLayerNN.evaluateROCMultiClass(
      dataSets("validation").asInstanceOf[EmnistDataSetIterator],0).asInstanceOf[ROCMultiClass]
    println(s"4. AUC - ${evalROC.calculateAUC(0)}")
    // 3. All Stats
    println(s"5. Basic Statistics - \n${evalBasic.stats()}")
    println(s"6. ROC Metrics - \n${evalROC.stats()}")
  }
}
