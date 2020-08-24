import java.util
import java.util.{Collections, Comparator}
import org.nd4j.linalg.factory.Nd4j
import org.apache.commons.lang3.tuple.{ImmutablePair, Pair}
import org.deeplearning4j.api.storage.StatsStorage
import org.nd4j.linalg.activations.Activation
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.NeuralNetConfiguration
import org.nd4j.linalg.learning.config.AdaGrad
import org.deeplearning4j.nn.conf.layers.DenseLayer
import org.deeplearning4j.nn.conf.layers.OutputLayer
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.optimize.listeners.ScoreIterationListener
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.lossfunctions.LossFunctions

object AutoEncoder {

  var dataParams : Map[String, Int] = Map(
    "imageHeight" -> 28,
    "imageWidth" -> 28,
    "numClass" -> 10,
    "numSamples" -> 50000
  )
  val hyperParams : Map[String, Any] = Map(
    "batchSize" -> 300,
    "rngSeed" -> 123,
    "numEpochs" -> 5,
    "trainSplit" -> 0.80
  )

  def prepareData(trainSplit : Double, batchSize : Int, numSamples : Int) : Map[String, util.ArrayList[INDArray]] = {
    val mnistSet : MnistDataSetIterator = new MnistDataSetIterator(batchSize, numSamples, false)
    val dataSets : Map[String, util.ArrayList[INDArray]] = Map(
      "trainFeatures" -> new util.ArrayList[INDArray],
      "validationFeatures" -> new util.ArrayList[INDArray],
      "validationLabels" -> new util.ArrayList[INDArray]
    )
    while(mnistSet.hasNext) {
      val split = mnistSet.next().splitTestAndTrain(trainSplit) // perform split as : trainSplit = train/validation
      dataSets("trainFeatures").add(split.getTrain.getFeatures)
      val validationSplit : DataSet = split.getTest
      dataSets("validationFeatures").add(validationSplit.getFeatures)
      dataSets("validationLabels").add(Nd4j.argMax(validationSplit.getLabels,1))
    }
    dataSets
  }

  def prepareModelLayers(imageHeight : Int, imageWidth : Int) : Map[String, Any] = {
    val modelLayers : Map[String, Map[String, Int]] = Map(
      "hidden 1" -> Map(
        "nodesInput" -> imageHeight * imageWidth, // #inputs to a neuron in the layer
        "nodesOutput" -> 250 // #neurons in the layer
      ),
      "hidden 2" -> Map(
        "nodesInput" -> 250, // #inputs to a neuron in the layer
        "nodesOutput" -> 10 // #neurons in the layer
      ),
      "hidden 3" -> Map(
        "nodesInput" -> 10, // #inputs to a neuron in the layer
        "nodesOutput" -> 250 // #neurons in the layer
      ),
      "output" -> Map(
        "nodesInput" -> 250,
        "nodesOutput" -> imageHeight * imageWidth
      )
    )
    val layersNN : Map[String, Any]= Map(
      // NOTE -
      // 1. Input layer has no activation i.e. why first layer is Hidden
      // 2. Each layer considers
      //    a. #weights = #inputs
      //    b. #activations = #outputs
      "hidden 1" -> new DenseLayer.Builder()
        .nIn(modelLayers("hidden 1")("nodesInput"))
        .nOut(modelLayers("hidden 1")("nodesOutput"))
        .build(),
      "hidden 2" -> new DenseLayer.Builder()
        .nIn(modelLayers("hidden 2")("nodesInput"))
        .nOut(modelLayers("hidden 2")("nodesOutput"))
        .build(),
      "hidden 3" -> new DenseLayer.Builder()
        .nIn(modelLayers("hidden 3")("nodesInput"))
        .nOut(modelLayers("hidden 3")("nodesOutput"))
        .build(),
      "output" -> new OutputLayer.Builder()
        .nIn(modelLayers("output")("nodesInput"))
        .nOut(modelLayers("output")("nodesOutput"))
        .lossFunction(LossFunctions.LossFunction.MSE)
        .build()
    )
    layersNN
  }

  def prepareModel(seedValue : Int, layersNN : Map[String, Any]) : MultiLayerNetwork = {
    // Model Configuration
    val configNN = new NeuralNetConfiguration.Builder()
      .seed(seedValue) // seed value is used for random initialization of learnable paramters between runs
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .activation(Activation.RELU)
      .updater(new AdaGrad(0.05))
      .l2(1e-4) // Regularization L2, lambda(l2 coefficient) = 0.0001
      .list()
      .layer(0, layersNN("hidden 1").asInstanceOf[DenseLayer])
      .layer(1, layersNN("hidden 2").asInstanceOf[DenseLayer])
      .layer(2, layersNN("hidden 3").asInstanceOf[DenseLayer])
      .layer(3, layersNN("output").asInstanceOf[OutputLayer])
      .build()

    // Build Raw Neural Net Model
    val neuralNetModel = new MultiLayerNetwork(configNN)
    neuralNetModel.init() // Initializes all learnable parameters

    neuralNetModel
  }


  def evaluateModel(model : MultiLayerNetwork, validationFeatures : util.ArrayList[INDArray],
                    validationLabels : util.ArrayList[INDArray]) : (util.ArrayList[Pair[Integer, Double]], util.ArrayList[Pair[Integer, Double]]) = {

    // Map to store label wise samples score and sample in Validation Data ( For Anomaly detection )
    val listsByDigit = new util.HashMap[Integer, util.ArrayList[Pair[Double, INDArray]]]
    (0 to 9).foreach{label => listsByDigit.put(label, new util.ArrayList[Pair[Double, INDArray]])}

    // Visit each Validation set(split out from Train)
    (0 to validationFeatures.size()).foreach{ i =>
      val validationSample : INDArray = validationFeatures.get(i)
      val validationLabel : INDArray = validationLabels.get(i)

      // Add (MSE score, features) for each Validation sample
      (0 until validationSample.rows()).foreach{ j =>
        val score : Double = model.score(new DataSet(validationSample.getRow(j,true), validationSample.getRow(j, true)))
        listsByDigit.get(validationLabel.getInt(j)).add(
          new ImmutablePair[Double, INDArray](score, validationSample.getRow(j))
        )
      }
    }

    // Sort each set by MSE score
    val validationComparator = new Comparator[Pair[Double, INDArray]] {
      override def compare(o1: Pair[Double, INDArray], o2: Pair[Double, INDArray]): Int =
        java.lang.Double.compare(o1.getLeft, o2.getLeft)
    }
    listsByDigit.values().forEach(validationPair => Collections.sort(validationPair, validationComparator))

    // Rank by N best/worst cases. 5 best/worst for each class
    val best = new util.ArrayList[Pair[Integer, Double]](50)
    val worst = new util.ArrayList[Pair[Integer, Double]](50)

    (0 to 9).foreach{label =>
      val arrayList : util.ArrayList[Pair[Double, INDArray]] = listsByDigit.get(label)

      (0 to 4).foreach{j =>
        best.add(new ImmutablePair[Integer, Double](label, arrayList.get(j).getLeft))
        worst.add(new ImmutablePair[Integer, Double](label, arrayList.get(arrayList.size()-j-1).getLeft))
      }
    }

    (best, worst)
  }

  def main(args: Array[String]): Unit = {
    // Prepare data ,Model layers and Multilayer Model
    val dataSets : Map[String, util.ArrayList[INDArray]] = prepareData(
      trainSplit = hyperParams("trainSplit").asInstanceOf[Double],
      batchSize = hyperParams("batchSize").asInstanceOf[Int],
      numSamples = dataParams("numSamples")
    )
    val modelLayersNN : Map[String, Any] = prepareModelLayers(imageHeight = dataParams("imageHeight"),
      imageWidth = dataParams("imageWidth"))
    val multiLayerNN : MultiLayerNetwork = prepareModel(
      seedValue = hyperParams("rngSeed").asInstanceOf[Int], layersNN = modelLayersNN)

    // Initialise UI Server
    val uiServer : UIServer = UIServer.getInstance()
    val statsStorage : StatsStorage = new InMemoryStatsStorage()
    uiServer.attach(statsStorage)

    // Train model
    multiLayerNN.addListeners(new StatsListener(statsStorage), new ScoreIterationListener(50)) // returns avg value of loss function after 100 iterations
    for(epoch <- 1 to hyperParams("numEpochs").asInstanceOf[Int]) {
      dataSets("trainFeatures").forEach( set => multiLayerNN.fit(set, set))
      println(f"Epoch $epoch%d complete\n")
    }

    // Evaluate model
    val ratedSamples = evaluateModel(model = multiLayerNN, validationFeatures = dataSets("validationFeatures"),
      validationLabels = dataSets("validationLabels"))

    println("Best Samples MSE")
    (0 to ratedSamples._1.size()).foreach{k =>
      println(f"Sample $k%d : Label - ${ratedSamples._1.get(k).getLeft}%d, Score - ${ratedSamples._1.get(k).getRight}%2.2f")
    }
    println("\nWorst Samples MSE")
    (0 to ratedSamples._2.size()).foreach{k =>
      println(f"Sample $k%d : Label - ${ratedSamples._2.get(k).getLeft}%d, Score - ${ratedSamples._2.get(k).getRight}%2.2f")
    }
  }
}
