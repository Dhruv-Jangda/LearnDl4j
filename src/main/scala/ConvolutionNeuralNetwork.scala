import org.datavec.image.loader.LFWLoader
import org.deeplearning4j.api.storage.StatsStorage
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.storage.InMemoryStatsStorage
import org.nd4j.linalg.activations.Activation

object ConvolutionNeuralNetwork {

  var dataParams : Map[String, Int] = Map(
    "imageHeight" -> 96,
    "imageWidth" -> 96,
    "numChannels" -> 3,
    "numSamples" -> LFWLoader.NUM_IMAGES,
    "numClass" -> LFWLoader.NUM_LABELS
  )
  val hyperParams : Map[String, Any] = Map(
    "batchSize" -> 48,
    "rngSeed" -> 123,
    "numEpochs" -> 5,
    "trainTestSplit" -> 1.0,
    "iterations" -> 1,
    "embeddingSize" -> 128,
    "activation" -> Activation.RELU,
    "embeddingSize" -> 128
  )

  def prepareData(batchSize : Int) : Unit = {

  }

  def prepareModelLayers(imageHeight : Int, imageWidth : Int, numClass : Int, numChannels : Int, embeddingSize : Int) : Map[String, Map[String, Any]] = {
    val modelLayers : Map[String, Map[String, Any]] = Map(
      // Stem Layer
      "stemCNN1" -> Map(
        "kernelSize" -> Array[Int](7,7), // Filter Matrix of [7,7] for convolution
        "strideLength" -> Array[Int](2,2), // Stride Length of 2 across rows and 2 across columns
        "padding" -> Array[Int](3,3), // Pad 3 rows of 0 and 3 columns of 0 before Convolution
        "channelsInput" -> numChannels, // Generally #inputs = # features to First layer and in CNN #features = #channels
        "channelsOutput" -> 64 // #channels at output
      ),
      "stemBatch1" -> Map(
        "channelsInput" -> 64,
        "channelsOutput" -> 64
      ),
      "stemPool1" -> Map(
        "kernelSize" -> Array[Int](3,3),
        "strideLength" -> Array[Int](2,2),
        "padding" -> Array[Int](1,1)
      ),
      // LRN - Local Response Normalization, non-trainable layer with (k, alpha, beta, n) as hyper-parameters.
      // k is used to avoid any singularities (division by zero),
      // alpha is used as a normalization constant, beta is contrast constant
      // n is neighborhood length i.e. how many consecutive pixel values to consider for normalization.
      // The case of (k, alpha, beta, n) = (0,1,1,N) is the standard normalization
      "stemLRN1" -> Map(
        "k" -> 1,
        "n" -> 5,
        "alpha" -> 1e-4,
        "beta" -> 0.75
      ),
      // Inception Layer
      "inception2_CNN1" ->  Map(
        "kernelSize" -> Array[Int](1,1),
        "channelsInput" -> 64,
        "channelsOutput"-> 64
      ),
      "inception2_Batch1" ->  Map(
        "channelsInput" -> 64,
        "channelsOutput"-> 64
      ),
      "inception2_CNN2" ->  Map(
        "kernelSize" -> Array[Int](3,3),
        "strideLength" -> Array[Int](1,1),
        "padding" -> Array[Int](1,1),
        "channelsInput" -> 64,
        "channelsOutput"-> 192
      ),
      "inception2_Batch1" ->  Map(
        "channelsInput" -> 192,
        "channelsOutput"-> 192
      ),
      "inception2_LRN1" -> Map(
        "k" -> 1,
        "n" -> 5,
        "alpha" -> 1e-4,
        "beta" -> 0.75
      ),
      "inception2_Pool" -> Map(
        "kernelSize" -> Array[Int](3,3),
        "strideLength" -> Array[Int](2,2),
        "padding" -> Array[Int](1,1)
      ),
      // NOTE - The above pattern of layer can be directly made by Model Zoo - FaceNetHelper
      // Inception 3a Layer
      "inception3A" -> Map(
        "inputChannels" -> 192,
        "kernelSize" -> Array[Int](3,5),
        "kernelStride" -> Array[Int](1,1),
        "outputSize" -> Array[Int](128,32),
        "reduceSize" -> Array[Int](96,16,32,64)
      ),
      // Inception 3b Layer
      "inception3B" -> Map(
        "inputChannels" -> 256,
        "kernelSize" -> Array[Int](3,5),
        "kernelStride" -> Array[Int](1,1),
        "outputSize" -> Array[Int](128,32),
        "reduceSize" -> Array[Int](96,32,64,64),
        "poolSize" -> 2
      ),
      // Inception 3c Layer
      "inception3C_CNN1" -> Map(
        "kernelSize" -> Array[Int](1,1),
        "strideLength" -> Array[Int](1,1),
        "channelsInput" -> 320,
        "channelsOutput" -> 128
      ),
      "inception3C_Batch1" ->  Map(
        "channelsInput" -> 128,
        "channelsOutput"-> 128
      ),
      "inception3C_CNN2" -> Map(
        "kernelSize" -> Array[Int](3,3),
        "strideLength" -> Array[Int](2,2),
        "channelsInput" -> 128,
        "channelsOutput" -> 256
      ),
      "inception3C_Batch2" -> Map(
        "channelsInput" -> 256,
        "channelsOutput"-> 256
      ),
      "inception3C_CNN3" -> Map(
        "kernelSize" -> Array[Int](1,1),
        "strideLength" -> Array[Int](1,1),
        "channelsInput" -> 320,
        "channelsOutput" -> 32
      ),
      "inception3C_Batch3" -> Map(
        "channelsInput" -> 32,
        "channelsOutput"-> 32
      ),
      "inception3C_CNN4" -> Map(
        "kernelSize" -> Array[Int](3,3),
        "strideLength" -> Array[Int](2,2),
        "channelsInput" -> 32,
        "channelsOutput" -> 64
      ),
      "inception3C_Batch4" -> Map(
        "channelsInput" -> 64,
        "channelsOutput"-> 64
      ),
      "inception3C_Pool" -> Map(
        "kernelSize" -> Array[Int](3,3),
        "strideLength" -> Array[Int](2,2),
        "padding" -> Array[Int](1,1)
      ),
      // Inception 4a
      "inception4A" -> Map(
        "inputChannels" -> 640,
        "kernelSize" -> Array[Int](3,5),
        "kernelStride" -> Array[Int](1,1),
        "outputSize" -> Array[Int](192,64),
        "reduceSize" -> Array[Int](96,32,128,256),
        "poolSize" -> 2
      ),
      // Inception 4e
      "inception4E_CNN1"-> Map(
        "kernelSize" -> Array[Int](1,1),
        "strideLength" -> Array[Int](1,1),
        "channelsInput" -> 640,
        "channelsOutput" -> 160
      ),
      "inception4E_Batch1" ->  Map(
        "channelsInput" -> 160,
        "channelsOutput"-> 160
      ),
      "inception4E_CNN2" -> Map(
        "kernelSize" -> Array[Int](3,3),
        "strideLength" -> Array[Int](2,2),
        "channelsInput" -> 160,
        "channelsOutput" -> 256
      ),
      "inception4E_Batch2" -> Map(
        "channelsInput" -> 256,
        "channelsOutput"-> 256
      ),
      "inception4E_CNN3" -> Map(
        "kernelSize" -> Array[Int](1,1),
        "strideLength" -> Array[Int](1,1),
        "channelsInput" -> 640,
        "channelsOutput" -> 64
      ),
      "inception4E_Batch3" -> Map(
        "channelsInput" -> 64,
        "channelsOutput"-> 64
      ),
      "inception4E_CNN4" -> Map(
        "kernelSize" -> Array[Int](3,3),
        "strideLength" -> Array[Int](2,2),
        "channelsInput" -> 64,
        "channelsOutput" -> 128
      ),
      "inception4E_Batch4" -> Map(
        "channelsInput" -> 128,
        "channelsOutput"-> 128
      ),
      "inception4E_Pool" -> Map(
        "kernelSize" -> Array[Int](3,3),
        "strideLength" -> Array[Int](2,2),
        "padding" -> Array[Int](1,1)
      ),
      // Inception 5a
      "inception5A_CNN1"-> Map(
        "kernelSize" -> Array[Int](1,1),
        "strideLength" -> Array[Int](1,1),
        "channelsInput" -> 1024,
        "channelsOutput" -> 256
      ),
      "inception5A_Batch1" ->  Map(
        "channelsInput" -> 256,
        "channelsOutput"-> 256
      ),
      "inception5A_CNN2" -> Map(
        "kernelSize" -> Array[Int](1,1),
        "strideLength" -> Array[Int](1,1),
        "channelsInput" -> 1024,
        "channelsOutput" -> 96
      ),
      "inception5A_Batch2" -> Map(
        "channelsInput" -> 96,
        "channelsOutput"-> 96
      ),
      "inception5A_CNN3" -> Map(
        "kernelSize" -> Array[Int](3,3),
        "strideLength" -> Array[Int](1,1),
        "channelsInput" -> 96,
        "channelsOutput" -> 384
      ),
      "inception5A_Batch3" -> Map(
        "channelsInput" -> 384,
        "channelsOutput"-> 384
      ),
      "inception5A_Pool" -> Map(
        "kernelSize" -> Array[Int](3,3),
        "strideLength" -> Array[Int](1,1),
        "norm" -> 2
      ),
      "inception5A_CNN4_Reduce" -> Map(
        "kernelSize" -> Array[Int](1,1),
        "strideLength" -> Array[Int](1,1),
        "channelsInput" -> 1024,
        "channelsOutput" -> 96
      ),
      "inception5A_Batch4_Reduce" -> Map(
        "channelsInput" -> 96,
        "channelsOutput"-> 96
      ),
      // Inception 5b
      "inception5B_CNN1"-> Map(
        "kernelSize" -> Array[Int](1,1),
        "strideLength" -> Array[Int](1,1),
        "channelsInput" -> 736,
        "channelsOutput" -> 256
      ),
      "inception5B_Batch1" ->  Map(
        "channelsInput" -> 256,
        "channelsOutput"-> 256
      ),
      "inception5B_CNN2" -> Map(
        "kernelSize" -> Array[Int](1,1),
        "strideLength" -> Array[Int](1,1),
        "channelsInput" -> 736,
        "channelsOutput" -> 96
      ),
      "inception5B_Batch2" -> Map(
        "channelsInput" -> 96,
        "channelsOutput"-> 96
      ),
      "inception5B_CNN3" -> Map(
        "kernelSize" -> Array[Int](3,3),
        "strideLength" -> Array[Int](1,1),
        "channelsInput" -> 96,
        "channelsOutput" -> 384
      ),
      "inception5B_Batch3" -> Map(
        "channelsInput" -> 384,
        "channelsOutput"-> 384
      ),
      "inception5B_Pool" -> Map(
        "kernelSize" -> Array[Int](3,3),
        "strideLength" -> Array[Int](1,1),
        "padding" -> Array[Int](1,1)
      ),
      "inception5B_CNN4_Reduce" -> Map(
        "kernelSize" -> Array[Int](1,1),
        "strideLength" -> Array[Int](1,1),
        "channelsInput" -> 736,
        "channelsOutput" -> 96
      ),
      "inception5B_Batch4_Reduce" -> Map(
        "channelsInput" -> 96,
        "channelsOutput"-> 96
      ),
      "output_AvgPool" -> Map(
        "kernelSize" -> Array[Int](3,3),
        "strideLength" -> Array[Int](3,3)
      ),
      "output_Dense" -> Map(
        "nodesInput" -> 736,
        "nodesOutput" -> embeddingSize
      ),
      "output_Loss" -> Map(
        "nodesInput" -> 128,
        "nodesOutput" -> numClass,
        "lambda" -> 1e-4,
        "alpha" -> 0.9
      )
    )
    modelLayers
  }

  def prepareModel(seedValue : Int, layersNN : Map[String, Map[String, Any]]) : Unit = {
    // Model Configuration

  }

  def main(args: Array[String]): Unit = {
    // Prepare data ,Model layers and Multilayer Model

    // Initialise UI Server
    val uiServer : UIServer = UIServer.getInstance()
    val statsStorage : StatsStorage = new InMemoryStatsStorage()
    uiServer.attach(statsStorage)

    // Train model

    // Evaluate model

  }
}
