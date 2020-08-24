name := "LearnDl4j"

version := "0.1"

scalaVersion := "2.12.11"

libraryDependencies ++=  Seq(
  "org.deeplearning4j" % "deeplearning4j-core" % "1.0.0-beta6",
  "org.deeplearning4j" % "deeplearning4j-ui" % "1.0.0-beta6",
  "org.deeplearning4j" % "deeplearning4j-zoo" % "1.0.0-beta6",
  "org.nd4j" % "nd4j-native-platform" % "1.0.0-beta6",
  "org.slf4j" % "slf4j-simple" % "1.7.25"
)
