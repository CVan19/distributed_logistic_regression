import scala.collection.JavaConverters._
import scala.util.Random
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression._
import org.apache.spark.mllib.optimization.{GradientDescent,
LogisticGradient, SquaredL2Updater}

val n = 200
val p = 5
val NumofSlaves = 2
val sgd_points = sc.parallelize(0 until n, 2).map(iter => {
val random = new Random()
val u = random.nextDouble()
val y = if(u>0.5) 1.0 else 0.0
(y, Vectors.dense(Array.fill(p)(random.nextDouble())))
})
val points = sgd_points.map(point => {
    val y = point._1
    val x = point._2
    (y, DenseVector(x.toArray))
})

val gradient = new LogisticGradient()
val updater = new SquaredL2Updater()
val stepSize = 0.1
val numIterations = 1000
val regParam = 1.0
val miniBatchFrac = 1.0
val Tol=0.001
val (weights, loss) = GradientDescent.runMiniBatchSGD(
sgd_points,
gradient,
updater,
stepSize,
numIterations,
regParam,
miniBatchFrac,
Vectors.dense(new Array[Double](p)),
Tol)


