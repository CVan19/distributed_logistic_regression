import org.scalatest.FunSuite
import scala.io.Source
import breeze.linalg._
import breeze.numerics._
import scala.util.control._
import scala.util.Random
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics

//加了L2惩罚项的逻辑回归，优化方法为牛顿法
class LogisticRidgeNewton(val x: DenseMatrix[Double], val y: DenseVector[Double]) {
    // Dimension constants
    private val dim_n = x.rows
    private val dim_p = x.cols
    // Penalty parameters
    private var lambda: Double = 0.0
    private var v: DenseVector[Double] = DenseVector.zeros[Double](dim_p)
    // Parameters related to convergence
    private var max_iter: Int = 100
    private var eps_abs: Double = 1e-6
    private var eps_rel: Double = 1e-6
    // Variables to be returned
    private val bhat = DenseVector.zeros[Double](dim_p)
    private var iter = 0

    // pi(x, b) = 1 / (1 + exp(-x * b))
    private def pi(x: DenseMatrix[Double], b: DenseVector[Double]): DenseVector[Double] = {
        return 1.0 / (exp(- x * b) + 1.0)
    }

    def set_opts(max_iter: Int = 100, eps_abs: Double = 1e-6, eps_rel: Double = 1e-6) {
        this.max_iter = max_iter
        this.eps_abs = eps_abs
        this.eps_rel = eps_rel
    }

    def set_lambda(lambda: Double) {
        this.lambda = lambda
    }

    def set_v(v: DenseVector[Double]) {
        this.v = v
    }

    def run() {
        bhat := 0.0

        val loop = new Breaks
        loop.breakable {
            for(i <- 0 until max_iter) {
                val mu = pi(x, bhat)
                val w = mu :* (1.0 - mu)
                // Gradient = -X'(y-mu) + lambda*(beta-v)
                val grad = x.t * (mu - y) + lambda * (bhat - v)
                // Hessian = X'WX + lambda * I
                val hessian = (x.t) * diag(w) * x + lambda * DenseMatrix.eye[Double](dim_p)

                val delta = inv(hessian) * grad
                bhat -= delta
                iter = i
                val r = norm(delta)
                if(r < eps_abs * math.sqrt(dim_p) || r < eps_rel * norm(bhat)) {
                    loop.break
                }
            }
        }
    }
    def coef = bhat.copy
    def niter = iter
}

//加了L1惩罚项的逻辑回归，优化方法为ADMM
class LogisticLassoADMM(val x: DenseMatrix[Double], val y: DenseVector[Double]) {
    val dim_x = x.cols
    // Penalty parameter
    protected var lambda: Double = 0.0
    // Parameters related to convergence
    private var max_iter: Int = 1000
    private var eps_abs: Double = 1e-6
    private var eps_rel: Double = 1e-6
    protected var rho: Double = 1.0
    private var logs: Boolean = false
    // Main variable
    protected val admm_x = DenseVector.zeros[Double](dim_x)
    // Auxiliary variable
    protected var admm_z = new VectorBuilder[Double](dim_x).toSparseVector
    // Dual variable
    protected val admm_y = DenseVector.zeros[Double](dim_x)
    // Number of iterations
    private var iter = 0
    // Residuals and tolerance
    private var eps_primal = 0.0;
    private var eps_dual = 0.0;
    private var resid_primal = 0.0;
    private var resid_dual = 0.0;
    // Logistic Ridge
    private val xsolver = new LogisticRidgeNewton(x, y)
    xsolver.set_lambda(rho)
    xsolver.set_opts(100, 1e-3, 1e-3)

    // Soft threshold
    private def soft_shreshold(vec: DenseVector[Double], penalty: Double): SparseVector[Double] = {
        val builder = new VectorBuilder[Double](vec.size)
        for(ind <- 0 until vec.size) {
            val v = vec(ind)
            if(v > penalty) {
                builder.add(ind, v - penalty)
            } else if(v < -penalty) {
                builder.add(ind, v + penalty)
            }
        }
        return builder.toSparseVector(true, true)
    }

    // Convenience function
    private def max2(x: Double, y: Double) = if(x > y) x else y

    // Tolerance for primal residual
    private def compute_eps_primal(): Double = {
        val r = max2(norm(admm_x), norm(admm_z))
        return r * eps_rel + math.sqrt(dim_x) * eps_abs
    }
    // Tolerance for dual residual
    private def compute_eps_dual(): Double = {
        return norm(admm_y) * eps_rel + math.sqrt(dim_x) * eps_abs
    }
    // Dual residual
    private def compute_resid_dual(new_z: SparseVector[Double]): Double = {
        return rho * norm(new_z - admm_z)
    }
    // Changing rho
    protected def rho_changed_action() {
         xsolver.set_lambda(rho)
    }
    private def update_rho() {
        if(resid_primal / eps_primal > 10 * resid_dual / eps_dual) {
            rho *= 2
            rho_changed_action()
        } else if(resid_dual / eps_dual > 10 * resid_primal / eps_primal) {
            rho /= 2
            rho_changed_action()
        }

        if(resid_primal < eps_primal) {
            rho /= 1.2
            rho_changed_action()
        }

        if(resid_dual < eps_dual) {
            rho *= 1.2
            rho_changed_action()
        }
    }
    // Update x -- abstract method
    protected def update_x() {
        val v = admm_z - admm_y / rho;
        xsolver.set_v(v)
        xsolver.run()
        admm_x := xsolver.coef
    }
    //log
    protected def logging(iter: Int) {
        if(iter % 10 == 0) {
            val xb = x * admm_z.toDenseVector
            val ll = sum((y :* xb) - log(exp(xb) + 1.0))
            val penalty = lambda * sum(abs(admm_z))
            val obj = -ll + penalty
            println("Iteration #" + iter + ": obj = " + obj)
        }
    }

    def set_opts(max_iter: Int = 1000, eps_abs: Double = 1e-6, eps_rel: Double = 1e-6,
                 rho: Double = 1, logs: Boolean = false) {
        this.max_iter = max_iter
        this.eps_abs = eps_abs
        this.eps_rel = eps_rel
        this.rho = rho
        this.logs = logs
    }

    def set_lambda(lambda: Double) {
        this.lambda = lambda
    }

    def run() {
        val loop = new Breaks
        loop.breakable {
            for(i <- 0 until max_iter) {
                // Calculate tolerance values
                eps_primal = compute_eps_primal()
                eps_dual = compute_eps_dual()

                // x step
                update_x()

                // z step
                val new_z = soft_shreshold(admm_x + admm_y / rho, lambda / rho)
                resid_dual = compute_resid_dual(new_z)
                admm_z = new_z

                // y step
                val resid = admm_x - admm_z
                resid_primal = norm(resid)
                admm_y :+= rho * resid

                iter = i

                if(logs)
                    logging(iter)

                // Convergence test
                if(resid_primal < eps_primal && resid_dual < eps_dual) {
                    loop.break
                }

                if(i > 3)
                    update_rho()
            }
        }
    }

    def coef = admm_z.toDenseVector
    def niter = iter
}


//读取数据
def read_data(f: String): (DenseMatrix[Double], DenseVector[Double]) = {
    var source = Source.fromFile(f)
    val lines = source.getLines()
    val first = lines.take(1).toArray
    val n = lines.length + 1
    val p = first(0).split(',').length - 1
    source.close()

    val x = DenseMatrix.zeros[Double](n, p)
    val y = DenseVector.zeros[Double](n)
    source = Source.fromFile(f)
    var i = 0
    for(line <- source.getLines()) {
        val l = line.split(',')
        y(i) = l(0).toDouble
        x(i, ::) := (new DenseVector(l.drop(1).map(x => x.toDouble))).t
        i += 1
    }
    source.close()
    return (x, y)
}

//读取已经标准化的训练集与测试集
val train_data = read_data("Nonheader_train.csv")
val Xtrain = train_data._1
val ytrain = train_data._2
val test_data = read_data("Nonheader_test.csv")
val Xtest = test_data._1
val ytest = test_data._2
//常量
val trainNum = Xtrain.rows
val testNum = Xtest.rows
val dim = Xtrain.cols
val slaveNum = 10  //节点数
val partNum = trainNum / slaveNum //每个节点上的样本数
//将训练数据随机打乱，并分成slaveNum份
val indices = Random.shuffle(0 to (trainNum-1))
val sf_Xtrain = Xtrain(indices, ::).toDenseMatrix
val sf_ytrain = ytrain(indices).toDenseVector
val sf_traindata = new Array[(DenseVector[Double], DenseMatrix[Double])](slaveNum)
var i = 0
while(i < slaveNum){
    if(i == (slaveNum-1)){
        sf_traindata(i) = (sf_ytrain(i*partNum until trainNum), sf_Xtrain(i*partNum until trainNum, ::))
    }
    else{
        sf_traindata(i) = (sf_ytrain(i*partNum until (i+1)*partNum), sf_Xtrain(i*partNum until (i+1)*partNum, ::))
    }
    i += 1
}
//divide and conquer，在每一个节点上拟合一个beta，最后取平均
val distributed_data = sc.parallelize(sf_traindata, slaveNum)
val tmp = distributed_data.map(point => {
    val part_y = point._1
    val part_x = point._2
    val logisticL2 = new LogisticRidgeNewton(part_x, part_y)
    logisticL2.set_lambda(1.0)
    logisticL2.run()
    val logisticL1 = new LogisticLassoADMM(part_x, part_y)
    logisticL1.set_lambda(1.0)
    logisticL1.run()

    (logisticL2.coef, logisticL1.coef, 1.0)
}).reduce((x, y) => (x._1+y._1, x._2+y._2, x._3+y._3))
val L2_bhat = tmp._1 :/ tmp._3
val L1_bhat = tmp._2 :/ tmp._3
//测试与评估
val score = Xtest * L2_bhat
val pred = score.map(p => if(p > 0) 1.0 else 0.0)
val scoreAndLabel = new Array[(Double, Double)](ytest.size)
var right = 0
for(i <- 0 until ytest.size){
    if(pred(i) == ytest(i)){right += 1}
    scoreAndLabel(i) = (score(i), ytest(i))
}
val accuracy = right.toDouble / ytest.size
val rdd_scoreAndLabel = sc.parallelize(scoreAndLabel)
val metrics = new BinaryClassificationMetrics(rdd_scoreAndLabel)
val AUC = metrics.areaUnderROC() //0.733


val score = Xtest * L1_bhat
val pred = score.map(p => if(p > 0) 1.0 else 0.0)
val scoreAndLabel = new Array[(Double, Double)](ytest.size)
var right = 0
for(i <- 0 until ytest.size){
    if(pred(i) == ytest(i)){right += 1}
    scoreAndLabel(i) = (score(i), ytest(i))
}
val accuracy = right.toDouble / ytest.size
val rdd_scoreAndLabel = sc.parallelize(scoreAndLabel)
val metrics = new BinaryClassificationMetrics(rdd_scoreAndLabel)
val AUC = metrics.areaUnderROC() //0.734








//用MLlib自带的包优化
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization.{LBFGS, LogisticGradient, L1Updater, SquaredL2Updater}
import org.apache.spark.mllib.util.MLUtils

val train_data = MLUtils.loadLibSVMFile(sc, "train.libsvm").map(x => (x.label, x.features)).cache()
val test_data = MLUtils.loadLibSVMFile(sc, "test.libsvm").cache()
val numFeatures = train_data.take(1)(0)._2.size - 1

// Run training algorithm to build the model
val numCorrections = 10
val convergenceTol = 1e-4
val maxNumIterations = 20
val regParam = 1.0
val initialWeightsWithIntercept = Vectors.dense(new Array[Double](numFeatures + 1))

val (weightsWithIntercept, loss) = LBFGS.runLBFGS(
  train_data,
  new LogisticGradient(),
  new SquaredL2Updater(),
  numCorrections,
  convergenceTol,
  maxNumIterations,
  regParam,
  initialWeightsWithIntercept)

val model = new LogisticRegressionModel(
  Vectors.dense(weightsWithIntercept.toArray.slice(0, weightsWithIntercept.size - 1)),
  weightsWithIntercept(weightsWithIntercept.size - 1))

// Clear the default threshold.
 model.clearThreshold()

// Compute raw scores on the test set.
val scoreAndLabels = test_data.map { point =>
  val score = model.predict(point.features)
  (score, point.label)
}

// Get evaluation metrics.
val metrics = new BinaryClassificationMetrics(scoreAndLabels)
val AUC = metrics.areaUnderROC() //0.661
println("Loss of each step in training process")
loss.foreach(println)
println(s"Area under ROC = $AUC")



