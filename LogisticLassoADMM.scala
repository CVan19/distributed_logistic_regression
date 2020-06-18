import breeze.linalg._
import breeze.numerics._
import scala.util.control._

// Minimize
//      -loglik(beta) + lambda * ||beta||_1
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

    def coef = admm_z.copy
    def niter = iter
}

//divide and conquer
val sampleNum = 10000
val p = 5
val slaveNum = 4
val partNum = sampleNum / slaveNum
val normal01 = breeze.stats.distributions.Gaussian(0, 1)
val x = DenseMatrix.rand(sampleNum, p, normal01)
val tmp = (x :> 0.0).map(i => if(i) 1 else 0)
val y = sum(tmp(*, ::)).map(i => if(i > (p/2)) 1.0 else 0.0)
val data = new Array[(DenseVector[Double], DenseMatrix[Double])](slaveNum)
var i = 0
while(i < slaveNum){
    data(i) = (y(i*partNum until (i+1)*partNum), x(i*partNum until (i+1)*partNum, ::))
    i += 1
}
val distributed_data = sc.parallelize(data, slaveNum)
val tmp = distributed_data.map(point => {
    val part_y = point._1
    val part_x = point._2
    val logisticL1 = new LogisticLassoADMM(part_x, part_y)
    logisticL1.set_lambda(1.0)
    logisticL1.run()
    (logisticL1.coef, 1.0)
}).reduce((x, y) => (x._1+y._1, x._2+y._2))
val bhat = tmp._1 :/ tmp._2


