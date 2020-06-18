import java.util.concurrent.ThreadLocalRandom

val N = 100000
val MU1 = 10.0
val Sig1 = 2.0
val MU2 = 0.0
val Sig2 = 1.0
val MU3 = 22.0
val Sig3 = math.sqrt(5.0)
val P1 = 0.1
val P2 = 0.5
val P3 = 0.4
val NumOfSlaves = 5
val random = ThreadLocalRandom.current
val data = Array.ofDim[Double](N)
for(i <- 0 until N){
    if(random.nextDouble <= P1){data(i) = MU1 + Sig1*random.nextGaussian}
    else if(random.nextDouble <= P1+P2){data(i) = MU2 + Sig2*random.nextGaussian}
    else{data(i) = MU3 + Sig3*random.nextGaussian}
    }
var ParData = sc.parallelize(data, NumOfSlaves)
val ParDataStr = ParData.map("%.4f" format _)
ParData = ParDataStr.map(_.toDouble)
//设置初始值
val InitialP1 = 0.3
val InitialP2 = 0.3
val InitialP3 = 0.4
var N1 = N.toDouble*InitialP1
var N2 = N.toDouble*InitialP2
var N3 = N.toDouble*InitialP3
var EstMu1 = ParData.reduce((x,y) => x+y)/N.toDouble
var EstSig1 = math.sqrt(ParData.map(x => x*x).
    reduce((x,y) => x+y)/N.toDouble - EstMu1*EstMu1)
var EstMu2 = EstMu1-1
var EstSig2 = EstSig1
var EstMu3 = EstMu1+1
var EstSig3 = EstSig1
var Diff = 0.0
var OldEstMu1 = 0.0
var OldEstMu2 = 0.0
var OldEstMu3 = 0.0
var OldEstSig1 = 0.0
var OldEstSig2 = 0.0
var OldEstSig3 = 0.0
var ii = 0
var eps = 0.001
do{
    ii += 1
    OldEstMu1 = EstMu1
    OldEstSig1 = EstSig1
    OldEstMu2 = EstMu2
    OldEstSig2 = EstSig2
    OldEstMu3 = EstMu3
    OldEstSig3 = EstSig3
    var SufficientStatistics = ParData.map(line => {
        val idx1 = -math.pow((line-EstMu1)/EstSig1, 2)/2.0
        val idx2 = -math.pow((line-EstMu2)/EstSig2, 2)/2.0
        val idx3 = -math.pow((line-EstMu3)/EstSig3, 2)/2.0
        val gamma1 = N1*EstSig2*EstSig3/(N1*EstSig2*EstSig3 + 
            N2*EstSig1*EstSig3*math.exp(idx2-idx1) + 
            N3*EstSig1*EstSig2*math.exp(idx3-idx1))
        val gamma2 = N2*EstSig1*EstSig3/(N2*EstSig1*EstSig3 + 
            N1*EstSig2*EstSig3*math.exp(idx1-idx2) + 
            N3*EstSig1*EstSig2*math.exp(idx3-idx2))
        val gamma3 = N3*EstSig1*EstSig2/(N3*EstSig1*EstSig2 + 
            N2*EstSig1*EstSig3*math.exp(idx2-idx3) + 
            N1*EstSig2*EstSig3*math.exp(idx1-idx3))
        (line, gamma1, line*gamma1, line*line*gamma1,
            gamma2, line*gamma2, line*line*gamma2,
            gamma3, line*gamma3, line*line*gamma3)
    })
    val Results = SufficientStatistics.reduce((x,y) => 
    (x._1+y._1, x._2+y._2, x._3+y._3,
        x._4+y._4, x._5+y._5, x._6+y._6, x._7+y._7,
        x._8+y._8, x._9+y._9, x._10+y._10))
    N1 = Results._2
    N2 = Results._5
    N3 = Results._8
    EstMu1 = Results._3/N1
    EstSig1 = math.sqrt(Results._4/N1 - EstMu1*EstMu1)
    EstMu2 = Results._6/N2
    EstSig2 = math.sqrt(Results._7/N2 - EstMu2*EstMu2)
    EstMu3 = Results._9/N3
    EstSig3 = math.sqrt(Results._10/N3 - EstMu3*EstMu3)
    Diff = math.abs(EstMu1-OldEstMu1) + math.abs(EstMu2-OldEstMu2) +  math.abs(EstMu3-OldEstMu3)
    Diff += math.abs(EstSig1-OldEstSig1) + math.abs(EstSig2-OldEstSig2) + + math.abs(EstSig3-OldEstSig3)
}while(Diff > eps)