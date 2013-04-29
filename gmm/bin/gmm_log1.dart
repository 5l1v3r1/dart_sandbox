import 'dart:math';

class DiagonalGaussian {

  List<double> means;
  List<double> variances;
  List<double> negativeHalfPrecisions;
  double logPrecomputedDistance;

  DiagonalGaussian(this.means, this.variances) {
    // instead of using [-0.5 * 1/var[d]] during likelihood calculation we pre-compute the values.
    // This saves 1 mul 1 div operation.
    negativeHalfPrecisions = new List<double>(variances.length);
    for (int i = 0; i < negativeHalfPrecisions.length; i++) {
      negativeHalfPrecisions[i] = -0.5 / variances[i];
    } 
    // calculate the precomputed distance.
    // -0.5*SUM[d=1..D] ( log(2*PI) + log(var[d]) ) = -0.5*log(2*PI)*D -0.5 SUM[d=1..D](log(var[d]))
    double val = -0.5 * log(2 * PI) * variances.length;
    for (double variance in variances) {
      val -= (0.5 * log(variance));
    }
    logPrecomputedDistance = val;      
  }

  /// Calculates linear likelihood of a given vector.
  double logLikelihood(List<double> data) {
    double res = 0.0;
    for (int i = 0; i < means.length; i++) {
        final double dif = data[i] - means[i];
        res += ((dif * dif) * negativeHalfPrecisions[i]);     
    }    
    return logPrecomputedDistance + res;
  } 

  int get dimension => means.length;
}

class Gmm  {

  List<double> mixtureWeights;
  List<DiagonalGaussian> gaussians;

  Gmm(this.mixtureWeights, this.gaussians) {
    for(int i = 0; i< mixtureWeights.length; i++) {
      mixtureWeights[i] = log(mixtureWeights[i]);
    }
  }
  
  final double LOG0 = log(0);
  
  double logLikelihood(List<double> data) {
    double logSum = LOG0;
    for (int i = 0; i < gaussians.length; ++i) {
      var a = log(logSum);
      var b = mixtureWeights[i] + gaussians[i].logLikelihood(data);
      logSum = b + log(1+exp(a-b));
    }
    return logSum;
  } 
}