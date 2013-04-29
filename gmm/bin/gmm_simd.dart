library gmm_simd;

import 'dart:math';
import 'dart:typeddata';

class DiagonalGaussian {

  Float32x4List means;
  Float32x4List variances;
  Float32x4List negativeHalfPrecisions;
  double logPrecomputedDistance;

  DiagonalGaussian(this.means, this.variances) {
    // instead of using [-0.5 * 1/var[d]] during likelihood calculation we pre-compute the values.
    negativeHalfPrecisions = new Float32x4List(means.length);
    for (int i = 0; i < negativeHalfPrecisions.length; i++) {
      Float32x4 v = variances[i];
      negativeHalfPrecisions[i] = new Float32x4(-0.5/v.x,-0.5/v.y,-0.5/v.z,-0.5/v.w);
    }
 
    // calculate the precomputed distance.
    // -0.5*SUM[d=1..D] ( log(2*PI) + log(var[d]) ) = -0.5*log(2*PI)*D -0.5 SUM[d=1..D](log(var[d]))
    List<double> variances = toDoubleList(this.variances);
    double val = -0.5 * log(2 * PI) * variances.length;
    for (double variance in variances) {
        val -= (0.5 * log(variance));
    }
    logPrecomputedDistance = val;     
  }

  /// Calculates linear likelihood of a given vector.
  double logLikelihood(Float32x4List data) {
    Float32x4 res = new Float32x4.zero();
    for (int i = 0; i < means.length; i++) {
        final Float32x4 dif = data[i] - means[i];
        res += ((dif * dif) * negativeHalfPrecisions[i]);     
    }    
    return logPrecomputedDistance + res.x + res.y + res.w + res.z;
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
  
  double score(Float32x4List data) {
    double result = LOG0;
    for (int i = 0; i < gaussians.length; ++i) {
      var b = mixtureWeights[i] + gaussians[i].logLikelihood(data);
      result = logSum(result,b);
    }
    return result;
  } 
}

List<double> toDoubleList(Float32x4List lst) {
  List<double> result = new List(lst.length*4);
  for(int i =0; i<lst.length;i++){
    result[i*4] = lst[i].x;
    result[i*4+1] = lst[i].y;
    result[i*4+2] = lst[i].z;
    result[i*4+3] = lst[i].w;      
  }
  return result;
}
 
const double _SCALE = 1000.0;

LogMath logMath = new LogMath();

/**
 * Calculates an approximation of log(a+b) when log(a) and log(b) are given using the formula
 * log(a+b) = log(b) + log(1 + exp(log(a)-log(b))) where log(b)>log(a)
 * This method is an approximation because it uses a lookup table for log(1 + exp(log(b)-log(a))) part
 * This is useful for log-probabilities where values vary between -30 < log(p) <= 0
 * if difference between values is larger than 20 (which means sum of the numbers will be very close to the larger
 * value in linear domain) large value is returned instead of the logSum calculation because effect of the other
 * value is negligible
 */
double logSum(double logA, double logB) {
  if (logA > logB) {
    double dif = logA - logB; // logA-logB because during lookup calculation dif is multiplied with -1
    return dif >= 20.0 ? logA : logA + logMath.logSumLookup[(dif * _SCALE).toInt()];
  } else {
    final double dif = logB - logA;
    return dif >= 20.0 ? logB : logB + logMath.logSumLookup[(dif * _SCALE).toInt()];
  }
}

double LN0 = log(0);

class LogMath {  
  
  static final LogMath _singleton = new LogMath._internal();
  
  final logSumLookup = new Float64List(20000);
  
  factory LogMath() {
    return _singleton;
  }
  
  LogMath._internal() {
    for (int i = 0; i < logSumLookup.length; i++) {
      logSumLookup[i] = log(1.0 + exp(-i / _SCALE));
    }
  }    
}