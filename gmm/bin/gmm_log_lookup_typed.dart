library gmm_log_lookup_typed;

import 'dart:math';
import 'dart:typed_data';
import 'log_math.dart';

class TypedDiagonalGaussian {

  Float32List means;
  Float32List variances;
  Float32List negativeHalfPrecisions;
  double C;

  TypedDiagonalGaussian(List<double> means, List<double> variances) {
    this.means = new Float32List.fromList(means); 
    this.variances = new Float32List.fromList(variances);

    // instead of using [-0.5 * 1/var[d]] during likelihood calculation we pre-compute the values.
    // This saves 1 mul 1 div operation.    
    negativeHalfPrecisions = new Float32List(variances.length);
    for (int i = 0; i < negativeHalfPrecisions.length; i++) {
      negativeHalfPrecisions[i] = -0.5 / variances[i];
    } 
    // calculate the precomputed distance.
    // -0.5*SUM[d=1..D] ( log(2*PI) + log(var[d]) ) = -0.5*log(2*PI)*D -0.5 SUM[d=1..D](log(var[d]))
    double val = -0.5 * log(2 * PI) * variances.length;
    for (double variance in variances) {
      val -= (0.5 * log(variance));
    }
    C = val;      
  }

  double logLikelihood(Float32List data) {
    double res = 0.0;
    for (int i = 0; i < means.length; i++) {
        final double dif = data[i] - means[i];
        res += ((dif * dif) * negativeHalfPrecisions[i]);     
    }    
    return C + res;
  } 
}

class TypedGmm  {

  Float64List mixtureWeights;
  List<TypedDiagonalGaussian> gaussians;

  TypedGmm(this.mixtureWeights, this.gaussians) {
    this.mixtureWeights = new Float64List.fromList(mixtureWeights);
    for(int i = 0; i< mixtureWeights.length; i++) {
      mixtureWeights[i] = log(mixtureWeights[i]);
    }
  }
  
  double score(Float32List data) {
    var result = mixtureWeights[0] + gaussians[0].logLikelihood(data);
    for (int i = 1; i < gaussians.length; ++i) {
      var b = mixtureWeights[i] + gaussians[i].logLikelihood(data);
      result = logMath.logSumTyped(result, b);
    }
    return result;
  } 
}