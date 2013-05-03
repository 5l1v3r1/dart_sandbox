library gmm_simd;

import 'dart:math';
import 'dart:typed_data';
import 'log_math.dart';

class SimdDiagonalGaussian {

  Float32x4List means;
  Float32x4List negativeHalfPrecisions;
  double C;

  SimdDiagonalGaussian(List<double> _means, List<double> _variances) {
    
    this.means = toFloat32x4List(_means);
    Float32x4List variances = toFloat32x4List(_variances);    
    
    // instead of using [-0.5 * 1/var[d]] during likelihood calculation we pre-compute the values.
    negativeHalfPrecisions = new Float32x4List(means.length);
    for (int i = 0; i < negativeHalfPrecisions.length; i++) {
      Float32x4 v = variances[i];
      negativeHalfPrecisions[i] = new Float32x4(-0.5/v.x,-0.5/v.y,-0.5/v.z,-0.5/v.w);
    }
 
    // calculate the precomputed distance.
    // -0.5*SUM[d=1..D] ( log(2*PI) + log(var[d]) ) = -0.5*log(2*PI)*D -0.5 SUM[d=1..D](log(var[d]))
    double val = -0.5 * log(2 * PI) * _variances.length;
    for (double variance in _variances) {
        val -= (0.5 * log(variance));
    }
    C = val;     
  }

  /// Calculates linear likelihood of a given vector.
  double logLikelihood(Float32x4List data) {
    Float32x4 res = new Float32x4.zero();
    for (int i = 0; i < means.length; i++) {
        final Float32x4 dif = data[i] - means[i];
        res += ((dif * dif) * negativeHalfPrecisions[i]);     
    }    
    return C + res.x + res.y + res.w + res.z;
  } 
}

class SimdGmm  {

  Float64List mixtureWeights;
  List<SimdDiagonalGaussian> gaussians;

  SimdGmm(List<double> mixtureWeights, this.gaussians) {
    this.mixtureWeights = new Float64List.fromList(mixtureWeights);
    for(int i = 0; i< mixtureWeights.length; i++) {
      mixtureWeights[i] = log(mixtureWeights[i]);
    }
  }
  
  double score(Float32x4List data) {
    double result = mixtureWeights[0] + gaussians[0].logLikelihood(data);    
    for (int i = 1; i < gaussians.length; ++i) {
      double b = mixtureWeights[i] + gaussians[i].logLikelihood(data);
      result = logMath.logSumTyped(result,b);
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

Float32x4List toFloat32x4List(List<double> list) {
  int dsize = list.length~/4;
  Float32x4List res = new Float32x4List(dsize);
  for (int j = 0; j < dsize; j++) {        
    res[j] = new Float32x4(
        list[j*4],
        list[j*4+1],
        list[j*4+2],
        list[j*4+3]);        
  }  
  return res;
}