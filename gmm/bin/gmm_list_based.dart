library gmm_log_lookup;

import 'dart:math';
import 'log_math.dart';

class ListDiagonalGaussian {

  List<double> means;
  List<double> variances;
  List<double> negativeHalfPrecisions;
  double C;

  ListDiagonalGaussian(this.means, this.variances) {
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
    C = val;      
  }

  /// Calculates linear likelihood of a given vector.
  double logLikelihood(List<double> data) {
    double res = 0.0;
    for (int i = 0; i < means.length; i++) {
        final double dif = data[i] - means[i];
        res += ((dif * dif) * negativeHalfPrecisions[i]);     
    }    
    return C + res;
  }
  
  /// Calculates linear likelihood of a given vector.
  double likelihood(List<double> data) {
    double result = 1.0;
    for (int i = 0; i < means.length; i++) {      
      double meanDif = data[i] - means[i];      
      result *= (1 / sqrt(2 * PI * variances[i])) * 
          exp(-0.5 * meanDif * meanDif / variances[i]);
    }
    return result;
  }  
}

class LinearGmm  {

  List<double> mixtureWeights;
  List<ListDiagonalGaussian> gaussians;

  LinearGmm(this.mixtureWeights, this.gaussians);

  double score(List<double> data) {
    double sum = 0.0;
    for (int i = 0; i < gaussians.length; ++i) {
      sum += mixtureWeights[i]*gaussians[i].likelihood(data);
    }
    return sum;
  }
}

class LogGmm  {

  List<double> logMixtureWeights;  
  List<ListDiagonalGaussian> gaussians;

  LogGmm(List<double> mixtureWeights, this.gaussians) {
    this.logMixtureWeights = new List(mixtureWeights.length);
    for(int i = 0; i< mixtureWeights.length; i++) {
      logMixtureWeights[i] = log(mixtureWeights[i]);
    }
  }
  
  double score(List<double> data) {
    double result = logMixtureWeights[0] + gaussians[0].logLikelihood(data);
    for (int i = 1; i < gaussians.length; ++i) {
      var b = logMixtureWeights[i] + gaussians[i].logLikelihood(data);
      result = b + log(1 + exp(result-b));
    }
    return result;
  }  
}

class LogsumLookupGmm  {

  List<double> logMixtureWeights;  
  List<ListDiagonalGaussian> gaussians;

  LogsumLookupGmm(List<double> mixtureWeights, this.gaussians) {
    this.logMixtureWeights = new List(mixtureWeights.length);
    for(int i = 0; i< mixtureWeights.length; i++) {
      logMixtureWeights[i] = log(mixtureWeights[i]);
    }
  }

  double score(List<double> data) {
    double result = logMixtureWeights[0] + gaussians[0].logLikelihood(data);
    for (int i = 1; i < gaussians.length; ++i) {
      var b = logMixtureWeights[i] + gaussians[i].logLikelihood(data);
      result = logMath.logSum(result,b);
    }
    return result;
  }  
}
