library gmm_base; 

import 'dart:math';

class DiagonalGaussian {

  List<double> means;
  List<double> variances;

  DiagonalGaussian(this.means, this.variances);

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

class Gmm  {

  List<double> mixtureWeights;
  List<DiagonalGaussian> gaussians;

  Gmm(this.mixtureWeights, this.gaussians);

  double score(List<double> data) {
    double sum = 0.0;
    for (int i = 0; i < gaussians.length; ++i) {
      sum += mixtureWeights[i]*gaussians[i].likelihood(data);
    }
    return sum;
  }
}