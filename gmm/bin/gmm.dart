library gmm;

import 'dart:math';
import 'dart:typeddata';

class DiagonalGaussian {

  Float32List means;
  Float32List variances;
  Float32List negativeHalfPrecisions;
  double logPrecomputedDistance;

  DiagonalGaussian(this.means, this.variances) {
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
    logPrecomputedDistance = val;      
  }

  /// Calculates linear likelihood of a given vector.
  double logLikelihood(Float32List data) {
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

  Float32List mixtureWeights;
  List<DiagonalGaussian> gaussians;

  Gmm(this.mixtureWeights, this.gaussians) {
    for(int i = 0; i< mixtureWeights.length; i++) {
      mixtureWeights[i] = log(mixtureWeights[i]);
    }
  }
  
  final double LOG0 = log(0);
  
  double logLikelihood(List<double> data) {
    double result = LOG0;
    for (int i = 0; i < gaussians.length; ++i) {
      var b = mixtureWeights[i] + gaussians[i].logLikelihood(data);
      result = logSum(result,b);
    }
    return result;
  } 
}



Random random = new Random(0xcafebabe);
double _random() {
    return random.nextInt(10) / 10 + 0.1;
}
 
class GaussData {
  List<double> means;
  List<double> variances;
  int dimension;
  double weight;
  
  GaussData(this.means, this.variances){
    this.dimension = means.length;
  }
  
  GaussData.random(this.dimension) {
    means = new List(dimension);
    variances = new List(dimension);    
    for(int i = 0 ; i<dimension; i++){
      means[i]=_random();
      variances[i] = _random();
    }
    weight = _random();
  }
}

Gmm getGmm(List<GaussData> gd) {
  var gaussList = new List();
  var weightList = new List();  
  for(int a = 0; a<gd.length; a++) {
    gaussList.add(new DiagonalGaussian(gd[a].means, gd[a].variances));
    weightList.add(gd[a].weight);       
  }
  return new Gmm(new Float32List.fromList(weightList), gaussList);
}

class InputData {
  List<List<double>> data;
  int size;
  int dimension;
  
  InputData.random(this.size, this.dimension){
    data = new List<List<double>>(size);
    for (int i = 0; i < data.length; i++) {
      data[i] = new List<double>(dimension);
      for (int j = 0; j < dimension; j++) {
        data[i][j] = _random();
      }
    }    
  }
  
  List<Float32List> getAsFloat32List() {
    List<Float32List> result = new List(data.length);
    for (int i = 0; i < data.length; i++) {
      result[i]=new Float32List.fromList(data[i]);      
    }     
    return result;    
  }
}
 
void calculate(DiagonalGaussian gaussian, List data) {
  Stopwatch sw = new Stopwatch()..start();
  double tot = 0.0;    
  for (int i = 0; i<data.length; ++i) {
    tot= tot+ gaussian.logLikelihood(data[i]);
  }
  print("log-likelihood=$tot");      
} 
 
int perfList(GaussData gauss, InputData input) {
    var d = new DiagonalGaussian(gauss.means, gauss.variances);
    Stopwatch sw = new Stopwatch()..start();
    double tot = 0.0;    
    for (int i = 0; i<input.size; ++i) {
      tot= tot+ d.logLikelihood(input.data[i]);
    }
    sw.stop();
    print("log-likelihood=$tot");
    print("List Elapsed: ${sw.elapsedMilliseconds}");    
    return sw.elapsedMilliseconds;
}

main() {

  print("Generating test data");
  int gmmCount = 1000;
  int gaussPerGmm = 8;
  int dimension = 40;
  int inputAmount = 1000;
  int iterationCount = 7;
  List<Gmm> gmms = new List(gmmCount);
  for(int i = 0; i<gmmCount;i++) {
    List<GaussData> gaussDataList = new List(gaussPerGmm);
    for(int j = 0; j<gaussPerGmm;j++) {
      gaussDataList[j]= new GaussData.random(40);
    }
    gmms[i]=getGmm(gaussDataList);
  } 
  //InputData dataLarge = new InputData.random(inputAmount,40);
  List<Float32List> dataLarge = new InputData.random(inputAmount,40).getAsFloat32List();

  List<int> times = new List(iterationCount);
  print("calculating..");
  for(int it = 0; it<iterationCount; it++) {
    Stopwatch sw = new Stopwatch()..start();
    double total = 0.0;  
    for(int i = 0; i<inputAmount;i++) {
      for(int j = 0; j<gmmCount;j++) {
        total+= gmms[j].logLikelihood(dataLarge[i]);
      }
    }
    sw.stop();
    times[it] = sw.elapsedMilliseconds;
    print("List Elapsed: ${sw.elapsedMilliseconds}");
  }
  
  times.sort();
  double tot = 0.0;
  for(int it = 0; it<iterationCount; it++) {
    if(it!=0 && it!=iterationCount-1)
      tot = tot + times[it];
  }
  print("mean = ${tot/(iterationCount-2).toDouble()}");
  
  print("Done.");
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
    return dif >= 30.0 ? logA : logA + logMath.logSumLookup[(dif * _SCALE).toInt()];
  } else {
    final double dif = logB - logA;
    return dif >= 30.0 ? logB : logB + logMath.logSumLookup[(dif * _SCALE).toInt()];
  }
}

double LN0 = log(0);

/**
 * Calculates approximate logSum of log values using the <code> logSum(logA,logB) </code>
*
 * @param logValues log values to use in logSum calculation.
 * @return <p>log(a+b) value approximation
 */
double logSumAll(List<double> logValues) {
  double result = LN0;
  for (double logValue in logValues) {
    result = logSum(result, logValue);
  }
  return result;
}

/**
 * Exact calculation of log(a+b) using log(a) and log(b) with formula
 * log(a+b) = log(b) + log(1 + exp(log(b)-log(a))) where log(b)>log(a)
 */
double logSumExact(double logA, double logB) {
  if (logA == double.INFINITY || logA==double.NEGATIVE_INFINITY)
    return logB;
  if (logB == double.INFINITY || logB==double.NEGATIVE_INFINITY)
    return logA;
  if (logA > logB) {
    double dif = logA - logB;
    return dif >= 30 ? logA : logA + log(1 + exp(-dif));
  } else {
    double dif = logB - logA;
    return dif >= 30 ? logB : logB + log(1 + exp(-dif));
  }
}

class LogMath {  
  
  static final LogMath _singleton = new LogMath._internal();
  
  final logSumLookup = new List<double>(30000);
  
  factory LogMath() {
    return _singleton;
  }
  
  LogMath._internal() {
    for (int i = 0; i < logSumLookup.length; i++) {
      logSumLookup[i] = log(1.0 + exp(-i / _SCALE));
    }
  }    
}



