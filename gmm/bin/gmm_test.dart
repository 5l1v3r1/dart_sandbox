library gmm;

import 'dart:math';
import 'dart:typed_data';
import 'gmm_log_lookup.dart';
import 'gmm_log_lookup_typed.dart';
import 'gmm_simd.dart';

Random random = new Random(0xbeefcafe);
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
  
  List<Float32x4List> getAsFloat32x4List() {
    List<Float32x4List> result = new List(data.length);
    for (int i = 0; i < data.length; i++) {
      result[i]=toFloat32x4List(data[i]);      
    }     
    return result;    
  }
}

List<LinearGmm> getLinearGmms() {
  var gmms = new List(gmmCount);
  for(int i = 0; i<gmmCount;i++) {
    var gaussList = new List(gaussPerGmm);
    var weightList = new List(gaussPerGmm);
    for(int j = 0; j<gaussPerGmm;j++) {
      var gaussData = new GaussData.random(dimension);
      gaussList[j] = new ListDiagonalGaussian(gaussData.means, gaussData.variances);
      weightList[j]= gaussData.weight;      
    }  
    gmms[i]=new LinearGmm(weightList, gaussList);
  }   
  return gmms;  
}

List<LogGmm> getLogGmms() {
  var gmms = new List(gmmCount);
  for(int i = 0; i<gmmCount;i++) {
    var gaussList = new List(gaussPerGmm);
    var weightList = new List(gaussPerGmm);
    for(int j = 0; j<gaussPerGmm;j++) {
      var gaussData = new GaussData.random(dimension);
      gaussList[j] = new ListDiagonalGaussian(gaussData.means, gaussData.variances);
      weightList[j]= gaussData.weight;      
    }  
    gmms[i]=new LogGmm(weightList, gaussList);
  }   
  return gmms;  
}

List<LogsumLookupGmm> getLogsumLookupGmms() {
  var gmms = new List(gmmCount);
  for(int i = 0; i<gmmCount;i++) {
    var gaussList = new List(gaussPerGmm);
    var weightList = new List(gaussPerGmm);
    for(int j = 0; j<gaussPerGmm;j++) {
      var gaussData = new GaussData.random(dimension);
      gaussList[j] = new ListDiagonalGaussian(gaussData.means, gaussData.variances);
      weightList[j]= gaussData.weight;      
    }  
    gmms[i]=new LogsumLookupGmm(weightList, gaussList);
  }   
  return gmms;  
}

List<TypedGmm> getTypedGmms() {
  var gmms = new List(gmmCount);
  for(int i = 0; i<gmmCount;i++) {
    var gaussList = new List(gaussPerGmm);
    var weightList = new List(gaussPerGmm);
    for(int j = 0; j<gaussPerGmm;j++) {
      var gaussData = new GaussData.random(dimension);
      gaussList[j] = new TypedDiagonalGaussian(gaussData.means, gaussData.variances);
      weightList[j]= gaussData.weight;      
    }  
    gmms[i]=new TypedGmm(weightList, gaussList);
  }   
  return gmms;  
}

List<SimdGmm> getSimdGmms() {
  var gmms = new List(gmmCount);
  for(int i = 0; i<gmmCount;i++) {
    var gaussList = new List(gaussPerGmm);
    var weightList = new List(gaussPerGmm);
    for(int j = 0; j<gaussPerGmm;j++) {
      var gaussData = new GaussData.random(dimension);
      gaussList[j] = new SimdDiagonalGaussian(gaussData.means, gaussData.variances);
      weightList[j]= gaussData.weight;      
    }  
    gmms[i]=new SimdGmm(weightList, gaussList);
  }   
  return gmms;
}

test(List<List> data, List gmms, int inputAmount, int iterationCount) { 
  List<int> times = new List(iterationCount);
  for(int it = 0; it<iterationCount; it++) {
    Stopwatch sw = new Stopwatch()..start();
    double total = 0.0;  
    for(int i = 0; i<inputAmount;i++) {
      for(int j = 0; j<gmmCount;j++) {
        total+= gmms[j].score(data[i]);
      }
    }
    sw.stop();
    times[it] = sw.elapsedMilliseconds;
    print("Iteration=$it, elapsed: ${sw.elapsedMilliseconds}");
  }
  times.sort();
  double tot = 0.0;
  for(int it = 0; it<iterationCount; it++) {
    if(it!=0 && it!=iterationCount-1)
      tot = tot + times[it];
  }
  print("mean = ${tot/(iterationCount-2).toDouble()}");
  print("");
}

int gmmCount = 1000;
int gaussPerGmm = 8;
int dimension = 40;

testAll(int inputLength, int iterationCount) {

  InputData input = new InputData.random(inputLength , dimension);
  // below test is commented out because it is too slow.
  // print("Linear Gmm , List<double>");
  // test(input.data, getLinearGmms(), inputLength, iterationCount);
  print("Log Gmm, List<double>");
  test(input.data, getLogGmms(), inputLength, iterationCount);
  print("Log Gmm with Logsum lookup, List<double>");
  test(input.data, getLogsumLookupGmms(), inputLength, iterationCount);  
  print("Log Gmm with Logsum lookup, Typed Data");  
  test(input.getAsFloat32List(), getTypedGmms(), inputLength, iterationCount);  
  print("Log Gmm with Logsum lookup, SIMD");  
  test(input.getAsFloat32x4List(), getSimdGmms(), inputLength, iterationCount);  
}


main() {  
  print("Warm-up");
  testAll(1,3);
  print("Testing...");
  testAll(1000,7);
  print("Done.");
}

