library gmm;

import 'dart:math';
import 'dart:typed_data';
import 'gmm_log_lookup_typed.dart';
//import 'gmm_log_lookup.dart';
//import 'gmm_log1.dart';
//import 'gmm_simd.dart';

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
  
  Float32List getMeansTyped() {
    return new Float32List.fromList(means);
  }
  
  Float32List getVariancesTyped() {
    return new Float32List.fromList(variances);
  }
  
  Float32x4List getMeansSimd() {
    return toFloat32x4List(means);
  }
  
  Float32x4List getVariancesSimd() {
    return toFloat32x4List(variances);
  }
  
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

Gmm getGmm(List<GaussData> gd) {
  var gaussList = new List();
  var weightList = new List();  
  for(int a = 0; a<gd.length; a++) {
    //gaussList.add(new DiagonalGaussian(gd[a].means, gd[a].variances));    
    gaussList.add(new DiagonalGaussian(gd[a].getMeansTyped(), gd[a].getVariancesTyped()));    
    //gaussList.add(new DiagonalGaussian(gd[a].getMeansSimd(), gd[a].getVariancesSimd()));
    weightList.add(gd[a].weight);       
  }
  return new Gmm(weightList, gaussList);
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
 
void calculate(DiagonalGaussian gaussian, List data) {
  Stopwatch sw = new Stopwatch()..start();
  double tot = 0.0;    
  for (int i = 0; i<data.length; ++i) {
    tot= tot+ gaussian.logLikelihood(data[i]);
  }
  print("log-likelihood=$tot");      
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
      gaussDataList[j]= new GaussData.random(dimension);
    }
    gmms[i]=getGmm(gaussDataList);
  } 
  //List<List<double>> dataLarge = new InputData.random(inputAmount,40).data;
  List<Float32List> dataLarge = new InputData.random(inputAmount,40).getAsFloat32List();
  //List<Float32x4List> dataLarge = new InputData.random(inputAmount,40).getAsFloat32x4List();  

  List<int> times = new List(iterationCount);
  print("calculating..");
  for(int it = 0; it<iterationCount; it++) {
    Stopwatch sw = new Stopwatch()..start();
    double total = 0.0;  
    for(int i = 0; i<inputAmount;i++) {
      for(int j = 0; j<gmmCount;j++) {
        total+= gmms[j].score(dataLarge[i]);
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

