import 'dart:typed_data';
import 'dart:math';

final LogMath logMath = new LogMath();

class LogMath {  

  const _SCALE = 1000.0;  
  
  static final _singleton = new LogMath._internal();
  
  final logSumLookup = new List<double>(20000);
  final logSumLookupTyped = new Float32List(20000);
  
  factory LogMath() {
    return _singleton;
  }
  
  LogMath._internal() {
    for (int i = 0; i < logSumLookup.length; i++) {
      logSumLookup[i] = log(1.0 + exp(-i / _SCALE));
      logSumLookupTyped[i] = logSumLookup[i]; 
    }
  }
  
  double logSum(double logA, double logB) {
    if (logA > logB) {
      int index = ((logA - logB) * _SCALE).toInt(); 
      return index >= 20000 ? logA : logA + logSumLookup[index];
    } else {
      int index = ((logB - logA) * _SCALE).toInt(); 
      return index >= 20000 ? logB : logB + logSumLookup[index];
    }
  }  
  
  double logSumTyped(double logA, double logB) {
    if (logA > logB) {
      int index = ((logA - logB) * _SCALE).toInt(); 
      return index >= 20000 ? logA : logA + logSumLookupTyped[index];
    } else {
      int index = ((logB - logA) * _SCALE).toInt(); 
      return index >= 20000 ? logB : logB + logSumLookupTyped[index];
    }
  }   
}