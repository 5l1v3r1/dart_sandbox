import 'dart:typed_data';
import 'dart:math';

final LogMath logMath = new LogMath();

class LogMath {  

  const double _SCALE = 1000.0;  
  
  static final LogMath _singleton = new LogMath._internal();
  
  final List<double> logSumLookup = new List<double>(20000);
  final Float32List logSumLookupTyped = new Float32List(20000);
  
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
      double dif = logA - logB; // logA-logB because during lookup calculation dif is multiplied with -1
      return dif >= 20.0 ? logA : logA + logSumLookup[(dif * _SCALE).toInt()];
    } else {
      final double dif = logB - logA;
      return dif >= 20.0 ? logB : logB + logSumLookup[(dif * _SCALE).toInt()];
    }
  }  
  
  double logSumTyped(double logA, double logB) {
    if (logA > logB) {
      double dif = logA - logB; // logA-logB because during lookup calculation dif is multiplied with -1
      return dif >= 20.0 ? logA : logA + logSumLookupTyped[(dif * _SCALE).toInt()];
    } else {
      final double dif = logB - logA;
      return dif >= 20.0 ? logB : logB + logSumLookupTyped[(dif * _SCALE).toInt()];
    }
  }   
}