package gmm;

import java.util.Arrays;
import java.util.Random;

public class GmmTest {

    public static void main(String[] args) {

        System.out.println("Generating test data");
        int gmmCount = 1000;
        int gaussPerGmm = 8;
        int dimension = 40;
        int inputAmount = 1000;
        int iterationCount = 7;

        Gmm[] gmms = new Gmm[gmmCount];
        for (int i = 0; i < gmmCount; i++) {
            GaussData[] gaussDataList = new GaussData[gaussPerGmm];
            for (int j = 0; j < gaussPerGmm; j++) {
                gaussDataList[j] = GaussData.random(dimension);
            }
            gmms[i] = getGmm(gaussDataList);
        }

        double[][] dataLarge = InputData.random(inputAmount, 40).data;
        long[] times = new long[iterationCount];
        System.out.println("calculating..");
        for (int it = 0; it < iterationCount; it++) {
            long start = System.currentTimeMillis();
            double total = 0.0;
            for (int i = 0; i < inputAmount; i++) {
                for (int j = 0; j < gmmCount; j++) {
                   // total += gmms[j].scoreLinear(dataLarge[i]);
                   // total += gmms[j].scoreLog(dataLarge[i]);
                   total += gmms[j].scoreLogSum(dataLarge[i]);
                }
            }
            long end = System.currentTimeMillis();
            times[it] = end - start;
            System.out.println("Elapsed:" + times[it]);
            System.out.println("Result:" + total);
        }

        Arrays.sort(times);
        double tot = 0.0;
        for (int it = 0; it < iterationCount; it++) {
            if (it != 0 && it != iterationCount - 1)
                tot = tot + times[it];
        }
        System.out.println("mean =" + tot / ((double) iterationCount - 2));
    }

    static Gmm getGmm(GaussData[] gd) {
        DiagonalGaussian[] gaussList = new DiagonalGaussian[gd.length];
        double[] weightList = new double[gd.length];
        for (int a = 0; a < gd.length; a++) {
            gaussList[a] = new DiagonalGaussian(gd[a].means, gd[a].variances);
            weightList[a] = gd[a].weight;
        }
        return new Gmm(weightList, gaussList);
    }

    static Random rnd = new Random(0xbeefcafe);

    static double _random() {
        return rnd.nextDouble() + 0.1;
    }

    static class GaussData {
        double[] means;
        double[] variances;
        int dimension;
        double weight;

        GaussData(double[] means, double[] variances, double weight) {
            this.means = means;
            this.variances = variances;
            this.dimension = means.length;
            this.weight = weight;
        }

        static GaussData random(int dimension) {
            double[] means = new double[dimension];
            double[] variances = new double[dimension];
            for (int i = 0; i < dimension; i++) {
                means[i] = _random();
                variances[i] = _random();
            }
            double weight = _random();
            return new GaussData(means, variances, weight);
        }
    }

    static class InputData {
        double[][] data;
        int size;
        int dimension;

        InputData(double[][] data, int size, int dimension) {
            this.data = data;
            this.size = size;
            this.dimension = dimension;
        }

        static InputData random(int size, int dimension) {
            double[][] data = new double[size][];
            for (int i = 0; i < data.length; i++) {
                data[i] = new double[dimension];
                for (int j = 0; j < dimension; j++) {
                    data[i][j] = _random();
                }
            }
            return new InputData(data, size, dimension);
        }
    }

    static class DiagonalGaussian {
        double[] means;
        double[] variances;
        double[] negativeHalfPrecisions;
        double C;

        DiagonalGaussian(double[] means, double[] variances) {
            this.means = means;
            this.variances = variances;
            // instead of using [-0.5 * 1/var[d]] during likelihood calculation we pre-compute the values.
            // This saves 1 mul 1 div operation.
            negativeHalfPrecisions = new double[variances.length];
            for (int i = 0; i < negativeHalfPrecisions.length; i++) {
                negativeHalfPrecisions[i] = -0.5 / variances[i];
            }
            // calculate the precomputed distance.
            // -0.5*SUM[d=1..D] ( log(2*PI) + log(var[d]) ) = -0.5*log(2*PI)*D -0.5 SUM[d=1..D](log(var[d]))
            double val = -0.5 * Math.log(2 * Math.PI) * variances.length;
            for (double variance : variances) {
                val -= (0.5 * Math.log(variance));
            }
            C = val;
        }

        /// Calculates linear likelihood of a given vector.
        double likelihood(double[] data) {
            double result = 1.0;
            for (int i = 0; i < means.length; i++) {
                double meanDif = data[i] - means[i];
                result *= (1 / Math.sqrt(2 * Math.PI * variances[i])) *
                        Math.exp(-0.5 * meanDif * meanDif / variances[i]);
            }
            return result;
        }

        /// Calculates linear likelihood of a given vector.
        double logLikelihood(double[] data) {
            double res = 0.0;
            for (int i = 0; i < means.length; i++) {
                final double dif = data[i] - means[i];
                res += ((dif * dif) * negativeHalfPrecisions[i]);
            }
            return C + res;
        }

    }

    static class Gmm {

        double[] mixtureWeights;
        double[] logMixtureWeights;
        DiagonalGaussian[] gaussians;

        Gmm(double[] mixtureWeights, DiagonalGaussian[] gaussians) {
            this.mixtureWeights = mixtureWeights;
            this.gaussians = gaussians;
            logMixtureWeights = new double[mixtureWeights.length];
            for (int i = 0; i < mixtureWeights.length; i++) {
                logMixtureWeights[i] = Math.log(mixtureWeights[i]);
            }
        }

        double scoreLinear(double[] data) {
            double sum = 0.0;
            for (int i = 0; i < gaussians.length; ++i) {
                sum += mixtureWeights[i] * gaussians[i].likelihood(data);
            }
            return sum;
        }

        double scoreLog(double[] data) {
            double result = mixtureWeights[0] + gaussians[0].logLikelihood(data);
            for (int i = 1; i < gaussians.length; ++i) {
                double b = mixtureWeights[i] + gaussians[i].logLikelihood(data);
                result = b + Math.log(1 + Math.exp(result - b));
            }
            return result;
        }

        double scoreLogSum(double[] data) {
            double result = mixtureWeights[0] + gaussians[0].logLikelihood(data);
            for (int i = 1; i < gaussians.length; ++i) {
                double b = mixtureWeights[i] + gaussians[i].logLikelihood(data);
                result = LogMath.logSum(result, b);
            }
            return result;
        }

    }

    static class LogMath {

        static final double _SCALE = 1000.0;

        static double[] logSumLookup = new double[20000];

        static {
            for (int i = 0; i < logSumLookup.length; i++) {
                logSumLookup[i] = Math.log(1.0 + Math.exp(-i / _SCALE));
            }
        }

        /**
         * Calculates an approximation of log(a+b) when log(a) and log(b) are given using the formula
         * log(a+b) = log(b) + log(1 + exp(log(a)-log(b))) where log(b)>log(a)
         * This method is an approximation because it uses a lookup table for log(1 + exp(log(b)-log(a))) part
         * This is useful for log-probabilities where values vary between -30 < log(p) <= 0
         * if difference between values is larger than 20 (which means sum of the numbers will be very close to the larger
         * value in linear domain) large value is returned instead of the logSum calculation because effect of the other
         * value is negligible
         */
        static double logSum(double logA, double logB) {
            if (logA > logB) {
                double dif = logA - logB; // logA-logB because during lookup calculation dif is multiplied with -1
                return dif >= 20.0 ? logA : logA + logSumLookup[(int) (dif * _SCALE)];
            } else {
                final double dif = logB - logA;
                return dif >= 20.0 ? logB : logB + logSumLookup[(int) (dif * _SCALE)];
            }
        }
    }

}


