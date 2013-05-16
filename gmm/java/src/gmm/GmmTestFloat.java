package gmm;

import java.util.Arrays;
import java.util.Random;

public class GmmTestFloat {

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

        float[][] dataLarge = InputData.random(inputAmount, 40).data;
        long[] times = new long[iterationCount];
        System.out.println("calculating..");
        for (int it = 0; it < iterationCount; it++) {
            long start = System.currentTimeMillis();
            float total = 0.0f;
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
        float tot = 0.0f;
        for (int it = 0; it < iterationCount; it++) {
            if (it != 0 && it != iterationCount - 1)
                tot = tot + times[it];
        }
        System.out.println("mean =" + tot / ((float) iterationCount - 2));
    }

    static Gmm getGmm(GaussData[] gd) {
        DiagonalGaussian[] gaussList = new DiagonalGaussian[gd.length];
        float[] weightList = new float[gd.length];
        for (int a = 0; a < gd.length; a++) {
            gaussList[a] = new DiagonalGaussian(gd[a].means, gd[a].variances);
            weightList[a] = gd[a].weight;
        }
        return new Gmm(weightList, gaussList);
    }

    static Random rnd = new Random(0xbeefcafe);

    static float _random() {
        return rnd.nextFloat() + 0.1f;
    }

    static class GaussData {
        float[] means;
        float[] variances;
        int dimension;
        float weight;

        GaussData(float[] means, float[] variances, float weight) {
            this.means = means;
            this.variances = variances;
            this.dimension = means.length;
            this.weight = weight;
        }

        static GaussData random(int dimension) {
            float[] means = new float[dimension];
            float[] variances = new float[dimension];
            for (int i = 0; i < dimension; i++) {
                means[i] = _random();
                variances[i] = _random();
            }
            float weight = _random();
            return new GaussData(means, variances, weight);
        }
    }

    static class InputData {
        float[][] data;
        int size;
        int dimension;

        InputData(float[][] data, int size, int dimension) {
            this.data = data;
            this.size = size;
            this.dimension = dimension;
        }

        static InputData random(int size, int dimension) {
            float[][] data = new float[size][];
            for (int i = 0; i < data.length; i++) {
                data[i] = new float[dimension];
                for (int j = 0; j < dimension; j++) {
                    data[i][j] = _random();
                }
            }
            return new InputData(data, size, dimension);
        }
    }

    static class DiagonalGaussian {
        float[] means;
        float[] negativeHalfPrecisions;
        float C;

        DiagonalGaussian(float[] means, float[] variances) {
            this.means = means;
            // instead of using [-0.5 * 1/var[d]] during likelihood calculation we pre-compute the values.
            // This saves 1 mul 1 div operation.
            negativeHalfPrecisions = new float[variances.length];
            for (int i = 0; i < negativeHalfPrecisions.length; i++) {
                negativeHalfPrecisions[i] = -0.5f / variances[i];
            }
            // calculate the precomputed distance.
            // -0.5*SUM[d=1..D] ( log(2*PI) + log(var[d]) ) = -0.5*log(2*PI)*D -0.5 SUM[d=1..D](log(var[d]))
            float val = (float) (-0.5 * Math.log(2 * Math.PI) * variances.length);
            for (float variance : variances) {
                val -= (0.5 * Math.log(variance));
            }
            C = val;
        }

        /// Calculates linear likelihood of a given vector.
        double logLikelihood(float[] data) {
            double res = 0.0f;
            for (int i = 0; i < means.length; i++) {
                final double dif = data[i] - means[i];
                res += ((dif * dif) * negativeHalfPrecisions[i]);
            }
            return C + res;
        }

    }

    static class Gmm {

        float[] logMixtureWeights;
        DiagonalGaussian[] gaussians;

        Gmm(float[] mixtureWeights, DiagonalGaussian[] gaussians) {
            this.gaussians = gaussians;
            logMixtureWeights = new float[mixtureWeights.length];
            for (int i = 0; i < mixtureWeights.length; i++) {
                logMixtureWeights[i] = (float) Math.log(mixtureWeights[i]);
            }
        }

        double scoreLog(float[] data) {
            double result = logMixtureWeights[0] + gaussians[0].logLikelihood(data);
            for (int i = 1; i < gaussians.length; ++i) {
                double b = logMixtureWeights[i] + gaussians[i].logLikelihood(data);
                result = (float) (b + Math.log(1 + Math.exp(result - b)));
            }
            return result;
        }

        double scoreLogSum(float[] data) {
            double result = logMixtureWeights[0] + gaussians[0].logLikelihood(data);
            for (int i = 1; i < gaussians.length; ++i) {
                double b = logMixtureWeights[i] + gaussians[i].logLikelihood(data);
                result = LogMath.logSum(result, b);
            }
            return result;
        }

    }

    static class LogMath {

        static final float _SCALE = 1000.0f;

        static double[] logSumLookup = new double[20000];

        static {
            for (int i = 0; i < logSumLookup.length; i++) {
                logSumLookup[i] = Math.log(1.0 + Math.exp(-i / _SCALE));
            }
        }

        static double logSum(double logA, double logB) {
            if (logA > logB) {
                double dif = logA - logB; // logA-logB because during lookup calculation dif is multiplied with -1
                return dif >= 20.0 ? logA : logA + logSumLookup[(int) (dif * _SCALE)];
            } else {
                double dif = logB - logA;
                return dif >= 20.0 ? logB : logB + logSumLookup[(int) (dif * _SCALE)];
            }
        }
    }

}


