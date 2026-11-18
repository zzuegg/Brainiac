package io.github.zzuegg.brainiac.experiments.probabilistic;

import io.github.zzuegg.brainiac.experimental.ml.KernelFunction;
import io.github.zzuegg.brainiac.experimental.ml.KernelSvm;
import io.github.zzuegg.brainiac.experimental.ml.NystromKernelRidge;
import io.github.zzuegg.brainiac.experimental.ml.RandomFourierFeatures;
import io.github.zzuegg.brainiac.experimental.train.RidgeRegression;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Kernel machines as the training set grows (the scaling split of {@code ClassicMlShootout}):
 * two-arm spirals (angle noise 0.8, overlapping arms) and two noisy concentric circles, labels {@code +-1}, an RBF
 * kernel. Exact kernel ridge regression and a kernel SVM where they are affordable, Nyström KRR
 * and FALKON (25 CG iterations) with {@code m} landmarks, and random Fourier features with ridge regression at the
 * same feature count. Scores: test error and fit time.
 */
public final class KernelScaling {

    private KernelScaling() {
    }

    private record Split(double[][] x, double[] y, int[] labels, double[][] xt, int[] lt) {
    }

    private static Split data(String task, int n, long seed) {
        double[][][] xs = new double[2][][];
        int[][] ls = new int[2][];
        int[] sizes = {n, 2000};
        for (int s = 0; s < 2; s++) {
            if (task.equals("spirals")) {
                ToyTasks.Data d = ToyTasks.spiral(sizes[s] / 2, 2, 0.8, seed + s);
                xs[s] = new double[d.size()][];
                for (int i = 0; i < d.size(); i++) {
                    xs[s][i] = new double[] {d.x()[i][0], d.x()[i][1]};
                }
                ls[s] = d.y();
            } else {
                SplittableRandom r = new SplittableRandom(seed + s);
                xs[s] = new double[sizes[s]][];
                ls[s] = new int[sizes[s]];
                for (int i = 0; i < sizes[s]; i++) {
                    int c = r.nextInt(2);
                    double a = 2 * Math.PI * r.nextDouble();
                    double rad = (c == 0 ? 0.4 : 0.8) + 0.12 * r.nextGaussian();
                    xs[s][i] = new double[] {rad * Math.cos(a), rad * Math.sin(a)};
                    ls[s][i] = c;
                }
            }
        }
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            y[i] = ls[0][i] == 1 ? 1 : -1;
        }
        return new Split(xs[0], y, ls[0], xs[1], ls[1]);
    }

    private static double error(double[] scores, int[] labels) {
        int wrong = 0;
        for (int i = 0; i < scores.length; i++) {
            wrong += (scores[i] > 0 ? 1 : 0) != labels[i] ? 1 : 0;
        }
        return wrong / (double) scores.length;
    }

    /**
     * Test error and fit time per method, size and landmark count.
     *
     * @param task {@code "spirals"} or {@code "circles"}
     * @param sizes the training-set sizes
     * @param landmarks the landmark (feature) counts
     * @param exactUpTo the largest size for exact KRR and the SVM
     * @param seed the seed
     * @return {@code "method n=.. m=.."} to {@code {test error, fit seconds}}
     */
    public static Map<String, double[]> compare(String task, int[] sizes, int[] landmarks, int exactUpTo, long seed) {
        KernelFunction k = KernelFunction.rbf(task.equals("spirals") ? 20 : 5);
        double lambda = 1e-6;
        Map<String, double[]> out = new LinkedHashMap<>();
        for (int n : sizes) {
            Split s = data(task, n, seed);
            if (n <= exactUpTo) {
                long t = System.nanoTime();
                NystromKernelRidge exact = NystromKernelRidge.exact(s.x(), s.y(), k, lambda);
                out.put("exact KRR n=" + n, new double[] {error(exact.predict(s.xt()), s.lt()), secs(t)});
                t = System.nanoTime();
                KernelSvm svm = KernelSvm.fit(s.x(), s.labels(), 2, k, 10, 1e-3);
                int[] pred = svm.predict(s.xt());
                double wrong = 0;
                for (int i = 0; i < pred.length; i++) {
                    wrong += pred[i] != s.lt()[i] ? 1 : 0;
                }
                out.put("kernel SVM n=" + n, new double[] {wrong / pred.length, secs(t)});
            }
            for (int m : landmarks) {
                long t = System.nanoTime();
                NystromKernelRidge ny = NystromKernelRidge.nystrom(s.x(), s.y(), k, m, lambda, seed);
                out.put("Nystrom n=" + n + " m=" + m, new double[] {error(ny.predict(s.xt()), s.lt()), secs(t)});
                t = System.nanoTime();
                NystromKernelRidge fa = NystromKernelRidge.falkon(s.x(), s.y(), k, m, lambda, 25, seed);
                out.put("FALKON n=" + n + " m=" + m, new double[] {error(fa.predict(s.xt()), s.lt()), secs(t)});
                t = System.nanoTime();
                RandomFourierFeatures rff = new RandomFourierFeatures(2, m, task.equals("spirals") ? 20 : 5,
                        new SplittableRandom(seed));
                double[][] yy = new double[n][];
                for (int i = 0; i < n; i++) {
                    yy[i] = new double[] {s.y()[i]};
                }
                double[][] w = RidgeRegression.fit(rff.transform(s.x()), yy, n * lambda);
                double[][] ft = rff.transform(s.xt());
                double[] p = new double[ft.length];
                for (int i = 0; i < ft.length; i++) {
                    p[i] = RidgeRegression.predict(w, ft[i])[0];
                }
                out.put("RFF+ridge n=" + n + " m=" + m, new double[] {error(p, s.lt()), secs(t)});
            }
        }
        return out;
    }

    private static double secs(long start) {
        return (System.nanoTime() - start) / 1e9;
    }
}
