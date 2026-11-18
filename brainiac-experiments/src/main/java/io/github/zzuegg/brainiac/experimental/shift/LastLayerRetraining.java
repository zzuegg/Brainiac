package io.github.zzuegg.brainiac.experimental.shift;

import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Deep feature reweighting (DFR; Kirichenko et al., ICLR 2023,
 * <a href="https://arxiv.org/abs/2204.02937">arXiv:2204.02937</a>): ERM features already
 * encode the core attribute even when the ERM classifier leans on a spurious one, so only the
 * last layer is retrained, as an {@code l1}-regularised logistic regression on standardised
 * features, on group-balanced subsamples of a small held-out set; the weights of several
 * subsamples are averaged. Group labels are needed only on that held-out set.
 *
 * <p>The fit is proximal gradient descent (ISTA) on the multinomial logistic loss; the result
 * is a core {@link Dense} head with the standardisation folded in, a drop-in replacement for
 * the ERM head.
 */
public final class LastLayerRetraining {

    private LastLayerRetraining() {
    }

    /**
     * Group-balanced subsample: every group cut to the size of the smallest.
     *
     * @param groups the group of each row
     * @param groupCount the number of groups
     * @param random the sampling source
     * @return the chosen rows
     */
    public static int[] balancedSubsample(int[] groups, int groupCount, SplittableRandom random) {
        List<List<Integer>> by = new ArrayList<>();
        for (int g = 0; g < groupCount; g++) {
            by.add(new ArrayList<>());
        }
        for (int i = 0; i < groups.length; i++) {
            by.get(groups[i]).add(i);
        }
        int min = Integer.MAX_VALUE;
        for (List<Integer> l : by) {
            if (!l.isEmpty()) {
                min = Math.min(min, l.size());
            }
        }
        List<Integer> out = new ArrayList<>();
        for (List<Integer> l : by) {
            List<Integer> copy = new ArrayList<>(l);
            for (int i = 0; i < Math.min(min, copy.size()); i++) {
                int j = i + random.nextInt(copy.size() - i);
                Integer t = copy.get(i);
                copy.set(i, copy.get(j));
                copy.set(j, t);
                out.add(copy.get(i));
            }
        }
        return out.stream().mapToInt(Integer::intValue).toArray();
    }

    /**
     * Retrains the last layer.
     *
     * @param features held-out features {@code [n, d]}
     * @param labels their classes
     * @param groups their groups (class times spurious attribute)
     * @param groupCount the number of groups
     * @param classes the number of classes
     * @param l1 the {@code l1} strength (on standardised features), such as 0.01
     * @param subsamples the number of balanced subsamples to average, 10 in the paper
     * @param seed the seed
     * @return the new head
     */
    public static Dense retrain(Tensor features, int[] labels, int[] groups, int groupCount, int classes, double l1,
            int subsamples, long seed) {
        int n = labels.length;
        int d = features.shape().dim(1);
        double[] f = features.toDoubleArray();
        double[] mean = new double[d];
        double[] sd = new double[d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                mean[j] += f[i * d + j] / n;
            }
        }
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                double x = f[i * d + j] - mean[j];
                sd[j] += x * x / n;
            }
        }
        for (int j = 0; j < d; j++) {
            sd[j] = Math.sqrt(sd[j]) + 1e-6;
        }
        double[] z = new double[n * d];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                z[i * d + j] = (f[i * d + j] - mean[j]) / sd[j];
            }
        }
        SplittableRandom r = new SplittableRandom(seed);
        double[] w = new double[d * classes];
        double[] b = new double[classes];
        for (int s = 0; s < subsamples; s++) {
            int[] rows = balancedSubsample(groups, groupCount, r);
            double[][] fit = logistic(z, d, labels, rows, classes, l1, 300);
            for (int i = 0; i < w.length; i++) {
                w[i] += fit[0][i] / subsamples;
            }
            for (int c = 0; c < classes; c++) {
                b[c] += fit[1][c] / subsamples;
            }
        }
        // fold the standardisation: z W + b = x (W / sd) + (b - mean / sd W)
        float[] wf = new float[d * classes];
        float[] bf = new float[classes];
        for (int c = 0; c < classes; c++) {
            double bias = b[c];
            for (int j = 0; j < d; j++) {
                wf[j * classes + c] = (float) (w[j * classes + c] / sd[j]);
                bias -= mean[j] / sd[j] * w[j * classes + c];
            }
            bf[c] = (float) bias;
        }
        return new Dense(d, classes, Activation.IDENTITY, Tensor.of(Shape.of(d, classes), wf),
                Tensor.of(Shape.of(classes), bf));
    }

    /** The largest eigenvalue of {@code Z^T Z / m} over the rows, by power iteration. */
    private static double topEigenvalue(double[] z, int d, int[] rows) {
        double[] v = new double[d];
        java.util.Arrays.fill(v, 1 / Math.sqrt(d));
        double lambda = 1;
        for (int it = 0; it < 30; it++) {
            double[] next = new double[d];
            for (int i : rows) {
                double dot = 0;
                for (int j = 0; j < d; j++) {
                    dot += z[i * d + j] * v[j];
                }
                for (int j = 0; j < d; j++) {
                    next[j] += dot * z[i * d + j] / rows.length;
                }
            }
            double norm = 0;
            for (double x : next) {
                norm += x * x;
            }
            lambda = Math.sqrt(norm);
            for (int j = 0; j < d; j++) {
                v[j] = next[j] / Math.max(1e-12, lambda);
            }
        }
        return lambda;
    }

    /** ISTA on the mean multinomial logistic loss plus {@code l1 |W|_1}. */
    private static double[][] logistic(double[] z, int d, int[] labels, int[] rows, int classes, double l1,
            int iterations) {
        double[] w = new double[d * classes];
        double[] b = new double[classes];
        double step = 1 / (0.5 * (topEigenvalue(z, d, rows) + 1) + 1e-9);
        double[] logits = new double[classes];
        for (int it = 0; it < iterations; it++) {
            double[] gw = new double[w.length];
            double[] gb = new double[classes];
            for (int i : rows) {
                double max = Double.NEGATIVE_INFINITY;
                for (int c = 0; c < classes; c++) {
                    double s = b[c];
                    for (int j = 0; j < d; j++) {
                        s += z[i * d + j] * w[j * classes + c];
                    }
                    logits[c] = s;
                    max = Math.max(max, s);
                }
                double sum = 0;
                for (int c = 0; c < classes; c++) {
                    logits[c] = Math.exp(logits[c] - max);
                    sum += logits[c];
                }
                for (int c = 0; c < classes; c++) {
                    double g = logits[c] / sum - (labels[i] == c ? 1 : 0);
                    gb[c] += g / rows.length;
                    for (int j = 0; j < d; j++) {
                        gw[j * classes + c] += g * z[i * d + j] / rows.length;
                    }
                }
            }
            for (int k = 0; k < w.length; k++) {
                double v = w[k] - step * gw[k];
                w[k] = Math.signum(v) * Math.max(0, Math.abs(v) - step * l1);
            }
            for (int c = 0; c < classes; c++) {
                b[c] -= step * gb[c];
            }
        }
        return new double[][] {w, b};
    }
}
