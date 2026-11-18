package io.github.zzuegg.brainiac.experimental.data;

import java.util.SplittableRandom;

/**
 * A prior over small tabular classification tasks, the training distribution of prior-fitted
 * networks (TabPFN; Hollmann et al., 2023, <a href="https://arxiv.org/abs/2207.01848">
 * arXiv:2207.01848</a>). Each draw is a whole dataset. Three families:
 * <ul>
 *   <li>{@link Family#SCM}: a random structural causal model. Nodes of a random DAG are noisy
 *       non-linear functions of their parents; some nodes are observed as features and one is
 *       thresholded at its median into the label, so features can be causes or effects of it;</li>
 *   <li>{@link Family#RANDOM_MLP}: Gaussian inputs through a random two-layer network with a
 *       random activation, thresholded at the median;</li>
 *   <li>{@link Family#TREES}: a random axis-aligned decision tree of depth 3 with random leaf
 *       labels, a family the other two do not contain (a held-out prior for testing).</li>
 * </ul>
 * Features are standardized per dataset, and the label noise is a few percent.
 */
public final class TabularPrior {

    /** A family of dataset generators. */
    public enum Family {
        /** Random structural causal models. */
        SCM,
        /** Random MLPs on Gaussian inputs. */
        RANDOM_MLP,
        /** Random axis-aligned decision trees. */
        TREES
    }

    /**
     * One dataset.
     *
     * @param x {@code [rows][features]} standardized features
     * @param y 0/1 labels
     */
    public record Task(double[][] x, int[] y) {
        /** @return the number of rows */
        public int rows() {
            return y.length;
        }
    }

    private TabularPrior() {
    }

    /**
     * Draws a dataset whose labels are not all equal.
     *
     * @param family the generator family
     * @param rows the rows
     * @param features the features
     * @param random the random source
     * @return the dataset
     */
    public static Task sample(Family family, int rows, int features, SplittableRandom random) {
        for (int attempt = 0; ; attempt++) {
            Task t = switch (family) {
                case SCM -> scm(rows, features, random);
                case RANDOM_MLP -> mlp(rows, features, random);
                case TREES -> trees(rows, features, random);
            };
            int ones = 0;
            for (int v : t.y()) {
                ones += v;
            }
            if ((ones >= rows / 10 && ones <= rows - rows / 10) || attempt > 20) {
                return t;
            }
        }
    }

    /**
     * Draws a dataset from {@link Family#SCM} or {@link Family#RANDOM_MLP} with equal odds, the
     * mixture a prior-fitted network is trained on.
     *
     * @param rows the rows
     * @param features the features
     * @param random the random source
     * @return the dataset
     */
    public static Task sample(int rows, int features, SplittableRandom random) {
        return sample(random.nextBoolean() ? Family.SCM : Family.RANDOM_MLP, rows, features, random);
    }

    private static double act(int kind, double v) {
        return switch (kind) {
            case 0 -> Math.tanh(v);
            case 1 -> Math.max(0, v);
            case 2 -> Math.sin(v);
            default -> v;
        };
    }

    private static Task scm(int rows, int features, SplittableRandom r) {
        int nodes = features + 1 + r.nextInt(4);
        double[][] w = new double[nodes][nodes];
        for (int i = 0; i < nodes; i++) {
            for (int j = 0; j < i; j++) {
                if (r.nextDouble() < 0.5) {
                    w[i][j] = r.nextGaussian();
                }
            }
        }
        int[] kind = new int[nodes];
        double[] noise = new double[nodes];
        for (int i = 0; i < nodes; i++) {
            kind[i] = r.nextInt(4);
            noise[i] = 0.1 + 0.4 * r.nextDouble();
        }
        int[] perm = permutation(nodes, r);
        int target = perm[0];
        double[][] v = new double[rows][nodes];
        for (int n = 0; n < rows; n++) {
            for (int i = 0; i < nodes; i++) {
                double s = noise[i] * r.nextGaussian();
                for (int j = 0; j < i; j++) {
                    s += w[i][j] * v[n][j];
                }
                v[n][i] = i == 0 ? r.nextGaussian() : act(kind[i], s);
            }
        }
        double[][] x = new double[rows][features];
        double[] t = new double[rows];
        for (int n = 0; n < rows; n++) {
            for (int f = 0; f < features; f++) {
                x[n][f] = v[n][perm[1 + f]];
            }
            t[n] = v[n][target];
        }
        return finish(x, t, r);
    }

    private static Task mlp(int rows, int features, SplittableRandom r) {
        int hidden = 8 + r.nextInt(9);
        double[][] w1 = new double[features][hidden];
        double[] w2 = new double[hidden];
        for (int f = 0; f < features; f++) {
            for (int h = 0; h < hidden; h++) {
                w1[f][h] = r.nextDouble() < 0.7 ? r.nextGaussian() * 1.5 / Math.sqrt(features) : 0;
            }
        }
        for (int h = 0; h < hidden; h++) {
            w2[h] = r.nextGaussian();
        }
        int kind = r.nextInt(3);
        double[][] x = new double[rows][features];
        double[] t = new double[rows];
        for (int n = 0; n < rows; n++) {
            for (int f = 0; f < features; f++) {
                x[n][f] = r.nextGaussian();
            }
            double s = 0;
            for (int h = 0; h < hidden; h++) {
                double a = 0;
                for (int f = 0; f < features; f++) {
                    a += x[n][f] * w1[f][h];
                }
                s += w2[h] * act(kind, a + 0.3 * r.nextGaussian());
            }
            t[n] = s;
        }
        return finish(x, t, r);
    }

    private static Task trees(int rows, int features, SplittableRandom r) {
        int depth = 3;
        int[] feature = new int[(1 << depth) - 1];
        double[] threshold = new double[feature.length];
        for (int i = 0; i < feature.length; i++) {
            feature[i] = r.nextInt(features);
            threshold[i] = 0.8 * r.nextGaussian();
        }
        double[] leaf = new double[1 << depth];
        for (int i = 0; i < leaf.length; i++) {
            leaf[i] = i % 2 == 0 ? r.nextDouble() : -r.nextDouble();
        }
        double[][] x = new double[rows][features];
        double[] t = new double[rows];
        for (int n = 0; n < rows; n++) {
            for (int f = 0; f < features; f++) {
                x[n][f] = r.nextGaussian();
            }
            int node = 0;
            for (int d = 0; d < depth; d++) {
                node = 2 * node + (x[n][feature[node]] < threshold[node] ? 1 : 2);
            }
            t[n] = leaf[node - feature.length] + 0.1 * r.nextGaussian();
        }
        return finish(x, t, r);
    }

    private static Task finish(double[][] x, double[] t, SplittableRandom r) {
        int rows = t.length;
        double[] sorted = t.clone();
        java.util.Arrays.sort(sorted);
        double median = sorted[rows / 2];
        int[] y = new int[rows];
        for (int n = 0; n < rows; n++) {
            y[n] = t[n] > median ? 1 : 0;
            if (r.nextDouble() < 0.03) {
                y[n] = 1 - y[n];
            }
        }
        int features = x[0].length;
        for (int f = 0; f < features; f++) {
            double mean = 0;
            for (double[] row : x) {
                mean += row[f];
            }
            mean /= rows;
            double var = 0;
            for (double[] row : x) {
                var += (row[f] - mean) * (row[f] - mean);
            }
            double sd = Math.sqrt(var / rows) + 1e-9;
            for (double[] row : x) {
                row[f] = (row[f] - mean) / sd;
            }
        }
        return new Task(x, y);
    }

    private static int[] permutation(int n, SplittableRandom r) {
        int[] p = new int[n];
        for (int i = 0; i < n; i++) {
            p[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = p[i];
            p[i] = p[j];
            p[j] = t;
        }
        return p;
    }
}
