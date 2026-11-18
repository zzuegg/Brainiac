package io.github.zzuegg.brainiac.experimental.data;

import io.github.zzuegg.brainiac.experimental.ml.RandomForest;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * Cheap data valuation: a value per training example without retraining a network for every
 * subset, next to {@link InRunShapley} and {@link DataPruner}.
 *
 * <ul>
 *   <li>{@link #knnShapley}: the exact Shapley value of each training point for a
 *       {@code k}-nearest-neighbour classifier's validation accuracy, by the closed-form
 *       recursion over points sorted by distance to each validation point, in
 *       {@code O(N log N)} per validation point (Jia et al., VLDB 2019,
 *       <a href="https://arxiv.org/abs/1908.08619">arXiv:1908.08619</a>).</li>
 *   <li>{@link #dataOob}: Data-OOB, the out-of-bag accuracy of a bagging ensemble of decision
 *       trees on each point, using only the trees that did not see it (Kwon and Zou, ICML 2023,
 *       <a href="https://arxiv.org/abs/2304.07718">arXiv:2304.07718</a>).</li>
 *   <li>{@link #leverageScores}: the ridge leverage {@code x_iᵀ (XᵀX + λI)⁻¹ x_i}, how much a
 *       point extends the span of the data, a training-free and label-free value (Geometric
 *       Data Valuation via Leverage Scores, NeurIPS 2025,
 *       <a href="https://arxiv.org/abs/2511.02100">arXiv:2511.02100</a>); {@link #leverageSelect}
 *       picks diverse subsets greedily by it.</li>
 * </ul>
 */
public final class DataValuation {

    private DataValuation() {
    }

    /**
     * KNN-Shapley values, averaged over the validation points.
     *
     * @param trainX training features
     * @param trainY training labels
     * @param validX validation features
     * @param validY validation labels
     * @param k the number of neighbours
     * @return one value per training point (higher is more helpful)
     */
    public static double[] knnShapley(double[][] trainX, int[] trainY, double[][] validX, int[] validY, int k) {
        int n = trainX.length;
        double[] value = new double[n];
        Integer[] order = new Integer[n];
        double[] dist = new double[n];
        double[] s = new double[n];
        for (int t = 0; t < validX.length; t++) {
            for (int i = 0; i < n; i++) {
                order[i] = i;
                double d = 0;
                for (int j = 0; j < validX[t].length; j++) {
                    double e = trainX[i][j] - validX[t][j];
                    d += e * e;
                }
                dist[i] = d;
            }
            Arrays.sort(order, (a, b) -> Double.compare(dist[a], dist[b]));
            int y = validY[t];
            // s(α_N) = 1[y_αN = y] / N; s(α_i) = s(α_(i+1)) + (1[y_αi = y] - 1[y_α(i+1) = y]) / K * min(K, i) / i
            s[n - 1] = (trainY[order[n - 1]] == y ? 1.0 : 0.0) / n;
            for (int i = n - 2; i >= 0; i--) {
                int rank = i + 1;
                double diff = (trainY[order[i]] == y ? 1.0 : 0.0) - (trainY[order[i + 1]] == y ? 1.0 : 0.0);
                s[i] = s[i + 1] + diff / k * Math.min(k, rank) / rank;
            }
            for (int i = 0; i < n; i++) {
                value[order[i]] += s[i] / validX.length;
            }
        }
        return value;
    }

    /**
     * Data-OOB values from bagged decision trees.
     *
     * @param x features
     * @param y labels
     * @param classes the number of classes
     * @param bags bootstrap models
     * @param maxDepth the depth of each tree
     * @param seed the seed
     * @return each point's out-of-bag accuracy (NaN if it was in every bag)
     */
    public static double[] dataOob(double[][] x, int[] y, int classes, int bags, int maxDepth, long seed) {
        int n = x.length;
        SplittableRandom r = new SplittableRandom(seed);
        double[] right = new double[n];
        int[] counted = new int[n];
        for (int b = 0; b < bags; b++) {
            int[] sample = new int[n];
            boolean[] in = new boolean[n];
            double[][] bx = new double[n][];
            int[] by = new int[n];
            for (int i = 0; i < n; i++) {
                sample[i] = r.nextInt(n);
                in[sample[i]] = true;
                bx[i] = x[sample[i]];
                by[i] = y[sample[i]];
            }
            RandomForest tree = RandomForest.fit(bx, by, classes, new RandomForest.Settings(
                    RandomForest.Mode.EXTRA_TREES, 1, maxDepth, 1, 0, r.nextLong()));
            for (int i = 0; i < n; i++) {
                if (!in[i]) {
                    double[] p = tree.predictProbability(x[i]);
                    int best = 0;
                    for (int c = 1; c < p.length; c++) {
                        best = p[c] > p[best] ? c : best;
                    }
                    right[i] += best == y[i] ? 1 : 0;
                    counted[i]++;
                }
            }
        }
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            out[i] = counted[i] == 0 ? Double.NaN : right[i] / counted[i];
        }
        return out;
    }

    /**
     * Ridge leverage scores.
     *
     * @param x features {@code [n][d]}
     * @param lambda the ridge; 0 for plain leverage (then {@code XᵀX} must be invertible)
     * @return {@code x_iᵀ (XᵀX + λI)⁻¹ x_i} per point; they sum to the statistical dimension
     */
    public static double[] leverageScores(double[][] x, double lambda) {
        int d = x[0].length;
        double[] gram = new double[d * d];
        for (double[] row : x) {
            for (int a = 0; a < d; a++) {
                for (int b = 0; b < d; b++) {
                    gram[a * d + b] += row[a] * row[b];
                }
            }
        }
        for (int a = 0; a < d; a++) {
            gram[a * d + a] += lambda;
        }
        double[] inv = Linalg.inverse(Tensor.of(Shape.of(d, d), gram, DType.FLOAT64, Device.cpu())).toDoubleArray();
        double[] out = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            double s = 0;
            for (int a = 0; a < d; a++) {
                double t = 0;
                for (int b = 0; b < d; b++) {
                    t += inv[a * d + b] * x[i][b];
                }
                s += x[i][a] * t;
            }
            out[i] = s;
        }
        return out;
    }

    /**
     * Greedy leverage selection: repeatedly picks the candidate with the largest leverage
     * {@code x_iᵀ A⁻¹ x_i} relative to the points chosen so far,
     * {@code A = λI + Σ_(chosen) x xᵀ}, updating {@code A⁻¹} by Sherman-Morrison. Each pick is
     * the point least explained by the span of the others, so the subset is diverse.
     *
     * @param x candidate features
     * @param chosen indices already chosen (they only shape {@code A})
     * @param candidates indices that may be picked
     * @param k how many to pick
     * @param lambda the ridge
     * @return the picked indices, in order
     */
    public static int[] leverageSelect(double[][] x, int[] chosen, int[] candidates, int k, double lambda) {
        int d = x[0].length;
        double[][] inv = new double[d][d];
        for (int a = 0; a < d; a++) {
            inv[a][a] = 1 / lambda;
        }
        for (int c : chosen) {
            shermanMorrison(inv, x[c]);
        }
        boolean[] taken = new boolean[x.length];
        int[] out = new int[Math.min(k, candidates.length)];
        for (int p = 0; p < out.length; p++) {
            int best = -1;
            double bestScore = Double.NEGATIVE_INFINITY;
            for (int i : candidates) {
                if (taken[i]) {
                    continue;
                }
                double s = quadratic(inv, x[i]);
                if (s > bestScore) {
                    bestScore = s;
                    best = i;
                }
            }
            out[p] = best;
            taken[best] = true;
            shermanMorrison(inv, x[best]);
        }
        return out;
    }

    private static double quadratic(double[][] inv, double[] v) {
        double s = 0;
        for (int a = 0; a < v.length; a++) {
            double t = 0;
            for (int b = 0; b < v.length; b++) {
                t += inv[a][b] * v[b];
            }
            s += v[a] * t;
        }
        return s;
    }

    /** {@code inv <- (A + v vᵀ)⁻¹} from {@code inv = A⁻¹}. */
    private static void shermanMorrison(double[][] inv, double[] v) {
        int d = v.length;
        double[] u = new double[d];
        for (int a = 0; a < d; a++) {
            for (int b = 0; b < d; b++) {
                u[a] += inv[a][b] * v[b];
            }
        }
        double denom = 1;
        for (int a = 0; a < d; a++) {
            denom += v[a] * u[a];
        }
        for (int a = 0; a < d; a++) {
            for (int b = 0; b < d; b++) {
                inv[a][b] -= u[a] * u[b] / denom;
            }
        }
    }

    /**
     * The indices of the lowest-valued share of points.
     *
     * @param value the values (NaN counts as highest)
     * @param fraction the share to return
     * @return the indices, lowest first
     */
    public static int[] lowest(double[] value, double fraction) {
        Integer[] idx = new Integer[value.length];
        for (int i = 0; i < idx.length; i++) {
            idx[i] = i;
        }
        Arrays.sort(idx, (a, b) -> Double.compare(Double.isNaN(value[a]) ? Double.MAX_VALUE : value[a],
                Double.isNaN(value[b]) ? Double.MAX_VALUE : value[b]));
        int m = (int) Math.round(fraction * value.length);
        int[] out = new int[m];
        for (int i = 0; i < m; i++) {
            out[i] = idx[i];
        }
        return out;
    }
}
