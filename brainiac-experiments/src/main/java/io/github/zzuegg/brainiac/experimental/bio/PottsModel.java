package io.github.zzuegg.brainiac.experimental.bio;

import java.util.SplittableRandom;

/**
 * A Potts model {@code p(s) ~ exp(sum_i h_i(s_i) + sum_(i<j) J_ij(s_i, s_j))} over aligned
 * sequences, fitted by asymmetric pseudo-likelihood maximisation (plmDCA): one L2-penalised
 * softmax regression per column predicts {@code s_i} from all other positions, fitted by
 * full-batch Adam on sparse one-hot features, and the two estimates of each coupling are
 * averaged. Couplings are ranked by the Frobenius norm of {@code J_ij} in the zero-sum gauge,
 * with the average product correction (APC) removing background; {@link #sample} draws new
 * sequences by Gibbs sampling.
 *
 * <p>Ekeberg et al., "Improved contact prediction in proteins: using pseudolikelihoods to infer
 * Potts models" (PRE 2013), <a href="https://arxiv.org/abs/1211.1281">arXiv:1211.1281</a>;
 * Weigt et al. (PNAS 2009); Dunn, Wahl and Gloor, "Mutual information without the influence of
 * phylogeny or entropy" (Bioinformatics 2008).
 */
public final class PottsModel {

    private final int length;
    private final int states;
    private final double[][] h;
    private final double[][][][] j;

    private PottsModel(int length, int states) {
        this.length = length;
        this.states = states;
        this.h = new double[length][states];
        this.j = new double[length][length][states][states];
    }

    /**
     * Fits by asymmetric pseudo-likelihood.
     *
     * @param seqs aligned sequences {@code [sample][position]}
     * @param states the number of states q
     * @param lambdaJ the coupling L2 penalty (on the mean negative log pseudo-likelihood)
     * @param steps Adam steps per column
     * @return the model
     */
    public static PottsModel fitPseudoLikelihood(int[][] seqs, int states, double lambdaJ, int steps) {
        int n = seqs.length;
        int length = seqs[0].length;
        PottsModel model = new PottsModel(length, states);
        double lambdaH = 0.01;
        for (int i = 0; i < length; i++) {
            double[][] w = new double[length * states][states];
            double[] b = new double[states];
            double[][] mW = new double[length * states][states];
            double[][] vW = new double[length * states][states];
            double[] mB = new double[states];
            double[] vB = new double[states];
            double lr = 0.05;
            for (int step = 1; step <= steps; step++) {
                double[][] gW = new double[length * states][states];
                double[] gB = new double[states];
                double[] z = new double[states];
                for (int[] s : seqs) {
                    System.arraycopy(b, 0, z, 0, states);
                    for (int k = 0; k < length; k++) {
                        if (k == i) {
                            continue;
                        }
                        double[] row = w[k * states + s[k]];
                        for (int a = 0; a < states; a++) {
                            z[a] += row[a];
                        }
                    }
                    double max = Double.NEGATIVE_INFINITY;
                    for (double v : z) {
                        max = Math.max(max, v);
                    }
                    double sum = 0;
                    for (int a = 0; a < states; a++) {
                        z[a] = Math.exp(z[a] - max);
                        sum += z[a];
                    }
                    for (int a = 0; a < states; a++) {
                        double r = (z[a] / sum - (s[i] == a ? 1 : 0)) / n;
                        gB[a] += r;
                        for (int k = 0; k < length; k++) {
                            if (k != i) {
                                gW[k * states + s[k]][a] += r;
                            }
                        }
                    }
                }
                double c1 = 1 - Math.pow(0.9, step);
                double c2 = 1 - Math.pow(0.999, step);
                for (int a = 0; a < states; a++) {
                    double g = gB[a] + 2 * lambdaH * b[a];
                    mB[a] = 0.9 * mB[a] + 0.1 * g;
                    vB[a] = 0.999 * vB[a] + 0.001 * g * g;
                    b[a] -= lr * (mB[a] / c1) / (Math.sqrt(vB[a] / c2) + 1e-8);
                }
                for (int r = 0; r < length * states; r++) {
                    if (r / states == i) {
                        continue;
                    }
                    for (int a = 0; a < states; a++) {
                        double g = gW[r][a] + 2 * lambdaJ * w[r][a];
                        mW[r][a] = 0.9 * mW[r][a] + 0.1 * g;
                        vW[r][a] = 0.999 * vW[r][a] + 0.001 * g * g;
                        w[r][a] -= lr * (mW[r][a] / c1) / (Math.sqrt(vW[r][a] / c2) + 1e-8);
                    }
                }
            }
            model.h[i] = b;
            for (int k = 0; k < length; k++) {
                if (k == i) {
                    continue;
                }
                for (int x = 0; x < states; x++) {
                    for (int a = 0; a < states; a++) {
                        // w[k, x][a] estimates J_ik(a, x); average with the other column's estimate
                        model.j[i][k][a][x] += 0.5 * w[k * states + x][a];
                        model.j[k][i][x][a] += 0.5 * w[k * states + x][a];
                    }
                }
            }
        }
        return model;
    }

    /**
     * The coupling strength of each pair: the Frobenius norm of {@code J_ij} after centring its
     * rows and columns (zero-sum gauge).
     *
     * @return the symmetric score matrix (zero diagonal)
     */
    public double[][] frobeniusScores() {
        double[][] f = new double[length][length];
        for (int a = 0; a < length; a++) {
            for (int b = a + 1; b < length; b++) {
                double[][] m = j[a][b];
                double[] rowMean = new double[states];
                double[] colMean = new double[states];
                double all = 0;
                for (int x = 0; x < states; x++) {
                    for (int y = 0; y < states; y++) {
                        rowMean[x] += m[x][y] / states;
                        colMean[y] += m[x][y] / states;
                        all += m[x][y] / (states * states);
                    }
                }
                double s = 0;
                for (int x = 0; x < states; x++) {
                    for (int y = 0; y < states; y++) {
                        double c = m[x][y] - rowMean[x] - colMean[y] + all;
                        s += c * c;
                    }
                }
                f[a][b] = Math.sqrt(s);
                f[b][a] = f[a][b];
            }
        }
        return f;
    }

    /**
     * The average product correction {@code S_ij - S_i. S_.j / S_..} of a symmetric score matrix.
     *
     * @param s scores with a zero diagonal
     * @return corrected scores
     */
    public static double[][] apc(double[][] s) {
        int n = s.length;
        double[] mean = new double[n];
        double all = 0;
        for (int a = 0; a < n; a++) {
            for (int b = 0; b < n; b++) {
                if (a != b) {
                    mean[a] += s[a][b] / (n - 1);
                    all += s[a][b] / (n * (n - 1.0));
                }
            }
        }
        double[][] out = new double[n][n];
        for (int a = 0; a < n; a++) {
            for (int b = 0; b < n; b++) {
                if (a != b) {
                    out[a][b] = s[a][b] - mean[a] * mean[b] / all;
                }
            }
        }
        return out;
    }

    /**
     * The precision of the top-{@code k} pairs with {@code |i - j| >= minSeparation}.
     *
     * @param scores symmetric pair scores
     * @param contacts the true contacts
     * @param k how many top pairs
     * @param minSeparation the minimum sequence separation counted
     * @return the share of the top pairs that are contacts
     */
    public static double topPrecision(double[][] scores, boolean[][] contacts, int k, int minSeparation) {
        int n = scores.length;
        java.util.List<int[]> pairs = new java.util.ArrayList<>();
        for (int a = 0; a < n; a++) {
            for (int b = a + minSeparation; b < n; b++) {
                pairs.add(new int[] {a, b});
            }
        }
        pairs.sort((x, y) -> Double.compare(scores[y[0]][y[1]], scores[x[0]][x[1]]));
        int hits = 0;
        for (int t = 0; t < Math.min(k, pairs.size()); t++) {
            hits += contacts[pairs.get(t)[0]][pairs.get(t)[1]] ? 1 : 0;
        }
        return hits / (double) Math.min(k, pairs.size());
    }

    /**
     * Gibbs sampling from the fitted model.
     *
     * @param n samples
     * @param sweeps sweeps between samples
     * @param seed the seed
     * @return sampled sequences
     */
    public int[][] sample(int n, int sweeps, long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        int[] s = new int[length];
        for (int i = 0; i < length; i++) {
            s[i] = rnd.nextInt(states);
        }
        int[][] out = new int[n][];
        double[] p = new double[states];
        for (int t = 0; t < n; t++) {
            for (int sw = 0; sw < (t == 0 ? 50 : sweeps); sw++) {
                for (int i = 0; i < length; i++) {
                    double max = Double.NEGATIVE_INFINITY;
                    for (int a = 0; a < states; a++) {
                        double v = h[i][a];
                        for (int k = 0; k < length; k++) {
                            if (k != i) {
                                v += j[i][k][a][s[k]];
                            }
                        }
                        p[a] = v;
                        max = Math.max(max, v);
                    }
                    double sum = 0;
                    for (int a = 0; a < states; a++) {
                        p[a] = Math.exp(p[a] - max);
                        sum += p[a];
                    }
                    double u = rnd.nextDouble() * sum;
                    int pick = states - 1;
                    for (int a = 0; a < states; a++) {
                        u -= p[a];
                        if (u <= 0) {
                            pick = a;
                            break;
                        }
                    }
                    s[i] = pick;
                }
            }
            out[t] = s.clone();
        }
        return out;
    }
}
