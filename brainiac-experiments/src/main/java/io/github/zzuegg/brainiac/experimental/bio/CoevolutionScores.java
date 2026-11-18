package io.github.zzuegg.brainiac.experimental.bio;

/**
 * Local (non-Potts) coevolution scores between alignment columns: mutual information with a
 * pseudocount, which confounds direct and transitive couplings, and a lasso stand-in that
 * regresses each column's one-hot indicators on all other columns' one-hot indicators with an
 * L1 penalty (solved by ISTA) and scores a pair by the Frobenius norm of its coefficient block.
 * Combine either with {@link PottsModel#apc} for the average product correction.
 */
public final class CoevolutionScores {

    private CoevolutionScores() {
    }

    /**
     * Mutual information between columns.
     *
     * @param seqs aligned sequences
     * @param states the number of states
     * @param pseudocount the pseudocount added to every pair count
     * @return the symmetric MI matrix (zero diagonal)
     */
    public static double[][] mutualInformation(int[][] seqs, int states, double pseudocount) {
        int n = seqs.length;
        int length = seqs[0].length;
        double[][] mi = new double[length][length];
        for (int a = 0; a < length; a++) {
            for (int b = a + 1; b < length; b++) {
                double[][] pab = new double[states][states];
                for (int[] s : seqs) {
                    pab[s[a]][s[b]] += 1;
                }
                double total = n + pseudocount * states * states;
                double[] pa = new double[states];
                double[] pb = new double[states];
                for (int x = 0; x < states; x++) {
                    for (int y = 0; y < states; y++) {
                        pab[x][y] = (pab[x][y] + pseudocount) / total;
                        pa[x] += pab[x][y];
                        pb[y] += pab[x][y];
                    }
                }
                double v = 0;
                for (int x = 0; x < states; x++) {
                    for (int y = 0; y < states; y++) {
                        v += pab[x][y] * Math.log(pab[x][y] / (pa[x] * pb[y]));
                    }
                }
                mi[a][b] = v;
                mi[b][a] = v;
            }
        }
        return mi;
    }

    /**
     * The lasso on one-hot pairs: for each column {@code i} and state {@code a}, the centred
     * indicator {@code 1[s_i = a]} is regressed on the centred one-hot encoding of every other
     * column with penalty {@code lambda ||beta||_1}, by ISTA with the step from the largest
     * eigenvalue of the design's Gram matrix (power iteration).
     *
     * @param seqs aligned sequences
     * @param states the number of states
     * @param lambda the L1 penalty (on the mean squared error)
     * @param iterations ISTA iterations per column
     * @return symmetric scores (zero diagonal)
     */
    public static double[][] lasso(int[][] seqs, int states, double lambda, int iterations) {
        int n = seqs.length;
        int length = seqs[0].length;
        int p = length * states;
        double[][] x = new double[n][p];
        double[] mean = new double[p];
        for (int r = 0; r < n; r++) {
            for (int k = 0; k < length; k++) {
                x[r][k * states + seqs[r][k]] = 1;
            }
        }
        for (int c = 0; c < p; c++) {
            for (int r = 0; r < n; r++) {
                mean[c] += x[r][c] / n;
            }
            for (int r = 0; r < n; r++) {
                x[r][c] -= mean[c];
            }
        }
        // the Gram matrix G = X^T X / n; the target is itself a design column, so X^T y = G[:, t]
        double[][] g = new double[p][p];
        for (int r = 0; r < n; r++) {
            for (int a = 0; a < p; a++) {
                if (x[r][a] == 0) {
                    continue;
                }
                for (int b = 0; b < p; b++) {
                    g[a][b] += x[r][a] * x[r][b] / n;
                }
            }
        }
        // Lipschitz constant of the gradient: the largest eigenvalue of G by power iteration
        double[] v = new double[p];
        java.util.Arrays.fill(v, 1 / Math.sqrt(p));
        double lip = 1;
        for (int it = 0; it < 50; it++) {
            double[] next = new double[p];
            for (int a = 0; a < p; a++) {
                for (int b = 0; b < p; b++) {
                    next[a] += g[a][b] * v[b];
                }
            }
            double norm = 0;
            for (double t : next) {
                norm += t * t;
            }
            norm = Math.sqrt(norm);
            lip = norm;
            for (int c = 0; c < p; c++) {
                v[c] = next[c] / norm;
            }
        }
        double step = 1 / lip;
        double[][] score = new double[length][length];
        for (int i = 0; i < length; i++) {
            for (int a = 0; a < states; a++) {
                int t = i * states + a;
                double[] beta = new double[p];
                for (int it = 0; it < iterations; it++) {
                    double[] grad = new double[p];
                    for (int c = 0; c < p; c++) {
                        double s2 = -g[c][t];
                        for (int d = 0; d < p; d++) {
                            if (beta[d] != 0) {
                                s2 += g[c][d] * beta[d];
                            }
                        }
                        grad[c] = s2;
                    }
                    for (int c = 0; c < p; c++) {
                        if (c / states == i) {
                            continue;
                        }
                        double z = beta[c] - step * grad[c];
                        beta[c] = Math.signum(z) * Math.max(Math.abs(z) - step * lambda, 0);
                    }
                }
                for (int k = 0; k < length; k++) {
                    if (k == i) {
                        continue;
                    }
                    for (int b = 0; b < states; b++) {
                        double tb = beta[k * states + b];
                        score[i][k] += tb * tb;
                    }
                }
            }
        }
        double[][] out = new double[length][length];
        for (int a = 0; a < length; a++) {
            for (int b = a + 1; b < length; b++) {
                out[a][b] = Math.sqrt(score[a][b]) + Math.sqrt(score[b][a]);
                out[b][a] = out[a][b];
            }
        }
        return out;
    }
}
