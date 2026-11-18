package io.github.zzuegg.brainiac.experimental.labels;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.interp.AttentionRollout;

import java.util.Arrays;

/**
 * Positive-unlabelled (PU) learning: only some positives are labelled, the rest of the data is
 * unlabelled, and the class prior {@code pi} is known or estimated.
 *
 * <ul>
 *   <li>{@link #biasedRisk}: the naive baseline, unlabelled data taken as negative.</li>
 *   <li>{@link #unbiasedRisk} (uPU; du Plessis et al. 2015): the negative-class risk rewritten as
 *       the unlabelled risk minus {@code pi} times the positives' risk as negatives,
 *       {@code pi R_p^+ + R_u^- - pi R_p^-}. With a flexible model the last two terms go
 *       negative and the model overfits.</li>
 *   <li>{@link #nonNegative} (nnPU; Kiryo et al., NeurIPS 2017,
 *       <a href="https://arxiv.org/abs/1703.00593">arXiv:1703.00593</a>): when the estimated
 *       negative risk {@code R_u^- - pi R_p^-} falls below {@code -beta}, the step instead
 *       minimises {@code -gamma} times it, pushing it back to zero.</li>
 *   <li>{@link #elkanNotoPrior} (Elkan and Noto, KDD 2008,
 *       <a href="https://doi.org/10.1145/1401890.1401920">doi:10.1145/1401890.1401920</a>): a
 *       labelled-versus-unlabelled classifier {@code g} gives {@code c = P(labelled | positive)} as
 *       the mean of {@code g} over held-out labelled positives, and from it the prior.</li>
 *   <li>{@link #twoHnc} (2-HNC, 2025, <a href="https://arxiv.org/abs/2505.08212">arXiv:2505.08212</a>):
 *       a k-nearest-neighbour similarity graph with the labelled positives tied to the source; a
 *       parametric minimum cut {@code min C(S, S') + lambda sum_{i in S} d_i} over increasing
 *       {@code lambda} gives nested partitions that rank unlabelled nodes by when they leave the
 *       positive side. Stage two ties the lowest-ranked nodes to the sink as likely negatives and
 *       sweeps again; of all partitions of both stages, the one whose positive share of the
 *       unlabelled nodes is closest to the prior is returned. The cuts use the Edmonds-Karp
 *       solver of {@link AttentionRollout#minCutSourceSide}. This is a reading of the paper's
 *       abstract (Hochbaum's normalised cut with a degree-weighted penalty); the paper's exact
 *       parametrisation may differ.</li>
 * </ul>
 * Scores are raw outputs {@code z}, positive meaning positive; the surrogate loss is the sigmoid
 * loss {@code l(z, y) = sigmoid(-y z)} the nnPU paper uses.
 */
public final class PositiveUnlabeled {

    private PositiveUnlabeled() {
    }

    /**
     * One nnPU step's objective.
     *
     * @param objective what to back-propagate
     * @param risk the full (unclipped) PU risk estimate, for monitoring
     * @param negativeRisk the estimated negative-class risk {@code R_u^- - pi R_p^-}
     * @param corrective whether this step pushes the negative risk back up
     */
    public record Step(Var objective, double risk, double negativeRisk, boolean corrective) {
    }

    /**
     * @param z scores
     * @param y the label, +1 or -1
     * @return the mean sigmoid loss
     */
    public static Var sigmoidLoss(Var z, float y) {
        return z.mul(-y).sigmoid().mean();
    }

    /**
     * Unlabelled as negative.
     *
     * @param positives scores of labelled positives
     * @param unlabelled scores of unlabelled rows
     * @param prior the class prior, weighting the two terms as uPU does
     * @return {@code pi R_p^+ + (1 - pi) R_u^-}
     */
    public static Var biasedRisk(Var positives, Var unlabelled, double prior) {
        return sigmoidLoss(positives, 1).mul((float) prior).add(sigmoidLoss(unlabelled, -1).mul((float) (1 - prior)));
    }

    /**
     * The unbiased PU risk.
     *
     * @param positives scores of labelled positives
     * @param unlabelled scores of unlabelled rows
     * @param prior the class prior
     * @return {@code pi R_p^+ + R_u^- - pi R_p^-}
     */
    public static Var unbiasedRisk(Var positives, Var unlabelled, double prior) {
        return sigmoidLoss(positives, 1).mul((float) prior).add(negativeRisk(positives, unlabelled, prior));
    }

    private static Var negativeRisk(Var positives, Var unlabelled, double prior) {
        return sigmoidLoss(unlabelled, -1).sub(sigmoidLoss(positives, -1).mul((float) prior));
    }

    /**
     * The non-negative PU step.
     *
     * @param positives scores of labelled positives
     * @param unlabelled scores of unlabelled rows
     * @param prior the class prior
     * @param beta the tolerated negative risk, 0 in the paper
     * @param gamma the corrective step's weight, 1 in the paper
     * @return the step
     */
    public static Step nonNegative(Var positives, Var unlabelled, double prior, double beta, double gamma) {
        Var pos = sigmoidLoss(positives, 1).mul((float) prior);
        Var neg = negativeRisk(positives, unlabelled, prior);
        double negValue = neg.value().item();
        double risk = pos.value().item() + negValue;
        if (negValue < -beta) {
            return new Step(neg.mul((float) -gamma), risk, negValue, true);
        }
        return new Step(pos.add(neg), risk, negValue, false);
    }

    /**
     * Elkan and Noto's prior estimate.
     *
     * @param heldOutPositiveScores {@code g(x) = P(labelled | x)} on held-out labelled positives
     * @param labelledShare the share of all rows that are labelled
     * @return the estimated prior among the unlabelled rows, {@code (s / c - s) / (1 - s)}
     */
    public static double elkanNotoPrior(double[] heldOutPositiveScores, double labelledShare) {
        double c = 0;
        for (double g : heldOutPositiveScores) {
            c += g / heldOutPositiveScores.length;
        }
        double p = Math.min(1, labelledShare / Math.max(c, 1e-9));
        return Math.clamp((p - labelledShare) / (1 - labelledShare), 0, 1);
    }

    /**
     * 2-HNC on a set of rows.
     *
     * @param x the rows: labelled positives and unlabelled rows together
     * @param labelledPositive which rows are labelled positives
     * @param prior the positive share expected among the unlabelled rows
     * @param k neighbours per node in the similarity graph
     * @param lambdas sweep points per stage
     * @return the predicted positive set (labelled positives always included)
     */
    public static boolean[] twoHnc(float[][] x, boolean[] labelledPositive, double prior, int k, int lambdas) {
        int n = x.length;
        double[][] w = knnGraph(x, k);
        double[] degree = new double[n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                degree[i] += w[i][j];
            }
        }
        int unlabelled = 0;
        for (boolean b : labelledPositive) {
            unlabelled += b ? 0 : 1;
        }
        boolean[] negativeSeed = new boolean[n];
        // stage one: rank by the largest lambda at which a node stays on the positive side
        double[] score = new double[n];
        boolean[] best = null;
        double bestGap = Double.POSITIVE_INFINITY;
        for (int stage = 0; stage < 2; stage++) {
            for (int li = 0; li < lambdas; li++) {
                double lambda = 1e-3 * Math.pow(2.0 / 1e-3, li / (double) (lambdas - 1));
                boolean[] side = cut(w, degree, labelledPositive, negativeSeed, lambda);
                int pos = 0;
                for (int i = 0; i < n; i++) {
                    if (side[i]) {
                        if (!labelledPositive[i]) {
                            pos++;
                        }
                        if (stage == 0) {
                            score[i] = Math.max(score[i], li + 1);
                        }
                    }
                }
                double gap = Math.abs(pos / (double) Math.max(1, unlabelled) - prior);
                if (gap < bestGap) {
                    bestGap = gap;
                    best = side;
                }
            }
            if (stage == 0) {
                // seed the sink with the unlabelled nodes most likely negative: the lowest-ranked
                // (1 - prior) / 2 share of them
                Integer[] order = new Integer[n];
                for (int i = 0; i < n; i++) {
                    order[i] = i;
                }
                Arrays.sort(order, (a, b) -> Double.compare(score[a], score[b]));
                int seeds = (int) Math.round(unlabelled * (1 - prior) / 2);
                for (int i = 0, taken = 0; i < n && taken < seeds; i++) {
                    if (!labelledPositive[order[i]]) {
                        negativeSeed[order[i]] = true;
                        taken++;
                    }
                }
            }
        }
        return best;
    }

    private static boolean[] cut(double[][] w, double[] degree, boolean[] posSeed, boolean[] negSeed, double lambda) {
        int n = w.length;
        int s = n;
        int t = n + 1;
        double[][] cap = new double[n + 2][n + 2];
        double inf = 1e12;
        for (int i = 0; i < n; i++) {
            System.arraycopy(w[i], 0, cap[i], 0, n);
            if (posSeed[i]) {
                cap[s][i] = inf;
            } else if (negSeed[i]) {
                cap[i][t] = inf;
            } else {
                cap[i][t] = lambda * degree[i];
            }
        }
        boolean[] side = AttentionRollout.minCutSourceSide(cap, s, t);
        return Arrays.copyOf(side, n);
    }

    /**
     * A symmetric k-nearest-neighbour graph with Gaussian weights (bandwidth: the mean
     * k-th-neighbour distance).
     *
     * @param x the rows
     * @param k neighbours per node
     * @return {@code [n][n]} weights
     */
    public static double[][] knnGraph(float[][] x, int k) {
        int n = x.length;
        double[][] d = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double s = 0;
                for (int f = 0; f < x[i].length; f++) {
                    double v = x[i][f] - x[j][f];
                    s += v * v;
                }
                d[i][j] = s;
                d[j][i] = s;
            }
        }
        int[][] nn = new int[n][];
        double sigma = 0;
        for (int i = 0; i < n; i++) {
            final int row = i;
            Integer[] order = new Integer[n];
            for (int j = 0; j < n; j++) {
                order[j] = j;
            }
            Arrays.sort(order, (a, b) -> Double.compare(d[row][a], d[row][b]));
            nn[i] = new int[k];
            for (int m = 0; m < k; m++) {
                nn[i][m] = order[m + 1];
            }
            sigma += d[i][nn[i][k - 1]] / n;
        }
        double[][] w = new double[n][n];
        for (int i = 0; i < n; i++) {
            for (int j : nn[i]) {
                double v = Math.exp(-d[i][j] / Math.max(sigma, 1e-12));
                w[i][j] = Math.max(w[i][j], v);
                w[j][i] = w[i][j];
            }
        }
        return w;
    }
}
