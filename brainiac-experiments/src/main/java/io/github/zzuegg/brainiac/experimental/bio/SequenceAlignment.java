package io.github.zzuegg.brainiac.experimental.bio;

import java.util.Arrays;

/**
 * Pairwise sequence alignment by dynamic programming: global Needleman-Wunsch and local
 * Smith-Waterman, both with Gotoh's three-matrix recursion for affine gaps (a gap of length
 * {@code g} costs {@code open + (g - 1) extend}) and traceback, plus a differentiable soft
 * Smith-Waterman in which every max is replaced by a temperature-scaled log-sum-exp. The soft
 * score is the log partition function over all local alignments; its gradient with respect to
 * the position-pair scores is the posterior probability that the two positions are aligned
 * (the expected alignment), which is what end-to-end learned alignment (SMURF) trains through.
 *
 * <p>Needleman and Wunsch (JMB 1970), Smith and Waterman (JMB 1981), Gotoh, "An improved
 * algorithm for matching biological sequences" (JMB 1982); Petti et al., "End-to-end learning
 * of multiple sequence alignments with differentiable Smith-Waterman" (Bioinformatics 2023).
 */
public final class SequenceAlignment {

    private static final double NEG = -1e18;

    private SequenceAlignment() {
    }

    /**
     * A scoring scheme.
     *
     * @param substitution {@code [a][b]} the score of aligning residue {@code a} with {@code b}
     * @param gapOpen the cost of the first gap position (positive)
     * @param gapExtend the cost of each further gap position (positive)
     */
    public record Scoring(double[][] substitution, double gapOpen, double gapExtend) {

        /**
         * A match/mismatch scheme.
         *
         * @param alphabet the alphabet size
         * @param match the match score
         * @param mismatch the mismatch score
         * @param gapOpen the gap-open cost
         * @param gapExtend the gap-extend cost
         * @return the scoring
         */
        public static Scoring identity(int alphabet, double match, double mismatch, double gapOpen, double gapExtend) {
            double[][] s = new double[alphabet][alphabet];
            for (int a = 0; a < alphabet; a++) {
                for (int b = 0; b < alphabet; b++) {
                    s[a][b] = a == b ? match : mismatch;
                }
            }
            return new Scoring(s, gapOpen, gapExtend);
        }

        /**
         * The position-pair score matrix of two sequences.
         *
         * @param a the first sequence
         * @param b the second sequence
         * @return {@code [i][j]} the score of aligning {@code a[i]} with {@code b[j]}
         */
        public double[][] pairScores(int[] a, int[] b) {
            double[][] s = new double[a.length][b.length];
            for (int i = 0; i < a.length; i++) {
                for (int j = 0; j < b.length; j++) {
                    s[i][j] = substitution[a[i]][b[j]];
                }
            }
            return s;
        }
    }

    /**
     * An alignment.
     *
     * @param score the alignment score
     * @param match for each position of the first sequence, the aligned position of the second,
     *     or -1
     */
    public record Alignment(double score, int[] match) {

        /**
         * The share of the true aligned pairs that this alignment reproduces.
         *
         * @param truth the true {@code match} array
         * @return the alignment accuracy in [0, 1]
         */
        public double accuracy(int[] truth) {
            int total = 0;
            int ok = 0;
            for (int i = 0; i < truth.length; i++) {
                if (truth[i] >= 0) {
                    total++;
                    ok += match[i] == truth[i] ? 1 : 0;
                }
            }
            return total == 0 ? 1 : ok / (double) total;
        }
    }

    /**
     * Global alignment with affine gaps (Needleman-Wunsch with Gotoh's recursion).
     *
     * @param a the first sequence
     * @param b the second sequence
     * @param scoring the scores
     * @return the optimal global alignment
     */
    public static Alignment needlemanWunsch(int[] a, int[] b, Scoring scoring) {
        return align(a, b, scoring.pairScores(a, b), scoring.gapOpen(), scoring.gapExtend(), false);
    }

    /**
     * Local alignment with affine gaps (Smith-Waterman with Gotoh's recursion).
     *
     * @param a the first sequence
     * @param b the second sequence
     * @param scoring the scores
     * @return the optimal local alignment
     */
    public static Alignment smithWaterman(int[] a, int[] b, Scoring scoring) {
        return align(a, b, scoring.pairScores(a, b), scoring.gapOpen(), scoring.gapExtend(), true);
    }

    /**
     * Affine-gap alignment on a precomputed position-pair score matrix (for learned or
     * embedding-based scores).
     *
     * @param a the first sequence (only its length is used)
     * @param b the second sequence (only its length is used)
     * @param s {@code [i][j]} pair scores
     * @param open the gap-open cost
     * @param extend the gap-extend cost
     * @param local Smith-Waterman when true, Needleman-Wunsch otherwise
     * @return the optimal alignment
     */
    public static Alignment align(int[] a, int[] b, double[][] s, double open, double extend, boolean local) {
        int n = a.length;
        int m = b.length;
        // M: ends in an aligned pair; X: ends with a[i] against a gap; Y: ends with b[j] against a gap
        double[][] mm = new double[n + 1][m + 1];
        double[][] xx = new double[n + 1][m + 1];
        double[][] yy = new double[n + 1][m + 1];
        byte[][] pm = new byte[n + 1][m + 1];
        byte[][] px = new byte[n + 1][m + 1];
        byte[][] py = new byte[n + 1][m + 1];
        for (int i = 0; i <= n; i++) {
            Arrays.fill(mm[i], NEG);
            Arrays.fill(xx[i], NEG);
            Arrays.fill(yy[i], NEG);
        }
        mm[0][0] = 0;
        if (!local) {
            for (int i = 1; i <= n; i++) {
                xx[i][0] = -open - (i - 1) * extend;
                px[i][0] = (byte) (i == 1 ? 1 : 2);
            }
            for (int j = 1; j <= m; j++) {
                yy[0][j] = -open - (j - 1) * extend;
                py[0][j] = (byte) (j == 1 ? 1 : 3);
            }
        }
        double best = local ? 0 : NEG;
        int bi = 0;
        int bj = 0;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                double prev = mm[i - 1][j - 1];
                byte from = 1;
                if (xx[i - 1][j - 1] > prev) {
                    prev = xx[i - 1][j - 1];
                    from = 2;
                }
                if (yy[i - 1][j - 1] > prev) {
                    prev = yy[i - 1][j - 1];
                    from = 3;
                }
                if (local && prev <= 0) {
                    prev = 0;
                    from = 0;
                }
                mm[i][j] = s[i - 1][j - 1] + prev;
                pm[i][j] = from;
                double o = mm[i - 1][j] - open;
                double e = xx[i - 1][j] - extend;
                xx[i][j] = Math.max(o, e);
                px[i][j] = (byte) (o >= e ? 1 : 2);
                o = mm[i][j - 1] - open;
                e = yy[i][j - 1] - extend;
                yy[i][j] = Math.max(o, e);
                py[i][j] = (byte) (o >= e ? 1 : 3);
                if (local && mm[i][j] > best) {
                    best = mm[i][j];
                    bi = i;
                    bj = j;
                }
            }
        }
        int[] match = new int[n];
        Arrays.fill(match, -1);
        int i;
        int j;
        int state;
        if (local) {
            i = bi;
            j = bj;
            state = best > 0 ? 1 : 0;
        } else {
            i = n;
            j = m;
            best = Math.max(mm[n][m], Math.max(xx[n][m], yy[n][m]));
            state = best == mm[n][m] ? 1 : best == xx[n][m] ? 2 : 3;
        }
        while (state != 0 && (i > 0 || j > 0)) {
            if (state == 1) {
                if (i == 0 || j == 0) {
                    break;
                }
                match[i - 1] = j - 1;
                state = pm[i][j];
                i--;
                j--;
            } else if (state == 2) {
                state = px[i][j];
                i--;
            } else {
                state = py[i][j];
                j--;
            }
            if (!local && state == 1 && (i == 0 || j == 0)) {
                break;
            }
        }
        return new Alignment(best, match);
    }

    /**
     * The result of a soft Smith-Waterman pass.
     *
     * @param value {@code T log sum_alignments exp(score / T)}
     * @param pairGradient {@code d value / d s[i][j]}: the posterior that {@code i} and {@code j}
     *     are aligned
     * @param openGradient {@code d value / d open}: minus the expected number of gap opens
     * @param extendGradient {@code d value / d extend}: minus the expected number of extensions
     */
    public record Soft(double value, double[][] pairGradient, double openGradient, double extendGradient) {
    }

    /**
     * Differentiable Smith-Waterman with affine gaps: the forward pass replaces every max by
     * {@code T logsumexp(./T)}, and a reverse pass through the same recursion returns the
     * gradients (no autograd tape needed: the dynamic program is its own adjoint).
     *
     * @param s {@code [i][j]} pair scores
     * @param open the gap-open cost
     * @param extend the gap-extend cost
     * @param temperature {@code T}; the hard Smith-Waterman score is the limit {@code T -> 0}
     * @return the value and its gradients
     */
    public static Soft softSmithWaterman(double[][] s, double open, double extend, double temperature) {
        int n = s.length;
        int m = s[0].length;
        double t = temperature;
        double[][] mm = new double[n + 1][m + 1];
        double[][] xx = new double[n + 1][m + 1];
        double[][] yy = new double[n + 1][m + 1];
        for (int i = 0; i <= n; i++) {
            Arrays.fill(mm[i], NEG);
            Arrays.fill(xx[i], NEG);
            Arrays.fill(yy[i], NEG);
        }
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                mm[i][j] = s[i - 1][j - 1] + t * lse(0, mm[i - 1][j - 1] / t, xx[i - 1][j - 1] / t,
                        yy[i - 1][j - 1] / t);
                xx[i][j] = t * lse((mm[i - 1][j] - open) / t, (xx[i - 1][j] - extend) / t);
                yy[i][j] = t * lse((mm[i][j - 1] - open) / t, (yy[i][j - 1] - extend) / t);
            }
        }
        double max = NEG;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                max = Math.max(max, mm[i][j] / t);
            }
        }
        double z = 0;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                z += Math.exp(mm[i][j] / t - max);
            }
        }
        double value = t * (max + Math.log(z));
        double[][] am = new double[n + 2][m + 2];
        double[][] ax = new double[n + 2][m + 2];
        double[][] ay = new double[n + 2][m + 2];
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                am[i][j] = Math.exp(mm[i][j] / t - max) / z;
            }
        }
        double gOpen = 0;
        double gExtend = 0;
        double[][] grad = new double[n][m];
        for (int i = n; i >= 1; i--) {
            for (int j = m; j >= 1; j--) {
                // Y[i][j] = T lse((M[i][j-1] - open)/T, (Y[i][j-1] - extend)/T)
                if (ay[i][j] != 0) {
                    double w1 = Math.exp((mm[i][j - 1] - open - yy[i][j]) / t);
                    double w2 = Math.exp((yy[i][j - 1] - extend - yy[i][j]) / t);
                    am[i][j - 1] += ay[i][j] * w1;
                    ay[i][j - 1] += ay[i][j] * w2;
                    gOpen -= ay[i][j] * w1;
                    gExtend -= ay[i][j] * w2;
                }
                if (ax[i][j] != 0) {
                    double w1 = Math.exp((mm[i - 1][j] - open - xx[i][j]) / t);
                    double w2 = Math.exp((xx[i - 1][j] - extend - xx[i][j]) / t);
                    am[i - 1][j] += ax[i][j] * w1;
                    ax[i - 1][j] += ax[i][j] * w2;
                    gOpen -= ax[i][j] * w1;
                    gExtend -= ax[i][j] * w2;
                }
                double a = am[i][j];
                grad[i - 1][j - 1] = a;
                if (a != 0) {
                    double inner = mm[i][j] - s[i - 1][j - 1];
                    am[i - 1][j - 1] += a * Math.exp((mm[i - 1][j - 1] - inner) / t);
                    ax[i - 1][j - 1] += a * Math.exp((xx[i - 1][j - 1] - inner) / t);
                    ay[i - 1][j - 1] += a * Math.exp((yy[i - 1][j - 1] - inner) / t);
                }
            }
        }
        return new Soft(value, grad, gOpen, gExtend);
    }

    private static double lse(double... v) {
        double max = NEG;
        for (double x : v) {
            max = Math.max(max, x);
        }
        if (max <= NEG / 2) {
            return NEG;
        }
        double s = 0;
        for (double x : v) {
            s += Math.exp(x - max);
        }
        return max + Math.log(s);
    }

    /**
     * The edit (Levenshtein) distance, the unweighted baseline.
     *
     * @param a the first sequence
     * @param b the second sequence
     * @return the minimum number of substitutions, insertions and deletions
     */
    public static int editDistance(int[] a, int[] b) {
        int[] prev = new int[b.length + 1];
        int[] cur = new int[b.length + 1];
        for (int j = 0; j <= b.length; j++) {
            prev[j] = j;
        }
        for (int i = 1; i <= a.length; i++) {
            cur[0] = i;
            for (int j = 1; j <= b.length; j++) {
                cur[j] = Math.min(Math.min(prev[j] + 1, cur[j - 1] + 1), prev[j - 1] + (a[i - 1] == b[j - 1] ? 0 : 1));
            }
            int[] t = prev;
            prev = cur;
            cur = t;
        }
        return prev[b.length];
    }

    /**
     * The edit-distance alignment (unit costs, global), for alignment-accuracy baselines.
     *
     * @param a the first sequence
     * @param b the second sequence
     * @return the alignment, its score minus the edit distance
     */
    public static Alignment editAlignment(int[] a, int[] b) {
        return needlemanWunsch(a, b, Scoring.identity(Math.max(max(a), max(b)) + 1, 0, -1, 1, 1));
    }

    private static int max(int[] a) {
        int m = 0;
        for (int x : a) {
            m = Math.max(m, x);
        }
        return m;
    }
}
