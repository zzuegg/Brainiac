package io.github.zzuegg.brainiac.experimental.bio;

import java.util.Arrays;
import java.util.List;

/**
 * A profile hidden Markov model: per alignment column a match state (column-specific emission),
 * an insert state (background emission) and a silent delete state, in the Krogh/Haussler
 * topology without delete-insert transitions (as in HMMER's Plan 7). Built from a multiple
 * alignment with pseudocounts ({@link #fromAlignment}), from unaligned family members by a star
 * alignment to a medoid member ({@link #fromUnaligned}), and refined by Baum-Welch
 * ({@link #baumWelch}). Sequences are scored by the forward algorithm (or Viterbi) as log-odds
 * against an i.i.d. background null model.
 *
 * <p>Krogh et al., "Hidden Markov models in computational biology: applications to protein
 * modeling" (JMB 1994); Eddy, "Profile hidden Markov models" (Bioinformatics 1998).
 */
public final class ProfileHmm {

    private static final double NEG = Double.NEGATIVE_INFINITY;

    private final int length;
    private final int alphabet;
    private final double[] logBackground;
    // emissions of match states 1..L
    private final double[][] eM;
    // transitions, index k = 0..L (k = 0 is the begin state)
    private final double[] tMM;
    private final double[] tMI;
    private final double[] tMD;
    private final double[] tIM;
    private final double[] tII;
    private final double[] tDM;
    private final double[] tDD;

    private ProfileHmm(int length, double[] background) {
        this.length = length;
        this.alphabet = background.length;
        this.logBackground = new double[alphabet];
        for (int a = 0; a < alphabet; a++) {
            logBackground[a] = Math.log(background[a]);
        }
        eM = new double[length + 1][alphabet];
        tMM = new double[length + 1];
        tMI = new double[length + 1];
        tMD = new double[length + 1];
        tIM = new double[length + 1];
        tII = new double[length + 1];
        tDM = new double[length + 1];
        tDD = new double[length + 1];
    }

    /** @return the number of match columns */
    public int length() {
        return length;
    }

    /**
     * Builds a profile from a multiple alignment: columns with fewer than half gaps become match
     * states; counts of emissions and transitions get Dirichlet pseudocounts.
     *
     * @param alignment rows of the alignment, residue indices or -1 for a gap
     * @param background the background distribution
     * @param pseudocount the emission pseudocount mass (spread by the background)
     * @return the model
     */
    public static ProfileHmm fromAlignment(int[][] alignment, double[] background, double pseudocount) {
        int cols = alignment[0].length;
        boolean[] matchCol = new boolean[cols];
        int length = 0;
        for (int c = 0; c < cols; c++) {
            int gaps = 0;
            for (int[] row : alignment) {
                gaps += row[c] < 0 ? 1 : 0;
            }
            matchCol[c] = gaps * 2 < alignment.length;
            length += matchCol[c] ? 1 : 0;
        }
        ProfileHmm h = new ProfileHmm(length, background);
        Counts counts = h.new Counts();
        for (int[] row : alignment) {
            int k = 0;
            int state = 0; // 0 = M (begin at k = 0), 1 = I, 2 = D
            for (int c = 0; c < cols; c++) {
                if (matchCol[c]) {
                    int next = row[c] < 0 ? 2 : 0;
                    counts.transition(k, state, next, 1);
                    k++;
                    if (next == 0) {
                        counts.eM[k][row[c]] += 1;
                    }
                    state = next;
                } else if (row[c] >= 0) {
                    counts.transition(k, state, 1, 1);
                    state = 1;
                }
            }
            counts.transition(k, state, 0, 1);
        }
        h.maximise(counts, pseudocount);
        return h;
    }

    /**
     * Builds a profile from unaligned members: each is aligned globally to the medoid member
     * (affine gaps, identity scores) and the induced star alignment is passed to
     * {@link #fromAlignment}; positions inserted relative to the medoid are left out.
     *
     * @param members the family members
     * @param background the background distribution
     * @param pseudocount the emission pseudocount mass
     * @return the model
     */
    public static ProfileHmm fromUnaligned(List<int[]> members, double[] background, double pseudocount) {
        int alphabet = background.length;
        SequenceAlignment.Scoring scoring = SequenceAlignment.Scoring.identity(alphabet, 3, -1, 4, 1);
        int candidates = Math.min(8, members.size());
        int seed = 0;
        double best = Double.NEGATIVE_INFINITY;
        for (int c = 0; c < candidates; c++) {
            double total = 0;
            for (int o = 0; o < candidates; o++) {
                if (o != c) {
                    total += SequenceAlignment.needlemanWunsch(members.get(c), members.get(o), scoring).score();
                }
            }
            if (total > best) {
                best = total;
                seed = c;
            }
        }
        int[] ref = members.get(seed);
        int[][] rows = new int[members.size()][ref.length];
        for (int r = 0; r < members.size(); r++) {
            Arrays.fill(rows[r], -1);
            int[] seq = members.get(r);
            SequenceAlignment.Alignment al = SequenceAlignment.needlemanWunsch(seq, ref, scoring);
            for (int i = 0; i < seq.length; i++) {
                if (al.match()[i] >= 0) {
                    rows[r][al.match()[i]] = seq[i];
                }
            }
        }
        return fromAlignment(rows, background, pseudocount);
    }

    /** Expected or observed counts of a profile's events. */
    private final class Counts {
        final double[][] eM = new double[length + 1][alphabet];
        final double[][] t = new double[length + 1][7];

        // from-state (0 M, 1 I, 2 D) at node k to next state (0 M or end, 1 I, 2 D)
        void transition(int k, int from, int to, double w) {
            int idx = switch (from) {
                case 0 -> to;
                case 1 -> to == 0 ? 3 : 4;
                default -> to == 0 ? 5 : 6;
            };
            if (from == 1 && to == 2 || from == 2 && to == 1) {
                return;
            }
            t[k][idx] += w;
        }
    }

    private void maximise(Counts c, double pseudocount) {
        for (int k = 1; k <= length; k++) {
            double total = 0;
            for (int a = 0; a < alphabet; a++) {
                total += c.eM[k][a] + pseudocount * Math.exp(logBackground[a]);
            }
            for (int a = 0; a < alphabet; a++) {
                eM[k][a] = Math.log((c.eM[k][a] + pseudocount * Math.exp(logBackground[a])) / total);
            }
        }
        for (int k = 0; k <= length; k++) {
            boolean last = k == length;
            double mm = c.t[k][0] + 0.7;
            double mi = c.t[k][1] + 0.15;
            double md = last ? 0 : c.t[k][2] + 0.15;
            double sm = mm + mi + md;
            tMM[k] = Math.log(mm / sm);
            tMI[k] = Math.log(mi / sm);
            tMD[k] = last ? NEG : Math.log(md / sm);
            double im = c.t[k][3] + 0.6;
            double ii = c.t[k][4] + 0.4;
            tIM[k] = Math.log(im / (im + ii));
            tII[k] = Math.log(ii / (im + ii));
            if (k == 0) {
                tDM[k] = NEG;
                tDD[k] = NEG;
            } else {
                double dm = c.t[k][5] + 0.6;
                double dd = last ? 0 : c.t[k][6] + 0.4;
                tDM[k] = Math.log(dm / (dm + dd));
                tDD[k] = last ? NEG : Math.log(dd / (dm + dd));
            }
        }
    }

    private static double lse(double a, double b) {
        if (a == NEG) {
            return b;
        }
        if (b == NEG) {
            return a;
        }
        return Math.max(a, b) + Math.log1p(Math.exp(-Math.abs(a - b)));
    }

    private static double lse(double a, double b, double c) {
        return lse(lse(a, b), c);
    }

    private double comb(double a, double b, boolean viterbi) {
        return viterbi ? Math.max(a, b) : lse(a, b);
    }

    private double[][][] forward(int[] x, boolean viterbi) {
        int n = x.length;
        double[][] fM = new double[length + 1][n + 1];
        double[][] fI = new double[length + 1][n + 1];
        double[][] fD = new double[length + 1][n + 1];
        for (int k = 0; k <= length; k++) {
            Arrays.fill(fM[k], NEG);
            Arrays.fill(fI[k], NEG);
            Arrays.fill(fD[k], NEG);
        }
        fM[0][0] = 0;
        for (int i = 0; i <= n; i++) {
            for (int k = 0; k <= length; k++) {
                if (k >= 1 && i >= 1) {
                    double v = comb(comb(fM[k - 1][i - 1] + tMM[k - 1], fI[k - 1][i - 1] + tIM[k - 1], viterbi),
                            fD[k - 1][i - 1] + tDM[k - 1], viterbi);
                    fM[k][i] = v + eM[k][x[i - 1]];
                }
                if (i >= 1) {
                    fI[k][i] = logBackground[x[i - 1]]
                            + comb(fM[k][i - 1] + tMI[k], fI[k][i - 1] + tII[k], viterbi);
                }
                if (k >= 1) {
                    fD[k][i] = comb(fM[k - 1][i] + tMD[k - 1], fD[k - 1][i] + tDD[k - 1], viterbi);
                }
            }
        }
        return new double[][][] {fM, fI, fD};
    }

    private double end(double[][][] f, int n, boolean viterbi) {
        return comb(comb(f[0][length][n] + tMM[length], f[1][length][n] + tIM[length], viterbi),
                f[2][length][n] + tDM[length], viterbi);
    }

    /**
     * The forward log-likelihood {@code log P(x | model)}.
     *
     * @param x the sequence
     * @return the log-likelihood
     */
    public double logLikelihood(int[] x) {
        return end(forward(x, false), x.length, false);
    }

    /**
     * The forward log-odds score against the i.i.d. background null model.
     *
     * @param x the sequence
     * @return {@code log P(x | model) - log P(x | null)}
     */
    public double logOdds(int[] x) {
        return logLikelihood(x) - nullLogLikelihood(x);
    }

    /**
     * The Viterbi (best single path) log-odds score.
     *
     * @param x the sequence
     * @return the best path's log-probability minus the null log-likelihood
     */
    public double viterbiLogOdds(int[] x) {
        return end(forward(x, true), x.length, true) - nullLogLikelihood(x);
    }

    private double nullLogLikelihood(int[] x) {
        double s = 0;
        for (int v : x) {
            s += logBackground[v];
        }
        return s;
    }

    /**
     * Baum-Welch: expected counts from forward-backward over all sequences, then pseudocount
     * re-estimation.
     *
     * @param sequences unaligned training sequences
     * @param iterations EM iterations
     * @param pseudocount the emission pseudocount mass
     * @return the mean log-likelihood per sequence after the last iteration
     */
    public double baumWelch(List<int[]> sequences, int iterations, double pseudocount) {
        double mean = 0;
        for (int it = 0; it < iterations; it++) {
            Counts c = new Counts();
            mean = 0;
            for (int[] x : sequences) {
                mean += accumulate(x, c) / sequences.size();
            }
            maximise(c, pseudocount);
        }
        return mean;
    }

    private double accumulate(int[] x, Counts c) {
        int n = x.length;
        double[][][] f = forward(x, false);
        double[][] fM = f[0];
        double[][] fI = f[1];
        double[][] fD = f[2];
        double logP = end(f, n, false);
        double[][] bM = new double[length + 1][n + 1];
        double[][] bI = new double[length + 1][n + 1];
        double[][] bD = new double[length + 1][n + 1];
        for (int k = 0; k <= length; k++) {
            Arrays.fill(bM[k], NEG);
            Arrays.fill(bI[k], NEG);
            Arrays.fill(bD[k], NEG);
        }
        for (int i = n; i >= 0; i--) {
            for (int k = length; k >= 0; k--) {
                double m = NEG;
                double ins = NEG;
                double d = NEG;
                if (k == length && i == n) {
                    m = tMM[length];
                    ins = tIM[length];
                    d = tDM[length];
                }
                if (k < length && i < n) {
                    double next = eM[k + 1][x[i]] + bM[k + 1][i + 1];
                    m = lse(m, tMM[k] + next);
                    ins = lse(ins, tIM[k] + next);
                    d = lse(d, tDM[k] + next);
                }
                if (i < n) {
                    double next = logBackground[x[i]] + bI[k][i + 1];
                    m = lse(m, tMI[k] + next);
                    ins = lse(ins, tII[k] + next);
                }
                if (k < length) {
                    m = lse(m, tMD[k] + bD[k + 1][i]);
                    d = lse(d, tDD[k] + bD[k + 1][i]);
                }
                bM[k][i] = m;
                bI[k][i] = ins;
                bD[k][i] = k == 0 ? NEG : d;
            }
        }
        for (int i = 0; i <= n; i++) {
            for (int k = 0; k <= length; k++) {
                if (k >= 1 && i >= 1) {
                    double p = Math.exp(fM[k][i] + bM[k][i] - logP);
                    c.eM[k][x[i - 1]] += p;
                }
                if (k < length && i < n) {
                    double next = eM[k + 1][x[i]] + bM[k + 1][i + 1] - logP;
                    c.t[k][0] += Math.exp(fM[k][i] + tMM[k] + next);
                    c.t[k][3] += Math.exp(fI[k][i] + tIM[k] + next);
                    if (k >= 1) {
                        c.t[k][5] += Math.exp(fD[k][i] + tDM[k] + next);
                    }
                }
                if (i < n) {
                    double next = logBackground[x[i]] + bI[k][i + 1] - logP;
                    c.t[k][1] += Math.exp(fM[k][i] + tMI[k] + next);
                    c.t[k][4] += Math.exp(fI[k][i] + tII[k] + next);
                }
                if (k < length) {
                    c.t[k][2] += Math.exp(fM[k][i] + tMD[k] + bD[k + 1][i] - logP);
                    if (k >= 1) {
                        c.t[k][6] += Math.exp(fD[k][i] + tDD[k] + bD[k + 1][i] - logP);
                    }
                }
            }
        }
        c.t[length][0] += Math.exp(fM[length][n] + tMM[length] - logP);
        c.t[length][3] += Math.exp(fI[length][n] + tIM[length] - logP);
        c.t[length][5] += Math.exp(fD[length][n] + tDM[length] - logP);
        return logP;
    }

    /**
     * The match-state emission distribution of a column.
     *
     * @param column the column, 1-based
     * @return the probabilities
     */
    public double[] matchEmission(int column) {
        double[] p = new double[alphabet];
        for (int a = 0; a < alphabet; a++) {
            p[a] = Math.exp(eM[column][a]);
        }
        return p;
    }
}
