package io.github.zzuegg.brainiac.experimental.bio;

import io.github.zzuegg.brainiac.experimental.data.BioSequences;

import java.util.List;
import java.util.SplittableRandom;

/**
 * Alignment scores learned end to end through the differentiable Smith-Waterman of
 * {@link SequenceAlignment#softSmithWaterman}. Training maximises the conditional
 * log-likelihood of the true alignment, {@code score(true path) - log Z}, where {@code log Z}
 * is the soft Smith-Waterman value at temperature 1: its gradient is the true alignment's pair
 * counts minus the expected alignment, the CRF view of pairwise alignment. Two
 * parameterisations: a symmetric substitution matrix ({@link Mode#MATRIX}) and a linear 1-D
 * convolutional embedding of each residue's 3-residue window whose dot products give the pair
 * scores ({@link Mode#CONV_EMBEDDING}, as in SMURF's learned alignment). Gap costs are learned
 * too (in log space, so they stay positive).
 *
 * <p>Petti et al., "End-to-end learning of multiple sequence alignments with differentiable
 * Smith-Waterman" (Bioinformatics 2023).
 */
public final class LearnedScoring {

    /** The parameterisation of the pair scores. */
    public enum Mode {
        /** a symmetric {@code A x A} substitution matrix. */
        MATRIX,
        /** context embeddings from a window-3 linear convolution, scored by dot products. */
        CONV_EMBEDDING
    }

    private final Mode mode;
    private final int alphabet;
    private final double[][] sub;
    private final double[][] w;
    private double bias;
    private double logOpen;
    private double logExtend;
    private final int dim;

    /**
     * Creates scores initialised to a match/mismatch scheme (matrix mode) or small random
     * embeddings.
     *
     * @param mode the parameterisation
     * @param alphabet the alphabet size
     * @param dim the embedding dimension (embedding mode)
     * @param seed the seed
     */
    public LearnedScoring(Mode mode, int alphabet, int dim, long seed) {
        this.mode = mode;
        this.alphabet = alphabet;
        this.dim = dim;
        this.sub = new double[alphabet][alphabet];
        for (int a = 0; a < alphabet; a++) {
            for (int b = 0; b < alphabet; b++) {
                sub[a][b] = a == b ? 2 : -1;
            }
        }
        SplittableRandom rnd = new SplittableRandom(seed);
        this.w = new double[dim][3 * alphabet];
        for (double[] row : w) {
            for (int k = 0; k < row.length; k++) {
                row[k] = 0.3 * (rnd.nextDouble() - 0.5);
            }
        }
        // the centre of the window starts as an identity-like code so training starts from identity scores
        for (int a = 0; a < alphabet && a < dim; a++) {
            w[a][alphabet + a] = 1.4;
        }
        this.bias = -0.5;
        this.logOpen = Math.log(3);
        this.logExtend = Math.log(1);
    }

    private double[][] embed(int[] seq) {
        double[][] e = new double[seq.length][dim];
        for (int i = 0; i < seq.length; i++) {
            for (int off = -1; off <= 1; off++) {
                int p = i + off;
                if (p < 0 || p >= seq.length) {
                    continue;
                }
                int col = (off + 1) * alphabet + seq[p];
                for (int d = 0; d < dim; d++) {
                    e[i][d] += w[d][col];
                }
            }
        }
        return e;
    }

    /**
     * The pair score matrix of two sequences.
     *
     * @param a the first sequence
     * @param b the second sequence
     * @return {@code [i][j]} scores
     */
    public double[][] scores(int[] a, int[] b) {
        if (mode == Mode.MATRIX) {
            double[][] s = new double[a.length][b.length];
            for (int i = 0; i < a.length; i++) {
                for (int j = 0; j < b.length; j++) {
                    s[i][j] = sub[a[i]][b[j]];
                }
            }
            return s;
        }
        double[][] ea = embed(a);
        double[][] eb = embed(b);
        double[][] s = new double[a.length][b.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b.length; j++) {
                double v = bias;
                for (int d = 0; d < dim; d++) {
                    v += ea[i][d] * eb[j][d];
                }
                s[i][j] = v;
            }
        }
        return s;
    }

    /** @return the gap-open cost */
    public double gapOpen() {
        return Math.exp(logOpen);
    }

    /** @return the gap-extend cost */
    public double gapExtend() {
        return Math.exp(logExtend);
    }

    /** @return a copy of the substitution matrix (matrix mode) */
    public double[][] substitution() {
        double[][] out = new double[alphabet][];
        for (int a = 0; a < alphabet; a++) {
            out[a] = sub[a].clone();
        }
        return out;
    }

    /**
     * Local alignment under the learned scores.
     *
     * @param a the first sequence
     * @param b the second sequence
     * @return the Smith-Waterman alignment
     */
    public SequenceAlignment.Alignment align(int[] a, int[] b) {
        return SequenceAlignment.align(a, b, scores(a, b), gapOpen(), gapExtend(), true);
    }

    /**
     * Trains on pairs with known alignments by Adam on the alignment CRF likelihood.
     *
     * @param pairs training pairs
     * @param epochs passes over the pairs
     * @param learningRate the Adam step size
     * @return the mean negative log-likelihood per pair in the last epoch
     */
    public double fit(List<BioSequences.Pair> pairs, int epochs, double learningRate) {
        int params = mode == Mode.MATRIX ? alphabet * alphabet + 2 : dim * 3 * alphabet + 3;
        double[] m1 = new double[params];
        double[] m2 = new double[params];
        int step = 0;
        double last = 0;
        for (int epoch = 0; epoch < epochs; epoch++) {
            last = 0;
            for (BioSequences.Pair pair : pairs) {
                step++;
                int[] a = pair.a();
                int[] b = pair.b();
                double[][] s = scores(a, b);
                double open = gapOpen();
                double ext = gapExtend();
                SequenceAlignment.Soft soft = SequenceAlignment.softSmithWaterman(s, open, ext, 1);
                // g[i][j] = d(-loss)/ds = truth - posterior
                double[][] g = new double[a.length][b.length];
                for (int i = 0; i < a.length; i++) {
                    for (int j = 0; j < b.length; j++) {
                        g[i][j] = -soft.pairGradient()[i][j];
                    }
                }
                double truthScore = 0;
                int opens = 0;
                int extends_ = 0;
                int pi = -1;
                int pj = -1;
                for (int i = 0; i < a.length; i++) {
                    int j = pair.match()[i];
                    if (j < 0) {
                        continue;
                    }
                    g[i][j] += 1;
                    truthScore += s[i][j];
                    if (pi >= 0) {
                        for (int gap : new int[] {i - pi - 1, j - pj - 1}) {
                            if (gap > 0) {
                                opens++;
                                extends_ += gap - 1;
                            }
                        }
                    }
                    pi = i;
                    pj = j;
                }
                truthScore -= opens * open + extends_ * ext;
                last += (soft.value() - truthScore) / pairs.size();
                double[] grad = new double[params];
                // gradient ascent direction on (truth - logZ); Adam below minimises, so negate
                double gOpen = -opens - soft.openGradient();
                double gExt = -extends_ - soft.extendGradient();
                if (mode == Mode.MATRIX) {
                    for (int i = 0; i < a.length; i++) {
                        for (int j = 0; j < b.length; j++) {
                            grad[a[i] * alphabet + b[j]] -= g[i][j] / 2;
                            grad[b[j] * alphabet + a[i]] -= g[i][j] / 2;
                        }
                    }
                } else {
                    double[][] ea = embed(a);
                    double[][] eb = embed(b);
                    double[][] da = new double[a.length][dim];
                    double[][] db = new double[b.length][dim];
                    double gb = 0;
                    for (int i = 0; i < a.length; i++) {
                        for (int j = 0; j < b.length; j++) {
                            double gij = g[i][j];
                            if (gij == 0) {
                                continue;
                            }
                            gb += gij;
                            for (int d = 0; d < dim; d++) {
                                da[i][d] += gij * eb[j][d];
                                db[j][d] += gij * ea[i][d];
                            }
                        }
                    }
                    accumulate(grad, a, da);
                    accumulate(grad, b, db);
                    grad[params - 3] = -gb;
                }
                grad[params - 2] = -gOpen * open;
                grad[params - 1] = -gExt * ext;
                double scale = 1.0 / Math.max(1, a.length);
                for (int k = 0; k < params; k++) {
                    double gk = grad[k] * scale;
                    m1[k] = 0.9 * m1[k] + 0.1 * gk;
                    m2[k] = 0.999 * m2[k] + 0.001 * gk * gk;
                    double upd = learningRate * (m1[k] / (1 - Math.pow(0.9, step)))
                            / (Math.sqrt(m2[k] / (1 - Math.pow(0.999, step))) + 1e-8);
                    apply(k, params, upd);
                }
            }
        }
        return last;
    }

    private void accumulate(double[] grad, int[] seq, double[][] de) {
        for (int i = 0; i < seq.length; i++) {
            for (int off = -1; off <= 1; off++) {
                int p = i + off;
                if (p < 0 || p >= seq.length) {
                    continue;
                }
                int col = (off + 1) * alphabet + seq[p];
                for (int d = 0; d < dim; d++) {
                    grad[d * 3 * alphabet + col] -= de[i][d];
                }
            }
        }
    }

    private void apply(int k, int params, double upd) {
        if (k == params - 2) {
            logOpen = Math.max(Math.log(0.2), logOpen - upd);
        } else if (k == params - 1) {
            logExtend = Math.max(Math.log(0.05), logExtend - upd);
        } else if (mode == Mode.MATRIX) {
            sub[k / alphabet][k % alphabet] -= upd;
        } else if (k == params - 3) {
            bias -= upd;
        } else {
            w[k / (3 * alphabet)][k % (3 * alphabet)] -= upd;
        }
    }
}
