package io.github.zzuegg.brainiac.experimental.testing;

import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;
import io.github.zzuegg.brainiac.experimental.stats.PenalizedRegression;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * Model-X knockoffs for variable selection with finite-sample FDR control (Candes, Fan, Janson
 * and Lv, JRSS-B 2018, <a href="https://arxiv.org/abs/1610.02351">arXiv:1610.02351</a>; Barber and
 * Candes, Annals 2015). For Gaussian features {@code X ~ N(0, Sigma)}, knockoffs
 * {@code X~ | X ~ N(X - X Sigma^-1 D, 2D - D Sigma^-1 D)} with {@code D = diag(s)} are
 * exchangeable with {@code X} yet carry no information about {@code y}. A lasso on
 * {@code [X, X~]} gives the coefficient-difference statistics {@code W_j = |b_j| - |b~_j|},
 * which are sign-symmetric for nulls, and the knockoff(+) threshold
 * {@code T = min{t : (c + #{W <= -t}) / #{W >= t} <= q}} controls the FDR at {@code q}
 * ({@code c = 1} for knockoff+, exact control; {@code c = 0}, a modified FDR).
 *
 * <p>The diagonal {@code s} comes from the equicorrelated construction
 * {@code s_j = min(1, 2 lambda_min(C))} on the correlation matrix {@code C}, or from the SDP
 * {@code max sum s_j} s.t. {@code 2C - diag(s) >= 0}, solved approximately by coordinate ascent
 * with Sherman-Morrison updates of {@code (2C - diag(s))^-1}.
 */
public final class Knockoffs {

    /** The construction of {@code s}. */
    public enum Construction {
        /** Equicorrelated. */
        EQUICORRELATED,
        /** SDP by coordinate ascent. */
        SDP
    }

    /**
     * A selection.
     *
     * @param selected the selected features
     * @param statistics the {@code W} statistics
     * @param threshold the data-dependent threshold
     */
    public record Selection(boolean[] selected, double[] statistics, double threshold) {
    }

    private Knockoffs() {
    }

    /**
     * Runs the full pipeline: estimate {@code Sigma}, sample knockoffs, lasso (CV) on the
     * augmented design, threshold.
     *
     * @param x the design {@code [n][p]}
     * @param y the response
     * @param q the target FDR
     * @param construction the {@code s} construction
     * @param plus knockoff+ (exact) rather than knockoff
     * @param seed the seed
     * @return the selection
     */
    public static Selection select(double[][] x, double[] y, double q, Construction construction, boolean plus,
            long seed) {
        int p = x[0].length;
        double[][] xk = sample(x, construction, seed);
        double[][] aug = new double[x.length][2 * p];
        for (int i = 0; i < x.length; i++) {
            System.arraycopy(x[i], 0, aug[i], 0, p);
            System.arraycopy(xk[i], 0, aug[i], p, p);
        }
        PenalizedRegression lasso = PenalizedRegression.of(PenalizedRegression.Penalty.LASSO);
        PenalizedRegression.Fit f = lasso.crossValidate(aug, y, 30, 0.01, 5, seed).min();
        double[] w = new double[p];
        for (int j = 0; j < p; j++) {
            w[j] = Math.abs(f.beta()[j]) - Math.abs(f.beta()[j + p]);
        }
        double t = threshold(w, q, plus);
        boolean[] sel = new boolean[p];
        for (int j = 0; j < p; j++) {
            sel[j] = w[j] >= t;
        }
        return new Selection(sel, w, t);
    }

    /**
     * The knockoff(+) threshold.
     *
     * @param w the statistics
     * @param q the target FDR
     * @param plus whether to add one to the numerator
     * @return the threshold ({@code +infinity} when no {@code t} qualifies)
     */
    public static double threshold(double[] w, double q, boolean plus) {
        double[] cand = Arrays.stream(w).map(Math::abs).filter(v -> v > 0).sorted().toArray();
        for (double t : cand) {
            int neg = 0;
            int pos = 0;
            for (double v : w) {
                if (v <= -t) {
                    neg++;
                }
                if (v >= t) {
                    pos++;
                }
            }
            if ((plus ? 1.0 : 0.0) + neg <= q * Math.max(1, pos)) {
                return t;
            }
        }
        return Double.POSITIVE_INFINITY;
    }

    /**
     * Samples Gaussian knockoffs with {@code Sigma} estimated from the data (standardised, with
     * a little shrinkage towards the identity).
     *
     * @param x the design
     * @param construction the {@code s} construction
     * @param seed the seed
     * @return the knockoff copy, on the original scale of {@code x}
     */
    public static double[][] sample(double[][] x, Construction construction, long seed) {
        int n = x.length;
        int p = x[0].length;
        double[] mean = MlMatrices.columnMeans(x);
        double[][] cov = GraphicalLasso.covariance(x);
        double[] sd = new double[p];
        for (int j = 0; j < p; j++) {
            sd[j] = Math.sqrt(cov[j][j]);
        }
        double[][] c = new double[p][p];
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < p; j++) {
                c[i][j] = 0.95 * cov[i][j] / (sd[i] * sd[j]) + (i == j ? 0.05 : 0);
            }
        }
        double[] s = construction == Construction.EQUICORRELATED ? equicorrelated(c) : sdp(c);
        double[][] cInv = MlMatrices.choleskyInverse(MlMatrices.cholesky(c));
        // V = 2D - D C^-1 D
        double[][] v = new double[p][p];
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < p; j++) {
                v[i][j] = (i == j ? 2 * s[i] : 0) - s[i] * cInv[i][j] * s[j];
            }
        }
        double[][] l = MlMatrices.cholesky(v);
        SplittableRandom r = new SplittableRandom(seed);
        double[][] out = new double[n][p];
        double[] z = new double[p];
        double[] u = new double[p];
        for (int t = 0; t < n; t++) {
            for (int j = 0; j < p; j++) {
                u[j] = (x[t][j] - mean[j]) / sd[j];
                z[j] = r.nextGaussian();
            }
            for (int j = 0; j < p; j++) {
                // mean: u - u C^-1 D
                double m = u[j];
                for (int k = 0; k < p; k++) {
                    m -= u[k] * cInv[k][j] * s[j];
                }
                double e = 0;
                for (int k = 0; k <= j; k++) {
                    e += l[j][k] * z[k];
                }
                out[t][j] = mean[j] + sd[j] * (m + e);
            }
        }
        return out;
    }

    /**
     * The equicorrelated diagonal {@code min(1, 2 lambda_min(C))}, slightly shrunk.
     *
     * @param c a correlation matrix
     * @return {@code s}
     */
    public static double[] equicorrelated(double[][] c) {
        double[] ev = MlMatrices.eigh(c).values();
        double lmin = ev[ev.length - 1];
        double[] s = new double[c.length];
        Arrays.fill(s, 0.999 * Math.min(1, 2 * lmin));
        return s;
    }

    /**
     * An approximate SDP diagonal by coordinate ascent: each {@code s_j} grows to (nearly) the
     * largest value keeping {@code G = 2C - diag(s)} positive definite, which by the Schur
     * complement is {@code s_j + 1 / (G^-1)_jj}, capped at 1; {@code G^-1} is kept current by
     * rank-one Sherman-Morrison updates. Starts from the equicorrelated solution.
     *
     * @param c a correlation matrix
     * @return {@code s}
     */
    public static double[] sdp(double[][] c) {
        int p = c.length;
        double[] s = equicorrelated(c);
        for (int j = 0; j < p; j++) {
            s[j] *= 0.9;
        }
        double[][] g = new double[p][p];
        for (int i = 0; i < p; i++) {
            for (int j = 0; j < p; j++) {
                g[i][j] = 2 * c[i][j] - (i == j ? s[i] : 0);
            }
        }
        double[][] gi = MlMatrices.choleskyInverse(MlMatrices.cholesky(g));
        for (int sweep = 0; sweep < 3; sweep++) {
            for (int j = 0; j < p; j++) {
                double room = 1 / gi[j][j];
                double target = Math.min(1, s[j] + 0.9 * room);
                double delta = target - s[j];
                if (delta <= 1e-9) {
                    continue;
                }
                // G <- G - delta e_j e_j^T; (G - d e e^T)^-1 = Gi + d Gi e e^T Gi / (1 - d Gi_jj)
                double denom = 1 - delta * gi[j][j];
                double[] col = new double[p];
                for (int i = 0; i < p; i++) {
                    col[i] = gi[i][j];
                }
                for (int i = 0; i < p; i++) {
                    for (int k = 0; k < p; k++) {
                        gi[i][k] += delta * col[i] * col[k] / denom;
                    }
                }
                s[j] = target;
            }
        }
        return s;
    }
}
