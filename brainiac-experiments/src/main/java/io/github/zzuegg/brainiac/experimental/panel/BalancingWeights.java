package io.github.zzuegg.brainiac.experimental.panel;

import io.github.zzuegg.brainiac.experimental.discovery.LeastSquares;
import io.github.zzuegg.brainiac.experimental.ml.CausalEstimators;
import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;

import java.util.Arrays;

/**
 * Covariate balancing weights for the average treatment effect on the treated: control weights
 * (non-negative, summing to 1) chosen so that the weighted controls look like the treated.
 *
 * <ul>
 *   <li>{@link Kind#ENTROPY_BALANCING} (Hainmueller, Political Analysis 2012,
 *       <a href="https://doi.org/10.1093/pan/mpr025">doi:10.1093/pan/mpr025</a>): the weights
 *       closest to uniform in KL divergence that match the treated means exactly; the convex dual
 *       {@code min_l log sum_i exp(l . (c_i - m))} is solved by Newton's method and
 *       {@code w_i proportional to exp(l . c_i)}.</li>
 *   <li>{@link Kind#CBPS} (Imai and Ratkovic, JRSS-B 2014,
 *       <a href="https://doi.org/10.1111/rssb.12027">doi:10.1111/rssb.12027</a>): the
 *       over-identified covariate balancing propensity score: a logistic propensity fitted by GMM
 *       on both the likelihood score and the ATT balance conditions
 *       {@code sum (t_i - (1 - t_i) e_i / (1 - e_i)) x_i = 0} (Gauss-Newton from the logistic
 *       MLE), with control weights {@code e / (1 - e)}.</li>
 *   <li>{@link Kind#STABLE_WEIGHTS} (Zubizarreta, JASA 2015,
 *       <a href="https://doi.org/10.1080/01621459.2015.1023805">doi:10.1080/01621459.2015.1023805</a>):
 *       the minimum-variance weights on the simplex whose standardised imbalances stay within a
 *       tolerance {@code delta}, by accelerated projected gradient on a quadratic-penalty form of
 *       the QP.</li>
 *   <li>{@link Kind#IPW}, the plain logistic propensity odds, as the reference.</li>
 * </ul>
 * Covariates are standardised internally; balance is on means (first moments).
 */
public final class BalancingWeights {

    private BalancingWeights() {
    }

    /** A weighting scheme. */
    public enum Kind {
        /** Logistic propensity odds. */
        IPW,
        /** Entropy balancing. */
        ENTROPY_BALANCING,
        /** Covariate balancing propensity score (over-identified GMM). */
        CBPS,
        /** Stable balancing weights. */
        STABLE_WEIGHTS
    }

    /**
     * Control weights.
     *
     * @param x covariates {@code [n][d]}
     * @param t treatment
     * @param kind the scheme
     * @param delta the standardised imbalance tolerance of {@link Kind#STABLE_WEIGHTS}
     * @return per-unit weights: {@code 1/n_t} on treated units, weights summing to 1 on controls
     */
    public static double[] weights(double[][] x, int[] t, Kind kind, double delta) {
        double[][] z = LeastSquares.standardise(x);
        int n = z.length;
        double[] w = switch (kind) {
            case IPW -> odds(CausalEstimators.propensity(z, t, 1e-6), t);
            case ENTROPY_BALANCING -> entropy(z, t);
            case CBPS -> cbps(z, t);
            case STABLE_WEIGHTS -> stable(z, t, delta);
        };
        double sc = 0;
        int nt = 0;
        for (int i = 0; i < n; i++) {
            if (t[i] == 0) {
                sc += w[i];
            } else {
                nt++;
            }
        }
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            out[i] = t[i] == 1 ? 1.0 / nt : w[i] / sc;
        }
        return out;
    }

    private static double[] odds(double[] e, int[] t) {
        double[] w = new double[e.length];
        for (int i = 0; i < e.length; i++) {
            double p = Math.min(e[i], 0.999);
            w[i] = t[i] == 0 ? p / (1 - p) : 1;
        }
        return w;
    }

    private static double[] treatedMean(double[][] z, int[] t) {
        int d = z[0].length;
        double[] m = new double[d];
        int nt = 0;
        for (int i = 0; i < z.length; i++) {
            if (t[i] == 1) {
                nt++;
                for (int j = 0; j < d; j++) {
                    m[j] += z[i][j];
                }
            }
        }
        for (int j = 0; j < d; j++) {
            m[j] /= nt;
        }
        return m;
    }

    private static double[] entropy(double[][] z, int[] t) {
        int d = z[0].length;
        double[] m = treatedMean(z, t);
        double[] lambda = new double[d];
        double[] w = new double[z.length];
        for (int it = 0; it < 100; it++) {
            double max = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < z.length; i++) {
                if (t[i] == 0) {
                    max = Math.max(max, dot(lambda, z[i], m));
                }
            }
            double s = 0;
            for (int i = 0; i < z.length; i++) {
                w[i] = t[i] == 0 ? Math.exp(dot(lambda, z[i], m) - max) : 0;
                s += w[i];
            }
            double[] g = new double[d];
            double[][] h = new double[d][d];
            for (int i = 0; i < z.length; i++) {
                if (t[i] == 1) {
                    continue;
                }
                double p = w[i] / s;
                for (int a = 0; a < d; a++) {
                    g[a] += p * (z[i][a] - m[a]);
                }
            }
            for (int i = 0; i < z.length; i++) {
                if (t[i] == 1) {
                    continue;
                }
                double p = w[i] / s;
                for (int a = 0; a < d; a++) {
                    for (int b = 0; b < d; b++) {
                        h[a][b] += p * (z[i][a] - m[a] - g[a]) * (z[i][b] - m[b] - g[b]);
                    }
                }
            }
            double gn = 0;
            for (double v : g) {
                gn += v * v;
            }
            if (Math.sqrt(gn) < 1e-9) {
                break;
            }
            double[] step = MlMatrices.choleskySolve(MlMatrices.cholesky(h), g);
            for (int a = 0; a < d; a++) {
                lambda[a] -= Math.max(-5, Math.min(5, step[a]));
            }
        }
        return w;
    }

    private static double dot(double[] l, double[] z, double[] m) {
        double s = 0;
        for (int a = 0; a < l.length; a++) {
            s += l[a] * (z[a] - m[a]);
        }
        return s;
    }

    /** The stacked GMM moments of CBPS (score and balance), each averaged over units. */
    private static double[] moments(double[][] z, int[] t, double[] beta) {
        int d = z[0].length + 1;
        int n = z.length;
        double[] g = new double[2 * d];
        for (int i = 0; i < n; i++) {
            double eta = beta[0];
            for (int a = 1; a < d; a++) {
                eta += beta[a] * z[i][a - 1];
            }
            eta = Math.max(-30, Math.min(30, eta));
            double e = 1 / (1 + Math.exp(-eta));
            double odds = Math.exp(eta);
            for (int a = 0; a < d; a++) {
                double xa = a == 0 ? 1 : z[i][a - 1];
                g[a] += (t[i] - e) * xa / n;
                g[d + a] += (t[i] - (1 - t[i]) * odds) * xa / n;
            }
        }
        return g;
    }

    private static double[] cbps(double[][] z, int[] t) {
        int d = z[0].length + 1;
        // start from the logistic MLE
        double[] e0 = CausalEstimators.propensity(z, t, 1e-6);
        double[][] design = new double[z.length][];
        double[] logit = new double[z.length];
        for (int i = 0; i < z.length; i++) {
            design[i] = z[i];
            double p = Math.max(1e-6, Math.min(1 - 1e-6, e0[i]));
            logit[i] = Math.log(p / (1 - p));
        }
        double[] beta = LeastSquares.fit(design, logit, true, 1e-9).beta();
        for (int it = 0; it < 30; it++) {
            double[] g = moments(z, t, beta);
            double[][] jac = new double[g.length][d];
            for (int a = 0; a < d; a++) {
                double[] bp = beta.clone();
                bp[a] += 1e-5;
                double[] gp = moments(z, t, bp);
                for (int k = 0; k < g.length; k++) {
                    jac[k][a] = (gp[k] - g[k]) / 1e-5;
                }
            }
            double[][] jtj = MlMatrices.multiply(MlMatrices.transpose(jac), jac);
            for (int a = 0; a < d; a++) {
                jtj[a][a] += 1e-8;
            }
            double[] jtg = new double[d];
            for (int a = 0; a < d; a++) {
                for (int k = 0; k < g.length; k++) {
                    jtg[a] += jac[k][a] * g[k];
                }
            }
            double[] step = MlMatrices.choleskySolve(MlMatrices.cholesky(jtj), jtg);
            double before = norm(g);
            double s = 1;
            double[] next = beta.clone();
            for (int ls = 0; ls < 20; ls++) {
                for (int a = 0; a < d; a++) {
                    next[a] = beta[a] - s * step[a];
                }
                if (norm(moments(z, t, next)) < before) {
                    break;
                }
                s /= 2;
            }
            if (Math.abs(before - norm(moments(z, t, next))) < 1e-12) {
                beta = next;
                break;
            }
            beta = next;
        }
        double[] w = new double[z.length];
        for (int i = 0; i < z.length; i++) {
            double eta = beta[0];
            for (int a = 1; a < d; a++) {
                eta += beta[a] * z[i][a - 1];
            }
            w[i] = t[i] == 0 ? Math.exp(Math.max(-30, Math.min(30, eta))) : 1;
        }
        return w;
    }

    private static double norm(double[] g) {
        double s = 0;
        for (double v : g) {
            s += v * v;
        }
        return s;
    }

    private static double[] stable(double[][] z, int[] t, double delta) {
        int d = z[0].length;
        double[] m = treatedMean(z, t);
        int[] controls = java.util.stream.IntStream.range(0, z.length).filter(i -> t[i] == 0).toArray();
        int nc = controls.length;
        double rho = 200;
        double[] w = new double[nc];
        Arrays.fill(w, 1.0 / nc);
        double[] y = w.clone();
        double tk = 1;
        // Lipschitz bound of 2 + 2 rho ||C||^2 (C the control covariates)
        double cn = 0;
        for (int c : controls) {
            for (int a = 0; a < d; a++) {
                cn += z[c][a] * z[c][a];
            }
        }
        double lip = 2 + 2 * rho * cn;
        for (int it = 0; it < 3000; it++) {
            double[] bal = new double[d];
            for (int k = 0; k < nc; k++) {
                for (int a = 0; a < d; a++) {
                    bal[a] += y[k] * z[controls[k]][a];
                }
            }
            double[] viol = new double[d];
            for (int a = 0; a < d; a++) {
                double gap = bal[a] - m[a];
                viol[a] = Math.signum(gap) * Math.max(0, Math.abs(gap) - delta);
            }
            double[] step = new double[nc];
            for (int k = 0; k < nc; k++) {
                double g = 2 * y[k];
                for (int a = 0; a < d; a++) {
                    g += 2 * rho * viol[a] * z[controls[k]][a];
                }
                step[k] = y[k] - g / lip;
            }
            double[] next = SyntheticControl.projectSimplex(step);
            double tn = (1 + Math.sqrt(1 + 4 * tk * tk)) / 2;
            for (int k = 0; k < nc; k++) {
                y[k] = next[k] + (tk - 1) / tn * (next[k] - w[k]);
            }
            w = next;
            tk = tn;
        }
        double[] out = new double[z.length];
        for (int k = 0; k < nc; k++) {
            out[controls[k]] = w[k];
        }
        for (int i = 0; i < z.length; i++) {
            if (t[i] == 1) {
                out[i] = 1;
            }
        }
        return out;
    }

    /**
     * The weighted ATT.
     *
     * @param y outcomes
     * @param t treatment
     * @param w weights from {@link #weights}
     * @return {@code mean over treated of y - sum over controls of w y}
     */
    public static double att(double[] y, int[] t, double[] w) {
        double s = 0;
        for (int i = 0; i < y.length; i++) {
            s += (t[i] == 1 ? 1 : -1) * w[i] * y[i];
        }
        return s;
    }

    /**
     * The largest absolute standardised mean difference after weighting.
     *
     * @param x covariates
     * @param t treatment
     * @param w weights from {@link #weights}
     * @return {@code max_j |mean_t x_j - sum_c w x_j| / sd(x_j)}
     */
    public static double maxSmd(double[][] x, int[] t, double[] w) {
        double[][] z = LeastSquares.standardise(x);
        double worst = 0;
        for (int j = 0; j < z[0].length; j++) {
            double s = 0;
            for (int i = 0; i < z.length; i++) {
                s += (t[i] == 1 ? 1 : -1) * w[i] * z[i][j];
            }
            worst = Math.max(worst, Math.abs(s));
        }
        return worst;
    }

    /**
     * Kish's effective sample size of the control weights.
     *
     * @param t treatment
     * @param w weights
     * @return {@code (sum w)^2 / sum w^2} over controls
     */
    public static double effectiveControls(int[] t, double[] w) {
        double s = 0;
        double s2 = 0;
        for (int i = 0; i < t.length; i++) {
            if (t[i] == 0) {
                s += w[i];
                s2 += w[i] * w[i];
            }
        }
        return s * s / s2;
    }
}
