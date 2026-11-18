package io.github.zzuegg.brainiac.experimental.events;

import java.util.List;

/**
 * A multivariate linear Hawkes process (Hawkes, "Spectra of some self-exciting and mutually
 * exciting point processes", Biometrika 1971,
 * <a href="https://doi.org/10.1093/biomet/58.1.83">doi:10.1093/biomet/58.1.83</a>):
 *
 * <pre>
 * lambda_k(t) = mu_k + sum_(t_i &lt; t) sum_b alpha[k][k_i][b] g_b(t - t_i)
 * </pre>
 *
 * <p>where every past event excites future events of each type through a non-negative mixture of
 * <em>basis kernels</em> {@code g_b}, each a probability density on {@code [0, inf)}, so
 * {@code sum_b alpha[k][k'][b]} is the expected number of type-{@code k} children of one
 * type-{@code k'} event. Three fitted families share one EM algorithm, whose latent variable is the
 * branching structure (which earlier event, or the background, caused each event):
 * <ul>
 *   <li>{@link #fitExponential}: one exponential kernel whose decay is also re-estimated;</li>
 *   <li>{@link #fitSumOfExponentials}: a fixed bank of decays (a flexible, multi-scale kernel);</li>
 *   <li>{@link #fitSpline}: linear B-splines (hat functions) on a grid of lags, fitted by EMS
 *       (EM with a smoothing step, Silverman et al. 1990), the penalised non-parametric kernel
 *       that a NeurIPS 2026 re-evaluation of neural point processes found closes most of their
 *       gap to Hawkes models.</li>
 * </ul>
 * The log-likelihood and compensator are closed-form, and simulation uses thinning with each
 * kernel's supremum over the remaining lags as the bound.
 */
public final class HawkesProcess implements TemporalPointProcess {

    /** A basis kernel: a probability density on the lags {@code [0, inf)}. */
    public interface Kernel {
        /**
         * @param s a lag
         * @return the density at {@code s}
         */
        double value(double s);

        /**
         * @param s a lag
         * @return {@code int_0^s value}
         */
        double integral(double s);

        /**
         * @param s a lag
         * @return the largest density at any lag {@code >= s}
         */
        double supAfter(double s);

        /** @return a lag beyond which the kernel is negligible (used to skip old events) */
        double support();
    }

    /**
     * @param beta the decay rate
     * @return {@code beta exp(-beta s)}
     */
    public static Kernel exponential(double beta) {
        return new Kernel() {
            @Override
            public double value(double s) {
                return s < 0 ? 0 : beta * Math.exp(-beta * s);
            }

            @Override
            public double integral(double s) {
                return s <= 0 ? 0 : 1 - Math.exp(-beta * s);
            }

            @Override
            public double supAfter(double s) {
                return value(Math.max(0, s));
            }

            @Override
            public double support() {
                return 12 / beta;
            }

            @Override
            public String toString() {
                return "exp(" + beta + ")";
            }
        };
    }

    /**
     * A heavy-tailed (Omori-law) kernel.
     *
     * @param c the time offset
     * @param p the tail exponent
     * @return {@code p c^p / (s + c)^(p + 1)}
     */
    public static Kernel powerLaw(double c, double p) {
        return new Kernel() {
            @Override
            public double value(double s) {
                return s < 0 ? 0 : p * Math.pow(c, p) / Math.pow(s + c, p + 1);
            }

            @Override
            public double integral(double s) {
                return s <= 0 ? 0 : 1 - Math.pow(c / (s + c), p);
            }

            @Override
            public double supAfter(double s) {
                return value(Math.max(0, s));
            }

            @Override
            public double support() {
                return c * (Math.pow(1e4, 1 / p) - 1);
            }
        };
    }

    /**
     * A delayed (non-monotone) kernel: a normal bump truncated to positive lags.
     *
     * @param delay the peak lag
     * @param width the bump's standard deviation
     * @return the normalised bump
     */
    public static Kernel delayed(double delay, double width) {
        double mass = 1 - normalCdf(-delay / width);
        return new Kernel() {
            @Override
            public double value(double s) {
                if (s < 0) {
                    return 0;
                }
                double z = (s - delay) / width;
                return Math.exp(-0.5 * z * z) / (width * Math.sqrt(2 * Math.PI) * mass);
            }

            @Override
            public double integral(double s) {
                return s <= 0 ? 0 : (normalCdf((s - delay) / width) - normalCdf(-delay / width)) / mass;
            }

            @Override
            public double supAfter(double s) {
                return s <= delay ? value(delay) : value(s);
            }

            @Override
            public double support() {
                return delay + 6 * width;
            }
        };
    }

    /**
     * A linear B-spline (hat) basis function, normalised to a density.
     *
     * @param center the peak lag
     * @param width the half-width
     * @return the hat, truncated to positive lags
     */
    public static Kernel hat(double center, double width) {
        double lo = Math.max(0, center - width);
        double hi = center + width;
        double mass = raw(hi, center, width) - raw(lo, center, width);
        return new Kernel() {
            @Override
            public double value(double s) {
                return s < 0 ? 0 : Math.max(0, 1 - Math.abs(s - center) / width) / mass;
            }

            @Override
            public double integral(double s) {
                if (s <= lo) {
                    return 0;
                }
                return (raw(Math.min(s, hi), center, width) - raw(lo, center, width)) / mass;
            }

            @Override
            public double supAfter(double s) {
                return s <= center ? value(center) : value(s);
            }

            @Override
            public double support() {
                return hi;
            }
        };
    }

    /** The antiderivative of the unnormalised hat on its support. */
    private static double raw(double s, double c, double w) {
        double d = s - c;
        return d <= 0 ? d + d * d / (2 * w) + w / 2 : w / 2 + d - d * d / (2 * w);
    }

    static double normalCdf(double z) {
        double x = -z / Math.sqrt(2);
        double a = Math.abs(x);
        double t = 1 / (1 + 0.5 * a);
        double r = t * Math.exp(-a * a - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418
                + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223
                + t * 0.17087277)))))))));
        double erfc = x >= 0 ? r : 2 - r;
        return 0.5 * erfc;
    }

    private final double[] mu;
    private final double[][][] alpha;   // [type][parent type][basis]
    private final Kernel[] basis;
    private final double support;

    /**
     * Creates a process with given parameters.
     *
     * @param mu background rates per type
     * @param alpha non-negative weights {@code [type][parent type][basis]}
     * @param basis the basis kernels
     */
    public HawkesProcess(double[] mu, double[][][] alpha, Kernel[] basis) {
        this.mu = mu.clone();
        this.alpha = new double[alpha.length][][];
        for (int k = 0; k < alpha.length; k++) {
            this.alpha[k] = new double[alpha[k].length][];
            for (int j = 0; j < alpha[k].length; j++) {
                this.alpha[k][j] = alpha[k][j].clone();
            }
        }
        this.basis = basis.clone();
        double s = 0;
        for (Kernel g : basis) {
            s = Math.max(s, g.support());
        }
        this.support = s;
    }

    @Override
    public int types() {
        return mu.length;
    }

    /** @return the branching matrix {@code [type][parent type]}: expected children per event */
    public double[][] branching() {
        int k = mu.length;
        double[][] out = new double[k][k];
        for (int a = 0; a < k; a++) {
            for (int b = 0; b < k; b++) {
                for (double v : alpha[a][b]) {
                    out[a][b] += v;
                }
            }
        }
        return out;
    }

    /** @return the background rates */
    public double[] background() {
        return mu.clone();
    }

    @Override
    public double[] intensities(double t, EventSequence h, int count) {
        double[] l = mu.clone();
        for (int i = count - 1; i >= 0; i--) {
            double lag = t - h.times()[i];
            if (lag > support) {
                break;
            }
            int p = h.types()[i];
            for (int b = 0; b < basis.length; b++) {
                double g = basis[b].value(lag);
                if (g == 0) {
                    continue;
                }
                for (int k = 0; k < l.length; k++) {
                    l[k] += alpha[k][p][b] * g;
                }
            }
        }
        return l;
    }

    @Override
    public double[] bound(double t, EventSequence h, int count) {
        double total = 0;
        for (double m : mu) {
            total += m;
        }
        for (int i = count - 1; i >= 0; i--) {
            double lag = t - h.times()[i];
            if (lag > support) {
                break;
            }
            int p = h.types()[i];
            for (int b = 0; b < basis.length; b++) {
                double g = basis[b].supAfter(lag);
                for (double[][] a : alpha) {
                    total += a[p][b] * g;
                }
            }
        }
        return new double[] {total, Double.POSITIVE_INFINITY};
    }

    @Override
    public double compensator(double from, double to, EventSequence h, int count) {
        double c = 0;
        for (double m : mu) {
            c += m * (to - from);
        }
        for (int i = count - 1; i >= 0; i--) {
            double t = h.times()[i];
            if (from - t > support) {
                break;
            }
            int p = h.types()[i];
            for (int b = 0; b < basis.length; b++) {
                double inc = basis[b].integral(to - t) - basis[b].integral(from - t);
                for (double[][] a : alpha) {
                    c += a[p][b] * inc;
                }
            }
        }
        return c;
    }

    /**
     * Fits one exponential kernel by EM, re-estimating its decay each iteration.
     *
     * @param data sequences
     * @param types event types
     * @param iterations EM iterations
     * @return the fitted process
     */
    public static HawkesProcess fitExponential(List<EventSequence> data, int types, int iterations) {
        double beta = 1;
        HawkesProcess p = initial(data, types, new Kernel[] {exponential(beta)});
        for (int it = 0; it < iterations; it++) {
            double[] stats = new double[2];
            p = p.emStep(data, 0, stats);
            if (stats[1] > 0) {
                beta = Math.clamp(stats[0] / stats[1], 1e-3, 1e3);
            }
            p = new HawkesProcess(p.mu, p.alpha, new Kernel[] {exponential(beta)});
        }
        return p;
    }

    /**
     * Fits a sum of exponential kernels with fixed decays by EM.
     *
     * @param data sequences
     * @param types event types
     * @param decays the decay bank
     * @param iterations EM iterations
     * @return the fitted process
     */
    public static HawkesProcess fitSumOfExponentials(List<EventSequence> data, int types, double[] decays,
            int iterations) {
        Kernel[] basis = new Kernel[decays.length];
        for (int b = 0; b < decays.length; b++) {
            basis[b] = exponential(decays[b]);
        }
        HawkesProcess p = initial(data, types, basis);
        for (int it = 0; it < iterations; it++) {
            p = p.emStep(data, 0, null);
        }
        return p;
    }

    /**
     * Fits a linear-spline kernel on {@code [0, maxLag]} by EM with a smoothing step.
     *
     * @param data sequences
     * @param types event types
     * @param knots the number of hat functions
     * @param maxLag the largest lag the kernel covers
     * @param smoothing the share of each weight replaced by its neighbours' mean after every
     *     M-step (0 = plain EM)
     * @param iterations EM iterations
     * @return the fitted process
     */
    public static HawkesProcess fitSpline(List<EventSequence> data, int types, int knots, double maxLag,
            double smoothing, int iterations) {
        double width = maxLag / (knots - 1);
        Kernel[] basis = new Kernel[knots];
        for (int b = 0; b < knots; b++) {
            basis[b] = hat(b * width, width);
        }
        HawkesProcess p = initial(data, types, basis);
        for (int it = 0; it < iterations; it++) {
            p = p.emStep(data, smoothing, null);
        }
        return p;
    }

    private static HawkesProcess initial(List<EventSequence> data, int types, Kernel[] basis) {
        double total = 0;
        double[] counts = new double[types];
        for (EventSequence s : data) {
            total += s.horizon();
            for (int k : s.types()) {
                counts[k]++;
            }
        }
        double[] mu = new double[types];
        double[][][] alpha = new double[types][types][basis.length];
        for (int k = 0; k < types; k++) {
            mu[k] = 0.5 * counts[k] / total;
            for (int j = 0; j < types; j++) {
                java.util.Arrays.fill(alpha[k][j], 0.3 / (types * basis.length));
            }
        }
        return new HawkesProcess(mu, alpha, basis);
    }

    /**
     * One EM step.
     *
     * @param stats if not null, receives {@code {sum p_ij, sum p_ij (t_i - t_j)}} for basis 0
     *     (the exponential decay update)
     */
    private HawkesProcess emStep(List<EventSequence> data, double smoothing, double[] stats) {
        int types = mu.length;
        int nb = basis.length;
        double[] muNum = new double[types];
        double[][][] num = new double[types][types][nb];
        double[][] den = new double[types][nb];   // [parent type][basis]: sum of integrals to the horizon
        double total = 0;
        double[] resp = new double[0];
        for (EventSequence s : data) {
            total += s.horizon();
            double[] t = s.times();
            int[] k = s.types();
            for (int i = 0; i < s.size(); i++) {
                for (int b = 0; b < nb; b++) {
                    den[k[i]][b] += basis[b].integral(s.horizon() - t[i]);
                }
                int first = i;
                while (first > 0 && t[i] - t[first - 1] <= support) {
                    first--;
                }
                int m = (i - first) * nb;
                if (resp.length < m) {
                    resp = new double[m];
                }
                double lambda = mu[k[i]];
                for (int j = first; j < i; j++) {
                    for (int b = 0; b < nb; b++) {
                        double v = alpha[k[i]][k[j]][b] * basis[b].value(t[i] - t[j]);
                        resp[(j - first) * nb + b] = v;
                        lambda += v;
                    }
                }
                if (lambda <= 0) {
                    continue;
                }
                muNum[k[i]] += mu[k[i]] / lambda;
                for (int j = first; j < i; j++) {
                    for (int b = 0; b < nb; b++) {
                        double p = resp[(j - first) * nb + b] / lambda;
                        num[k[i]][k[j]][b] += p;
                        if (stats != null && b == 0) {
                            stats[0] += p;
                            stats[1] += p * (t[i] - t[j]);
                        }
                    }
                }
            }
        }
        double[] newMu = new double[types];
        double[][][] newAlpha = new double[types][types][nb];
        for (int a = 0; a < types; a++) {
            newMu[a] = Math.max(muNum[a] / total, 1e-8);
            for (int p = 0; p < types; p++) {
                for (int b = 0; b < nb; b++) {
                    newAlpha[a][p][b] = den[p][b] > 0 ? num[a][p][b] / den[p][b] : 0;
                }
                if (smoothing > 0 && nb > 2) {
                    double[] v = newAlpha[a][p].clone();
                    for (int b = 0; b < nb; b++) {
                        double left = b > 0 ? v[b - 1] : v[b];
                        double right = b + 1 < nb ? v[b + 1] : v[b];
                        newAlpha[a][p][b] = (1 - smoothing) * v[b] + smoothing * (left + right) / 2;
                    }
                }
            }
        }
        return new HawkesProcess(newMu, newAlpha, basis);
    }

    /**
     * The fitted kernel from parent type to child type at a lag, {@code sum_b alpha g_b(s)}.
     *
     * @param type the child type
     * @param parent the parent type
     * @param lag the lag
     * @return the kernel's value
     */
    public double kernel(int type, int parent, double lag) {
        double v = 0;
        for (int b = 0; b < basis.length; b++) {
            v += alpha[type][parent][b] * basis[b].value(lag);
        }
        return v;
    }
}
