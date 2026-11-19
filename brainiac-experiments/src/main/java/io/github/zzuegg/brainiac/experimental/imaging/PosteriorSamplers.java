package io.github.zzuegg.brainiac.experimental.imaging;

import io.github.zzuegg.brainiac.tensor.Matrices;

import java.util.SplittableRandom;

/**
 * Diffusion posterior sampling for {@code y = F(x) + n}, {@code n ~ N(0, sigma_y^2 I)}, with any
 * prior given as a {@link Score} model, the minimum-mean-square-error denoiser
 * {@code D(x, sigma) = E[x0 | x0 + sigma e = x]} of a variance-exploding diffusion (the score is
 * {@code (D - x) / sigma^2} by Tweedie's formula). Every sampler runs the same geometric noise
 * schedule from {@code sigmaMax} to {@code sigmaMin} and returns the last denoised estimate:
 * <ul>
 *   <li>{@link Method#DPS} (Chung et al., ICLR 2023): ancestral steps plus the gradient of
 *       {@code -|y - F(D(x_t))|}, scaled by {@code zeta / |y - F(D)|}, through the denoiser;</li>
 *   <li>{@link Method#PI_GDM} (Song et al., ICLR 2023): pseudo-inverse guidance with the Gaussian
 *       approximation {@code x0 | x_t ~ N(D, r_t^2 I)}, {@code r_t^2 = sigma_t^2 v / (sigma_t^2 + v)},
 *       solved by conjugate gradients (linear {@code F} only);</li>
 *   <li>{@link Method#DDRM} (Kawar et al., NeurIPS 2022): in the SVD basis of {@code A}, observed
 *       components are pulled to {@code y}, unobserved ones follow DDIM (linear {@code F} only);</li>
 *   <li>{@link Method#DAPS} (Zhang et al., CVPR 2025): at each level, Langevin steps on
 *       {@code x0} under {@code N(D(x_t), sigma_t^2 I)} times the likelihood, then re-noising to
 *       the next level (decoupled steps, any differentiable {@code F}).</li>
 * </ul>
 * Vector-Jacobian products through the denoiser use central differences along the vector (the
 * Tweedie Jacobian is symmetric), so any black-box score model works; each call to
 * {@link Score#denoise} is counted.
 */
public final class PosteriorSamplers {

    /** The samplers. */
    public enum Method {
        /** Diffusion posterior sampling. */
        DPS,
        /** Pseudo-inverse guided diffusion. */
        PI_GDM,
        /** Denoising diffusion restoration models. */
        DDRM,
        /** Decoupled annealed posterior sampling. */
        DAPS
    }

    /** A diffusion prior as its denoiser. */
    @FunctionalInterface
    public interface Score {
        /**
         * The posterior mean of the clean signal.
         *
         * @param x the noisy point
         * @param sigma its noise level
         * @return {@code E[x0 | x]}
         */
        double[] denoise(double[] x, double sigma);
    }

    /** A measurement model with its vector-Jacobian product. */
    public interface Forward {
        /**
         * Measures.
         *
         * @param x the signal
         * @return {@code F(x)}
         */
        double[] apply(double[] x);

        /**
         * The vector-Jacobian product.
         *
         * @param x the point
         * @param v the measurement-space vector
         * @return {@code J_F(x)^T v}
         */
        double[] vjp(double[] x, double[] v);

        /** @return the linear operator when {@code F} is linear, else {@code null} */
        LinearOperator linear();

        /**
         * A linear measurement.
         *
         * @param a the operator
         * @return the model
         */
        static Forward linear(LinearOperator a) {
            return new Forward() {
                @Override
                public double[] apply(double[] x) {
                    return a.apply(x);
                }

                @Override
                public double[] vjp(double[] x, double[] v) {
                    return a.adjoint(v);
                }

                @Override
                public LinearOperator linear() {
                    return a;
                }
            };
        }

        /**
         * The magnitude measurement {@code |A x|} (elementwise), phase-retrieval-like.
         *
         * @param a the operator
         * @return the model
         */
        static Forward magnitude(LinearOperator a) {
            return new Forward() {
                @Override
                public double[] apply(double[] x) {
                    double[] z = a.apply(x);
                    for (int i = 0; i < z.length; i++) {
                        z[i] = Math.abs(z[i]);
                    }
                    return z;
                }

                @Override
                public double[] vjp(double[] x, double[] v) {
                    double[] z = a.apply(x);
                    double[] w = new double[z.length];
                    for (int i = 0; i < z.length; i++) {
                        w[i] = Math.signum(z[i]) * v[i];
                    }
                    return a.adjoint(w);
                }

                @Override
                public LinearOperator linear() {
                    return null;
                }
            };
        }
    }

    /**
     * The schedule and step sizes.
     *
     * @param steps the noise levels
     * @param sigmaMax the first level
     * @param sigmaMin the last level
     * @param dataVariance {@code v}, the prior's per-coordinate variance (PI-GDM)
     * @param dpsStep DPS's {@code zeta}
     * @param ddrmEta DDRM's {@code eta}
     * @param dapsLangevinSteps DAPS's Langevin steps per level
     * @param dapsStepScale DAPS's Langevin step as a share of the stable step
     */
    public record Settings(int steps, double sigmaMax, double sigmaMin, double dataVariance, double dpsStep,
            double ddrmEta, int dapsLangevinSteps, double dapsStepScale) {

        /**
         * The defaults used in the experiments.
         *
         * @param steps the noise levels
         * @param sigmaMax the first level
         * @param sigmaMin the last level
         * @param dataVariance the prior's variance
         * @return the settings
         */
        public static Settings defaults(int steps, double sigmaMax, double sigmaMin, double dataVariance) {
            return new Settings(steps, sigmaMax, sigmaMin, dataVariance, 1.0, 0.85, 20, 0.5);
        }
    }

    /** One measurement, with the spectral decomposition DDRM needs computed once. */
    public static final class Problem {

        private final Forward forward;
        private final double[] y;
        private final double sigmaY;
        private final int dimension;
        private final double lipschitz;
        private double[] singular2;
        private double[][] basis;

        /**
         * Creates the problem.
         *
         * @param forward the measurement model
         * @param y the measurement
         * @param sigmaY the noise level
         * @param dimension the signal dimension
         * @param normSquared an upper bound on {@code |J_F|^2}
         */
        public Problem(Forward forward, double[] y, double sigmaY, int dimension, double normSquared) {
            this.forward = forward;
            this.y = y.clone();
            this.sigmaY = sigmaY;
            this.dimension = dimension;
            this.lipschitz = normSquared;
        }

        private void spectral() {
            if (basis != null) {
                return;
            }
            LinearOperator a = forward.linear();
            int d = dimension;
            double[][] ata = new double[d][];
            for (int j = 0; j < d; j++) {
                double[] e = new double[d];
                e[j] = 1;
                ata[j] = a.adjoint(a.apply(e));
            }
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < i; j++) {
                    double s = 0.5 * (ata[i][j] + ata[j][i]);
                    ata[i][j] = s;
                    ata[j][i] = s;
                }
            }
            Matrices.SymmetricEigen eig = Matrices.symmetricEigen(ata);
            singular2 = eig.values().clone();
            basis = new double[d][d];
            for (int c = 0; c < d; c++) {
                for (int r = 0; r < d; r++) {
                    basis[c][r] = eig.vectors()[r][c];
                }
            }
        }
    }

    /** Counts denoiser calls. */
    private static final class Counter implements Score {
        private final Score inner;
        private long calls;

        Counter(Score inner) {
            this.inner = inner;
        }

        @Override
        public double[] denoise(double[] x, double sigma) {
            calls++;
            return inner.denoise(x, sigma);
        }
    }

    /**
     * One posterior sample with its cost.
     *
     * @param x the sample
     * @param scoreEvaluations the denoiser calls it took
     */
    public record Sample(double[] x, long scoreEvaluations) {
    }

    private PosteriorSamplers() {
    }

    private static double[] schedule(Settings s) {
        double[] sig = new double[s.steps() + 1];
        for (int i = 0; i < s.steps(); i++) {
            sig[i] = s.sigmaMax() * Math.pow(s.sigmaMin() / s.sigmaMax(), i / (double) (s.steps() - 1));
        }
        return sig;
    }

    /**
     * Draws one posterior sample.
     *
     * @param method the sampler
     * @param score the prior
     * @param problem the measurement
     * @param settings the schedule
     * @param r the random source
     * @return the sample and its cost
     */
    public static Sample sample(Method method, Score score, Problem problem, Settings settings, SplittableRandom r) {
        if ((method == Method.PI_GDM || method == Method.DDRM) && problem.forward.linear() == null) {
            throw new IllegalArgumentException(method + " needs a linear measurement");
        }
        Counter c = new Counter(score);
        double[] x = switch (method) {
            case DPS -> dps(c, problem, settings, r, false);
            case PI_GDM -> dps(c, problem, settings, r, true);
            case DDRM -> ddrm(c, problem, settings, r);
            case DAPS -> daps(c, problem, settings, r);
        };
        return new Sample(x, c.calls);
    }

    private static double[] noise(int d, double s, SplittableRandom r) {
        double[] z = new double[d];
        for (int i = 0; i < d; i++) {
            z[i] = s * gaussian(r);
        }
        return z;
    }

    static double gaussian(SplittableRandom r) {
        double u = 1 - r.nextDouble();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r.nextDouble());
    }

    /** {@code J_D(x)^T v} by a central difference along {@code v}. */
    private static double[] denoiserVjp(Score s, double[] x, double sigma, double[] v) {
        double n = ImageMath.norm(v);
        if (n == 0) {
            return new double[x.length];
        }
        double eps = 1e-3 * sigma;
        double[] up = s.denoise(ImageMath.axpy(x, eps / n, v), sigma);
        double[] dn = s.denoise(ImageMath.axpy(x, -eps / n, v), sigma);
        double[] g = new double[x.length];
        for (int i = 0; i < g.length; i++) {
            g[i] = (up[i] - dn[i]) * n / (2 * eps);
        }
        return g;
    }

    /** The ancestral step from {@code sigma} to {@code next} given the denoised estimate. */
    private static double[] ancestral(double[] x, double[] x0, double sigma, double next, SplittableRandom r) {
        double up2 = next * next * (sigma * sigma - next * next) / (sigma * sigma);
        double down = Math.sqrt(Math.max(next * next - up2, 0));
        double[] out = new double[x.length];
        double up = Math.sqrt(up2);
        for (int i = 0; i < x.length; i++) {
            out[i] = x0[i] + down * (x[i] - x0[i]) / sigma + up * gaussian(r);
        }
        return out;
    }

    private static double[] dps(Score s, Problem p, Settings st, SplittableRandom r, boolean pseudoInverse) {
        double[] sig = schedule(st);
        int d = p.dimension;
        double[] x = noise(d, sig[0], r);
        double[] x0 = x;
        for (int i = 0; i < st.steps(); i++) {
            double sg = sig[i];
            x0 = s.denoise(x, sg);
            double[] res = ImageMath.axpy(p.y, -1, p.forward.apply(x0));
            double[] g;
            if (pseudoInverse) {
                double rt2 = sg * sg * st.dataVariance() / (sg * sg + st.dataVariance());
                double[] w = solveMeasurementSystem(p.forward.linear(), rt2, p.sigmaY * p.sigmaY, res);
                g = denoiserVjp(s, x, sg, p.forward.vjp(x0, w));
            } else {
                g = denoiserVjp(s, x, sg, p.forward.vjp(x0, res));
            }
            if (i == st.steps() - 1) {
                break;
            }
            double next = sig[i + 1];
            double[] xn = ancestral(x, x0, sg, next, r);
            if (pseudoInverse) {
                x = ImageMath.axpy(xn, sg * sg - next * next, g);
            } else {
                double rn = Math.max(ImageMath.norm(res), 1e-12);
                x = ImageMath.axpy(xn, st.dpsStep() / rn, g);
            }
        }
        return x0;
    }

    /** Solves {@code (rt2 A A^T + s2 I) w = res} by conjugate gradients. */
    private static double[] solveMeasurementSystem(LinearOperator a, double rt2, double s2, double[] res) {
        int m = res.length;
        double[] w = new double[m];
        double[] rr = res.clone();
        double[] q = rr.clone();
        double rs = ImageMath.dot(rr, rr);
        for (int it = 0; it < 50 && rs > 1e-20; it++) {
            double[] aq = a.apply(a.adjoint(q));
            for (int i = 0; i < m; i++) {
                aq[i] = rt2 * aq[i] + s2 * q[i];
            }
            double alpha = rs / ImageMath.dot(q, aq);
            for (int i = 0; i < m; i++) {
                w[i] += alpha * q[i];
                rr[i] -= alpha * aq[i];
            }
            double nrs = ImageMath.dot(rr, rr);
            for (int i = 0; i < m; i++) {
                q[i] = rr[i] + nrs / rs * q[i];
            }
            rs = nrs;
        }
        return w;
    }

    private static double[] ddrm(Score s, Problem p, Settings st, SplittableRandom r) {
        p.spectral();
        double[] sig = schedule(st);
        int d = p.dimension;
        double[] s2 = p.singular2;
        double top = 0;
        for (double v : s2) {
            top = Math.max(top, v);
        }
        boolean[] observed = new boolean[d];
        double[] ybar = new double[d];
        double[] ynoise = new double[d];
        LinearOperator a = p.forward.linear();
        for (int c = 0; c < d; c++) {
            if (s2[c] > 1e-9 * top) {
                observed[c] = true;
                ybar[c] = ImageMath.dot(a.apply(p.basis[c]), p.y) / s2[c];
                ynoise[c] = p.sigmaY / Math.sqrt(s2[c]);
            }
        }
        double eta = st.ddrmEta();
        double[] xb = new double[d];
        for (int c = 0; c < d; c++) {
            xb[c] = observed[c] && ynoise[c] < sig[0]
                    ? ybar[c] + Math.sqrt(sig[0] * sig[0] - ynoise[c] * ynoise[c]) * gaussian(r)
                    : sig[0] * gaussian(r);
        }
        double[] x0 = null;
        for (int i = 0; i < st.steps(); i++) {
            double sg = sig[i];
            x0 = s.denoise(toPixels(p.basis, xb), sg);
            if (i == st.steps() - 1) {
                break;
            }
            double next = sig[i + 1];
            double[] x0b = toSpectral(p.basis, x0);
            for (int c = 0; c < d; c++) {
                if (!observed[c]) {
                    xb[c] = x0b[c] + Math.sqrt(1 - eta * eta) * next * (xb[c] - x0b[c]) / sg
                            + eta * next * gaussian(r);
                } else if (next < ynoise[c]) {
                    xb[c] = x0b[c] + Math.sqrt(1 - eta * eta) * next * (ybar[c] - x0b[c]) / ynoise[c]
                            + eta * next * gaussian(r);
                } else {
                    xb[c] = ybar[c] + Math.sqrt(next * next - ynoise[c] * ynoise[c]) * gaussian(r);
                }
            }
        }
        return x0;
    }

    private static double[] toPixels(double[][] basis, double[] xb) {
        double[] x = new double[basis.length];
        for (int c = 0; c < basis.length; c++) {
            if (xb[c] != 0) {
                for (int j = 0; j < x.length; j++) {
                    x[j] += xb[c] * basis[c][j];
                }
            }
        }
        return x;
    }

    private static double[] toSpectral(double[][] basis, double[] x) {
        double[] xb = new double[basis.length];
        for (int c = 0; c < basis.length; c++) {
            xb[c] = ImageMath.dot(basis[c], x);
        }
        return xb;
    }

    private static double[] daps(Score s, Problem p, Settings st, SplittableRandom r) {
        double[] sig = schedule(st);
        int d = p.dimension;
        double[] x = noise(d, sig[0], r);
        double[] x0 = null;
        double iy = 1 / (p.sigmaY * p.sigmaY);
        for (int i = 0; i < st.steps(); i++) {
            double sg = sig[i];
            double[] mean = s.denoise(x, sg);
            double rt2 = sg * sg;
            double eta = st.dapsStepScale() / (1 / rt2 + p.lipschitz * iy);
            x0 = mean.clone();
            for (int k = 0; k < st.dapsLangevinSteps(); k++) {
                double[] res = ImageMath.axpy(p.forward.apply(x0), -1, p.y);
                double[] gl = p.forward.vjp(x0, res);
                double sq = Math.sqrt(2 * eta);
                boolean last = i == st.steps() - 1;
                for (int j = 0; j < d; j++) {
                    double grad = -(x0[j] - mean[j]) / rt2 - gl[j] * iy;
                    x0[j] += eta * grad + (last ? 0 : sq * gaussian(r));
                }
            }
            if (i < st.steps() - 1) {
                x = ImageMath.axpy(x0, 1, noise(d, sig[i + 1], r));
            }
        }
        return x0;
    }
}
