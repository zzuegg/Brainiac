package io.github.zzuegg.brainiac.experimental.estimators;

import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Gradient estimators for expectations over discrete random variables,
 * {@code d/d theta E_{q_theta}[f(b)]}, for factorised Bernoulli and categorical distributions
 * parameterised by logits.
 *
 * <p>Sampling is piecewise constant in the logits, so backpropagation alone gives no gradient.
 * The estimators trade bias against variance:
 * <ul>
 *   <li>{@link Estimator#REINFORCE} (Williams 1992): {@code (f(b) - baseline) d log q(b)}, with an
 *       exponential moving-average baseline; unbiased, high variance;</li>
 *   <li>{@link Estimator#RLOO} (Kool et al. 2019): {@code K} samples, each baselined by the mean
 *       of the other {@code K - 1}; unbiased;</li>
 *   <li>{@link Estimator#STRAIGHT_THROUGH} and {@link Estimator#GUMBEL_ST} (Bengio et al. 2013;
 *       Jang et al. 2017): the gradient of a relaxed {@code f} at the hard sample, passed through
 *       the mean or through a tempered Gumbel-softmax; one evaluation, biased;</li>
 *   <li>{@link Estimator#REBAR} (Tucker et al. 2017, <a href="https://arxiv.org/abs/1703.07370">
 *       arXiv:1703.07370</a>): the concrete relaxation {@code f(sigma(z / tau))} as a control
 *       variate evaluated at the reparameterised sample {@code z} and at a conditional sample
 *       {@code z~ | b}, with its scale {@code eta} adapted online to minimise the variance;</li>
 *   <li>{@link Estimator#RELAX} (Grathwohl et al. 2018, <a href="https://arxiv.org/abs/1711.00123">
 *       arXiv:1711.00123</a>): REBAR plus a learned surrogate {@code r_w(z)}, here a per-coordinate
 *       quadratic in {@code sigma(z)}, trained by gradient descent on the estimator's own squared
 *       norm. Because the estimator is affine in the control-variate parameters, that variance
 *       gradient is exact;</li>
 *   <li>{@link Estimator#ARM} (Yin and Zhou 2019, <a href="https://arxiv.org/abs/1807.11143">
 *       arXiv:1807.11143</a>) and {@link Estimator#DISARM} (Dong, Mnih and Tucker 2020,
 *       <a href="https://arxiv.org/abs/2006.10680">arXiv:2006.10680</a>): two antithetic samples
 *       sharing one uniform; DisARM integrates out the uniform where the two agree;</li>
 *   <li>{@link Estimator#RODEO} (Shi et al. 2022, <a href="https://arxiv.org/abs/2202.09497">
 *       arXiv:2202.09497</a>): RLOO with control variates {@code (A h)(x)} from the random-scan
 *       Gibbs Stein operator {@code A = P - I}, which has mean zero under {@code q}. The surrogates
 *       {@code h_k(y) = mean_{j != k} H(f(x_j), grad f(x_j) . (y - x_j))} reuse the other samples'
 *       values and gradients, so no extra evaluation of {@code f} is needed. Here {@code H} and
 *       the global {@code H*} are linear in their two inputs (the paper uses a small MLP), which
 *       makes the Gibbs operator closed form and the variance gradient exact.</li>
 * </ul>
 * The categorical case supports REINFORCE, RLOO, straight-through and Gumbel straight-through.
 */
public final class DiscreteGradients {

    /** The available estimators. */
    public enum Estimator {
        /** Score function with a moving-average baseline. */
        REINFORCE(1),
        /** Score function with leave-one-out baselines. */
        RLOO(1),
        /** Relaxed gradient at the hard sample through the mean. */
        STRAIGHT_THROUGH(1),
        /** Relaxed gradient at the hard sample through a Gumbel-softmax. */
        GUMBEL_ST(1),
        /** Concrete-relaxation control variate with an adaptive scale. */
        REBAR(3),
        /** REBAR with a learned surrogate. */
        RELAX(3),
        /** Augment-REINFORCE-merge. */
        ARM(2),
        /** Augment-REINFORCE-swap-merge. */
        DISARM(2),
        /** RLOO with discrete Stein-operator control variates. */
        RODEO(1);

        private final int evaluations;

        Estimator(int evaluations) {
            this.evaluations = evaluations;
        }

        /**
         * Evaluations of {@code f} (hard or relaxed) per drawn sample; ARM and DisARM draw pairs.
         *
         * @return the evaluations per draw
         */
        public int evaluationsPerDraw() {
            return evaluations;
        }
    }

    /** A function of a binary vector with a gradient of its continuous relaxation. */
    public interface Objective {

        /**
         * Evaluates {@code f} at a point of {@code [0, 1]^d} (binary or relaxed).
         *
         * @param b the point
         * @return {@code f(b)}
         */
        double value(double[] b);

        /**
         * The gradient of the relaxed {@code f}.
         *
         * @param z the point
         * @return {@code df/dz}
         */
        double[] gradient(double[] z);
    }

    /** A function of {@code m} categorical variables with {@code k} classes each. */
    public interface CategoricalObjective {

        /**
         * Evaluates {@code f} at a point of the product of simplices (one-hot or relaxed).
         *
         * @param y the point {@code [m][k]}
         * @return {@code f(y)}
         */
        double value(double[][] y);

        /**
         * The gradient of the relaxed {@code f}.
         *
         * @param y the point {@code [m][k]}
         * @return {@code df/dy}
         */
        double[][] gradient(double[][] y);
    }

    private final Estimator estimator;
    private final int samples;
    private final double temperature;
    private final double cvRate;
    private double baseline;
    private boolean baselineSet;
    private double[] phi;
    private double[] phiM;
    private double[] phiV;
    private int adamSteps;
    private long evaluations;

    private DiscreteGradients(Estimator estimator, int samples, double temperature, double cvRate) {
        this.estimator = Objects.requireNonNull(estimator, "estimator");
        if (samples < 1 || (estimator == Estimator.RLOO || estimator == Estimator.RODEO) && samples < 2) {
            throw new IllegalArgumentException("too few samples for " + estimator + ": " + samples);
        }
        this.samples = samples;
        this.temperature = temperature;
        this.cvRate = cvRate;
    }

    /**
     * Creates an estimator.
     *
     * @param estimator the estimator
     * @param samples draws per gradient (pairs for ARM and DisARM; at least 2 for RLOO and RODEO)
     * @param temperature the relaxation temperature of Gumbel-ST, REBAR and RELAX, for example 0.5
     * @param cvLearningRate the Adam step of the control-variate parameters (REBAR, RELAX, RODEO),
     *     for example 0.01
     * @return the estimator
     */
    public static DiscreteGradients of(Estimator estimator, int samples, double temperature, double cvLearningRate) {
        return new DiscreteGradients(estimator, samples, temperature, cvLearningRate);
    }

    /**
     * The estimator kind.
     *
     * @return the kind
     */
    public Estimator estimator() {
        return estimator;
    }

    /**
     * Evaluations of {@code f} (hard or relaxed) so far.
     *
     * @return the count
     */
    public long evaluations() {
        return evaluations;
    }

    private static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    private static double logit(double u) {
        return Math.log(u) - Math.log1p(-u);
    }

    private static double uniform(SplittableRandom r) {
        return Math.min(1 - 1e-12, Math.max(1e-12, r.nextDouble()));
    }

    /**
     * Estimates {@code d/d theta E[f(b)]} for independent {@code b_i ~ Bernoulli(sigmoid(theta_i))}.
     *
     * @param logits the logits {@code theta}
     * @param f the objective
     * @param random the random source
     * @return the gradient estimate
     */
    public double[] bernoulli(double[] logits, Objective f, SplittableRandom random) {
        int d = logits.length;
        double[] p = new double[d];
        for (int i = 0; i < d; i++) {
            p[i] = sigmoid(logits[i]);
        }
        return switch (estimator) {
            case REINFORCE -> reinforce(p, f, random);
            case RLOO -> rloo(p, f, random);
            case STRAIGHT_THROUGH -> straightThrough(p, f, random);
            case GUMBEL_ST -> gumbelSt(logits, f, random);
            case REBAR, RELAX -> rebar(logits, p, f, random);
            case ARM -> arm(logits, f, random);
            case DISARM -> disarm(logits, p, f, random);
            case RODEO -> rodeo(p, f, random);
        };
    }

    private double[] draw(double[] p, SplittableRandom r) {
        double[] b = new double[p.length];
        for (int i = 0; i < p.length; i++) {
            b[i] = r.nextDouble() < p[i] ? 1 : 0;
        }
        return b;
    }

    private double eval(Objective f, double[] b) {
        evaluations++;
        return f.value(b);
    }

    private double[] reinforce(double[] p, Objective f, SplittableRandom r) {
        int d = p.length;
        double[] g = new double[d];
        double mean = 0;
        for (int k = 0; k < samples; k++) {
            double[] b = draw(p, r);
            double v = eval(f, b);
            mean += v / samples;
            double c = baselineSet ? baseline : 0;
            for (int i = 0; i < d; i++) {
                g[i] += (v - c) * (b[i] - p[i]) / samples;
            }
        }
        baseline = baselineSet ? 0.9 * baseline + 0.1 * mean : mean;
        baselineSet = true;
        return g;
    }

    private double[] rloo(double[] p, Objective f, SplittableRandom r) {
        int d = p.length;
        double[][] b = new double[samples][];
        double[] v = new double[samples];
        double sum = 0;
        for (int k = 0; k < samples; k++) {
            b[k] = draw(p, r);
            v[k] = eval(f, b[k]);
            sum += v[k];
        }
        double[] g = new double[d];
        for (int k = 0; k < samples; k++) {
            double base = (sum - v[k]) / (samples - 1);
            for (int i = 0; i < d; i++) {
                g[i] += (v[k] - base) * (b[k][i] - p[i]) / samples;
            }
        }
        return g;
    }

    private double[] straightThrough(double[] p, Objective f, SplittableRandom r) {
        int d = p.length;
        double[] g = new double[d];
        for (int k = 0; k < samples; k++) {
            double[] b = draw(p, r);
            evaluations++;
            double[] df = f.gradient(b);
            for (int i = 0; i < d; i++) {
                g[i] += df[i] * p[i] * (1 - p[i]) / samples;
            }
        }
        return g;
    }

    private double[] gumbelSt(double[] logits, Objective f, SplittableRandom r) {
        int d = logits.length;
        double[] g = new double[d];
        double[] b = new double[d];
        double[] s = new double[d];
        for (int k = 0; k < samples; k++) {
            for (int i = 0; i < d; i++) {
                double z = logits[i] + logit(uniform(r));
                b[i] = z > 0 ? 1 : 0;
                s[i] = sigmoid(z / temperature);
            }
            evaluations++;
            double[] df = f.gradient(b);
            for (int i = 0; i < d; i++) {
                g[i] += df[i] * s[i] * (1 - s[i]) / temperature / samples;
            }
        }
        return g;
    }

    private double[] arm(double[] logits, Objective f, SplittableRandom r) {
        int d = logits.length;
        double[] g = new double[d];
        double[] b1 = new double[d];
        double[] b2 = new double[d];
        double[] u = new double[d];
        for (int k = 0; k < samples; k++) {
            for (int i = 0; i < d; i++) {
                u[i] = r.nextDouble();
                b1[i] = u[i] > sigmoid(-logits[i]) ? 1 : 0;
                b2[i] = u[i] < sigmoid(logits[i]) ? 1 : 0;
            }
            double diff = eval(f, b1) - eval(f, b2);
            for (int i = 0; i < d; i++) {
                g[i] += diff * (u[i] - 0.5) / samples;
            }
        }
        return g;
    }

    private double[] disarm(double[] logits, double[] p, Objective f, SplittableRandom r) {
        int d = logits.length;
        double[] g = new double[d];
        double[] b = new double[d];
        double[] bt = new double[d];
        for (int k = 0; k < samples; k++) {
            for (int i = 0; i < d; i++) {
                double u = r.nextDouble();
                b[i] = u < p[i] ? 1 : 0;
                bt[i] = 1 - u < p[i] ? 1 : 0;
            }
            double diff = eval(f, b) - eval(f, bt);
            for (int i = 0; i < d; i++) {
                if (b[i] != bt[i]) {
                    g[i] += 0.5 * diff * (bt[i] == 1 ? -1 : 1) * sigmoid(Math.abs(logits[i])) / samples;
                }
            }
        }
        return g;
    }

    private void ensurePhi(int size) {
        if (phi == null || phi.length != size) {
            phi = new double[size];
            phiM = new double[size];
            phiV = new double[size];
            adamSteps = 0;
        }
    }

    /** One Adam step on the control-variate parameters given the gradient of the squared norm. */
    private void adam(double[] grad, boolean[] frozen) {
        adamSteps++;
        double b1 = 0.9;
        double b2 = 0.999;
        for (int j = 0; j < phi.length; j++) {
            if (frozen != null && frozen[j]) {
                continue;
            }
            phiM[j] = b1 * phiM[j] + (1 - b1) * grad[j];
            phiV[j] = b2 * phiV[j] + (1 - b2) * grad[j] * grad[j];
            double mh = phiM[j] / (1 - Math.pow(b1, adamSteps));
            double vh = phiV[j] / (1 - Math.pow(b2, adamSteps));
            phi[j] -= cvRate * mh / (Math.sqrt(vh) + 1e-8);
        }
    }

    /**
     * REBAR and RELAX. The control variate is
     * {@code c(z) = eta f(sigma(z / tau)) + c0 + sum_i (w1_i s_i + w2_i s_i^2)} with
     * {@code s = sigma(z)} (REBAR keeps only {@code eta} and the constant baseline {@code c0});
     * the estimate is {@code [f(b) - c(z~)] (b - p) + d c(z)/d theta - d c(z~)/d theta}. It is affine
     * in {@code phi = (eta, c0, w1, w2)}, so its Jacobian {@code J} is formed column by column and the
     * parameters follow {@code 2 J^T g}, the exact gradient of {@code |g|^2}.
     */
    private double[] rebar(double[] logits, double[] p, Objective f, SplittableRandom r) {
        int d = logits.length;
        boolean relax = estimator == Estimator.RELAX;
        int np = relax ? 2 + 2 * d : 2;
        boolean fresh = phi == null || phi.length != np;
        ensurePhi(np);
        if (fresh) {
            phi[0] = 1;   // eta starts at the plain REBAR value
        }
        double[] total = new double[d];
        double[] phiGrad = new double[np];
        double tau = temperature;
        for (int k = 0; k < samples; k++) {
            double[] z = new double[d];
            double[] zt = new double[d];
            double[] dzt = new double[d];
            double[] b = new double[d];
            for (int i = 0; i < d; i++) {
                z[i] = logits[i] + logit(uniform(r));
                b[i] = z[i] > 0 ? 1 : 0;
                double v = uniform(r);
                double up;
                double dup;
                if (b[i] == 1) {
                    up = (1 - p[i]) + v * p[i];
                    dup = -(1 - v) * p[i] * (1 - p[i]);
                } else {
                    up = v * (1 - p[i]);
                    dup = -v * p[i] * (1 - p[i]);
                }
                up = Math.min(1 - 1e-12, Math.max(1e-12, up));
                zt[i] = logits[i] + logit(up);
                dzt[i] = 1 + dup / (up * (1 - up));
            }
            double[] sz = new double[d];
            double[] szt = new double[d];
            for (int i = 0; i < d; i++) {
                sz[i] = sigmoid(z[i] / tau);
                szt[i] = sigmoid(zt[i] / tau);
            }
            double fb = eval(f, b);
            double fzt = eval(f, szt);
            evaluations++;   // the relaxed evaluation at z, made through its gradient
            double[] gz = f.gradient(sz);
            double[] gzt = f.gradient(szt);
            // g = g0 + J phi
            double[] g0 = new double[d];
            double[][] jac = new double[np][d];
            for (int i = 0; i < d; i++) {
                double score = b[i] - p[i];
                g0[i] = fb * score;
                jac[0][i] = -fzt * score + gz[i] * sz[i] * (1 - sz[i]) / tau
                        - gzt[i] * szt[i] * (1 - szt[i]) / tau * dzt[i];
                jac[1][i] = -score;   // a constant baseline inside the control variate
            }
            if (relax) {
                double[] s = new double[d];
                double[] st = new double[d];
                for (int i = 0; i < d; i++) {
                    s[i] = sigmoid(z[i]);
                    st[i] = sigmoid(zt[i]);
                }
                for (int m = 0; m < d; m++) {
                    double[] c1 = jac[2 + m];
                    double[] c2 = jac[2 + d + m];
                    for (int i = 0; i < d; i++) {
                        double score = b[i] - p[i];
                        c1[i] = -st[m] * score;
                        c2[i] = -st[m] * st[m] * score;
                    }
                    c1[m] += s[m] * (1 - s[m]) - st[m] * (1 - st[m]) * dzt[m];
                    c2[m] += 2 * s[m] * s[m] * (1 - s[m]) - 2 * st[m] * st[m] * (1 - st[m]) * dzt[m];
                }
            }
            double[] g = g0.clone();
            for (int j = 0; j < np; j++) {
                for (int i = 0; i < d; i++) {
                    g[i] += jac[j][i] * phi[j];
                }
            }
            for (int i = 0; i < d; i++) {
                total[i] += g[i] / samples;
            }
            for (int j = 0; j < np; j++) {
                double s = 0;
                for (int i = 0; i < d; i++) {
                    s += jac[j][i] * g[i];
                }
                phiGrad[j] += 2 * s / samples;
            }
        }
        adam(phiGrad, null);
        return total;
    }

    /**
     * RODEO with linear surrogate networks: {@code gamma = (a, b, c, a*, b*, c*)} with
     * {@code H(f, t) = a f + b t + c} and {@code H*(f, t) = a* f + b* t + c*}.
     */
    private double[] rodeo(double[] p, Objective f, SplittableRandom r) {
        int d = p.length;
        int kk = samples;
        ensurePhi(6);
        double[][] x = new double[kk][];
        double[] fv = new double[kk];
        double[][] gf = new double[kk][];
        double[] gx = new double[kk];
        for (int k = 0; k < kk; k++) {
            x[k] = draw(p, r);
            fv[k] = eval(f, x[k]);
            gf[k] = f.gradient(x[k]);
            for (int i = 0; i < d; i++) {
                gx[k] += gf[k][i] * x[k][i];
            }
        }
        // the estimate is affine in gamma: evaluate at gamma = 0 and at each unit vector
        double[] g0 = rodeoEstimate(p, x, fv, gf, gx, new double[6]);
        double[][] jac = new double[6][];
        for (int j = 0; j < 6; j++) {
            double[] e = new double[6];
            e[j] = 1;
            double[] gj = rodeoEstimate(p, x, fv, gf, gx, e);
            for (int i = 0; i < d; i++) {
                gj[i] -= g0[i];
            }
            jac[j] = gj;
        }
        double[] g = g0.clone();
        for (int j = 0; j < 6; j++) {
            for (int i = 0; i < d; i++) {
                g[i] += jac[j][i] * phi[j];
            }
        }
        double[] grad = new double[6];
        for (int j = 0; j < 6; j++) {
            for (int i = 0; i < d; i++) {
                grad[j] += 2 * jac[j][i] * g[i];
            }
        }
        adam(grad, null);
        return g;
    }

    private static double[] rodeoEstimate(double[] p, double[][] x, double[] fv, double[][] gf, double[] gx,
            double[] gamma) {
        int kk = x.length;
        int d = p.length;
        // surrogate k: h_k(y) = alpha_k + beta_k . y, from the samples other than k
        double[] alpha = new double[kk];
        double[][] beta = new double[kk][d];
        double[] alphaStar = new double[kk];
        double[][] betaStar = new double[kk][d];
        for (int k = 0; k < kk; k++) {
            for (int j = 0; j < kk; j++) {
                if (j == k) {
                    continue;
                }
                alpha[k] += (gamma[0] * fv[j] - gamma[1] * gx[j] + gamma[2]) / (kk - 1);
                alphaStar[k] += (gamma[3] * fv[j] - gamma[4] * gx[j] + gamma[5]) / (kk - 1);
                for (int i = 0; i < d; i++) {
                    beta[k][i] += gamma[1] * gf[j][i] / (kk - 1);
                    betaStar[k][i] += gamma[4] * gf[j][i] / (kk - 1);
                }
            }
        }
        // (A h_j)(x_j) = (1/d) sum_i beta_j,i (p_i - x_j,i) for a linear h and the Gibbs kernel
        double[] ah = new double[kk];
        double fsum = 0;
        for (int j = 0; j < kk; j++) {
            for (int i = 0; i < d; i++) {
                ah[j] += beta[j][i] * (p[i] - x[j][i]) / d;
            }
            fsum += fv[j] + ah[j];
        }
        double[] g = new double[d];
        for (int k = 0; k < kk; k++) {
            double base = (fsum - fv[k] - ah[k]) / (kk - 1);
            double hx = alphaStar[k];
            double sTot = 0;
            for (int i = 0; i < d; i++) {
                hx += betaStar[k][i] * x[k][i];
                sTot += betaStar[k][i] * (p[i] - x[k][i]);
            }
            for (int m = 0; m < d; m++) {
                double score = x[k][m] - p[m];
                double gibbsMean = (d - 1.0) / d * hx + (sTot - betaStar[k][m] * (p[m] - x[k][m])) / d;
                double global = score * gibbsMean + p[m] * (1 - p[m]) * betaStar[k][m] / d - hx * score;
                g[m] += ((fv[k] - base) * score + global) / kk;
            }
        }
        return g;
    }

    /**
     * Estimates {@code d/d theta E[f(y)]} for independent categorical variables
     * {@code y_m ~ softmax(theta_m)} (REINFORCE, RLOO, STRAIGHT_THROUGH and GUMBEL_ST only).
     *
     * @param logits the logits {@code [m][k]}
     * @param f the objective on one-hot rows
     * @param random the random source
     * @return the gradient estimate {@code [m][k]}
     */
    public double[][] categorical(double[][] logits, CategoricalObjective f, SplittableRandom random) {
        int m = logits.length;
        int k = logits[0].length;
        double[][] prob = new double[m][];
        for (int a = 0; a < m; a++) {
            prob[a] = softmax(logits[a], 1);
        }
        double[][] g = new double[m][k];
        switch (estimator) {
            case REINFORCE, RLOO -> {
                double[][][] y = new double[samples][][];
                double[] v = new double[samples];
                double sum = 0;
                for (int s = 0; s < samples; s++) {
                    y[s] = new double[m][k];
                    for (int a = 0; a < m; a++) {
                        y[s][a][sampleIndex(prob[a], random)] = 1;
                    }
                    evaluations++;
                    v[s] = f.value(y[s]);
                    sum += v[s];
                }
                double c = baselineSet ? baseline : 0;
                for (int s = 0; s < samples; s++) {
                    double base = estimator == Estimator.RLOO ? (sum - v[s]) / (samples - 1) : c;
                    for (int a = 0; a < m; a++) {
                        for (int j = 0; j < k; j++) {
                            g[a][j] += (v[s] - base) * (y[s][a][j] - prob[a][j]) / samples;
                        }
                    }
                }
                baseline = baselineSet ? 0.9 * baseline + 0.1 * sum / samples : sum / samples;
                baselineSet = true;
            }
            case STRAIGHT_THROUGH, GUMBEL_ST -> {
                boolean gumbel = estimator == Estimator.GUMBEL_ST;
                for (int s = 0; s < samples; s++) {
                    double[][] y = new double[m][k];
                    double[][] soft = new double[m][];
                    for (int a = 0; a < m; a++) {
                        if (gumbel) {
                            double[] pert = new double[k];
                            for (int j = 0; j < k; j++) {
                                pert[j] = logits[a][j] - Math.log(-Math.log(uniform(random)));
                            }
                            soft[a] = softmax(pert, temperature);
                            y[a][argmax(pert)] = 1;
                        } else {
                            soft[a] = prob[a];
                            y[a][sampleIndex(prob[a], random)] = 1;
                        }
                    }
                    evaluations++;
                    double[][] df = f.gradient(y);
                    double scale = gumbel ? 1 / temperature : 1;
                    for (int a = 0; a < m; a++) {
                        double dot = 0;
                        for (int j = 0; j < k; j++) {
                            dot += df[a][j] * soft[a][j];
                        }
                        for (int j = 0; j < k; j++) {
                            g[a][j] += scale * soft[a][j] * (df[a][j] - dot) / samples;
                        }
                    }
                }
            }
            default -> throw new UnsupportedOperationException(estimator + " is implemented for Bernoulli variables");
        }
        return g;
    }

    private static double[] softmax(double[] x, double tau) {
        double max = Double.NEGATIVE_INFINITY;
        for (double v : x) {
            max = Math.max(max, v / tau);
        }
        double[] out = new double[x.length];
        double s = 0;
        for (int j = 0; j < x.length; j++) {
            out[j] = Math.exp(x[j] / tau - max);
            s += out[j];
        }
        for (int j = 0; j < x.length; j++) {
            out[j] /= s;
        }
        return out;
    }

    private static int argmax(double[] x) {
        int best = 0;
        for (int j = 1; j < x.length; j++) {
            if (x[j] > x[best]) {
                best = j;
            }
        }
        return best;
    }

    private static int sampleIndex(double[] p, SplittableRandom r) {
        double u = r.nextDouble();
        double c = 0;
        for (int j = 0; j < p.length; j++) {
            c += p[j];
            if (u < c) {
                return j;
            }
        }
        return p.length - 1;
    }

    /**
     * The exact gradient {@code d/d theta E[f(b)]} by enumerating all {@code 2^d} states, for
     * {@code d <= 20}.
     *
     * @param logits the logits
     * @param f the objective
     * @return the exact gradient
     */
    public static double[] exactBernoulli(double[] logits, Objective f) {
        int d = logits.length;
        if (d > 20) {
            throw new IllegalArgumentException("too many variables to enumerate: " + d);
        }
        double[] p = new double[d];
        for (int i = 0; i < d; i++) {
            p[i] = sigmoid(logits[i]);
        }
        double[] g = new double[d];
        double[] b = new double[d];
        for (int s = 0; s < (1 << d); s++) {
            double q = 1;
            for (int i = 0; i < d; i++) {
                b[i] = (s >> i) & 1;
                q *= b[i] == 1 ? p[i] : 1 - p[i];
            }
            double v = f.value(b);
            for (int i = 0; i < d; i++) {
                g[i] += q * v * (b[i] - p[i]);
            }
        }
        return g;
    }
}
