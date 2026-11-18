package io.github.zzuegg.brainiac.experimental.survival;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;
import io.github.zzuegg.brainiac.experimental.ml.NormalTails;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Accelerated failure time models for right-censored data.
 *
 * <ul>
 *   <li>Parametric AFT (Wei 1992, <a href="https://doi.org/10.1002/sim.4780111409">doi</a>):
 *       {@code log T = b0 + x . b + sigma eps} with {@code eps} standard minimum extreme value
 *       ({@link Kind#WEIBULL}, a Weibull time with shape {@code 1 / sigma}), normal
 *       ({@link Kind#LOG_NORMAL}) or logistic ({@link Kind#LOG_LOGISTIC}), fitted by damped
 *       Newton steps on the censored log-likelihood (analytic gradient, Hessian by central
 *       differences of the gradient, a small ridge on {@code b}).</li>
 *   <li>{@link Kind#DEEP_SURVIVAL_MACHINES} (Nagpal, Li and Dubrawski 2021,
 *       <a href="https://arxiv.org/abs/2003.01176">arXiv:2003.01176</a>): a mixture of
 *       {@code K} Weibull distributions whose mixing weights and (bounded) offsets of each
 *       component's log-rate and log-shape come from a one-hidden-layer tanh network; the base
 *       log-rates and log-shapes start at the marginal {@link Weibull#fit} spread over the
 *       components. Trained by Adam on the censored mixture likelihood
 *       {@code sum_uncensored log sum_k w_k f_k(t) + alpha sum_censored log sum_k w_k S_k(t)}
 *       (the paper's loss with its censoring discount {@code alpha}).</li>
 * </ul>
 */
public final class AftModels {

    /** The model family. */
    public enum Kind {
        /** Weibull AFT. */
        WEIBULL,
        /** Log-normal AFT. */
        LOG_NORMAL,
        /** Log-logistic AFT. */
        LOG_LOGISTIC,
        /** A network-parametrised mixture of Weibulls. */
        DEEP_SURVIVAL_MACHINES
    }

    private final Kind kind;
    private final double[] beta;
    private final double sigma;
    private final List<Var> net;
    private final int components;

    private AftModels(Kind kind, double[] beta, double sigma, List<Var> net, int components) {
        this.kind = kind;
        this.beta = beta;
        this.sigma = sigma;
        this.net = net;
        this.components = components;
    }

    /** @return the family */
    public Kind kind() {
        return kind;
    }

    /** @return {@code {b0, b...}} of a parametric model (empty for the mixture) */
    public double[] coefficients() {
        return beta == null ? new double[0] : beta.clone();
    }

    /** @return {@code sigma} of a parametric model (NaN for the mixture) */
    public double sigma() {
        return sigma;
    }

    // ------------------------------------------------------------------ error distributions

    private static double logNormalSurvival(double z) {
        if (z > 6) {
            double logPhi = -0.5 * z * z - 0.5 * Math.log(2 * Math.PI);
            return logPhi - Math.log(z) + Math.log1p(-1 / (z * z) + 3 / (z * z * z * z));
        }
        return Math.log(Math.max(1e-300, 1 - NormalTails.cdf(z)));
    }

    /** {@code {log f(z), d log f / dz, log S(z), d log S / dz}}. */
    private static double[] terms(Kind kind, double z) {
        return switch (kind) {
            case WEIBULL -> {
                double ez = Math.exp(Math.min(z, 700));
                yield new double[] {z - ez, 1 - ez, -ez, -ez};
            }
            case LOG_NORMAL -> {
                double logPhi = -0.5 * z * z - 0.5 * Math.log(2 * Math.PI);
                double logS = logNormalSurvival(z);
                yield new double[] {logPhi, -z, logS, -Math.exp(logPhi - logS)};
            }
            case LOG_LOGISTIC -> {
                double sp = z > 0 ? z + Math.log1p(Math.exp(-z)) : Math.log1p(Math.exp(z));
                double sg = 1 / (1 + Math.exp(-z));
                yield new double[] {z - 2 * sp, 1 - 2 * sg, -sp, -sg};
            }
            default -> throw new IllegalStateException();
        };
    }

    private static double errorMedian(Kind kind) {
        return kind == Kind.WEIBULL ? Math.log(Math.log(2)) : 0;
    }

    private static double errorSurvival(Kind kind, double z) {
        return Math.exp(terms(kind, z)[2]);
    }

    /** {@code {log-likelihood, gradient...}} over {@code theta = {b0, b, log sigma}}. */
    private static double[] objective(Kind kind, double[][] x, double[] logT, int[] event, double[] theta,
            double ridge) {
        int d = x[0].length;
        double s = Math.exp(theta[d + 1]);
        double[] out = new double[d + 3];
        for (int i = 0; i < x.length; i++) {
            double mu = theta[0];
            for (int j = 0; j < d; j++) {
                mu += theta[j + 1] * x[i][j];
            }
            double z = (logT[i] - mu) / s;
            double[] t = terms(kind, z);
            boolean ev = event[i] != 0;
            double ll = ev ? t[0] - theta[d + 1] - logT[i] : t[2];
            double g = ev ? t[1] : t[3];
            out[0] += ll;
            double dMu = -g / s;
            out[1] += dMu;
            for (int j = 0; j < d; j++) {
                out[j + 2] += dMu * x[i][j];
            }
            out[d + 2] += -g * z - (ev ? 1 : 0);
        }
        for (int j = 1; j <= d; j++) {
            out[0] -= 0.5 * ridge * theta[j] * theta[j];
            out[j + 1] -= ridge * theta[j];
        }
        return out;
    }

    private static AftModels parametric(Kind kind, double[][] x, double[] time, int[] event, double ridge) {
        int d = x[0].length;
        int p = d + 2;
        double[] logT = new double[time.length];
        double mean = 0;
        double var = 0;
        for (int i = 0; i < time.length; i++) {
            logT[i] = Math.log(Math.max(time[i], 1e-12));
            mean += logT[i] / time.length;
        }
        for (double v : logT) {
            var += (v - mean) * (v - mean) / time.length;
        }
        double[] theta = new double[p];
        theta[0] = mean;
        theta[d + 1] = 0.5 * Math.log(Math.max(var, 1e-6));
        double current = objective(kind, x, logT, event, theta, ridge)[0];
        for (int it = 0; it < 60; it++) {
            double[] obj = objective(kind, x, logT, event, theta, ridge);
            double[] grad = java.util.Arrays.copyOfRange(obj, 1, p + 1);
            double[][] hess = new double[p][p];
            double eps = 1e-5;
            for (int j = 0; j < p; j++) {
                double[] tp = theta.clone();
                double[] tm = theta.clone();
                tp[j] += eps;
                tm[j] -= eps;
                double[] gp = objective(kind, x, logT, event, tp, ridge);
                double[] gm = objective(kind, x, logT, event, tm, ridge);
                for (int i = 0; i < p; i++) {
                    hess[i][j] = -(gp[i + 1] - gm[i + 1]) / (2 * eps);   // negative Hessian
                }
            }
            for (int i = 0; i < p; i++) {
                for (int j = 0; j < i; j++) {
                    double v = 0.5 * (hess[i][j] + hess[j][i]);
                    hess[i][j] = v;
                    hess[j][i] = v;
                }
                hess[i][i] += 1e-8;
            }
            double[] step;
            try {
                step = MlMatrices.choleskySolve(MlMatrices.cholesky(hess), grad);
            } catch (RuntimeException e) {
                step = grad.clone();
            }
            double rate = 1;
            double[] next = theta;
            double nextValue = current;
            for (int ls = 0; ls < 30; ls++) {
                double[] cand = theta.clone();
                for (int j = 0; j < p; j++) {
                    cand[j] += rate * step[j];
                }
                double v = objective(kind, x, logT, event, cand, ridge)[0];
                if (Double.isFinite(v) && v >= current) {
                    next = cand;
                    nextValue = v;
                    break;
                }
                rate /= 2;
            }
            boolean done = nextValue - current < 1e-9 * (1 + Math.abs(current));
            theta = next;
            current = nextValue;
            if (done) {
                break;
            }
        }
        return new AftModels(kind, java.util.Arrays.copyOf(theta, d + 1), Math.exp(theta[d + 1]), null, 0);
    }

    // ------------------------------------------------------------------ Deep Survival Machines

    private static Var param(int rows, int cols, double scale, SplittableRandom r) {
        float[] v = new float[rows * cols];
        for (int i = 0; i < v.length; i++) {
            v[i] = (float) (scale * r.nextGaussian());
        }
        return Var.param(Tensor.of(Shape.of(rows, cols), v));
    }

    /** {@code {log mixture weights [b, K], log-rates [b, K], log-shapes [b, K]}}. */
    private Var[] heads(double[][] x) {
        int b = x.length;
        int d = x[0].length;
        float[] xv = new float[b * d];
        for (int i = 0; i < b; i++) {
            for (int j = 0; j < d; j++) {
                xv[i * d + j] = (float) x[i][j];
            }
        }
        Var in = Var.of(Tensor.of(Shape.of(b, d), xv));
        Var hidden = in.matmul(net.get(0)).add(net.get(1)).tanh();
        Var o = hidden.matmul(net.get(2)).add(net.get(3));
        int k = components;
        Var logW = o.slice(1, 0, k).logSoftmax(1);
        Var eta = o.slice(1, k, 2 * k).tanh().mul(3f).add(net.get(4));
        Var kappa = o.slice(1, 2 * k, 3 * k).tanh().add(net.get(5));
        return new Var[] {logW, eta, kappa};
    }

    private static AftModels deepSurvivalMachines(double[][] x, double[] time, int[] event, int components,
            int hiddenUnits, int steps, int batch, float lr, float alpha, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int d = x[0].length;
        Weibull marginal = Weibull.fit(time, event);
        float[] eta0 = new float[components];
        float[] kappa0 = new float[components];
        for (int c = 0; c < components; c++) {
            eta0[c] = (float) (Math.log(marginal.lambda()) + 0.5 * (c - (components - 1) / 2.0));
            kappa0[c] = (float) Math.log(marginal.k());
        }
        List<Var> net = new ArrayList<>();
        net.add(param(d, hiddenUnits, 1 / Math.sqrt(d), r));
        net.add(Var.param(Tensor.zeros(Shape.of(1, hiddenUnits))));
        net.add(param(hiddenUnits, 3 * components, 0.1 / Math.sqrt(hiddenUnits), r));
        net.add(Var.param(Tensor.zeros(Shape.of(1, 3 * components))));
        net.add(Var.param(Tensor.of(Shape.of(1, components), eta0)));
        net.add(Var.param(Tensor.of(Shape.of(1, components), kappa0)));
        AftModels m = new AftModels(Kind.DEEP_SURVIVAL_MACHINES, null, Double.NaN, net, components);
        Optimizer opt = Optimizer.adam(lr);
        int bs = Math.min(batch, x.length);
        for (int s = 0; s < steps; s++) {
            double[][] bx = new double[bs][];
            float[] logT = new float[bs];
            float[] ev = new float[bs];
            float[] weight = new float[bs];
            for (int i = 0; i < bs; i++) {
                int j = r.nextInt(x.length);
                bx[i] = x[j];
                logT[i] = (float) Math.log(Math.max(time[j], 1e-12));
                ev[i] = event[j] != 0 ? 1 : 0;
                weight[i] = event[j] != 0 ? 1 : alpha;
            }
            Var[] h = m.heads(bx);
            Var lt = Var.of(Tensor.of(Shape.of(bs, 1), logT));
            Var e = Var.of(Tensor.of(Shape.of(bs, 1), ev));
            Var shape = h[2].exp();
            Var cum = h[1].add(shape.mul(lt)).exp();                      // lambda t^k
            Var logF = h[1].add(h[2]).add(shape.sub(1f).mul(lt)).sub(cum);
            Var logS = cum.neg();
            Var ll = e.mul(logF).add(e.neg().add(1f).mul(logS));          // [b, K]
            Var mix = h[0].add(ll).logsumexp(1, true);                    // [b, 1]
            Var w = Var.of(Tensor.of(Shape.of(bs, 1), weight));
            mix.mul(w).mean().neg().backward();
            opt.step(net);
        }
        return m;
    }

    // ------------------------------------------------------------------ fitting and prediction

    /**
     * Fits a parametric AFT model.
     *
     * @param kind {@link Kind#WEIBULL}, {@link Kind#LOG_NORMAL} or {@link Kind#LOG_LOGISTIC}
     * @param x covariates
     * @param time observed times
     * @param event event indicators (0 censored)
     * @param ridge the ridge on the slopes, for example 1e-3
     * @return the model
     */
    public static AftModels fit(Kind kind, double[][] x, double[] time, int[] event, double ridge) {
        if (kind == Kind.DEEP_SURVIVAL_MACHINES) {
            throw new IllegalArgumentException("use deepSurvivalMachines");
        }
        return parametric(kind, x, time, event, ridge);
    }

    /**
     * Fits Deep Survival Machines.
     *
     * @param x covariates
     * @param time observed times
     * @param event event indicators
     * @param components mixture components, for example 4
     * @param hiddenUnits hidden units
     * @param steps Adam steps
     * @param batch rows per step
     * @param lr the learning rate
     * @param alpha the weight of censored rows' terms (1 is the plain likelihood)
     * @param seed the seed
     * @return the model
     */
    public static AftModels deepSurvivalMachines(double[][] x, double[] time, int[] event, int components,
            int hiddenUnits, int steps, int batch, float lr, double alpha, long seed) {
        return deepSurvivalMachines(x, time, event, components, hiddenUnits, steps, batch, lr, (float) alpha, seed);
    }

    private double location(double[] x) {
        double mu = beta[0];
        for (int j = 0; j < x.length; j++) {
            mu += beta[j + 1] * x[j];
        }
        return mu;
    }

    /**
     * Survival curves.
     *
     * @param x covariates {@code [n][d]}
     * @param grid times
     * @return {@code S(grid[g] | x_i)}, {@code [n][grid]}
     */
    public double[][] survival(double[][] x, double[] grid) {
        double[][] out = new double[x.length][grid.length];
        if (kind != Kind.DEEP_SURVIVAL_MACHINES) {
            for (int i = 0; i < x.length; i++) {
                double mu = location(x[i]);
                for (int g = 0; g < grid.length; g++) {
                    out[i][g] = errorSurvival(kind, (Math.log(grid[g]) - mu) / sigma);
                }
            }
            return out;
        }
        float[][] h = mixture(x);
        for (int i = 0; i < x.length; i++) {
            for (int g = 0; g < grid.length; g++) {
                out[i][g] = mixtureSurvival(h, i, grid[g]);
            }
        }
        return out;
    }

    private float[][] mixture(double[][] x) {
        Var[] h = Var.noGrad(() -> heads(x));
        return new float[][] {h[0].value().toArray(), h[1].value().toArray(), h[2].value().toArray()};
    }

    private double mixtureSurvival(float[][] h, int i, double t) {
        double s = 0;
        double lt = Math.log(t);
        for (int c = 0; c < components; c++) {
            int idx = i * components + c;
            s += Math.exp(h[0][idx]) * Math.exp(-Math.exp(h[1][idx] + Math.exp(h[2][idx]) * lt));
        }
        return s;
    }

    /**
     * Median survival times: closed form for the parametric models, bisection on the mixture's
     * survival for Deep Survival Machines.
     *
     * @param x covariates
     * @return one median per row
     */
    public double[] median(double[][] x) {
        double[] out = new double[x.length];
        if (kind != Kind.DEEP_SURVIVAL_MACHINES) {
            for (int i = 0; i < x.length; i++) {
                out[i] = Math.exp(location(x[i]) + sigma * errorMedian(kind));
            }
            return out;
        }
        float[][] h = mixture(x);
        for (int i = 0; i < x.length; i++) {
            double lo = -20;
            double hi = 20;
            for (int it = 0; it < 50; it++) {
                double mid = (lo + hi) / 2;
                if (mixtureSurvival(h, i, Math.exp(mid)) > 0.5) {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            out[i] = Math.exp((lo + hi) / 2);
        }
        return out;
    }

    /**
     * A risk score for concordance: the negative log median time.
     *
     * @param x covariates
     * @return higher means earlier failure
     */
    public double[] risk(double[][] x) {
        double[] m = median(x);
        double[] out = new double[m.length];
        for (int i = 0; i < m.length; i++) {
            out[i] = -Math.log(m[i]);
        }
        return out;
    }
}
