package io.github.zzuegg.brainiac.experimental.markets;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.discovery.Distributions;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Arrays;
import java.util.SplittableRandom;
import java.util.function.ToDoubleFunction;

/**
 * Conditional variance models for daily returns, fitted by maximum likelihood.
 *
 * <ul>
 *   <li>{@link Kind#EWMA}: RiskMetrics {@code s2_t = lambda s2_(t-1) + (1 - lambda) r2_(t-1)}
 *       with {@code lambda = 0.94} (not fitted).</li>
 *   <li>{@link Kind#GARCH}: {@code s2_t = omega + alpha r2_(t-1) + beta s2_(t-1)} (Bollerslev
 *       1986).</li>
 *   <li>{@link Kind#GJR_GARCH}: adds {@code gamma r2_(t-1)} after negative returns (Glosten,
 *       Jagannathan and Runkle 1993).</li>
 *   <li>{@link Kind#EGARCH}: {@code ln s2_t = omega + alpha (|z| - E|z|) + gamma z + beta ln s2_(t-1)}
 *       (Nelson 1991).</li>
 * </ul>
 *
 * <p>Innovations are Gaussian or unit-variance Student-t (its log-density from
 * {@link Distributions#logGamma}); the likelihood is maximised by Nelder-Mead with invalid
 * parameters rejected. {@link #forecast} gives the {@code h}-day-ahead variance by the models'
 * mean-reversion recursions, {@link #valueAtRisk} the one-day quantile.
 * {@link #neuralForecaster} is the learned baseline: an MLP on the last 20 squared returns that
 * outputs log variances at several horizons, trained by Gaussian likelihood.
 */
public final class Volatility {

    /** The variance recursion. */
    public enum Kind {
        /** exponentially weighted moving average, lambda 0.94. */
        EWMA,
        /** GARCH(1,1). */
        GARCH,
        /** GARCH with the leverage term. */
        GJR_GARCH,
        /** exponential GARCH. */
        EGARCH
    }

    /** The innovation distribution. */
    public enum Innovation {
        /** standard normal. */
        GAUSSIAN,
        /** unit-variance Student-t with fitted degrees of freedom. */
        STUDENT_T
    }

    private final Kind kind;
    private final Innovation innovation;
    private final double[] params;
    private final double logLikelihood;

    private Volatility(Kind kind, Innovation innovation, double[] params, double logLikelihood) {
        this.kind = kind;
        this.innovation = innovation;
        this.params = params;
        this.logLikelihood = logLikelihood;
    }

    /** @return the recursion */
    public Kind kind() {
        return kind;
    }

    /** @return the innovation law */
    public Innovation innovation() {
        return innovation;
    }

    /**
     * The fitted parameters: {@code omega, alpha, gamma, beta} (gamma 0 for GARCH, lambda first
     * for EWMA), then {@code nu} for t innovations.
     *
     * @return a copy of the parameters
     */
    public double[] parameters() {
        return params.clone();
    }

    /** @return the maximised log-likelihood */
    public double logLikelihood() {
        return logLikelihood;
    }

    /**
     * Fits a model.
     *
     * @param kind the recursion
     * @param innovation the innovation law
     * @param returns the training returns
     * @return the fitted model
     */
    public static Volatility fit(Kind kind, Innovation innovation, double[] returns) {
        double var = 0;
        for (double r : returns) {
            var += r * r / returns.length;
        }
        boolean t = innovation == Innovation.STUDENT_T;
        if (kind == Kind.EWMA) {
            double[] p = t ? new double[] {0.94, 0, 0, 0, 8} : new double[] {0.94, 0, 0, 0};
            if (t) {
                double[] nu = nelderMead(x -> -logLik(kind, new double[] {0.94, 0, 0, 0, x[0]}, returns),
                        new double[] {8}, new double[] {2}, 100);
                p[4] = nu[0];
            }
            return new Volatility(kind, innovation, p, logLik(kind, p, returns));
        }
        double[] start = kind == Kind.EGARCH
                ? new double[] {0.03 * Math.log(var), 0.1, -0.05, 0.97}
                : new double[] {0.05 * var, 0.05, kind == Kind.GJR_GARCH ? 0.05 : 0, 0.9};
        double[] step = kind == Kind.EGARCH
                ? new double[] {0.05 * Math.abs(Math.log(var)), 0.05, 0.05, 0.02}
                : new double[] {0.03 * var, 0.03, 0.03, 0.03};
        // optimise omega on a log scale for the GARCH family
        double[] x0 = kind == Kind.EGARCH ? start.clone() : new double[] {Math.log(start[0]), start[1], start[2],
            start[3]};
        if (kind != Kind.EGARCH) {
            step[0] = 0.5;
        }
        boolean gamma = kind != Kind.GARCH;
        int dims = (gamma ? 4 : 3) + (t ? 1 : 0);
        double[] packed = new double[dims];
        double[] packedStep = new double[dims];
        int k = 0;
        for (int i = 0; i < 4; i++) {
            if (i == 2 && !gamma) {
                continue;
            }
            packed[k] = x0[i];
            packedStep[k++] = step[i];
        }
        if (t) {
            packed[k] = 8;
            packedStep[k] = 2;
        }
        ToDoubleFunction<double[]> objective = x -> -logLik(kind, unpack(kind, x, gamma, t), returns);
        double[] best = nelderMead(objective, packed, packedStep, 400);
        best = nelderMead(objective, best, scale(packedStep, 0.2), 200);
        double[] p = unpack(kind, best, gamma, t);
        return new Volatility(kind, innovation, p, logLik(kind, p, returns));
    }

    private static double[] scale(double[] v, double s) {
        double[] out = v.clone();
        for (int i = 0; i < out.length; i++) {
            out[i] *= s;
        }
        return out;
    }

    private static double[] unpack(Kind kind, double[] x, boolean gamma, boolean t) {
        double[] p = new double[t ? 5 : 4];
        int k = 0;
        p[0] = kind == Kind.EGARCH ? x[k++] : Math.exp(x[k++]);
        p[1] = x[k++];
        p[2] = gamma ? x[k++] : 0;
        p[3] = x[k++];
        if (t) {
            p[4] = x[k];
        }
        return p;
    }

    private static boolean valid(Kind kind, double[] p) {
        if (p.length == 5 && (p[4] <= 2.05 || p[4] > 200)) {
            return false;
        }
        return switch (kind) {
            case EWMA -> true;
            case GARCH, GJR_GARCH -> p[1] >= 0 && p[3] >= 0 && p[1] + p[2] >= 0 && p[1] + p[2] / 2 + p[3] < 0.9999;
            case EGARCH -> Math.abs(p[3]) < 0.9999;
        };
    }

    private static double expectedAbs(double[] p) {
        if (p.length < 5) {
            return Math.sqrt(2 / Math.PI);
        }
        double nu = p[4];
        return Math.sqrt(nu - 2) * Math.exp(Distributions.logGamma((nu - 1) / 2) - Distributions.logGamma(nu / 2))
                / Math.sqrt(Math.PI);
    }

    /**
     * The one-step-ahead variances {@code s2_t} for {@code t = 0..T} (entry {@code T} is the
     * forecast for the day after the data).
     */
    private static double[] filter(Kind kind, double[] p, double[] r) {
        double[] s2 = new double[r.length + 1];
        double init = 0;
        int warm = Math.min(r.length, 20);
        for (int i = 0; i < warm; i++) {
            init += r[i] * r[i] / Math.max(warm, 1);
        }
        s2[0] = Math.max(init, 1e-10);
        double eAbs = kind == Kind.EGARCH ? expectedAbs(p) : 0;
        for (int i = 0; i < r.length; i++) {
            double prev = s2[i];
            s2[i + 1] = switch (kind) {
                case EWMA -> p[0] * prev + (1 - p[0]) * r[i] * r[i];
                case GARCH, GJR_GARCH -> p[0] + (p[1] + (r[i] < 0 ? p[2] : 0)) * r[i] * r[i] + p[3] * prev;
                case EGARCH -> {
                    double z = r[i] / Math.sqrt(prev);
                    double ln = p[0] + p[1] * (Math.abs(z) - eAbs) + p[2] * z + p[3] * Math.log(prev);
                    yield Math.exp(Math.max(-50, Math.min(ln, 5)));
                }
            };
            s2[i + 1] = Math.max(s2[i + 1], 1e-12);
        }
        return s2;
    }

    private static double logLik(Kind kind, double[] p, double[] r) {
        if (!valid(kind, p)) {
            return -1e30;
        }
        double[] s2 = filter(kind, p, r);
        double ll = 0;
        boolean t = p.length == 5;
        double nu = t ? p[4] : 0;
        double c = t ? Distributions.logGamma((nu + 1) / 2) - Distributions.logGamma(nu / 2)
                - 0.5 * Math.log(Math.PI * (nu - 2)) : -0.5 * Math.log(2 * Math.PI);
        for (int i = 0; i < r.length; i++) {
            double e = r[i] * r[i] / s2[i];
            ll += t ? c - 0.5 * Math.log(s2[i]) - (nu + 1) / 2 * Math.log(1 + e / (nu - 2))
                    : c - 0.5 * Math.log(s2[i]) - 0.5 * e;
        }
        return Double.isFinite(ll) ? ll : -1e30;
    }

    /**
     * Minimises a function by Nelder-Mead.
     *
     * @param f the objective
     * @param x0 the start
     * @param step the initial simplex size per coordinate
     * @param iterations the iteration cap
     * @return the best point found
     */
    public static double[] nelderMead(ToDoubleFunction<double[]> f, double[] x0, double[] step, int iterations) {
        int d = x0.length;
        double[][] simplex = new double[d + 1][];
        double[] values = new double[d + 1];
        simplex[0] = x0.clone();
        for (int i = 0; i < d; i++) {
            simplex[i + 1] = x0.clone();
            simplex[i + 1][i] += step[i];
        }
        for (int i = 0; i <= d; i++) {
            values[i] = f.applyAsDouble(simplex[i]);
        }
        Integer[] order = new Integer[d + 1];
        for (int it = 0; it < iterations; it++) {
            for (int i = 0; i <= d; i++) {
                order[i] = i;
            }
            final double[] vv = values;
            Arrays.sort(order, (a, b) -> Double.compare(vv[a], vv[b]));
            double[][] s = new double[d + 1][];
            double[] v = new double[d + 1];
            for (int i = 0; i <= d; i++) {
                s[i] = simplex[order[i]];
                v[i] = values[order[i]];
            }
            simplex = s;
            values = v;
            double[] centroid = new double[d];
            for (int i = 0; i < d; i++) {
                for (int j = 0; j < d; j++) {
                    centroid[j] += simplex[i][j] / d;
                }
            }
            double[] reflected = blend(centroid, simplex[d], -1);
            double fr = f.applyAsDouble(reflected);
            if (fr < values[0]) {
                double[] expanded = blend(centroid, simplex[d], -2);
                double fe = f.applyAsDouble(expanded);
                simplex[d] = fe < fr ? expanded : reflected;
                values[d] = Math.min(fe, fr);
            } else if (fr < values[d - 1]) {
                simplex[d] = reflected;
                values[d] = fr;
            } else {
                double[] contracted = blend(centroid, simplex[d], 0.5);
                double fc = f.applyAsDouble(contracted);
                if (fc < values[d]) {
                    simplex[d] = contracted;
                    values[d] = fc;
                } else {
                    for (int i = 1; i <= d; i++) {
                        simplex[i] = blend(simplex[0], simplex[i], 0.5);
                        values[i] = f.applyAsDouble(simplex[i]);
                    }
                }
            }
        }
        int best = 0;
        for (int i = 1; i <= d; i++) {
            if (values[i] < values[best]) {
                best = i;
            }
        }
        return simplex[best];
    }

    private static double[] blend(double[] c, double[] x, double t) {
        double[] out = new double[c.length];
        for (int i = 0; i < c.length; i++) {
            out[i] = c[i] + t * (x[i] - c[i]);
        }
        return out;
    }

    /**
     * The conditional variances given the returns before each day.
     *
     * @param returns the returns
     * @return {@code returns.length + 1} one-step variances
     */
    public double[] variances(double[] returns) {
        return filter(kind, params, returns);
    }

    /**
     * Variance forecasts for day {@code t + h} made with the returns up to day {@code t}, for
     * every origin.
     *
     * @param returns the returns
     * @param horizon {@code h}, at least 1
     * @return {@code f[t]} forecasting the variance of {@code returns[t + h - 1]} from
     *     {@code returns[0..t-1]}, for {@code t = 0..T}
     */
    public double[] forecast(double[] returns, int horizon) {
        double[] s2 = variances(returns);
        double[] out = new double[s2.length];
        for (int t = 0; t < s2.length; t++) {
            out[t] = ahead(s2[t], horizon);
        }
        return out;
    }

    private double ahead(double oneStep, int h) {
        if (h == 1 || kind == Kind.EWMA) {
            return oneStep;
        }
        if (kind == Kind.EGARCH) {
            double mean = params[0] / (1 - params[3]);
            return Math.exp(mean + Math.pow(params[3], h - 1) * (Math.log(oneStep) - mean));
        }
        double persistence = params[1] + params[2] / 2 + params[3];
        double longRun = params[0] / (1 - persistence);
        return longRun + Math.pow(persistence, h - 1) * (oneStep - longRun);
    }

    /**
     * The one-day value at risk as a positive loss: {@code -q_(1-level) sqrt(s2)}.
     *
     * @param variance the forecast variance
     * @param level such as 0.99
     * @return the VaR
     */
    public double valueAtRisk(double variance, double level) {
        return quantile(params.length == 5 ? params[4] : Double.POSITIVE_INFINITY, level) * Math.sqrt(variance);
    }

    /**
     * The upper {@code level} quantile of a unit-variance Student-t (Gaussian for infinite
     * {@code nu}), by bisection on {@link Distributions#studentTwoSided}.
     *
     * @param nu degrees of freedom
     * @param level such as 0.99
     * @return the quantile
     */
    public static double quantile(double nu, double level) {
        double df = Double.isInfinite(nu) ? 1e7 : nu;
        double lo = 0;
        double hi = 100;
        for (int i = 0; i < 100; i++) {
            double mid = (lo + hi) / 2;
            if (Distributions.studentTwoSided(mid, df) / 2 > 1 - level) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        return (lo + hi) / 2 * (Double.isInfinite(nu) ? 1 : Math.sqrt((nu - 2) / nu));
    }

    /**
     * The QLIKE loss {@code truth / f - ln(truth / f) - 1}, averaged.
     *
     * @param forecast forecasts
     * @param truth true (or proxy) variances
     * @return the mean loss
     */
    public static double qlike(double[] forecast, double[] truth) {
        double s = 0;
        for (int i = 0; i < truth.length; i++) {
            double q = truth[i] / Math.max(forecast[i], 1e-14);
            s += q - Math.log(q) - 1;
        }
        return s / truth.length;
    }

    /**
     * Rolling historical variance forecasts.
     *
     * @param returns the returns
     * @param window the window
     * @return {@code returns.length + 1} forecasts (each the mean square of the last window)
     */
    public static double[] rolling(double[] returns, int window) {
        double[] out = new double[returns.length + 1];
        double sum = 0;
        for (int t = 0; t <= returns.length; t++) {
            int n = Math.min(t, window);
            out[t] = n == 0 ? returns[0] * returns[0] + 1e-8 : sum / n;
            if (t < returns.length) {
                sum += returns[t] * returns[t];
                if (t >= window) {
                    sum -= returns[t - window] * returns[t - window];
                }
            }
        }
        return out;
    }

    /**
     * The neural baseline: an MLP mapping the last {@code lags} squared returns (scaled by the
     * training variance) to log variances at each horizon, trained by Gaussian likelihood on
     * the realised returns with early stopping on the last fifth of the training days.
     *
     * @param returns the returns (the first {@code trainDays} are used for training)
     * @param trainDays training days
     * @param horizons the horizons
     * @param lags input lags
     * @param steps Adam steps
     * @param seed the seed
     * @return {@code [horizon][t]} forecasts aligned as {@link #forecast} for {@code t >= lags}
     *     (earlier origins get the training variance)
     */
    public static double[][] neuralForecaster(double[] returns, int trainDays, int[] horizons, int lags, int steps,
            long seed) {
        double var = 0;
        for (int t = 0; t < trainDays; t++) {
            var += returns[t] * returns[t] / trainDays;
        }
        int maxH = Arrays.stream(horizons).max().orElse(1);
        SplittableRandom rnd = new SplittableRandom(seed);
        Mlp net = new Mlp(new int[] {lags, 32, horizons.length}, rnd);
        Optimizer opt = Optimizer.adam(1e-2f);
        int samples = trainDays - maxH - lags + 1;
        int valid = samples / 5;
        int fit = samples - valid;
        float[] x = new float[samples * lags];
        float[] y = new float[samples * horizons.length];
        for (int s = 0; s < samples; s++) {
            int t = s + lags;
            fill(x, s * lags, returns, t, lags, var);
            for (int h = 0; h < horizons.length; h++) {
                double r = returns[t + horizons[h] - 1];
                y[s * horizons.length + h] = (float) (r * r / var);
            }
        }
        int hs = horizons.length;
        Tensor xt = Tensor.of(Shape.of(fit, lags), Arrays.copyOf(x, fit * lags));
        Tensor yt = Tensor.of(Shape.of(fit, hs), Arrays.copyOf(y, fit * hs));
        Tensor xv = Tensor.of(Shape.of(valid, lags), Arrays.copyOfRange(x, fit * lags, samples * lags));
        Tensor yv = Tensor.of(Shape.of(valid, hs), Arrays.copyOfRange(y, fit * hs, samples * hs));
        // the output is a bounded log variance; training stops early on the held-out last fifth
        double bestLoss = Double.POSITIVE_INFINITY;
        float[][] best = null;
        for (int step = 0; step < steps; step++) {
            Var logV = net.forward(Var.of(xt)).tanh().mul(4f);
            Var loss = logV.add(Var.of(yt).div(logV.exp())).mean();
            loss.backward();
            opt.step(net.parameters());
            if (step % 10 == 9 || step == steps - 1) {
                double v = Var.noGrad(() -> {
                    Var lv = net.forward(Var.of(xv)).tanh().mul(4f);
                    return lv.add(Var.of(yv).div(lv.exp())).mean();
                }).value().item();
                if (v < bestLoss) {
                    bestLoss = v;
                    best = new float[net.parameters().size()][];
                    for (int i = 0; i < best.length; i++) {
                        best[i] = net.parameters().get(i).value().toArray();
                    }
                }
            }
        }
        if (best != null) {
            for (int i = 0; i < best.length; i++) {
                Var p = net.parameters().get(i);
                p.assign(Tensor.of(p.shape(), best[i]));
            }
        }
        double[][] out = new double[horizons.length][returns.length + 1];
        float[] all = new float[(returns.length + 1) * lags];
        for (int t = lags; t <= returns.length; t++) {
            fill(all, t * lags, returns, t, lags, var);
        }
        float[] pred = Var.noGrad(() -> net.forward(Var.of(Tensor.of(Shape.of(returns.length + 1, lags), all))).tanh()
                .mul(4f)).value().toArray();
        for (int h = 0; h < horizons.length; h++) {
            for (int t = 0; t <= returns.length; t++) {
                out[h][t] = t < lags ? var : var * Math.exp(pred[t * horizons.length + h]);
            }
        }
        return out;
    }

    private static void fill(float[] x, int offset, double[] returns, int t, int lags, double var) {
        for (int l = 0; l < lags; l++) {
            double r = returns[t - 1 - l];
            x[offset + l] = (float) Math.min(r * r / var, 20);
        }
    }
}
