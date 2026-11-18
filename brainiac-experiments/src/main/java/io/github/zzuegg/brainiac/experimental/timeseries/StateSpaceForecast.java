package io.github.zzuegg.brainiac.experimental.timeseries;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ml.MlMatrices;
import io.github.zzuegg.brainiac.experimental.ml.StateSpaceFilter;
import io.github.zzuegg.brainiac.experimental.train.Lbfgs;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.function.Function;

/**
 * A Gaussian forecast with exact prediction intervals, and the shared machinery of the
 * linear-Gaussian state-space forecasters ({@link ExponentialSmoothing}, {@link Arima}): the
 * exact likelihood from {@link StateSpaceFilter#kalman}, maximised over a few unconstrained
 * parameters by {@link Lbfgs#fullBatch} (forward finite-difference gradients, since the
 * parameter count is tiny), the small-sample corrected AIC, and the {@code h}-step predictive
 * distribution by propagating the filtered state.
 *
 * @param mean the forecast means, one per step ahead
 * @param variance the forecast variances
 */
public record StateSpaceForecast(double[] mean, double[] variance) {

    /**
     * A central prediction interval.
     *
     * @param coverage the nominal coverage, for example 0.8
     * @return {@code [lower, upper]}, each one per step ahead
     */
    public double[][] interval(double coverage) {
        double z = normalQuantile(0.5 + coverage / 2);
        double[] lo = new double[mean.length];
        double[] hi = new double[mean.length];
        for (int h = 0; h < mean.length; h++) {
            double s = Math.sqrt(Math.max(0, variance[h]));
            lo[h] = mean[h] - z * s;
            hi[h] = mean[h] + z * s;
        }
        return new double[][] {lo, hi};
    }

    /**
     * The standard normal quantile, by Acklam's rational approximation (relative error about
     * 1e-9).
     *
     * @param p a probability in {@code (0, 1)}
     * @return {@code Phi^-1(p)}
     */
    public static double normalQuantile(double p) {
        double[] a = {-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02,
            -3.066479806614716e+01, 2.506628277459239e+00};
        double[] b = {-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01,
            -1.328068155288572e+01};
        double[] c = {-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00,
            4.374664141464968e+00, 2.938163982698783e+00};
        double[] d = {7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00};
        double pl = 0.02425;
        if (p < pl) {
            double q = Math.sqrt(-2 * Math.log(p));
            return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                    / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
        }
        if (p > 1 - pl) {
            return -normalQuantile(1 - p);
        }
        double q = p - 0.5;
        double r = q * q;
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
                / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
    }

    /**
     * Maximises a state-space log-likelihood with full-batch L-BFGS.
     *
     * @param logLikelihood the log-likelihood of unconstrained parameters
     * @param initial the starting parameters
     * @param iterations L-BFGS iterations (each runs a strong-Wolfe line search)
     * @return the maximising parameters
     */
    static double[] maximise(Function<double[], Double> logLikelihood, double[] initial, int iterations) {
        Var theta = Var.param(Tensor.of(Shape.of(initial.length), toFloat(initial)));
        List<Var> params = List.of(theta);
        Lbfgs opt = Lbfgs.fullBatch(1f, 8);
        Function<double[], Double> nll = x -> {
            double v = -logLikelihood.apply(x);
            return Double.isFinite(v) ? v : 1e12;
        };
        double[] best = initial.clone();
        double[] bestValue = {nll.apply(best)};
        io.github.zzuegg.brainiac.train.Optimizer.Reevaluation reevaluate = () -> {
            double[] x = toDouble(theta.value().toArray());
            double f = nll.apply(x);
            if (f < bestValue[0]) {
                bestValue[0] = f;
                System.arraycopy(x, 0, best, 0, x.length);
            }
            float[] g = new float[x.length];
            for (int i = 0; i < x.length; i++) {
                double e = 1e-4 * Math.max(1, Math.abs(x[i]));
                double[] up = x.clone();
                up[i] += e;
                g[i] = (float) ((nll.apply(up) - f) / e);
            }
            theta.setGrad(Tensor.of(Shape.of(x.length), g));
            return (float) f;
        };
        for (int it = 0; it < iterations; it++) {
            reevaluate.lossAndGradients();
            double before = bestValue[0];
            opt.step(params, reevaluate);
            if (before - bestValue[0] < 1e-6 * Math.max(1, Math.abs(before)) && it > 2) {
                break;
            }
        }
        return best;
    }

    /**
     * The small-sample corrected Akaike information criterion.
     *
     * @param logLikelihood the maximised log-likelihood
     * @param parameters the fitted parameter count
     * @param n the observations
     * @return {@code AICc}, lower is better
     */
    public static double aicc(double logLikelihood, int parameters, int n) {
        double k = parameters;
        return -2 * logLikelihood + 2 * k + 2 * k * (k + 1) / Math.max(1, n - k - 1);
    }

    /**
     * The predictive distribution {@code horizon} steps past the last filtered state.
     *
     * @param model the model
     * @param mean the filtered state mean at the last observation
     * @param cov the filtered state covariance
     * @param horizon steps ahead
     * @return the forecast
     */
    static StateSpaceForecast propagate(StateSpaceFilter.LinearModel model, double[] mean, double[][] cov,
            int horizon) {
        double[] m = mean.clone();
        double[][] p = cov;
        double[] out = new double[horizon];
        double[] var = new double[horizon];
        double[][] ft = MlMatrices.transpose(model.f());
        double[][] ht = MlMatrices.transpose(model.h());
        for (int h = 0; h < horizon; h++) {
            m = mv(model.f(), m);
            p = add(MlMatrices.multiply(MlMatrices.multiply(model.f(), p), ft), model.q());
            out[h] = mv(model.h(), m)[0];
            var[h] = MlMatrices.multiply(MlMatrices.multiply(model.h(), p), ht)[0][0] + model.r()[0][0];
        }
        return new StateSpaceForecast(out, var);
    }

    static double[] mv(double[][] a, double[] x) {
        double[] y = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            double s = 0;
            for (int j = 0; j < x.length; j++) {
                s += a[i][j] * x[j];
            }
            y[i] = s;
        }
        return y;
    }

    static double[][] add(double[][] a, double[][] b) {
        double[][] c = new double[a.length][a[0].length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < a[0].length; j++) {
                c[i][j] = a[i][j] + b[i][j];
            }
        }
        return c;
    }

    static double[][] column(double[] y) {
        double[][] ys = new double[y.length][1];
        for (int t = 0; t < y.length; t++) {
            ys[t][0] = y[t];
        }
        return ys;
    }

    static float[] toFloat(double[] x) {
        float[] f = new float[x.length];
        for (int i = 0; i < x.length; i++) {
            f[i] = (float) x[i];
        }
        return f;
    }

    static double[] toDouble(float[] x) {
        double[] d = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            d[i] = x[i];
        }
        return d;
    }

    static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    static double logit(double p) {
        return Math.log(p / (1 - p));
    }
}
