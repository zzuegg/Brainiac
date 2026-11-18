package io.github.zzuegg.brainiac.experimental.survival;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Lbfgs;

import java.util.List;

/**
 * The mixture cure model (Farewell, Biometrics 1982; Kuk and Chen, Biometrika 1992; Sy and Taylor,
 * Biometrics 2000): a share of subjects never has the event. The population survival is
 * {@code S(t | x) = 1 - pi(x) + pi(x) S_u(t | x)}, with the incidence {@code pi(x)}, the probability
 * of being susceptible, a logistic model, and the latency {@code S_u} the survival of the
 * susceptible, either a Weibull proportional-hazards model ({@link Latency#WEIBULL}) or a
 * semiparametric Cox model with a Breslow baseline ({@link Latency#COX}).
 *
 * <p>It is fitted by EM: the E-step gives every censored subject the posterior probability of
 * being susceptible, {@code w = pi S_u / (1 - pi + pi S_u)} (subjects with an event have
 * {@code w = 1}); the M-step refits the logistic incidence on the soft labels {@code w} and the
 * latency with weights {@code w}, each by L-BFGS ({@link Lbfgs}). A standard survival model fitted
 * to such data sees a long plateau it can only explain by a tiny hazard, and misreads both the
 * incidence and the latency.
 */
public final class CureModel {

    /** The latency model of the susceptible. */
    public enum Latency {
        /** Weibull proportional hazards: {@code S_u = exp(-exp(b0 + beta x) t^k)}. */
        WEIBULL,
        /** Cox proportional hazards with a weighted Breslow baseline. */
        COX
    }

    private final Latency latency;
    private final double[] gamma;
    private final double[] beta;
    private final double logScale;
    private final double shape;
    private final double[] baseTimes;
    private final double[] baseHazard;

    private CureModel(Latency latency, double[] gamma, double[] beta, double logScale, double shape,
            double[] baseTimes, double[] baseHazard) {
        this.latency = latency;
        this.gamma = gamma;
        this.beta = beta;
        this.logScale = logScale;
        this.shape = shape;
        this.baseTimes = baseTimes;
        this.baseHazard = baseHazard;
    }

    /**
     * Fits by EM.
     *
     * @param x covariates {@code [n][d]} (used by both parts)
     * @param time observed times
     * @param event 1 for an event, 0 if censored
     * @param latency the latency model
     * @param iterations EM iterations, such as 30
     * @return the model
     */
    public static CureModel fit(double[][] x, double[] time, int[] event, Latency latency, int iterations) {
        int n = x.length;
        int d = x[0].length;
        float[] xv = new float[n * (d + 1)];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < d; j++) {
                xv[i * (d + 1) + j] = (float) x[i][j];
            }
            xv[i * (d + 1) + d] = 1;
        }
        Tensor xt = Tensor.of(Shape.of(n, d + 1), xv);
        float[] ev = new float[n];
        float[] logT = new float[n];
        for (int i = 0; i < n; i++) {
            ev[i] = event[i] != 0 ? 1 : 0;
            logT[i] = (float) Math.log(Math.max(1e-6, time[i]));
        }
        Tensor evt = Tensor.of(Shape.of(n, 1), ev);
        Tensor logTt = Tensor.of(Shape.of(n, 1), logT);
        // risk-set matrix R_ij = 1 if time_j >= time_i (for the Cox latency)
        float[] risk = new float[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                risk[i * n + j] = time[j] >= time[i] ? 1 : 0;
            }
        }
        Tensor rt = Tensor.of(Shape.of(n, n), risk);
        Var g = Var.param(Tensor.zeros(Shape.of(d + 1, 1)));
        Var b = Var.param(Tensor.zeros(Shape.of(d + 1, 1)));
        Var logK = Var.param(Tensor.zeros(Shape.of(1)));
        double[] w = new double[n];
        for (int i = 0; i < n; i++) {
            w[i] = event[i] != 0 ? 1 : 0.5;
        }
        CureModel current = null;
        Lbfgs lbfgs = new Lbfgs(10, 1e-6);
        for (int it = 0; it < iterations; it++) {
            Tensor wt = column(w);
            // incidence: logistic regression on soft labels w
            lbfgs.minimize(List.of(g), () -> {
                Var z = Var.of(xt).matmul(g);
                return z.softplus().sub(z.mul(wt)).mean().add(g.square().sum().mul(1e-4f));
            }, 30);
            if (latency == Latency.WEIBULL) {
                // weighted Weibull log-likelihood: w [delta (log h) - H], h = k t^(k-1) e^eta, H = t^k e^eta
                lbfgs.minimize(List.of(b, logK), () -> {
                    Var eta = Var.of(xt).matmul(b);
                    Var k = logK.exp();
                    Var cumHaz = Var.of(logTt).mul(k).add(eta).exp();
                    Var logHaz = logK.add(Var.of(logTt).mul(k.sub(1f))).add(eta);
                    return Var.of(evt).mul(logHaz).sub(cumHaz.mul(wt)).mean().neg()
                            .add(b.square().sum().mul(1e-4f));
                }, 30);
            } else {
                // weighted Cox partial likelihood: risk set sums carry the susceptibility weights
                Tensor wRow = Tensor.of(Shape.of(n, 1), toFloat(w));
                lbfgs.minimize(List.of(b), () -> {
                    Var eta = Var.of(xt).matmul(b);
                    Var denom = Var.of(rt).matmul(eta.exp().mul(wRow)).add(1e-12f).log();
                    return Var.of(evt).mul(eta.sub(denom)).sum().mul(-1f / n).add(b.square().sum().mul(1e-4f));
                }, 30);
            }
            current = build(latency, g, b, logK, x, time, event, w);
            for (int i = 0; i < n; i++) {
                if (event[i] == 0) {
                    double pi = current.incidence(x[i]);
                    double su = current.latencySurvival(x[i], time[i]);
                    w[i] = pi * su / Math.max(1e-12, 1 - pi + pi * su);
                }
            }
        }
        return current;
    }

    private static Tensor column(double[] w) {
        return Tensor.of(Shape.of(w.length, 1), toFloat(w));
    }

    private static float[] toFloat(double[] w) {
        float[] f = new float[w.length];
        for (int i = 0; i < w.length; i++) {
            f[i] = (float) w[i];
        }
        return f;
    }

    private static CureModel build(Latency latency, Var g, Var b, Var logK, double[][] x, double[] time,
            int[] event, double[] w) {
        float[] gv = g.value().toArray();
        float[] bv = b.value().toArray();
        int d = gv.length - 1;
        double[] gamma = new double[d + 1];
        double[] beta = new double[d];
        for (int j = 0; j <= d; j++) {
            gamma[j] = gv[j];
        }
        for (int j = 0; j < d; j++) {
            beta[j] = bv[j];
        }
        if (latency == Latency.WEIBULL) {
            return new CureModel(latency, gamma, beta, bv[d], Math.exp(logK.value().toArray()[0]), null, null);
        }
        // weighted Breslow baseline cumulative hazard of the susceptible
        int n = time.length;
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        java.util.Arrays.sort(order, (p, q) -> Double.compare(time[p], time[q]));
        double[] r = new double[n];
        for (int i = 0; i < n; i++) {
            double s = 0;
            for (int j = 0; j < d; j++) {
                s += beta[j] * x[i][j];
            }
            r[i] = Math.exp(s);
        }
        double[] times = new double[n];
        double[] hazard = new double[n];
        double cum = 0;
        for (int k = 0; k < n; k++) {
            int i = order[k];
            if (event[i] != 0) {
                double denom = 0;
                for (int j = 0; j < n; j++) {
                    if (time[j] >= time[i]) {
                        denom += w[j] * r[j];
                    }
                }
                cum += 1 / Math.max(1e-12, denom);
            }
            times[k] = time[i];
            hazard[k] = cum;
        }
        return new CureModel(latency, gamma, beta, 0, 1, times, hazard);
    }

    /**
     * @param x covariates
     * @return {@code pi(x)}, the probability of being susceptible
     */
    public double incidence(double[] x) {
        double s = gamma[gamma.length - 1];
        for (int j = 0; j < x.length; j++) {
            s += gamma[j] * x[j];
        }
        return 1 / (1 + Math.exp(-s));
    }

    /**
     * @param x covariates
     * @param t a time
     * @return the latency survival {@code S_u(t | x)} of a susceptible subject
     */
    public double latencySurvival(double[] x, double t) {
        double eta = 0;
        for (int j = 0; j < beta.length; j++) {
            eta += beta[j] * x[j];
        }
        if (latency == Latency.WEIBULL) {
            return Math.exp(-Math.exp(logScale + eta) * Math.pow(Math.max(0, t), shape));
        }
        int k = java.util.Arrays.binarySearch(baseTimes, t);
        int idx = k >= 0 ? k : -k - 2;
        while (k >= 0 && idx + 1 < baseTimes.length && baseTimes[idx + 1] == t) {
            idx++;
        }
        double h = idx < 0 ? 0 : baseHazard[idx];
        return Math.exp(-h * Math.exp(eta));
    }

    /**
     * @param x covariates
     * @param t a time
     * @return the population survival {@code 1 - pi + pi S_u}
     */
    public double survival(double[] x, double t) {
        double pi = incidence(x);
        return 1 - pi + pi * latencySurvival(x, t);
    }

    /**
     * The estimated cured fraction of a sample.
     *
     * @param x covariates
     * @return {@code mean (1 - pi(x))}
     */
    public double curedFraction(double[][] x) {
        double s = 0;
        for (double[] r : x) {
            s += 1 - incidence(r);
        }
        return s / x.length;
    }

    /** @return the incidence coefficients, intercept last */
    public double[] incidenceCoefficients() {
        return gamma.clone();
    }

    /** @return the latency log-hazard coefficients */
    public double[] latencyCoefficients() {
        return beta.clone();
    }
}
