package io.github.zzuegg.brainiac.experimental.survival;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * Classical and neural survival models for right-censored data, where for censored units we only
 * know that the event had not happened by their observed time.
 * <ul>
 *   <li>{@link KaplanMeier} (Kaplan and Meier 1958): the nonparametric survival curve
 *       {@code S(t) = prod over event times t_i <= t of (1 - d_i / n_i)}.</li>
 *   <li>{@link Cox} (Cox 1972): proportional hazards {@code h(t | x) = h0(t) exp(beta . x)}, fitted
 *       by Newton's method on the partial likelihood (ties by Breslow's approximation), which
 *       cancels the unknown baseline hazard; {@link Breslow} then recovers the baseline cumulative
 *       hazard, giving full survival curves.</li>
 *   <li>{@link #coxLoss} and {@link DeepSurv} (Katzman et al. 2018,
 *       <a href="https://arxiv.org/abs/1606.00931">arXiv:1606.00931</a>): the negative log partial
 *       likelihood as an autograd loss over any network's log-risk, and an MLP trained with it.</li>
 * </ul>
 */
public final class SurvivalModels {

    private SurvivalModels() {
    }

    /** A survival curve as a right-continuous step function. */
    public static final class KaplanMeier {
        private final double[] times;
        private final double[] survival;

        /**
         * Estimates the curve.
         *
         * @param time observed times
         * @param event 0 if censored, otherwise an event (any cause)
         */
        public KaplanMeier(double[] time, int[] event) {
            int n = time.length;
            Integer[] order = new Integer[n];
            for (int i = 0; i < n; i++) {
                order[i] = i;
            }
            Arrays.sort(order, (a, b) -> Double.compare(time[a], time[b]));
            double[] ts = new double[n];
            double[] ss = new double[n];
            int k = 0;
            double s = 1;
            int atRisk = n;
            int i = 0;
            while (i < n) {
                double t = time[order[i]];
                int deaths = 0;
                int leaving = 0;
                while (i < n && time[order[i]] == t) {
                    deaths += event[order[i]] != 0 ? 1 : 0;
                    leaving++;
                    i++;
                }
                if (deaths > 0) {
                    s *= 1 - deaths / (double) atRisk;
                    ts[k] = t;
                    ss[k++] = s;
                }
                atRisk -= leaving;
            }
            this.times = Arrays.copyOf(ts, k);
            this.survival = Arrays.copyOf(ss, k);
        }

        /**
         * @param t a time
         * @return {@code S(t)}
         */
        public double survival(double t) {
            int idx = Arrays.binarySearch(times, t);
            int last = idx >= 0 ? idx : -idx - 2;
            return last < 0 ? 1 : survival[last];
        }
    }

    /** Breslow's baseline cumulative hazard for given log-risks. */
    public static final class Breslow {
        private final double[] times;
        private final double[] cumulative;

        /**
         * Estimates {@code H0(t) = sum over events t_i <= t of 1 / sum_{t_j >= t_i} exp(eta_j)}.
         *
         * @param logRisk {@code eta} per training unit
         * @param time observed times
         * @param event 0 if censored
         */
        public Breslow(double[] logRisk, double[] time, int[] event) {
            int n = time.length;
            Integer[] order = new Integer[n];
            for (int i = 0; i < n; i++) {
                order[i] = i;
            }
            Arrays.sort(order, (a, b) -> Double.compare(time[b], time[a]));
            double[] riskSum = new double[n];
            double acc = 0;
            for (int k = 0; k < n; k++) {
                acc += Math.exp(logRisk[order[k]]);
                riskSum[order[k]] = acc;
            }
            // units with equal times share the risk set of the earliest of them (the largest sum)
            for (int k = n - 2; k >= 0; k--) {
                if (time[order[k]] == time[order[k + 1]]) {
                    riskSum[order[k]] = riskSum[order[k + 1]];
                }
            }
            Integer[] asc = order.clone();
            Arrays.sort(asc, (a, b) -> Double.compare(time[a], time[b]));
            double[] ts = new double[n];
            double[] hs = new double[n];
            int m = 0;
            double h = 0;
            for (int idx : asc) {
                if (event[idx] != 0) {
                    h += 1 / riskSum[idx];
                    ts[m] = time[idx];
                    hs[m++] = h;
                }
            }
            this.times = Arrays.copyOf(ts, m);
            this.cumulative = Arrays.copyOf(hs, m);
        }

        /**
         * @param t a time
         * @return {@code H0(t)}
         */
        public double cumulativeHazard(double t) {
            int idx = Arrays.binarySearch(times, t);
            int last = idx >= 0 ? idx : -idx - 2;
            return last < 0 ? 0 : cumulative[Math.min(last, cumulative.length - 1)];
        }

        /**
         * @param logRisk {@code eta(x)}
         * @param t a time
         * @return {@code S(t | x) = exp(-H0(t) exp(eta))}
         */
        public double survival(double logRisk, double t) {
            return Math.exp(-cumulativeHazard(t) * Math.exp(logRisk));
        }
    }

    /** Linear Cox proportional hazards. */
    public static final class Cox {
        private final double[] beta;
        private final Breslow baseline;

        /**
         * Fits {@code beta} by Newton's method on the Breslow partial likelihood with a ridge.
         *
         * @param x covariates {@code [n][d]}
         * @param time observed times
         * @param event 0 if censored (any non-zero value counts as the event; pass a cause mask for
         *     a cause-specific model)
         * @param l2 the ridge penalty
         */
        public Cox(double[][] x, double[] time, int[] event, double l2) {
            int n = x.length;
            int d = x[0].length;
            Integer[] order = new Integer[n];
            for (int i = 0; i < n; i++) {
                order[i] = i;
            }
            Arrays.sort(order, (a, b) -> Double.compare(time[b], time[a]));
            double[] b = new double[d];
            for (int it = 0; it < 30; it++) {
                double[] grad = new double[d];
                double[][] hess = new double[d][d];
                double s0 = 0;
                double[] s1 = new double[d];
                double[][] s2 = new double[d][d];
                int k = 0;
                while (k < n) {
                    double t = time[order[k]];
                    int start = k;
                    while (k < n && time[order[k]] == t) {
                        int i = order[k];
                        double w = Math.exp(Math.clamp(dot(b, x[i]), -30, 30));
                        s0 += w;
                        for (int p = 0; p < d; p++) {
                            s1[p] += w * x[i][p];
                            for (int q = 0; q < d; q++) {
                                s2[p][q] += w * x[i][p] * x[i][q];
                            }
                        }
                        k++;
                    }
                    for (int j = start; j < k; j++) {
                        int i = order[j];
                        if (event[i] == 0) {
                            continue;
                        }
                        for (int p = 0; p < d; p++) {
                            grad[p] += x[i][p] - s1[p] / s0;
                            for (int q = 0; q < d; q++) {
                                hess[p][q] += s2[p][q] / s0 - s1[p] * s1[q] / (s0 * s0);
                            }
                        }
                    }
                }
                for (int p = 0; p < d; p++) {
                    grad[p] -= l2 * b[p];
                    hess[p][p] += l2 + 1e-9;
                }
                double[] step = solve(hess, grad);
                double change = 0;
                for (int p = 0; p < d; p++) {
                    b[p] += step[p];
                    change += Math.abs(step[p]);
                }
                if (change < 1e-8) {
                    break;
                }
            }
            this.beta = b;
            double[] eta = new double[n];
            for (int i = 0; i < n; i++) {
                eta[i] = dot(b, x[i]);
            }
            this.baseline = new Breslow(eta, time, event);
        }

        /** @return the coefficients */
        public double[] coefficients() {
            return beta.clone();
        }

        /**
         * @param x covariates
         * @return the log-risk {@code beta . x}
         */
        public double logRisk(double[] x) {
            return dot(beta, x);
        }

        /**
         * @param x covariates
         * @param t a time
         * @return {@code S(t | x)}
         */
        public double survival(double[] x, double t) {
            return baseline.survival(logRisk(x), t);
        }
    }

    /**
     * The mean negative log partial likelihood of Cox's model (Breslow ties) over a batch:
     * {@code -(1 / #events) sum over events i of (eta_i - log sum_{t_j >= t_i} exp(eta_j))}.
     *
     * @param logRisk {@code [n, 1]} the network's log-risks
     * @param time the batch's observed times
     * @param event 0 if censored
     * @return the scalar loss
     */
    public static Var coxLoss(Var logRisk, double[] time, int[] event) {
        int n = time.length;
        float[] mask = new float[n * n];
        float[] events = new float[n];
        int count = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                mask[i * n + j] = time[j] >= time[i] ? 0 : -1e9f;
            }
            if (event[i] != 0) {
                events[i] = 1;
                count++;
            }
        }
        Var eta = logRisk.reshape(n, 1);
        Var lse = eta.reshape(1, n).add(Tensor.of(Shape.of(n, n), mask)).logsumexp(1, true);
        return eta.sub(lse).mul(Tensor.of(Shape.of(n, 1), events)).sum().mul(-1f / Math.max(1, count));
    }

    /**
     * DeepSurv: an MLP log-risk trained on {@link #coxLoss} over random mini-batches, with Breslow's
     * baseline estimated from its training predictions.
     */
    public static final class DeepSurv {
        private final Model network;
        private Breslow baseline;

        /**
         * Creates an untrained network.
         *
         * @param features input covariates
         * @param hidden the width of its two hidden layers
         * @param seed the seed
         */
        public DeepSurv(int features, int hidden, long seed) {
            this.network = Model.sequential().input(features).dense(hidden, Activation.RELU)
                    .dense(hidden, Activation.RELU).dense(1).seed(seed).build();
        }

        /** @return the log-risk network */
        public Model network() {
            return network;
        }

        /**
         * Trains the network.
         *
         * @param x covariates
         * @param time observed times
         * @param event 0 if censored
         * @param steps Adam steps
         * @param batch rows per step (the partial likelihood is computed within the batch)
         * @param lr the learning rate
         * @param l2 weight decay on the log-risk output (keeps the risk scale bounded)
         * @param seed the batch seed
         * @return this
         */
        public DeepSurv fit(double[][] x, double[] time, int[] event, int steps, int batch, float lr, float l2,
                long seed) {
            int n = x.length;
            int d = x[0].length;
            Optimizer opt = Optimizer.adam(lr);
            SplittableRandom r = new SplittableRandom(seed);
            int b = Math.min(batch, n);
            for (int s = 0; s < steps; s++) {
                float[] bx = new float[b * d];
                double[] bt = new double[b];
                int[] be = new int[b];
                for (int i = 0; i < b; i++) {
                    int k = r.nextInt(n);
                    for (int j = 0; j < d; j++) {
                        bx[i * d + j] = (float) x[k][j];
                    }
                    bt[i] = time[k];
                    be[i] = event[k];
                }
                Var eta = network.forward(Var.of(Tensor.of(Shape.of(b, d), bx)), true);
                coxLoss(eta, bt, be).add(eta.square().mean().mul(l2)).backward();
                opt.step(network.parameters());
            }
            baseline = new Breslow(logRisk(x), time, event);
            return this;
        }

        /**
         * @param x covariates
         * @return the log-risk per row
         */
        public double[] logRisk(double[][] x) {
            int d = x[0].length;
            float[] f = new float[x.length * d];
            for (int i = 0; i < x.length; i++) {
                for (int j = 0; j < d; j++) {
                    f[i * d + j] = (float) x[i][j];
                }
            }
            float[] out = Var.noGrad(() -> network.forward(Var.of(Tensor.of(Shape.of(x.length, d), f)), false))
                    .value().toArray();
            double[] eta = new double[x.length];
            for (int i = 0; i < eta.length; i++) {
                eta[i] = out[i];
            }
            return eta;
        }

        /**
         * @param x covariates
         * @param grid times
         * @return {@code S(grid[k] | x_i)} as {@code [n][grid]}
         */
        public double[][] survival(double[][] x, double[] grid) {
            double[] eta = logRisk(x);
            double[][] out = new double[x.length][grid.length];
            for (int i = 0; i < x.length; i++) {
                for (int k = 0; k < grid.length; k++) {
                    out[i][k] = baseline.survival(eta[i], grid[k]);
                }
            }
            return out;
        }
    }

    private static double dot(double[] a, double[] b) {
        double s = 0;
        for (int i = 0; i < a.length; i++) {
            s += a[i] * b[i];
        }
        return s;
    }

    private static double[] solve(double[][] a, double[] b) {
        int n = b.length;
        double[] flat = new double[n * n];
        for (int i = 0; i < n; i++) {
            System.arraycopy(a[i], 0, flat, i * n, n);
        }
        return Linalg.solve(Tensor.of(Shape.of(n, n), flat, DType.FLOAT64, Device.cpu()),
                Tensor.of(Shape.of(n, 1), b, DType.FLOAT64, Device.cpu())).toDoubleArray();
    }
}
