package io.github.zzuegg.brainiac.experimental.science;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ml.StateSpaceFilter;
import io.github.zzuegg.brainiac.experimental.train.RidgeRegression;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Recurrent;
import io.github.zzuegg.brainiac.tensor.Matrices;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Decoders of 2-D cursor velocity from binned population spike counts.
 *
 * <ul>
 *   <li>{@link Method#WIENER}: ridge regression of velocity on the counts of the current and
 *       {@code lags - 1} previous bins.</li>
 *   <li>{@link Method#KALMAN} (Wu et al., Neural Comput. 2006): a linear-Gaussian model with
 *       velocity as the state, {@code v_t = A v_t-1 + w} and centred counts
 *       {@code z_t = H v_t + q} with full covariances; the parameters are the maximum-likelihood
 *       fit given the training kinematics (what EM converges to when the state is observed), and
 *       decoding runs the generic {@link StateSpaceFilter#kalman}.</li>
 *   <li>{@link Method#POISSON_GLM}: per-neuron Poisson encoding models
 *       {@code log lambda = b + w . v + h n_t-1} (with a spike-history term, Pillow et al.,
 *       Nature 2008) fitted by Newton's method, decoded by the point-process filter (Eden et al.,
 *       Neural Comput. 2004) with the Kalman decoder's velocity prior.</li>
 *   <li>{@link Method#RECURRENT}: a core {@link Recurrent#lstm} over square-root counts with a
 *       linear read-out (Glaser et al., eNeuro 2020).</li>
 *   <li>{@link Method#POPULATION_VECTOR} and {@link Method#RIDGE_CURRENT_BIN}: baselines.</li>
 * </ul>
 */
public final class NeuralDecoders {

    /** The decoder. */
    public enum Method {
        /** Ridge on lagged counts. */
        WIENER,
        /** Kalman filter with fitted parameters. */
        KALMAN,
        /** The same filter with naive parameters (identity dynamics and noise). */
        KALMAN_NAIVE,
        /** Poisson GLM with point-process filtering. */
        POISSON_GLM,
        /** LSTM decoder. */
        RECURRENT,
        /** Population vector from fitted preferred directions. */
        POPULATION_VECTOR,
        /** Ridge on the current bin only. */
        RIDGE_CURRENT_BIN
    }

    private final Method method;
    private final int lags;
    private double[] countMean;
    private double[][] ridge;
    private double[][] a;
    private double[][] w;
    private double[][] h;
    private double[][] r;
    private double[][] glm;
    private double dt;
    private Layer lstm;
    private Var head;
    private Var headBias;
    private double[] pvScale;
    private double[][] pvDirection;
    private int hidden = 32;
    private int lstmSteps = 150;

    /**
     * An untrained decoder.
     *
     * @param method the method
     * @param lags bins of history for {@link Method#WIENER}
     */
    public NeuralDecoders(Method method, int lags) {
        this.method = method;
        this.lags = lags;
    }

    /**
     * @param steps LSTM Adam steps (default 150)
     * @param units LSTM width (default 32)
     * @return this decoder
     */
    public NeuralDecoders recurrent(int steps, int units) {
        this.lstmSteps = steps;
        this.hidden = units;
        return this;
    }

    private double[] lagged(double[][] counts, int t, int l) {
        int n = counts[0].length;
        double[] f = new double[n * l + 1];
        for (int k = 0; k < l; k++) {
            int s = Math.max(0, t - k);
            for (int j = 0; j < n; j++) {
                f[k * n + j] = counts[s][j] - countMean[j];
            }
        }
        f[n * l] = 1;
        return f;
    }

    /**
     * Fits the decoder.
     *
     * @param counts {@code [bin][neuron]}
     * @param velocity {@code [bin][2]}
     * @param binSeconds the bin width
     * @param seed the seed
     * @return this decoder
     */
    public NeuralDecoders fit(double[][] counts, double[][] velocity, double binSeconds, long seed) {
        int t = counts.length;
        int n = counts[0].length;
        dt = binSeconds;
        countMean = new double[n];
        for (double[] c : counts) {
            for (int j = 0; j < n; j++) {
                countMean[j] += c[j] / t;
            }
        }
        switch (method) {
            case WIENER, RIDGE_CURRENT_BIN -> {
                int l = method == Method.WIENER ? lags : 1;
                RidgeRegression rr = RidgeRegression.of(n * l + 1, 2);
                for (int i = 0; i < t; i++) {
                    rr.add(lagged(counts, i, l), velocity[i]);
                }
                ridge = rr.solve(1.0);
            }
            case KALMAN, KALMAN_NAIVE, POISSON_GLM -> {
                fitKalman(counts, velocity);
                if (method == Method.POISSON_GLM) {
                    fitGlm(counts, velocity);
                }
            }
            case RECURRENT -> fitLstm(counts, velocity, seed);
            case POPULATION_VECTOR -> fitPopulationVector(counts, velocity);
            default -> throw new IllegalStateException();
        }
        return this;
    }

    private static double[][] lstsq(double[][] x, double[][] y, double ridgeLambda) {
        int p = x[0].length;
        int q = y[0].length;
        double[][] xtx = new double[p][p];
        double[][] xty = new double[p][q];
        for (int i = 0; i < x.length; i++) {
            for (int a1 = 0; a1 < p; a1++) {
                for (int b = 0; b < p; b++) {
                    xtx[a1][b] += x[i][a1] * x[i][b];
                }
                for (int b = 0; b < q; b++) {
                    xty[a1][b] += x[i][a1] * y[i][b];
                }
            }
        }
        for (int a1 = 0; a1 < p; a1++) {
            xtx[a1][a1] += ridgeLambda;
        }
        double[][] inv = Matrices.inverse(xtx);
        double[][] beta = new double[q][p];
        for (int a1 = 0; a1 < q; a1++) {
            for (int b = 0; b < p; b++) {
                for (int c = 0; c < p; c++) {
                    beta[a1][b] += inv[b][c] * xty[c][a1];
                }
            }
        }
        return beta;   // [outputs][inputs]
    }

    private static double[][] residualCov(double[][] x, double[][] y, double[][] beta) {
        int q = y[0].length;
        double[][] c = new double[q][q];
        for (int i = 0; i < x.length; i++) {
            double[] e = new double[q];
            for (int k = 0; k < q; k++) {
                e[k] = y[i][k];
                for (int j = 0; j < x[0].length; j++) {
                    e[k] -= beta[k][j] * x[i][j];
                }
            }
            for (int k = 0; k < q; k++) {
                for (int l = 0; l < q; l++) {
                    c[k][l] += e[k] * e[l] / x.length;
                }
            }
        }
        return c;
    }

    private void fitKalman(double[][] counts, double[][] velocity) {
        int t = counts.length;
        int n = counts[0].length;
        double[][] prev = new double[t - 1][];
        double[][] next = new double[t - 1][];
        for (int i = 1; i < t; i++) {
            prev[i - 1] = velocity[i - 1];
            next[i - 1] = velocity[i];
        }
        a = lstsq(prev, next, 1e-6);
        w = residualCov(prev, next, a);
        double[][] z = new double[t][n];
        for (int i = 0; i < t; i++) {
            for (int j = 0; j < n; j++) {
                z[i][j] = counts[i][j] - countMean[j];
            }
        }
        h = lstsq(velocity, z, 1e-6);
        r = residualCov(velocity, z, h);
        for (int j = 0; j < n; j++) {
            r[j][j] += 1e-3;
        }
        if (method == Method.KALMAN_NAIVE) {
            a = Matrices.identity(2);
            w = Matrices.identity(2);
            double v = 0;
            for (int j = 0; j < n; j++) {
                v += r[j][j] / n;
            }
            r = new double[n][n];
            for (int j = 0; j < n; j++) {
                r[j][j] = v;
            }
        }
    }

    private void fitGlm(double[][] counts, double[][] velocity) {
        int t = counts.length;
        int n = counts[0].length;
        glm = new double[n][4];
        for (int j = 0; j < n; j++) {
            double[] beta = {Math.log(Math.max(countMean[j], 1e-3)), 0, 0, 0};
            for (int it = 0; it < 25; it++) {
                double[] g = new double[4];
                double[][] hess = new double[4][4];
                for (int i = 1; i < t; i++) {
                    double[] x = {1, velocity[i][0], velocity[i][1], Math.log1p(counts[i - 1][j])};
                    double eta = 0;
                    for (int k = 0; k < 4; k++) {
                        eta += beta[k] * x[k];
                    }
                    double mu = Math.exp(Math.min(eta, 20));
                    for (int k = 0; k < 4; k++) {
                        g[k] += (counts[i][j] - mu) * x[k];
                        for (int l = 0; l < 4; l++) {
                            hess[k][l] += mu * x[k] * x[l];
                        }
                    }
                }
                for (int k = 0; k < 4; k++) {
                    hess[k][k] += 1e-6;
                }
                double[] step = Matrices.solve(hess, g);
                double norm = 0;
                for (int k = 0; k < 4; k++) {
                    beta[k] += step[k];
                    norm += step[k] * step[k];
                }
                if (norm < 1e-12) {
                    break;
                }
            }
            glm[j] = beta;
        }
    }

    private void fitPopulationVector(double[][] counts, double[][] velocity) {
        // preferred directions from the regression of centred counts on velocity
        int n = counts[0].length;
        double[][] z = new double[counts.length][n];
        for (int i = 0; i < counts.length; i++) {
            for (int j = 0; j < n; j++) {
                z[i][j] = counts[i][j] - countMean[j];
            }
        }
        double[][] b = lstsq(velocity, z, 1e-6);
        pvDirection = new double[n][2];
        for (int j = 0; j < n; j++) {
            double m = Math.hypot(b[j][0], b[j][1]) + 1e-12;
            pvDirection[j][0] = b[j][0] / m;
            pvDirection[j][1] = b[j][1] / m;
        }
        // one gain per axis by least squares on the raw population vector
        double[] num = new double[2];
        double[] den = new double[2];
        for (int i = 0; i < counts.length; i++) {
            double[] pv = rawPv(counts[i]);
            for (int d = 0; d < 2; d++) {
                num[d] += pv[d] * velocity[i][d];
                den[d] += pv[d] * pv[d];
            }
        }
        pvScale = new double[] {num[0] / den[0], num[1] / den[1]};
    }

    private double[] rawPv(double[] c) {
        double[] v = new double[2];
        for (int j = 0; j < c.length; j++) {
            double d = (c[j] - countMean[j]) / Math.sqrt(countMean[j] + 1e-3);
            v[0] += d * pvDirection[j][0];
            v[1] += d * pvDirection[j][1];
        }
        return v;
    }

    private void fitLstm(double[][] counts, double[][] velocity, long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        int n = counts[0].length;
        int window = 40;
        int burn = 5;
        int batch = 16;
        lstm = Recurrent.lstm(hidden).returnSequences(true).create(Shape.of(window, n), rnd);
        float[] hw = new float[hidden * 2];
        for (int i = 0; i < hw.length; i++) {
            hw[i] = (float) (0.1 * rnd.nextGaussian());
        }
        head = Var.param(Tensor.of(Shape.of(hidden, 2), hw));
        headBias = Var.param(Tensor.zeros(Shape.of(2)));
        List<Var> params = new ArrayList<>(lstm.parameters());
        params.add(head);
        params.add(headBias);
        Optimizer opt = Optimizer.adam(5e-3f);
        int t = counts.length;
        for (int s = 0; s < lstmSteps; s++) {
            float[] in = new float[batch * window * n];
            float[] target = new float[batch * (window - burn) * 2];
            for (int b = 0; b < batch; b++) {
                int start = rnd.nextInt(Math.max(1, t - window));
                for (int i = 0; i < window; i++) {
                    for (int j = 0; j < n; j++) {
                        in[(b * window + i) * n + j] = (float) Math.sqrt(counts[start + i][j]);
                    }
                    if (i >= burn) {
                        target[(b * (window - burn) + i - burn) * 2] = (float) velocity[start + i][0];
                        target[(b * (window - burn) + i - burn) * 2 + 1] = (float) velocity[start + i][1];
                    }
                }
            }
            Tensor xin = Tensor.of(Shape.of(batch, window, n), in);
            Tensor yt = Tensor.of(Shape.of(batch, window - burn, 2), target);
            opt.minimize(params, () -> {
                Var hs = lstm.forward(Var.of(xin)).slice(1, burn, window);
                Var pred = hs.reshape(batch * (window - burn), hidden).matmul(head).add(headBias)
                        .reshape(batch, window - burn, 2);
                return pred.sub(Var.of(yt)).square().mean();
            });
        }
    }

    /**
     * Decodes a session.
     *
     * @param counts {@code [bin][neuron]}
     * @return velocity {@code [bin][2]}
     */
    public double[][] decode(double[][] counts) {
        int t = counts.length;
        int n = counts[0].length;
        double[][] out = new double[t][2];
        switch (method) {
            case WIENER, RIDGE_CURRENT_BIN -> {
                int l = method == Method.WIENER ? lags : 1;
                for (int i = 0; i < t; i++) {
                    out[i] = RidgeRegression.predict(ridge, lagged(counts, i, l));
                }
            }
            case KALMAN, KALMAN_NAIVE -> {
                double[][] z = new double[t][n];
                for (int i = 0; i < t; i++) {
                    for (int j = 0; j < n; j++) {
                        z[i][j] = counts[i][j] - countMean[j];
                    }
                }
                StateSpaceFilter.Estimate e = StateSpaceFilter.kalman(new StateSpaceFilter.LinearModel(a, w, h, r,
                        new double[2], Matrices.identity(2)), z, false);
                for (int i = 0; i < t; i++) {
                    out[i] = e.means()[i].clone();
                }
            }
            case POISSON_GLM -> pointProcess(counts, out);
            case RECURRENT -> {
                float[] in = new float[t * n];
                for (int i = 0; i < t; i++) {
                    for (int j = 0; j < n; j++) {
                        in[i * n + j] = (float) Math.sqrt(counts[i][j]);
                    }
                }
                float[] o = Var.noGrad(() -> lstm.forward(Var.of(Tensor.of(Shape.of(1, t, n), in)))
                        .reshape(t, hidden).matmul(head).add(headBias)).value().toArray();
                for (int i = 0; i < t; i++) {
                    out[i][0] = o[2 * i];
                    out[i][1] = o[2 * i + 1];
                }
            }
            case POPULATION_VECTOR -> {
                for (int i = 0; i < t; i++) {
                    double[] pv = rawPv(counts[i]);
                    out[i][0] = pvScale[0] * pv[0];
                    out[i][1] = pvScale[1] * pv[1];
                }
            }
            default -> throw new IllegalStateException();
        }
        return out;
    }

    private void pointProcess(double[][] counts, double[][] out) {
        int t = counts.length;
        int n = counts[0].length;
        double[] x = new double[2];
        double[][] p = Matrices.identity(2);
        for (int i = 0; i < t; i++) {
            // predict
            double[] xp = {a[0][0] * x[0] + a[0][1] * x[1], a[1][0] * x[0] + a[1][1] * x[1]};
            double[][] pp = Matrices.multiply(Matrices.multiply(a, p), transpose(a));
            for (int k = 0; k < 2; k++) {
                for (int l = 0; l < 2; l++) {
                    pp[k][l] += w[k][l];
                }
            }
            double[][] info = Matrices.inverse(pp);
            double[] grad = new double[2];
            for (int j = 0; j < n; j++) {
                double hist = i > 0 ? Math.log1p(counts[i - 1][j]) : Math.log1p(countMean[j]);
                double lambda = Math.exp(Math.min(20, glm[j][0] + glm[j][1] * xp[0] + glm[j][2] * xp[1]
                        + glm[j][3] * hist));
                double[] wj = {glm[j][1], glm[j][2]};
                for (int k = 0; k < 2; k++) {
                    grad[k] += wj[k] * (counts[i][j] - lambda);
                    for (int l = 0; l < 2; l++) {
                        info[k][l] += lambda * wj[k] * wj[l];
                    }
                }
            }
            p = Matrices.inverse(info);
            x = new double[] {xp[0] + p[0][0] * grad[0] + p[0][1] * grad[1],
                xp[1] + p[1][0] * grad[0] + p[1][1] * grad[1]};
            out[i] = x.clone();
        }
    }

    private static double[][] transpose(double[][] m) {
        double[][] o = new double[m[0].length][m.length];
        for (int i = 0; i < m.length; i++) {
            for (int j = 0; j < m[0].length; j++) {
                o[j][i] = m[i][j];
            }
        }
        return o;
    }

    /**
     * The coefficient of determination averaged over the two velocity components.
     *
     * @param predicted decoded velocity
     * @param truth true velocity
     * @return mean R²
     */
    public static double r2(double[][] predicted, double[][] truth) {
        double total = 0;
        for (int d = 0; d < 2; d++) {
            double m = 0;
            for (double[] v : truth) {
                m += v[d] / truth.length;
            }
            double sse = 0;
            double sst = 0;
            for (int i = 0; i < truth.length; i++) {
                sse += (predicted[i][d] - truth[i][d]) * (predicted[i][d] - truth[i][d]);
                sst += (truth[i][d] - m) * (truth[i][d] - m);
            }
            total += 1 - sse / sst;
        }
        return total / 2;
    }
}
