package io.github.zzuegg.brainiac.experimental.science;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ScienceSims;
import io.github.zzuegg.brainiac.experimental.quantum.ParameterObjective;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Recurrent;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Data-driven and hybrid rainfall–runoff models.
 *
 * <ul>
 *   <li>{@link Model#LSTM}: a core {@link Recurrent#lstm} over daily forcings (precipitation,
 *       temperature, potential evaporation, season) with a linear read-out, the model that beats
 *       calibrated conceptual models on Nash–Sutcliffe efficiency (Kratzert et al., HESS 2018).</li>
 *   <li>{@link Model#DELTA_HBV_STATIC}: differentiable HBV (Feng et al., WRR 2022) whose
 *       parameters are learned end-to-end by gradient descent through the bucket equations
 *       (forward-mode derivatives of {@link Hbv#simulateDual}).</li>
 *   <li>{@link Model#DELTA_HBV_DYNAMIC}: the same with three parameters (CFMAX, LP, BETA) made
 *       time-varying by a learned map of the day's forcings, as the δ-HBV variants with dynamic
 *       parameters do with an LSTM.</li>
 * </ul>
 * The 2026 HESS study of Bohl et al. found LSTMs most accurate but least robust under warming,
 * and the hybrids best at reproducing observed change signals; {@code experiments.science.Hydrology}
 * tests that ranking on a synthetic catchment.
 */
public final class HybridHydrology {

    /** The model family. */
    public enum Model {
        /** LSTM on forcings. */
        LSTM,
        /** Differentiable HBV with learned static parameters. */
        DELTA_HBV_STATIC,
        /** Differentiable HBV with forcing-dependent parameters. */
        DELTA_HBV_DYNAMIC
    }

    private static final int[] DYNAMIC = {1, 6, 7};
    private static final int FEATURES = 5;

    private final Model model;
    private double[] theta;
    private double maxbas = 2.5;
    private Layer lstm;
    private Var head;
    private Var headBias;
    private double[] featMean;
    private double[] featSd;
    private double qScale = 1;

    /**
     * An untrained model.
     *
     * @param model the family
     */
    public HybridHydrology(Model model) {
        this.model = model;
    }

    private double[][] features(ScienceSims.Weather w) {
        int n = w.precipitation().length;
        double[][] x = new double[n][FEATURES];
        for (int t = 0; t < n; t++) {
            x[t][0] = w.precipitation()[t];
            x[t][1] = w.temperature()[t];
            x[t][2] = w.potentialEvaporation()[t];
            x[t][3] = Math.sin(2 * Math.PI * (t % 365) / 365.0);
            x[t][4] = Math.cos(2 * Math.PI * (t % 365) / 365.0);
        }
        if (featMean == null) {
            featMean = new double[FEATURES];
            featSd = new double[FEATURES];
            for (double[] r : x) {
                for (int f = 0; f < FEATURES; f++) {
                    featMean[f] += r[f] / n;
                }
            }
            for (double[] r : x) {
                for (int f = 0; f < FEATURES; f++) {
                    featSd[f] += (r[f] - featMean[f]) * (r[f] - featMean[f]) / n;
                }
            }
            for (int f = 0; f < FEATURES; f++) {
                featSd[f] = Math.sqrt(featSd[f]) + 1e-9;
            }
        }
        for (double[] r : x) {
            for (int f = 0; f < FEATURES; f++) {
                r[f] = (r[f] - featMean[f]) / featSd[f];
            }
        }
        return x;
    }

    private int thetaSize() {
        return model == Model.DELTA_HBV_DYNAMIC ? 13 + DYNAMIC.length * FEATURES : 13;
    }

    private static double sigmoid(double v) {
        return 1 / (1 + Math.exp(-v));
    }

    private Hbv.Dual[] dayParameters(double[] th, double[] x, int k, boolean withGrad) {
        Hbv.Dual[] p = new Hbv.Dual[13];
        for (int i = 0; i < 13; i++) {
            double z = th[i];
            double[] dz = new double[k];
            if (withGrad) {
                dz[i] = 1;
            }
            if (model == Model.DELTA_HBV_DYNAMIC) {
                for (int j = 0; j < DYNAMIC.length; j++) {
                    if (DYNAMIC[j] == i) {
                        for (int f = 0; f < FEATURES; f++) {
                            int idx = 13 + j * FEATURES + f;
                            z += th[idx] * x[f];
                            if (withGrad) {
                                dz[idx] = x[f];
                            }
                        }
                    }
                }
            }
            double s = sigmoid(z);
            double range = Hbv.UPPER[i] - Hbv.LOWER[i];
            double[] g = new double[k];
            for (int a = 0; a < k; a++) {
                g[a] = dz[a] * range * s * (1 - s);
            }
            p[i] = new Hbv.Dual(Hbv.LOWER[i] + range * s, g);
        }
        return p;
    }

    /**
     * Trains on a forcing series and observed runoff.
     *
     * @param w forcings
     * @param obs observed runoff
     * @param warmup days excluded from the loss
     * @param steps Adam steps (δ-HBV: full-series epochs; LSTM: mini-batches of windows)
     * @param learningRate the Adam step
     * @param initialUnit δ-HBV starting parameters in the unit cube (such as a CMA-ES calibration), or null
     * @param seed the seed
     * @return this model
     */
    public HybridHydrology train(ScienceSims.Weather w, double[] obs, int warmup, int steps, double learningRate,
            double[] initialUnit, long seed) {
        double[][] x = features(w);
        if (model == Model.LSTM) {
            trainLstm(x, obs, warmup, steps, learningRate, seed);
            return this;
        }
        int k = thetaSize();
        double[] th0 = new double[k];
        for (int i = 0; i < 13; i++) {
            double u = initialUnit == null ? 0.5 : Math.min(0.98, Math.max(0.02, initialUnit[i]));
            th0[i] = Math.log(u / (1 - u));
        }
        if (initialUnit != null) {
            maxbas = Hbv.LOWER[13] + initialUnit[13] * (Hbv.UPPER[13] - Hbv.LOWER[13]);
        }
        int n = obs.length;
        double var = 0;
        double mean = 0;
        for (int t = warmup; t < n; t++) {
            mean += obs[t] / (n - warmup);
        }
        for (int t = warmup; t < n; t++) {
            var += (obs[t] - mean) * (obs[t] - mean) / (n - warmup);
        }
        double scale = var;
        ParameterObjective f = (th, g) -> {
            Hbv.Dual[] q = Hbv.simulateDual(t -> dayParameters(th, x[t], k, true), w, maxbas, k);
            double loss = 0;
            java.util.Arrays.fill(g, 0);
            for (int t = warmup; t < n; t++) {
                double e = q[t].v - obs[t];
                loss += e * e / ((n - warmup) * scale);
                for (int a = 0; a < k; a++) {
                    g[a] += 2 * e * q[t].d[a] / ((n - warmup) * scale);
                }
            }
            return loss;
        };
        theta = f.minimize(Optimizer.adam((float) learningRate), th0, steps, null);
        return this;
    }

    private void trainLstm(double[][] x, double[] obs, int warmup, int steps, double lr, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int hidden = 20;
        int window = 365;
        int burn = 90;
        lstm = Recurrent.lstm(hidden).returnSequences(true).create(Shape.of(window, FEATURES), r);
        float[] hw = new float[hidden];
        for (int i = 0; i < hidden; i++) {
            hw[i] = (float) (0.1 * r.nextGaussian());
        }
        head = Var.param(Tensor.of(Shape.of(hidden, 1), hw));
        headBias = Var.param(Tensor.zeros(Shape.of(1)));
        double m = 0;
        for (double v : obs) {
            m += v / obs.length;
        }
        qScale = m;
        List<Var> params = new ArrayList<>(lstm.parameters());
        params.add(head);
        params.add(headBias);
        Optimizer opt = Optimizer.adam((float) lr);
        int batch = 8;
        for (int s = 0; s < steps; s++) {
            float[] in = new float[batch * window * FEATURES];
            float[] target = new float[batch * (window - burn)];
            for (int b = 0; b < batch; b++) {
                int start = warmup + r.nextInt(obs.length - window - warmup);
                for (int t = 0; t < window; t++) {
                    for (int f = 0; f < FEATURES; f++) {
                        in[(b * window + t) * FEATURES + f] = (float) x[start + t][f];
                    }
                    if (t >= burn) {
                        target[b * (window - burn) + t - burn] = (float) (obs[start + t] / qScale);
                    }
                }
            }
            Tensor xin = Tensor.of(Shape.of(batch, window, FEATURES), in);
            Tensor yt = Tensor.of(Shape.of(batch, window - burn, 1), target);
            opt.minimize(params, () -> {
                Var hs = lstm.forward(Var.of(xin)).slice(1, burn, window);
                Var pred = hs.reshape(batch * (window - burn), hidden).matmul(head).add(headBias)
                        .reshape(batch, window - burn, 1);
                return pred.sub(Var.of(yt)).square().mean();
            });
        }
    }

    /**
     * Predicts runoff for a forcing series (the features are standardised with the training
     * statistics).
     *
     * @param w forcings
     * @return runoff per day
     */
    public double[] predict(ScienceSims.Weather w) {
        double[][] x = features(w);
        int n = x.length;
        if (model == Model.LSTM) {
            float[] in = new float[n * FEATURES];
            for (int t = 0; t < n; t++) {
                for (int f = 0; f < FEATURES; f++) {
                    in[t * FEATURES + f] = (float) x[t][f];
                }
            }
            int hidden = head.shape().dim(0);
            float[] out = Var.noGrad(() -> lstm.forward(Var.of(Tensor.of(Shape.of(1, n, FEATURES), in)))
                    .reshape(n, hidden).matmul(head).add(headBias)).value().toArray();
            double[] q = new double[n];
            for (int t = 0; t < n; t++) {
                q[t] = Math.max(0, out[t] * qScale);
            }
            return q;
        }
        Hbv.Dual[] q = Hbv.simulateDual(t -> dayParameters(theta, x[t], 0, false), w, maxbas, 0);
        double[] out = new double[n];
        for (int t = 0; t < n; t++) {
            out[t] = q[t].v;
        }
        return out;
    }
}
