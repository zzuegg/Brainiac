package io.github.zzuegg.brainiac.experimental.timeseries;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;

/**
 * Adaptive nonlinear vector autoregression (Sherkhon et al., Sep 2025,
 * <a href="https://arxiv.org/abs/2507.08738">arXiv:2507.08738</a>), a follow-up of NG-RC
 * (Gauthier, Bollt, Griffith and Barbosa, Nature Communications 2021,
 * <a href="https://arxiv.org/abs/2106.07688">arXiv:2106.07688</a>). NG-RC regresses the next
 * increment on the delay-embedded state and all its quadratic monomials, whose number explodes
 * with the dimension; adaptive NVAR keeps the delay-embedded linear part and replaces the fixed
 * monomials by a small MLP on the same inputs, trained jointly with the linear readout by
 * gradient descent:
 * {@code x_t+1 - x_t = W_lin z_t + b + W_out tanh(W_h z_t + b_h)}, with
 * {@code z_t = [x_t, x_t-s, ..., x_t-(k-1)s]} standardised. No ridge or monomial grid is needed
 * and the parameter count grows linearly with the dimension. Training here is mini-batch Adam
 * (256 rows) with a cosine-decayed step size (the paper follows Adam with L-BFGS, which is not
 * done here); forecasts run in closed loop.
 */
public final class AdaptiveNvar {

    private final int d;
    private final int delays;
    private final int skip;
    private final double[] mean;
    private final double[] scale;
    private final double[] outScale;
    private final Var wLin;
    private final Var bias;
    private final Var wHidden;
    private final Var bHidden;
    private final Var wOut;

    private AdaptiveNvar(int d, int delays, int skip, int hidden, double[] mean, double[] scale, double[] outScale,
            long seed) {
        this.d = d;
        this.delays = delays;
        this.skip = skip;
        this.mean = mean;
        this.scale = scale;
        this.outScale = outScale;
        int in = d * delays;
        SplittableRandom r = new SplittableRandom(seed);
        this.wLin = Var.param(Tensor.zeros(Shape.of(in, d)));
        this.bias = Var.param(Tensor.zeros(Shape.of(1, d)));
        this.wHidden = Var.param(Tensor.normal(Shape.of(in, hidden), 0, (float) Math.sqrt(1.0 / in), r));
        this.bHidden = Var.param(Tensor.zeros(Shape.of(1, hidden)));
        this.wOut = Var.param(Tensor.normal(Shape.of(hidden, d), 0, (float) (0.1 / Math.sqrt(hidden)), r));
    }

    /**
     * Fits the model to the first {@code train} samples of a series.
     *
     * @param series {@code [time][dimension]}
     * @param train the training samples
     * @param delays the taps {@code k}
     * @param skip the tap spacing {@code s}
     * @param hidden the MLP width
     * @param epochs Adam steps on mini-batches of 256 rows
     * @param learningRate the initial Adam step size, decayed by a cosine to 5% of it
     * @param seed the seed
     * @return the fitted model
     */
    public static AdaptiveNvar fit(double[][] series, int train, int delays, int skip, int hidden, int epochs,
            double learningRate, long seed) {
        int d = series[0].length;
        double[] mean = new double[d];
        double[] sd = new double[d];
        for (int t = 0; t < train; t++) {
            for (int j = 0; j < d; j++) {
                mean[j] += series[t][j] / train;
            }
        }
        for (int t = 0; t < train; t++) {
            for (int j = 0; j < d; j++) {
                sd[j] += Math.pow(series[t][j] - mean[j], 2) / train;
            }
        }
        double[] outScale = new double[d];
        int warm = (delays - 1) * skip;
        int rows = train - 1 - warm;
        for (int t = warm; t < train - 1; t++) {
            for (int j = 0; j < d; j++) {
                outScale[j] += Math.pow(series[t + 1][j] - series[t][j], 2) / rows;
            }
        }
        for (int j = 0; j < d; j++) {
            sd[j] = Math.sqrt(sd[j]) + 1e-9;
            outScale[j] = Math.sqrt(outScale[j]) + 1e-9;
        }
        AdaptiveNvar m = new AdaptiveNvar(d, delays, skip, hidden, mean, sd, outScale, seed);
        float[] in = new float[rows * d * delays];
        float[] out = new float[rows * d];
        for (int i = 0; i < rows; i++) {
            int t = warm + i;
            float[] z = m.embed(series, t);
            System.arraycopy(z, 0, in, i * z.length, z.length);
            for (int j = 0; j < d; j++) {
                out[i * d + j] = (float) ((series[t + 1][j] - series[t][j]) / outScale[j]);
            }
        }
        List<Var> params = m.parameters();
        Optimizer opt = Optimizer.adam((float) learningRate);
        int batch = Math.min(256, rows);
        SplittableRandom r = new SplittableRandom(seed + 1);
        int width = d * delays;
        for (int e = 0; e < epochs; e++) {
            float[] bx = new float[batch * width];
            float[] by = new float[batch * d];
            for (int i = 0; i < batch; i++) {
                int k = batch == rows ? i : r.nextInt(rows);
                System.arraycopy(in, k * width, bx, i * width, width);
                System.arraycopy(out, k * d, by, i * d, d);
            }
            double c = 0.5 * (1 + Math.cos(Math.PI * e / Math.max(1, epochs)));
            opt.learningRate((float) (learningRate * (0.05 + 0.95 * c)));
            Var loss = m.forward(Var.of(Tensor.of(Shape.of(batch, width), bx)))
                    .sub(Tensor.of(Shape.of(batch, d), by)).square().mean();
            loss.backward();
            opt.step(params);
        }
        return m;
    }

    private float[] embed(double[][] series, int t) {
        float[] z = new float[d * delays];
        for (int k = 0; k < delays; k++) {
            double[] v = series[t - k * skip];
            for (int j = 0; j < d; j++) {
                z[k * d + j] = (float) ((v[j] - mean[j]) / scale[j]);
            }
        }
        return z;
    }

    private Var forward(Var z) {
        Var lin = z.matmul(wLin).add(bias);
        return lin.add(z.matmul(wHidden).add(bHidden).tanh().matmul(wOut));
    }

    /** @return the trainable parameters */
    public List<Var> parameters() {
        return List.of(wLin, bias, wHidden, bHidden, wOut);
    }

    /** @return the number of trainable numbers */
    public long parameterCount() {
        long n = 0;
        for (Var p : parameters()) {
            n += p.value().size();
        }
        return n;
    }

    /**
     * Forecasts in closed loop after the first {@code start} samples of a series.
     *
     * @param series the series (only the first {@code start} samples are read)
     * @param start the number of known samples
     * @param horizon the steps to forecast
     * @return {@code [horizon][dimension]}
     */
    public double[][] forecast(double[][] series, int start, int horizon) {
        int warm = (delays - 1) * skip;
        double[][] buf = new double[warm + 1 + horizon][];
        for (int i = 0; i <= warm; i++) {
            buf[i] = series[start - 1 - warm + i].clone();
        }
        double[][] out = new double[horizon][];
        for (int h = 0; h < horizon; h++) {
            int t = warm + h;
            float[] z = embed(buf, t);
            float[] dx = Var.noGrad(() -> forward(Var.of(Tensor.of(Shape.of(1, z.length), z)))).value().toArray();
            double[] next = new double[d];
            for (int j = 0; j < d; j++) {
                next[j] = buf[t][j] + dx[j] * outScale[j];
            }
            buf[t + 1] = next;
            out[h] = next.clone();
        }
        return out;
    }
}
