package io.github.zzuegg.brainiac.experimental.timeseries;

import io.github.zzuegg.brainiac.autograd.Complex;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * Autoformer's auto-correlation mechanism (Wu, Xu, Wang and Long, NeurIPS 2021,
 * <a href="https://arxiv.org/abs/2106.13008">arXiv:2106.13008</a>), a replacement for
 * self-attention on series. Queries and keys are correlated over all lags at once with the
 * Wiener-Khinchin theorem, {@code R(tau) = IFFT(FFT(Q) conj(FFT(K)))} (core {@link Var#rfft},
 * {@link Complex}), averaged over channels; the {@code k = factor ln L} lags with the
 * highest batch-mean correlation are kept, their correlations are softmax-normalised per
 * sample, and the output aggregates the values rolled by each lag ({@code Roll(V, tau)}),
 * which copies whole sub-series from one period to the next instead of mixing single points.
 * Input and output {@code [batch, time, width]}.
 */
public final class AutoCorrelationLayer implements Layer {

    private final int width;
    private final double factor;
    private final Var wq;
    private final Var wk;
    private final Var wv;
    private final Var wo;

    /**
     * Creates the layer.
     *
     * @param width the model width
     * @param factor {@code c} in {@code k = c ln L}, for example 1
     * @param random the initialisation source
     */
    public AutoCorrelationLayer(int width, double factor, SplittableRandom random) {
        this.width = width;
        this.factor = factor;
        float s = (float) Math.sqrt(1.0 / width);
        wq = Var.param(Tensor.normal(Shape.of(width, width), 0, s, random));
        wk = Var.param(Tensor.normal(Shape.of(width, width), 0, s, random));
        wv = Var.param(Tensor.normal(Shape.of(width, width), 0, s, random));
        wo = Var.param(Tensor.normal(Shape.of(width, width), 0, s, random));
    }

    /**
     * @param factor {@code c} in {@code k = c ln L}
     * @return a spec for model builders
     */
    public static LayerSpec spec(double factor) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return input;
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new AutoCorrelationLayer(input.dim(-1), factor, random);
            }
        };
    }

    /**
     * The channel-mean auto-correlation of queries and keys over all lags.
     *
     * @param q {@code [batch, time, width]}
     * @param k {@code [batch, time, width]}
     * @return {@code [batch, time]}, entry {@code tau} the correlation at lag {@code tau}
     */
    public static Var correlation(Var q, Var k) {
        int t = q.shape().dim(1);
        Var qf = q.permute(0, 2, 1).rfft();
        Var kf = k.permute(0, 2, 1).rfft();
        return Complex.mul(qf, Complex.conj(kf)).irfft(t).mean(1, false);
    }

    /** {@code out[t] = v[(t + tau) mod L]} along the time axis. */
    static Var roll(Var v, int tau) {
        int t = v.shape().dim(1);
        if (tau % t == 0) {
            return v;
        }
        return Var.concat(1, v.slice(1, tau, t), v.slice(1, 0, tau));
    }

    @Override
    public Var forward(Var input) {
        int b = input.shape().dim(0);
        int t = input.shape().dim(1);
        Var q = input.matmul(wq);
        Var k = input.matmul(wk);
        Var v = input.matmul(wv);
        Var corr = correlation(q, k);                                   // [b, t]
        int top = Math.max(1, Math.min(t - 1, (int) (factor * Math.log(t))));
        float[] mean = corr.value().mean(0, false).toArray();
        int[] lags = new int[top];
        boolean[] used = new boolean[t];
        for (int i = 0; i < top; i++) {
            int best = -1;
            for (int tau = 1; tau < t; tau++) {
                if (!used[tau] && (best < 0 || mean[tau] > mean[best])) {
                    best = tau;
                }
            }
            used[best] = true;
            lags[i] = best;
        }
        float[] idx = new float[top];
        for (int i = 0; i < top; i++) {
            idx[i] = lags[i];
        }
        Var selected = corr.gather(1, Tensor.of(Shape.of(top), idx));  // [b, top]
        Var weights = selected.sub(selected.logsumexp(1, true)).exp();
        Var out = null;
        for (int i = 0; i < top; i++) {
            Var term = roll(v, lags[i]).mul(weights.slice(1, i, i + 1).reshape(b, 1, 1));
            out = out == null ? term : out.add(term);
        }
        return out.matmul(wo);
    }

    @Override
    public List<Var> parameters() {
        return List.of(wq, wk, wv, wo);
    }

    @Override
    public String toString() {
        return "AutoCorrelationLayer(" + width + ", factor=" + factor + ")";
    }
}
