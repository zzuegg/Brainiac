package io.github.zzuegg.brainiac.experiments.timeseries;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.timeseries.AutoCorrelationLayer;
import io.github.zzuegg.brainiac.experimental.timeseries.FrequencyEnhancedBlock;
import io.github.zzuegg.brainiac.experimental.timeseries.SeriesDecomposition;
import io.github.zzuegg.brainiac.experimental.timeseries.TimesBlock;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerNorm;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.UnaryOperator;

/**
 * The shared multivariate forecasting harness: models map {@code [batch, lookback, channels]}
 * to {@code [batch, horizon, channels]}; univariate models are applied channel-independently.
 */
final class Forecasters {

    private Forecasters() {
    }

    /** A trainable forecaster. */
    record Model(String name, UnaryOperator<Var> forward, List<Var> parameters) {
        long size() {
            long n = 0;
            for (Var v : parameters) {
                n += v.shape().size();
            }
            return n;
        }
    }

    /** Applies a {@code [batch, lookback] -> [batch, horizon]} model to every channel. */
    static UnaryOperator<Var> channelIndependent(UnaryOperator<Var> univariate) {
        return x -> {
            int b = x.shape().dim(0);
            int l = x.shape().dim(1);
            int c = x.shape().dim(2);
            Var y = univariate.apply(x.permute(0, 2, 1).reshape(b * c, l));
            int h = y.shape().dim(1);
            return y.reshape(b, c, h).permute(0, 2, 1);
        };
    }

    /** Trains with Adam and MSE on random batches of 32 windows; returns the test MSE. */
    static double fit(Model m, Tensor trainX, Tensor trainY, Tensor testX, Tensor testY, int steps, float lr,
            long seed) {
        Optimizer opt = Optimizer.adam(lr);
        SplittableRandom r = new SplittableRandom(seed);
        int n = trainX.shape().dim(0);
        int b = Math.min(32, n);
        for (int s = 0; s < steps; s++) {
            float[] idx = new float[b];
            for (int i = 0; i < b; i++) {
                idx[i] = r.nextInt(n);
            }
            Tensor sel = Tensor.of(Shape.of(b), idx);
            Tensor bx = trainX.gather(0, sel);
            Tensor by = trainY.gather(0, sel);
            m.forward().apply(Var.of(bx)).sub(by).square().mean().backward();
            opt.step(m.parameters());
        }
        return predict(m, testX).sub(testY).square().mean().item();
    }

    /**
     * Trains like {@link #fit} (same batches, so the value at {@code checkpoints[i]} equals a
     * {@code fit} of that many steps) and records the test MSE at each checkpoint.
     */
    static double[] fitCurve(Model m, Tensor trainX, Tensor trainY, Tensor testX, Tensor testY, int[] checkpoints,
            float lr, long seed) {
        Optimizer opt = Optimizer.adam(lr);
        SplittableRandom r = new SplittableRandom(seed);
        int n = trainX.shape().dim(0);
        int b = Math.min(32, n);
        double[] out = new double[checkpoints.length];
        int next = 0;
        int last = checkpoints[checkpoints.length - 1];
        for (int s = 0; s <= last; s++) {
            while (next < checkpoints.length && checkpoints[next] == s) {
                out[next++] = predict(m, testX).sub(testY).square().mean().item();
            }
            if (s == last) {
                break;
            }
            float[] idx = new float[b];
            for (int i = 0; i < b; i++) {
                idx[i] = r.nextInt(n);
            }
            Tensor sel = Tensor.of(Shape.of(b), idx);
            m.forward().apply(Var.of(trainX.gather(0, sel))).sub(trainY.gather(0, sel)).square().mean().backward();
            opt.step(m.parameters());
        }
        return out;
    }

    /** Forecasts in chunks without gradients. */
    static Tensor predict(Model m, Tensor x) {
        int n = x.shape().dim(0);
        List<Tensor> parts = new ArrayList<>();
        for (int s = 0; s < n; s += 128) {
            Tensor chunk = x.slice(0, s, Math.min(n, s + 128));
            parts.add(Var.noGrad(() -> m.forward().apply(Var.of(chunk))).value());
        }
        return parts.size() == 1 ? parts.get(0) : Tensor.concat(0, parts.toArray(new Tensor[0]));
    }

    /** Per-channel window normalisation: {x_n, mean, std}. */
    static Var[] normalise(Var x) {
        Var mean = Var.of(x.value().mean(1, true));
        Var std = Var.of(x.value().sub(x.value().mean(1, true)).square().mean(1, true).add(1e-5f).sqrt());
        return new Var[] {x.sub(mean).div(std), mean, std};
    }

    /** The token mixer of an encoder-only transformer forecaster. */
    enum Mixer {
        /** Full self-attention (core {@link TransformerBlock}). */
        ATTENTION,
        /** Autoformer's auto-correlation with series decomposition. */
        AUTO_CORRELATION,
        /** FEDformer's Fourier block with series decomposition. */
        FOURIER
    }

    /**
     * An encoder-only transformer forecaster: window normalisation, (for the decomposition
     * mixers) a moving-average split whose trend goes through a linear time map, an embedding
     * with sinusoidal positions, {@code layers} encoder layers with the chosen mixer, and a
     * linear map over time and features to the horizon.
     */
    static Model transformer(String name, Mixer mixer, int lookback, int horizon, int channels, int width, int layers,
            int kernel, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<Var> params = new ArrayList<>();
        Var embed = param(params, Tensor.normal(Shape.of(channels, width), 0, 1f / (float) Math.sqrt(channels), r));
        Tensor pos = Functional.sinusoidal(lookback, width, 10000f, Tensor.zeros(Shape.of(1)).device())
                .mul(0.1f).reshape(1, lookback, width);
        Var timeHead = param(params, Tensor.normal(Shape.of(lookback, horizon), 0, 1f / (float) Math.sqrt(lookback),
                r));
        Var featHead = param(params, Tensor.normal(Shape.of(width, channels), 0, 1f / (float) Math.sqrt(width), r));
        Var trendHead = param(params, Tensor.full(Shape.of(lookback, horizon), 1f / lookback));
        SeriesDecomposition decomposition = new SeriesDecomposition(kernel);
        List<Layer> mixers = new ArrayList<>();
        List<Var[]> ffns = new ArrayList<>();
        List<LayerNorm> norms = new ArrayList<>();
        for (int l = 0; l < layers; l++) {
            Layer m = switch (mixer) {
                case ATTENTION -> TransformerBlock.of(2).ffnSize(2 * width).create(Shape.of(lookback, width), r);
                case AUTO_CORRELATION -> new AutoCorrelationLayer(width, 1.5, r);
                case FOURIER -> new FrequencyEnhancedBlock(lookback, width, 8, r);
            };
            mixers.add(m);
            params.addAll(m.parameters());
            if (mixer != Mixer.ATTENTION) {
                Var w1 = param(params, Tensor.normal(Shape.of(width, 2 * width), 0, (float) Math.sqrt(2.0 / width), r));
                Var w2 = param(params, Tensor.normal(Shape.of(2 * width, width), 0, (float) Math.sqrt(0.5 / width),
                        r));
                ffns.add(new Var[] {w1, w2});
            }
            LayerNorm norm = LayerNorm.of(width);
            params.addAll(norm.parameters());
            norms.add(norm);
        }
        boolean decompose = mixer != Mixer.ATTENTION;
        UnaryOperator<Var> f = x -> {
            Var[] n = normalise(x);
            Var season = n[0];
            Var trendOut = null;
            if (decompose) {
                Var[] st = decomposition.decompose(n[0]);
                season = st[0];
                trendOut = st[1].permute(0, 2, 1).matmul(trendHead).permute(0, 2, 1);
            }
            Var h = season.matmul(embed).add(pos);
            for (int l = 0; l < mixers.size(); l++) {
                if (decompose) {
                    h = decomposition.decompose(h.add(mixers.get(l).forward(h, true)))[0];
                    Var[] w = ffns.get(l);
                    h = decomposition.decompose(h.add(h.matmul(w[0]).relu().matmul(w[1])))[0];
                } else {
                    h = mixers.get(l).forward(h, true);
                }
                h = norms.get(l).forward(h);
            }
            Var y = h.permute(0, 2, 1).matmul(timeHead).permute(0, 2, 1).matmul(featHead);
            if (trendOut != null) {
                y = y.add(trendOut);
            }
            return y.mul(n[2]).add(n[1]);
        };
        return new Model(name, f, params);
    }

    /**
     * TimesNet for forecasting: window normalisation, a linear map of the lookback to
     * {@code lookback + horizon} steps, an embedding, {@link TimesBlock}s with layer norms, a
     * projection back to the channels, and the last {@code horizon} steps.
     */
    static Model timesNet(int lookback, int horizon, int channels, int width, int layers, int topK, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<Var> params = new ArrayList<>();
        int total = lookback + horizon;
        Var extend = param(params, Tensor.normal(Shape.of(lookback, total), 0, 1f / (float) Math.sqrt(lookback), r));
        Var embed = param(params, Tensor.normal(Shape.of(channels, width), 0, 1f / (float) Math.sqrt(channels), r));
        Var project = param(params, Tensor.normal(Shape.of(width, channels), 0, 1f / (float) Math.sqrt(width), r));
        List<TimesBlock> blocks = new ArrayList<>();
        List<LayerNorm> norms = new ArrayList<>();
        for (int l = 0; l < layers; l++) {
            TimesBlock b = new TimesBlock(width, width, topK, new int[] {1, 3}, r);
            blocks.add(b);
            params.addAll(b.parameters());
            LayerNorm norm = LayerNorm.of(width);
            norms.add(norm);
            params.addAll(norm.parameters());
        }
        UnaryOperator<Var> f = x -> {
            Var[] n = normalise(x);
            Var h = n[0].permute(0, 2, 1).matmul(extend).permute(0, 2, 1).matmul(embed);
            for (int l = 0; l < blocks.size(); l++) {
                h = norms.get(l).forward(blocks.get(l).forward(h));
            }
            Var y = h.matmul(project).slice(1, lookback, total);
            return y.mul(n[2]).add(n[1]);
        };
        return new Model("TimesNet", f, params);
    }

    static Var param(List<Var> params, Tensor init) {
        Var v = Var.param(init);
        params.add(v);
        return v;
    }
}
