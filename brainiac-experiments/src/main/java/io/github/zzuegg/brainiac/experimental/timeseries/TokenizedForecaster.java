package io.github.zzuegg.brainiac.experimental.timeseries;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A tiny Chronos (Ansari et al., 2024, <a href="https://arxiv.org/abs/2403.07815">arXiv:2403.07815</a>):
 * a decoder-only transformer language model over {@link ValueTokenizer} tokens. A series is
 * mean-scaled by its context, quantised, embedded with learned positions and passed through
 * causal core {@link TransformerBlock}s; training is next-token cross-entropy; a forecast is
 * a set of sampled token paths decoded back to values, so quantiles and intervals come from
 * the samples. The model always runs on a fixed {@code context + horizon - 1} input (unseen
 * positions are padding that the causal mask never lets earlier positions read).
 */
public final class TokenizedForecaster implements Layer {

    private final ValueTokenizer tokenizer;
    private final int context;
    private final int horizon;
    private final int steps;
    private final Var embed;
    private final Var position;
    private final List<Layer> blocks = new ArrayList<>();
    private final Var head;

    /**
     * Creates the model.
     *
     * @param tokenizer the value tokenizer
     * @param context the context length
     * @param horizon the forecast length
     * @param width the model width
     * @param layers transformer blocks
     * @param heads attention heads
     * @param random initialisation source
     */
    public TokenizedForecaster(ValueTokenizer tokenizer, int context, int horizon, int width, int layers, int heads,
            SplittableRandom random) {
        this.tokenizer = tokenizer;
        this.context = context;
        this.horizon = horizon;
        this.steps = context + horizon - 1;
        int v = tokenizer.vocabulary();
        embed = Var.param(Tensor.normal(Shape.of(v, width), 0, 0.3f, random));
        position = Var.param(Tensor.normal(Shape.of(steps, width), 0, 0.1f, random));
        for (int l = 0; l < layers; l++) {
            blocks.add(TransformerBlock.of(heads).ffnSize(2 * width).causal(true).rotary(1000f)
                    .create(Shape.of(steps, width), random));
        }
        head = Var.param(Tensor.normal(Shape.of(width, v), 0, (float) Math.sqrt(1.0 / width), random));
    }

    /** @return the tokenizer */
    public ValueTokenizer tokenizer() {
        return tokenizer;
    }

    /**
     * Next-token logits.
     *
     * @param input {@code [batch, context + horizon - 1]} token ids as floats
     * @param training whether in training mode
     * @return {@code [batch, steps, vocabulary]}
     */
    @Override
    public Var forward(Var input, boolean training) {
        int b = input.shape().dim(0);
        int v = tokenizer.vocabulary();
        Tensor oneHot = input.value().reshape(b * steps, 1).equalTo(Tensor.arange(v).to(input.value().device()))
                .to(embed.value().dtype());
        Var h = Var.of(oneHot).matmul(embed).reshape(b, steps, embed.shape().dim(1)).add(position);
        for (Layer block : blocks) {
            h = block.forward(h, training);
        }
        return h.matmul(head);
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    /**
     * The cross-entropy of one batch of windows.
     *
     * @param windows {@code context + horizon} values each
     * @return the mean next-token loss
     */
    public Var loss(float[][] windows) {
        int b = windows.length;
        float[] in = new float[b * steps];
        float[] target = new float[b * steps];
        for (int i = 0; i < b; i++) {
            double scale = ValueTokenizer.scale(Arrays.copyOf(windows[i], context));
            int[] ids = tokenizer.encode(windows[i], scale);
            for (int t = 0; t < steps; t++) {
                in[i * steps + t] = ids[t];
                target[i * steps + t] = ids[t + 1];
            }
        }
        Var logits = forward(Var.of(Tensor.of(Shape.of(b, steps), in)), true);
        return Functional.crossEntropy(logits.reshape(b * steps, tokenizer.vocabulary()),
                Tensor.of(Shape.of(b * steps), target));
    }

    /**
     * Pretrains with Adam on windows drawn by a generator.
     *
     * @param data draws one series of {@code context + horizon} values
     * @param iterations optimiser steps
     * @param batch windows per step
     * @param lr the learning rate
     * @return the final loss
     */
    public double fit(java.util.function.Supplier<float[]> data, int iterations, int batch, float lr) {
        Optimizer opt = Optimizer.adam(lr);
        double last = Double.NaN;
        for (int s = 0; s < iterations; s++) {
            float[][] w = new float[batch][];
            for (int i = 0; i < batch; i++) {
                w[i] = data.get();
            }
            Var l = loss(w);
            last = l.value().item();
            l.backward();
            opt.step(parameters());
        }
        return last;
    }

    /**
     * Samples forecast paths autoregressively.
     *
     * @param history at least {@code context} values (the last are used)
     * @param samples paths
     * @param temperature softmax temperature (1 for the model's distribution)
     * @param random the sampling source
     * @return {@code [samples][horizon]} values
     */
    public float[][] sample(float[] history, int samples, double temperature, SplittableRandom random) {
        float[] ctx = Arrays.copyOfRange(history, history.length - context, history.length);
        double scale = ValueTokenizer.scale(ctx);
        int[] ids = tokenizer.encode(ctx, scale);
        int v = tokenizer.vocabulary();
        float[] in = new float[samples * steps];
        for (int s = 0; s < samples; s++) {
            for (int t = 0; t < context; t++) {
                in[s * steps + t] = ids[t];
            }
        }
        float[][] out = new float[samples][horizon];
        for (int h = 0; h < horizon; h++) {
            Tensor x = Tensor.of(Shape.of(samples, steps), in.clone());
            float[] logits = Var.noGrad(() -> forward(Var.of(x), false)).value().toArray();
            int pos = context - 1 + h;
            for (int s = 0; s < samples; s++) {
                int base = (s * steps + pos) * v;
                double max = Double.NEGATIVE_INFINITY;
                for (int k = 0; k < v; k++) {
                    max = Math.max(max, logits[base + k]);
                }
                double[] p = new double[v];
                double z = 0;
                for (int k = 0; k < v; k++) {
                    p[k] = Math.exp((logits[base + k] - max) / temperature);
                    z += p[k];
                }
                double u = random.nextDouble() * z;
                int pick = v - 1;
                for (int k = 0; k < v; k++) {
                    u -= p[k];
                    if (u <= 0) {
                        pick = k;
                        break;
                    }
                }
                out[s][h] = (float) tokenizer.decode(pick, scale);
                if (pos + 1 < steps) {
                    in[s * steps + pos + 1] = pick;
                }
            }
        }
        return out;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(embed, position));
        for (Layer b : blocks) {
            p.addAll(b.parameters());
        }
        p.add(head);
        return p;
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(steps, tokenizer.vocabulary());
    }

    @Override
    public String toString() {
        return "TokenizedForecaster(context=" + context + ", horizon=" + horizon + ", vocabulary="
                + tokenizer.vocabulary() + ")";
    }
}
