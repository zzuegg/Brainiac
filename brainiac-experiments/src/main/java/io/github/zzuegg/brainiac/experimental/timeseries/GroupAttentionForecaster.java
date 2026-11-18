package io.github.zzuegg.brainiac.experimental.timeseries;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.QuantileLoss;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.MultiInputLayer;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Supplier;

/**
 * A tiny Chronos-2-style forecaster (Amazon, 2025, <a href="https://arxiv.org/abs/2510.15821">arXiv:2510.15821</a>):
 * a patch-based encoder over a group of related series (the target first, then covariates)
 * whose layers alternate <em>time attention</em> (within each series, over its patches) and
 * <em>group attention</em> (across the series of a group at the same patch). Every series
 * is standardised over its observed steps and squashed with {@code asinh}; a patch token is
 * its values and its observed mask, so past-only targets and known-future covariates share
 * one format. A quantile head reads the target's future patches and is trained with the
 * pinball loss. With a group of one it is a univariate forecaster; with covariates, group
 * attention carries their known future into the target's forecast in context, without
 * fine-tuning.
 *
 * <p>Inputs: {@code [values [batch, group, context + horizon], mask [batch, group, context +
 * horizon]]} (mask 1 where a value is observed); output {@code [batch, horizon, quantiles]}
 * on the target's original scale.
 */
public final class GroupAttentionForecaster implements MultiInputLayer {

    private final int context;
    private final int horizon;
    private final int patch;
    private final int patches;
    private final float[] quantiles;
    private final Var embed;
    private final Var embedBias;
    private final Var position;
    private final List<Layer> timeBlocks = new ArrayList<>();
    private final List<Layer> groupBlocks = new ArrayList<>();
    private final Var head;
    private final Var headBias;

    /**
     * Creates the model.
     *
     * @param context context steps (a multiple of {@code patch})
     * @param horizon forecast steps (a multiple of {@code patch})
     * @param patch the patch length
     * @param width the model width
     * @param layers pairs of time and group attention blocks
     * @param heads attention heads
     * @param quantiles the quantile levels predicted, ascending
     * @param random initialisation source
     */
    public GroupAttentionForecaster(int context, int horizon, int patch, int width, int layers, int heads,
            float[] quantiles, SplittableRandom random) {
        if (context % patch != 0 || horizon % patch != 0) {
            throw new IllegalArgumentException("context and horizon must be multiples of the patch length");
        }
        this.context = context;
        this.horizon = horizon;
        this.patch = patch;
        this.patches = (context + horizon) / patch;
        this.quantiles = quantiles.clone();
        embed = Var.param(Tensor.normal(Shape.of(2 * patch, width), 0, (float) Math.sqrt(1.0 / patch), random));
        embedBias = Var.param(Tensor.zeros(Shape.of(width)));
        position = Var.param(Tensor.normal(Shape.of(patches, width), 0, 0.1f, random));
        for (int l = 0; l < layers; l++) {
            timeBlocks.add(TransformerBlock.of(heads).ffnSize(2 * width).rotary(100f).create(Shape.of(patches, width),
                    random));
            groupBlocks.add(TransformerBlock.of(heads).ffnSize(2 * width).create(Shape.of(1, width), random));
        }
        head = Var.param(Tensor.normal(Shape.of(width, patch * quantiles.length), 0,
                (float) Math.sqrt(1.0 / width), random));
        headBias = Var.param(Tensor.zeros(Shape.of(patch * quantiles.length)));
    }

    /** @return the quantile levels */
    public float[] quantiles() {
        return quantiles.clone();
    }

    /** Standardised, squashed inputs and the target's statistics. */
    private record Prepared(Tensor features, float[] mean, float[] std, int batch, int group) {
    }

    private Prepared prepare(Tensor values, Tensor mask) {
        int b = values.shape().dim(0);
        int g = values.shape().dim(1);
        int t = context + horizon;
        float[] v = values.toArray();
        float[] m = mask.toArray();
        float[] feat = new float[b * g * patches * 2 * patch];
        float[] mean = new float[b];
        float[] std = new float[b];
        for (int i = 0; i < b * g; i++) {
            double s = 0;
            double c = 0;
            for (int k = 0; k < t; k++) {
                s += v[i * t + k] * m[i * t + k];
                c += m[i * t + k];
            }
            double mu = s / Math.max(1, c);
            double var = 0;
            for (int k = 0; k < t; k++) {
                var += m[i * t + k] * (v[i * t + k] - mu) * (v[i * t + k] - mu);
            }
            double sd = Math.sqrt(var / Math.max(1, c) + 1e-2);
            if (i % g == 0) {
                mean[i / g] = (float) mu;
                std[i / g] = (float) sd;
            }
            for (int k = 0; k < t; k++) {
                int p = k / patch;
                int o = ((i * patches + p) * 2 * patch) + (k % patch);
                double z = (v[i * t + k] - mu) / sd;
                feat[o] = (float) (m[i * t + k] * Math.log(z + Math.sqrt(z * z + 1)));
                feat[o + patch] = m[i * t + k];
            }
        }
        return new Prepared(Tensor.of(Shape.of(b, g, patches, 2 * patch), feat), mean, std, b, g);
    }

    /** Quantile forecasts of the target in standardised units, {@code [batch, horizon, quantiles]}. */
    private Var scaled(Prepared p, boolean training) {
        int b = p.batch();
        int g = p.group();
        int w = embed.shape().dim(1);
        Var h = Var.of(p.features()).matmul(embed).add(embedBias).add(position);      // [b, g, N, w]
        for (int l = 0; l < timeBlocks.size(); l++) {
            h = timeBlocks.get(l).forward(h.reshape(b * g, patches, w), training);
            h = h.reshape(b, g, patches, w).permute(0, 2, 1, 3).reshape(b * patches, g, w);
            h = groupBlocks.get(l).forward(h, training);
            h = h.reshape(b, patches, g, w).permute(0, 2, 1, 3);
        }
        int future = horizon / patch;
        Var target = h.slice(1, 0, 1).reshape(b, patches, w).slice(1, patches - future, patches);
        return target.matmul(head).add(headBias).reshape(b, horizon, quantiles.length);
    }

    @Override
    public Var forward(List<Var> inputs, boolean training) {
        Prepared p = prepare(inputs.get(0).value(), inputs.get(1).value());
        Var y = scaled(p, training);
        Tensor mean = Tensor.of(Shape.of(p.batch(), 1, 1), p.mean());
        Tensor std = Tensor.of(Shape.of(p.batch(), 1, 1), p.std());
        return y.mul(std).add(mean);
    }

    /**
     * The pinball loss of a batch of groups: the target (series 0) is observed only for its
     * context, covariates for all steps.
     *
     * @param groups {@code [batch][group][context + horizon]}, the same group size throughout
     * @return the mean pinball loss in the target's standardised units
     */
    public Var loss(float[][][] groups) {
        Tensor[] vm = tensors(groups);
        Prepared p = prepare(vm[0], vm[1]);
        int b = groups.length;
        float[] target = new float[b * horizon];
        for (int i = 0; i < b; i++) {
            for (int h = 0; h < horizon; h++) {
                target[i * horizon + h] = (groups[i][0][context + h] - p.mean()[i]) / p.std()[i];
            }
        }
        return QuantileLoss.pinball(scaled(p, true), Tensor.of(Shape.of(b, horizon, 1), target), quantiles);
    }

    private Tensor[] tensors(float[][][] groups) {
        int b = groups.length;
        int g = groups[0].length;
        int t = context + horizon;
        float[] v = new float[b * g * t];
        float[] m = new float[b * g * t];
        for (int i = 0; i < b; i++) {
            for (int j = 0; j < g; j++) {
                for (int k = 0; k < t; k++) {
                    boolean seen = j > 0 || k < context;
                    int o = (i * g + j) * t + k;
                    v[o] = seen ? groups[i][j][k] : 0;
                    m[o] = seen ? 1 : 0;
                }
            }
        }
        return new Tensor[] {Tensor.of(Shape.of(b, g, t), v), Tensor.of(Shape.of(b, g, t), m)};
    }

    /**
     * Pretrains with Adam on generated groups.
     *
     * @param data draws a batch of groups ({@code [batch][group][context + horizon]})
     * @param iterations optimiser steps
     * @param lr the learning rate
     * @return the last loss
     */
    public double fit(Supplier<float[][][]> data, int iterations, float lr) {
        Optimizer opt = Optimizer.adamW(lr, 0.01f);
        double last = Double.NaN;
        for (int s = 0; s < iterations; s++) {
            Var l = loss(data.get());
            last = l.value().item();
            l.backward();
            opt.step(parameters());
        }
        return last;
    }

    /**
     * Forecasts one target, optionally with covariates whose future is known.
     *
     * @param target the target's last {@code context} values (or more; the last are used)
     * @param covariates each {@code context + horizon} values aligned with the target window
     *     and its horizon (none for a univariate forecast)
     * @return {@code [horizon][quantiles]}
     */
    public float[][] forecast(float[] target, float[]... covariates) {
        float[][][] group = new float[1][1 + covariates.length][];
        float[] t = new float[context + horizon];
        System.arraycopy(target, target.length - context, t, 0, context);
        group[0][0] = t;
        for (int j = 0; j < covariates.length; j++) {
            group[0][j + 1] = covariates[j];
        }
        Tensor[] vm = tensors(group);
        float[] y = Var.noGrad(() -> forward(List.of(Var.of(vm[0]), Var.of(vm[1])), false)).value().toArray();
        float[][] out = new float[horizon][quantiles.length];
        for (int h = 0; h < horizon; h++) {
            for (int q = 0; q < quantiles.length; q++) {
                out[h][q] = y[h * quantiles.length + q];
            }
            java.util.Arrays.sort(out[h]);
        }
        return out;
    }

    @Override
    public Shape outputShape(List<Shape> inputs) {
        return Shape.of(horizon, quantiles.length);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(embed, embedBias, position));
        for (int l = 0; l < timeBlocks.size(); l++) {
            p.addAll(timeBlocks.get(l).parameters());
            p.addAll(groupBlocks.get(l).parameters());
        }
        p.addAll(List.of(head, headBias));
        return p;
    }

    @Override
    public String toString() {
        return "GroupAttentionForecaster(context=" + context + ", horizon=" + horizon + ", patch=" + patch + ")";
    }
}
