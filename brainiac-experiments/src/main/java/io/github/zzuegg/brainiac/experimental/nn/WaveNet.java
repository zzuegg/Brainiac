package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * A WaveNet over 1-D signals {@code [batch, channels, steps]}: stacks of dilated
 * <em>causal</em> convolutions (left padding only, so step {@code t} never sees later steps)
 * with dilations {@code 1, 2, 4, ...}, so the receptive field grows exponentially with depth.
 * Each block applies a gated activation {@code tanh(f) * sigmoid(g)}, a 1x1 residual
 * projection, and a 1x1 skip projection; the summed skips go through ReLU and two 1x1
 * convolutions to the output channels at every step. Also a strong baseline for time series
 * and the basis of temporal convolutional networks (TCN).
 *
 * <p>Paper: van den Oord et al., "WaveNet: A Generative Model for Raw Audio" (2016),
 * <a href="https://arxiv.org/abs/1609.03499">arXiv:1609.03499</a>.
 */
public final class WaveNet implements Layer {

    private final int inputs;
    private final int outputs;
    private final int[] dilations;
    private final Var inWeight;
    private final List<Var[]> blocks = new ArrayList<>();
    private final Var head1;
    private final Var head2;

    /**
     * Creates the network.
     *
     * @param inputs input channels
     * @param residual channels in the residual stream
     * @param outputs output channels per step
     * @param layers dilated layers; dilation doubles each layer
     * @param random the random source
     */
    public WaveNet(int inputs, int residual, int outputs, int layers, RandomGenerator random) {
        this.inputs = inputs;
        this.outputs = outputs;
        this.dilations = new int[layers];
        inWeight = conv(random, residual, inputs, 1);
        for (int l = 0; l < layers; l++) {
            dilations[l] = 1 << l;
            blocks.add(new Var[] {conv(random, 2 * residual, residual, 2), TechniqueOps.filled(0f, 2 * residual),
                conv(random, residual, residual, 1), conv(random, residual, residual, 1)});
        }
        head1 = conv(random, residual, residual, 1);
        head2 = conv(random, outputs, residual, 1);
    }

    private static Var conv(RandomGenerator random, int out, int in, int k) {
        return Var.param(Tensor.normal(Shape.of(out, in, k), 0, (float) Math.sqrt(1.0 / (in * k)), random));
    }

    /**
     * A causal dilated 1-D convolution: left padding of {@code (k - 1) * dilation}.
     *
     * @param x {@code [batch, channels, steps]}
     * @param weight {@code [filters, channels, k]}
     * @param bias {@code [filters]} or {@code null}
     * @param dilation the dilation
     * @return {@code [batch, filters, steps]}
     */
    public static Var causalConv(Var x, Var weight, Var bias, int dilation) {
        int k = weight.shape().dim(2);
        return Functional.conv(x, weight, bias, new int[] {1}, new int[] {(k - 1) * dilation}, new int[] {0},
                new int[] {dilation}, 1);
    }

    /** @return how many steps (including the current one) each output depends on */
    public int receptiveField() {
        int r = 1;
        for (int d : dilations) {
            r += d;
        }
        return r;
    }

    @Override
    public Var forward(Var input) {
        if (input.shape().rank() != 3 || input.shape().dim(1) != inputs) {
            throw new IllegalArgumentException(this + " expects [batch, " + inputs + ", steps], got " + input.shape());
        }
        Var x = causalConv(input, inWeight, null, 1);
        Var skips = null;
        for (int l = 0; l < blocks.size(); l++) {
            Var[] w = blocks.get(l);
            Var h = causalConv(x, w[0], w[1], dilations[l]);
            int r = h.shape().dim(1) / 2;
            Var gated = h.slice(1, 0, r).tanh().mul(h.slice(1, r, 2 * r).sigmoid());
            x = x.add(causalConv(gated, w[2], null, 1));
            Var skip = causalConv(gated, w[3], null, 1);
            skips = skips == null ? skip : skips.add(skip);
        }
        return causalConv(causalConv(skips.relu(), head1, null, 1).relu(), head2, null, 1);
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(outputs, input.dim(1));
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(inWeight));
        for (Var[] b : blocks) {
            p.addAll(List.of(b));
        }
        p.add(head1);
        p.add(head2);
        return p;
    }

    @Override
    public String toString() {
        return "WaveNet(" + inputs + " -> " + outputs + ", " + dilations.length + " layers)";
    }
}
