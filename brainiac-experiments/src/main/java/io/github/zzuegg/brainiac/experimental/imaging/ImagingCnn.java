package io.github.zzuegg.brainiac.experimental.imaging;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.vision.VisionOps;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * The small fully convolutional network the learned imaging blocks share: {@code depth}
 * same-padded 3x3 convolutions with ReLU between them, from {@code inputs} to {@code outputs}
 * channels, optionally residual (the output is added to the first {@code outputs} input
 * channels) and with the last layer initialised small so a residual network starts near the
 * identity. Used as the image- and sinogram-domain updates of learned primal-dual, MoDL's
 * denoiser, the variational network's regulariser, the demosaicking network, the flow-matching
 * prior and the deep image prior's refinement.
 */
public final class ImagingCnn implements Layer {

    private final List<Var> weights = new ArrayList<>();
    private final List<Var> biases = new ArrayList<>();
    private final int outputs;
    private final boolean residual;

    /**
     * Creates the network.
     *
     * @param inputs the input channels
     * @param hidden the hidden channels
     * @param outputs the output channels
     * @param depth the number of convolutions, at least 1
     * @param residual whether to add the input's first {@code outputs} channels
     * @param random the source of initial weights
     */
    public ImagingCnn(int inputs, int hidden, int outputs, int depth, boolean residual, RandomGenerator random) {
        if (depth < 1) {
            throw new IllegalArgumentException("depth must be at least 1, got " + depth);
        }
        this.outputs = outputs;
        this.residual = residual;
        int c = inputs;
        for (int l = 0; l < depth; l++) {
            int o = l == depth - 1 ? outputs : hidden;
            Var w = VisionOps.convWeight(random, o, c, 3);
            if (l == depth - 1) {
                w = Var.param(w.value().mul(0.1f));
            }
            weights.add(w);
            biases.add(VisionOps.zeros(o));
            c = o;
        }
    }

    @Override
    public Var forward(Var input) {
        Var h = input;
        for (int l = 0; l < weights.size(); l++) {
            h = VisionOps.conv(h, weights.get(l), biases.get(l));
            if (l < weights.size() - 1) {
                h = h.relu();
            }
        }
        if (residual) {
            Var skip = input.shape().dim(1) == outputs ? input : input.slice(1, 0, outputs);
            h = h.add(skip);
        }
        return h;
    }

    /**
     * Runs the network with a per-example scalar (a time or noise level) appended as a constant
     * channel; the network must have been built with one extra input channel.
     *
     * @param input {@code [b, c, h, w]}
     * @param condition {@code [b]}
     * @return the output
     */
    public Var forward(Var input, Tensor condition) {
        Shape s = input.shape();
        int b = s.dim(0);
        int hw = s.dim(2) * s.dim(3);
        float[] c = condition.toArray();
        float[] plane = new float[b * hw];
        for (int i = 0; i < b; i++) {
            java.util.Arrays.fill(plane, i * hw, (i + 1) * hw, c[i]);
        }
        Var cond = Var.of(Tensor.of(Shape.of(b, 1, s.dim(2), s.dim(3)), plane));
        Var h = Var.concat(1, input, cond);
        for (int l = 0; l < weights.size(); l++) {
            h = VisionOps.conv(h, weights.get(l), biases.get(l));
            if (l < weights.size() - 1) {
                h = h.relu();
            }
        }
        if (residual) {
            h = h.add(input.shape().dim(1) == outputs ? input : input.slice(1, 0, outputs));
        }
        return h;
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(input.dim(0), outputs, input.dim(2), input.dim(3));
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(weights);
        p.addAll(biases);
        return p;
    }

    /** @return the number of trainable values */
    public long parameterCount() {
        long n = 0;
        for (Var v : parameters()) {
            n += v.value().size();
        }
        return n;
    }

    @Override
    public String toString() {
        return "ImagingCnn(" + weights.size() + " convs, residual " + residual + ")";
    }
}
