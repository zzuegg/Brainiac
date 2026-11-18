package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;

/**
 * Filter response normalisation with a thresholded linear unit (FRN + TLU) over
 * {@code [batch, channels, h, w]}: each channel of each example is divided by the root mean
 * square of its own responses, {@code x / sqrt(mean_hw(x^2) + eps)}, then scaled and shifted
 * per channel and passed through {@code max(y, tau)} with a learned threshold {@code tau}.
 * No mean is subtracted and no batch statistics are used, so it behaves the same at batch 1
 * as at batch 1024; the learned threshold repairs the bias that skipping mean-centering
 * would cause for a ReLU. It replaces both the norm and the activation.
 *
 * <p>Paper: Singh and Krishnan, "Filter Response Normalization Layer: Eliminating Batch
 * Dependence in the Training of Deep Neural Networks" (CVPR 2020),
 * <a href="https://arxiv.org/abs/1911.09737">arXiv:1911.09737</a>.
 */
public final class FilterResponseNorm implements Layer {

    private final int channels;
    private final float epsilon;
    private final Var gamma;
    private final Var beta;
    private final Var tau;

    /**
     * Creates the layer.
     *
     * @param channels the channels
     * @param epsilon the stabiliser, such as 1e-6
     */
    public FilterResponseNorm(int channels, float epsilon) {
        this.channels = channels;
        this.epsilon = epsilon;
        this.gamma = Var.param(Tensor.ones(Shape.of(1, channels, 1, 1)));
        this.beta = Var.param(Tensor.zeros(Shape.of(1, channels, 1, 1)));
        this.tau = Var.param(Tensor.zeros(Shape.of(1, channels, 1, 1)));
    }

    /**
     * Describes the layer for model builders on {@code [channels, h, w]}.
     *
     * @return the spec
     */
    public static LayerSpec spec() {
        return TechniqueOps.sameShape((s, r) -> new FilterResponseNorm(s.dim(0), 1e-6f));
    }

    @Override
    public Var forward(Var input) {
        ClassicOps.requireImages(input, channels, this);
        Var nu2 = input.square().mean(3, true).mean(2, true);
        Var y = input.div(nu2.add(epsilon).sqrt()).mul(gamma).add(beta);
        return y.maximum(tau);
    }

    @Override
    public List<Var> parameters() {
        return List.of(gamma, beta, tau);
    }

    @Override
    public String toString() {
        return "FilterResponseNorm(" + channels + ")";
    }
}
