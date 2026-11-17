package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;

/**
 * A learnable scale on a residual branch, applied before the residual add:
 * {@code x + scale * f(x)}. Starting the scale small lets a deep stack begin close to the
 * identity and grow each block's contribution as it learns, which trains deep transformers
 * without warm-up tricks.
 *
 * <ul>
 *   <li>{@link #perChannel(int, float)} is <b>LayerScale</b> (CaiT, DINOv2, ConvNeXt): one
 *       scale per channel, starting at a small {@code epsilon} such as {@code 1e-5..0.1}.
 *       Touvron et al., <a href="https://arxiv.org/abs/2103.17239">arXiv:2103.17239</a>.</li>
 *   <li>{@link #rezero()} is <b>ReZero</b>: one scalar per branch that starts at exactly 0,
 *       so every block starts as the identity and no normalization is needed. Bachlechner
 *       et al., <a href="https://arxiv.org/abs/2003.04887">arXiv:2003.04887</a>.</li>
 * </ul>
 */
public final class LayerScale implements Layer {

    private final Var scale;

    private LayerScale(Var scale) {
        this.scale = scale;
    }

    /**
     * LayerScale: one learnable scale per channel (the last axis).
     *
     * @param features the channels
     * @param epsilon the starting value of every scale
     * @return the layer
     */
    public static LayerScale perChannel(int features, float epsilon) {
        return new LayerScale(Var.param(Tensor.full(Shape.of(features), epsilon)));
    }

    /**
     * ReZero: one learnable scalar that starts at 0.
     *
     * @return the layer
     */
    public static LayerScale rezero() {
        return new LayerScale(Var.param(Tensor.zeros(Shape.of(1))));
    }

    /**
     * Describes a per-channel LayerScale for {@code Model.sequential().add(...)}.
     *
     * @param epsilon the starting scale
     * @return the spec
     */
    public static LayerSpec spec(float epsilon) {
        return TechniqueOps.sameShape((in, r) -> perChannel(in.dim(in.rank() - 1), epsilon));
    }

    @Override
    public Var forward(Var input) {
        return input.mul(scale);
    }

    /** @return the learnable scale */
    public Var scale() {
        return scale;
    }

    @Override
    public List<Var> parameters() {
        return List.of(scale);
    }

    @Override
    public String toString() {
        return scale.shape().size() == 1 ? "ReZero" : "LayerScale(" + scale.shape().dim(0) + ")";
    }
}
