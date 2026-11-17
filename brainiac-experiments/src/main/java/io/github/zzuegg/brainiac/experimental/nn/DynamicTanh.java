package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;

/**
 * Dynamic Tanh (DyT): {@code gamma * tanh(alpha * x) + beta} over the last axis, a drop-in
 * replacement for {@link io.github.zzuegg.brainiac.nn.LayerNorm} in transformers. Layer
 * normalization in trained transformers maps inputs to outputs along tanh-like S-curves;
 * DyT produces that curve directly with one learnable scalar {@code alpha}, without
 * computing any per-token statistics. It is element-wise, so it costs less than a norm and
 * needs no reduction across features.
 *
 * <p>Paper: Zhu, Chen, He, LeCun, Liu, "Transformers without Normalization" (2025),
 * <a href="https://arxiv.org/abs/2503.10622">arXiv:2503.10622</a>.
 */
public final class DynamicTanh implements Layer {

    private final int features;
    private final Var alpha;
    private final Var gamma;
    private final Var beta;

    /**
     * Creates the layer.
     *
     * @param features the size of the last axis
     * @param initialAlpha the starting input scale; the paper uses 0.5 (lower for deep or
     *     wide models)
     */
    public DynamicTanh(int features, float initialAlpha) {
        this.features = features;
        this.alpha = Var.param(Tensor.full(Shape.of(1), initialAlpha));
        this.gamma = Var.param(Tensor.ones(Shape.of(features)));
        this.beta = Var.param(Tensor.zeros(Shape.of(features)));
    }

    /**
     * Creates the layer with the paper's default {@code alpha = 0.5}.
     *
     * @param features the size of the last axis
     * @return the layer
     */
    public static DynamicTanh of(int features) {
        return new DynamicTanh(features, 0.5f);
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}; reads the width from the
     * input's last axis, like {@code layerNorm()}.
     *
     * @param initialAlpha the starting input scale
     * @return the spec
     */
    public static LayerSpec spec(float initialAlpha) {
        return TechniqueOps.sameShape((in, r) -> new DynamicTanh(in.dim(in.rank() - 1), initialAlpha));
    }

    @Override
    public Var forward(Var input) {
        if (input.shape().dim(-1) != features) {
            throw new IllegalArgumentException(this + " expects " + features + " features, got " + input.shape());
        }
        return input.mul(alpha).tanh().mul(gamma).add(beta);
    }

    /** @return the learnable scalar input scale */
    public Var alpha() {
        return alpha;
    }

    @Override
    public List<Var> parameters() {
        return List.of(alpha, gamma, beta);
    }

    @Override
    public String toString() {
        return "DynamicTanh(" + features + ")";
    }
}
