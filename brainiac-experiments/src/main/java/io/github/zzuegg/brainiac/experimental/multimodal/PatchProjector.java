package io.github.zzuegg.brainiac.experimental.multimodal;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * The LLaVA bridge (Liu et al., NeurIPS 2023): project a frozen vision encoder's patch
 * features into the language model's input embedding space with a linear layer (LLaVA) or a
 * two-layer GELU MLP (LLaVA-1.5), and place them before the text as a visual prefix. It is
 * the cheapest of the bridges: one or two matrices, every patch kept.
 *
 * <p>As a {@link Layer} it maps {@code [batch, patches, visionDim]} to
 * {@code [batch, patches, lmDim]}.
 *
 * <p>Paper: <a href="https://arxiv.org/abs/2304.08485">arXiv:2304.08485</a>.
 */
public final class PatchProjector implements Layer {

    private final int out;
    private final Var w1;
    private final Var b1;
    private final Var w2;
    private final Var b2;

    /**
     * Creates the projector.
     *
     * @param visionDim the patch feature width
     * @param lmDim the language model width
     * @param mlp whether to use the two-layer MLP instead of a linear map
     * @param seed the seed
     */
    public PatchProjector(int visionDim, int lmDim, boolean mlp, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        this.out = lmDim;
        this.w1 = Var.param(Tensor.normal(Shape.of(visionDim, lmDim), 0, (float) Math.sqrt(1.0 / visionDim), r));
        this.b1 = Var.param(Tensor.zeros(Shape.of(lmDim)));
        this.w2 = mlp ? Var.param(Tensor.normal(Shape.of(lmDim, lmDim), 0, (float) Math.sqrt(1.0 / lmDim), r)) : null;
        this.b2 = mlp ? Var.param(Tensor.zeros(Shape.of(lmDim))) : null;
    }

    /**
     * A spec for model builders.
     *
     * @param lmDim the output width
     * @param mlp whether to use the two-layer MLP
     * @return the spec
     */
    public static LayerSpec spec(int lmDim, boolean mlp) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                int[] d = input.toArray();
                d[d.length - 1] = lmDim;
                return Shape.of(d);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new PatchProjector(input.dim(input.rank() - 1), lmDim, mlp, random.nextLong());
            }
        };
    }

    @Override
    public Var forward(Var input) {
        Var h = input.matmul(w1).add(b1);
        return w2 == null ? h : Activation.GELU.forward(h).matmul(w2).add(b2);
    }

    @Override
    public int outputSize(int inputSize) {
        return out;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(w1, b1));
        if (w2 != null) {
            p.add(w2);
            p.add(b2);
        }
        return p;
    }

    @Override
    public String toString() {
        return "PatchProjector(" + (w2 == null ? "linear" : "MLP") + " -> " + out + ")";
    }
}
