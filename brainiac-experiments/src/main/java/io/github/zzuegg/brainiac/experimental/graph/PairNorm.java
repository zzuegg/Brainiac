package io.github.zzuegg.brainiac.experimental.graph;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.SplittableRandom;

/**
 * PairNorm (Zhao and Akoglu, 2020, <a href="https://arxiv.org/abs/1909.12223">arXiv:1909.12223</a>):
 * a normalisation for node embeddings {@code [n, f]} that keeps the total pairwise
 * distance between nodes constant, so stacked graph convolutions cannot shrink every
 * node onto one point (over-smoothing):
 * <pre>
 * x_c = x - mean_nodes(x),   out = s * x_c / sqrt(mean_nodes |x_c|^2)
 * </pre>
 * It has no parameters; {@code s} sets the scale. Placed after every graph convolution.
 */
public final class PairNorm implements Layer {

    private final float scale;

    /**
     * Creates the normalisation.
     *
     * @param scale the target root-mean-square norm {@code s} (1 in the paper)
     */
    public PairNorm(float scale) {
        this.scale = scale;
    }

    /**
     * @param scale the target norm
     * @return a spec for model builders
     */
    public static LayerSpec spec(float scale) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return input;
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new PairNorm(scale);
            }
        };
    }

    @Override
    public Var forward(Var input) {
        Var c = input.sub(input.mean(0, true));
        return c.mul(scale).div(c.square().sum(1, true).mean().add(1e-6f).sqrt());
    }

    @Override
    public Shape outputShape(Shape inputShape) {
        return inputShape;
    }

    @Override
    public String toString() {
        return "PairNorm(" + scale + ")";
    }
}
