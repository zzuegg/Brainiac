package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * A bottleneck adapter (Houlsby et al. 2019, "Parameter-Efficient Transfer Learning for
 * NLP", <a href="https://arxiv.org/abs/1902.00751">arXiv:1902.00751</a>):
 * {@code x + up(act(down(x)))} with a small bottleneck, inserted after a frozen sub-layer.
 * {@code up} starts at zero, so the adapter starts as the identity. Two per transformer
 * block (after attention and after the feed-forward network) is the original placement;
 * {@code HookedDecoder.adapters} puts them there.
 */
public final class HoulsbyAdapter implements Layer {

    private final Var down;
    private final Var downBias;
    private final Var up;
    private final Var upBias;
    private final Activation activation;

    /**
     * @param features the width of the stream
     * @param bottleneck the bottleneck width
     * @param activation the non-linearity
     * @param random the source of {@code down}'s initial values
     */
    public HoulsbyAdapter(int features, int bottleneck, Activation activation, RandomGenerator random) {
        this.down = Var.param(Tensor.normal(Shape.of(features, bottleneck), 0, (float) (1 / Math.sqrt(features)), random));
        this.downBias = Var.param(Tensor.zeros(Shape.of(bottleneck)));
        this.up = Var.param(Tensor.zeros(Shape.of(bottleneck, features)));
        this.upBias = Var.param(Tensor.zeros(Shape.of(features)));
        this.activation = activation;
    }

    /**
     * @param bottleneck the bottleneck width
     * @param activation the non-linearity
     * @return a spec for model builders
     */
    public static LayerSpec spec(int bottleneck, Activation activation) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return input;
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new HoulsbyAdapter(input.dim(-1), bottleneck, activation, random);
            }
        };
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        Var h = activation.forward(input.matmul(down).add(downBias), training);
        return input.add(h.matmul(up).add(upBias));
    }

    @Override
    public List<Var> parameters() {
        return List.of(down, downBias, up, upBias);
    }

    @Override
    public String toString() {
        return "HoulsbyAdapter(" + down.shape() + ")";
    }
}
