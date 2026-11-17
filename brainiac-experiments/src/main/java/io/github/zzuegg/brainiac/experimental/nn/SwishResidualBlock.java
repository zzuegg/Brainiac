package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerNorm;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * The residual block of the 1000-layer contrastive RL networks (Wang et al.,
 * <a href="https://arxiv.org/abs/2503.14858">2503.14858</a>, NeurIPS 2025 best paper):
 * {@code layers} repetitions of Dense, LayerNorm and Swish, with one skip connection around
 * them all ({@code x + f(x)}). LayerNorm in every layer plus the residual path is what lets
 * these MLPs grow to hundreds of layers and keep improving goal-reaching. Shape-preserving;
 * see {@link #spec}.
 */
public final class SwishResidualBlock implements Layer {
    private final List<Dense> dense = new ArrayList<>();
    private final List<LayerNorm> norms = new ArrayList<>();

    /**
     * Creates a block.
     *
     * @param features the width
     * @param layers Dense-LayerNorm-Swish layers inside the skip (4 in the paper)
     * @param random the initialization randomness
     */
    public SwishResidualBlock(int features, int layers, SplittableRandom random) {
        for (int i = 0; i < layers; i++) {
            dense.add(new Dense(features, features, Activation.IDENTITY, Init.LECUN, random));
            norms.add(LayerNorm.of(features));
        }
    }

    /**
     * @param layers Dense-LayerNorm-Swish layers inside the skip
     * @return a spec for {@code Model.sequential().add(...)}
     */
    public static LayerSpec spec(int layers) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return input;
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new SwishResidualBlock(input.dim(-1), layers, random);
            }
        };
    }

    @Override
    public Var forward(Var input) {
        Var h = input;
        for (int i = 0; i < dense.size(); i++) {
            h = Activation.SILU.forward(norms.get(i).forward(dense.get(i).forward(h)));
        }
        return input.add(h);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        for (int i = 0; i < dense.size(); i++) {
            p.addAll(dense.get(i).parameters());
            p.addAll(norms.get(i).parameters());
        }
        return p;
    }

    @Override
    public String toString() {
        return "SwishResidualBlock(" + dense.size() + " layers)";
    }
}
