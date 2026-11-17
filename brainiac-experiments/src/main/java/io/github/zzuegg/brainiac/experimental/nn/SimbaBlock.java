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
 * The residual feed-forward block of SimBa (Lee et al., <a href="https://arxiv.org/abs/2410.09754">2410.09754</a>):
 * {@code x + W2 relu(W1 LayerNorm(x))} with an inverted bottleneck ({@code W1} widens by
 * {@code expansion}, 4 in the paper). Stacked after an observation normalizer
 * ({@link RunningObservationNorm}) and closed by a final LayerNorm, these blocks give deep RL
 * networks a bias towards simple functions, so scaling them up keeps improving returns
 * instead of overfitting the replay buffer. Shape-preserving, so it drops into
 * {@code Model.sequential()} via {@link #spec}.
 */
public final class SimbaBlock implements Layer {
    private final LayerNorm norm;
    private final Dense up;
    private final Dense down;

    /**
     * Creates a block.
     *
     * @param features the width
     * @param expansion the inner widening factor
     * @param random the initialization randomness
     */
    public SimbaBlock(int features, int expansion, SplittableRandom random) {
        norm = LayerNorm.of(features);
        up = new Dense(features, features * expansion, Activation.RELU, Init.HE, random);
        down = new Dense(features * expansion, features, Activation.IDENTITY, Init.HE, random);
    }

    /**
     * @param expansion the inner widening factor
     * @return a spec for {@code Model.sequential().add(...)}
     */
    public static LayerSpec spec(int expansion) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return input;
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new SimbaBlock(input.dim(-1), expansion, random);
            }
        };
    }

    @Override
    public Var forward(Var input) {
        return input.add(down.forward(up.forward(norm.forward(input))));
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(norm.parameters());
        p.addAll(up.parameters());
        p.addAll(down.parameters());
        return p;
    }

    @Override
    public String toString() {
        return "SimbaBlock";
    }
}
