package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.List;
import java.util.SplittableRandom;

/**
 * The Universal Transformer: a single transformer block applied recurrently over depth,
 * with per-position Adaptive Computation Time. At every step the block receives the
 * "coordinate embedding", the sum of a sinusoidal position embedding and a sinusoidal
 * time-step embedding, so one set of weights can behave differently at each depth and
 * position. Weight sharing across depth gives a recurrent inductive bias (it generalizes
 * better on algorithmic tasks than a fixed-depth stack) and ACT lets each position use its
 * own number of steps.
 *
 * <p>Built from a core {@link TransformerBlock} inside a {@link LoopedTransformer} with
 * {@link LoopedTransformer.Halting#ACT}; the loop adds the time-step embedding and this
 * class adds the position embedding at each step.
 *
 * <p>Paper: Dehghani et al., "Universal Transformers",
 * <a href="https://arxiv.org/abs/1807.03819">arXiv:1807.03819</a>.
 */
public final class UniversalTransformer implements Layer {

    private final LoopedTransformer loop;

    /**
     * Creates the model.
     *
     * @param features the width
     * @param heads attention heads
     * @param causal whether attention is causal
     * @param maxSteps the most recurrent steps
     * @param random the random source
     */
    public UniversalTransformer(int features, int heads, boolean causal, int maxSteps, SplittableRandom random) {
        Layer core = TransformerBlock.of(heads).causal(causal).rmsNorm(true).create(Shape.of(1, features), random);
        Layer withPositions = new Layer() {
            @Override
            public Var forward(Var input) {
                return forward(input, false);
            }

            @Override
            public Var forward(Var input, boolean training) {
                int t = input.shape().dim(1);
                return core.forward(input.add(Functional.sinusoidal(t, features, 10000f, input.value().device())),
                        training);
            }

            @Override
            public List<Var> parameters() {
                return core.parameters();
            }
        };
        this.loop = new LoopedTransformer(withPositions, features, maxSteps, LoopedTransformer.Halting.ACT, random);
    }

    /**
     * Describes the model for {@code Model.sequential().add(...)}.
     *
     * @param heads attention heads
     * @param causal whether attention is causal
     * @param maxSteps the most recurrent steps
     * @return the spec
     */
    public static LayerSpec of(int heads, boolean causal, int maxSteps) {
        return MixerOps.sameShape((in, r) -> new UniversalTransformer(in.dim(-1), heads, causal, maxSteps, r));
    }

    /** @return the underlying loop, for the ponder cost and tick counts */
    public LoopedTransformer loop() {
        return loop;
    }

    @Override
    public Var forward(Var input) {
        return loop.forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        return loop.forward(input, training);
    }

    @Override
    public Var auxiliaryLoss() {
        return loop.auxiliaryLoss();
    }

    @Override
    public List<Var> parameters() {
        return loop.parameters();
    }

    @Override
    public String toString() {
        return "UniversalTransformer(" + loop + ")";
    }
}
