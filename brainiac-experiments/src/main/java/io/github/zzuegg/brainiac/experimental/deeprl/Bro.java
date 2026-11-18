package io.github.zzuegg.brainiac.experimental.deeprl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.rl.RlNetworkFactory;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerNorm;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * BRO, "bigger, regularised, optimistic" (Nauman et al., NeurIPS 2024,
 * <a href="https://arxiv.org/abs/2405.16158">arXiv:2405.16158</a>): critic size improves sample
 * efficiency only with strong regularisation. The BroNet is a residual MLP with LayerNorm after
 * every dense layer ({@code Dense, LN, ReLU}, then blocks {@code x + LN(Dense(ReLU(LN(Dense x))))}),
 * trained with weight decay and periodic full resets; clipped double Q is replaced by the
 * ensemble mean, the critics predict return quantiles, and a separate optimistic actor
 * (maximising mean plus an epistemic bonus, held near the pessimistic actor by an adaptive KL
 * penalty) collects the data.
 *
 * <p>{@link #network} is the BroNet as an {@link RlNetworkFactory}, {@link Block} its residual
 * block as a layer; {@link #learner} assembles the recipe on {@link ModernSac}.
 */
public final class Bro {

    private Bro() {
    }

    /**
     * The BroNet residual block {@code x + LN(Dense(ReLU(LN(Dense(x)))))}, shape-preserving.
     */
    public static final class Block implements Layer {
        private final Dense first;
        private final LayerNorm firstNorm;
        private final Dense second;
        private final LayerNorm secondNorm;

        /**
         * Creates a block.
         *
         * @param width the width
         * @param random the initialisation randomness
         */
        public Block(int width, SplittableRandom random) {
            first = new Dense(width, width, Activation.IDENTITY, Init.HE, random);
            firstNorm = LayerNorm.of(width);
            second = new Dense(width, width, Activation.IDENTITY, Init.HE, random);
            secondNorm = LayerNorm.of(width);
        }

        /** @return a spec for {@code Model.sequential().add(...)} */
        public static LayerSpec spec() {
            return new LayerSpec() {
                @Override
                public Shape outputShape(Shape input) {
                    return input;
                }

                @Override
                public Layer create(Shape input, SplittableRandom random) {
                    return new Block(input.dim(-1), random);
                }
            };
        }

        @Override
        public Var forward(Var input) {
            Var h = firstNorm.forward(first.forward(input)).relu();
            return input.add(secondNorm.forward(second.forward(h)));
        }

        @Override
        public List<Var> parameters() {
            List<Var> p = new ArrayList<>(first.parameters());
            p.addAll(firstNorm.parameters());
            p.addAll(second.parameters());
            p.addAll(secondNorm.parameters());
            return p;
        }

        @Override
        public String toString() {
            return "BroBlock";
        }
    }

    /**
     * BroNet: {@code Dense(width), LN, ReLU}, {@code blocks} residual blocks, a linear head.
     *
     * @param width the width (512 in the paper)
     * @param blocks residual blocks (2 in the paper)
     * @return the factory
     */
    public static RlNetworkFactory network(int width, int blocks) {
        return (inputs, outputs, seed) -> {
            Model.Builder b = Model.sequential().input(inputs).dense(width).layerNorm().activation(Activation.RELU);
            for (int i = 0; i < blocks; i++) {
                b.add(Block.spec());
            }
            return b.dense(outputs).seed(seed).build();
        };
    }

    /**
     * Which parts of the recipe to use, for ablations.
     *
     * @param optimism whether the optimistic exploration actor is on
     * @param resetEvery gradient updates between full resets, 0 for none
     * @param quantiles quantiles per critic, 0 for scalar critics
     * @param replayRatio gradient updates per environment step (BRO uses 10, BRO-Fast 2)
     */
    public record Options(boolean optimism, int resetEvery, int quantiles, int replayRatio) {
        /** @return the paper's defaults at a toy scale: optimism, resets every 2,500 updates, 16 quantiles, UTD 2 */
        public static Options defaults() {
            return new Options(true, 2_500, 16, 2);
        }
    }

    /**
     * BRO on {@link ModernSac}: BroNet critics and actor, ensemble-mean targets, quantile
     * critics, weight decay, resets and the optimistic actor.
     *
     * @param observationDims values per observation
     * @param actionDims values per action
     * @param width the critic width
     * @param options the parts in use
     * @return the learner
     */
    public static ModernSac learner(int observationDims, int actionDims, int width, Options options) {
        ModernSac sac = ModernSac.of(observationDims, actionDims, network(Math.min(width, 64), 1), network(width, 2))
                .critics(2).pessimism(0).quantiles(options.quantiles()).weightDecay(1e-4f)
                .resetEvery(options.resetEvery()).updateToData(options.replayRatio());
        if (options.optimism()) {
            sac.optimism(1f, 0.05f);
        }
        return sac;
    }
}
