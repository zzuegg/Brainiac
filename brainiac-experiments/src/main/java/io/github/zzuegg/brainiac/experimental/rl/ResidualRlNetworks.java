package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.experimental.nn.RunningObservationNorm;
import io.github.zzuegg.brainiac.experimental.nn.SimbaBlock;
import io.github.zzuegg.brainiac.experimental.nn.SwishResidualBlock;
import io.github.zzuegg.brainiac.nn.Model;

/**
 * {@link RlNetworkFactory} recipes for the residual RL networks of 2024-2025, which made
 * scaling RL networks pay off:
 *
 * <ul>
 *   <li>{@link #simba}: SimBa (Lee et al., <a href="https://arxiv.org/abs/2410.09754">2410.09754</a>),
 *       running observation normalization, a linear embedding, pre-LayerNorm residual
 *       feed-forward blocks ({@link SimbaBlock}) and a final LayerNorm.</li>
 *   <li>{@link #deepResidual}: the very deep contrastive-RL networks (Wang et al.,
 *       <a href="https://arxiv.org/abs/2503.14858">2503.14858</a>), residual blocks of four
 *       Dense-LayerNorm-Swish layers ({@link SwishResidualBlock}).</li>
 * </ul>
 *
 * Every learner here that takes an {@link RlNetworkFactory} (SAC, TD3, IQL, ...) can use them
 * in place of {@link RlNetworkFactory#mlp}.
 */
public final class ResidualRlNetworks {
    private ResidualRlNetworks() {
    }

    /**
     * SimBa networks.
     *
     * @param width the residual stream width
     * @param blocks the number of residual blocks
     * @return the factory
     */
    public static RlNetworkFactory simba(int width, int blocks) {
        return (inputs, outputs, seed) -> {
            Model.Builder b = Model.sequential().input(inputs).add(RunningObservationNorm.spec()).dense(width);
            for (int i = 0; i < blocks; i++) {
                b.add(SimbaBlock.spec(4));
            }
            return b.layerNorm().dense(outputs).seed(seed).build();
        };
    }

    /**
     * Deep residual networks with LayerNorm and Swish.
     *
     * @param width the width
     * @param depth the number of Dense layers inside residual blocks (a multiple of 4)
     * @return the factory
     */
    public static RlNetworkFactory deepResidual(int width, int depth) {
        if (depth < 4 || depth % 4 != 0) {
            throw new IllegalArgumentException("depth must be a positive multiple of 4");
        }
        return (inputs, outputs, seed) -> {
            Model.Builder b = Model.sequential().input(inputs).dense(width);
            for (int i = 0; i < depth / 4; i++) {
                b.add(SwishResidualBlock.spec(4));
            }
            return b.dense(outputs).seed(seed).build();
        };
    }
}
