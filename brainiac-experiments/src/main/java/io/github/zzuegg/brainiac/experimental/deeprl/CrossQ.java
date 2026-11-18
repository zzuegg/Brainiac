package io.github.zzuegg.brainiac.experimental.deeprl;

import io.github.zzuegg.brainiac.experimental.nn.BatchRenorm;
import io.github.zzuegg.brainiac.experimental.rl.RlNetworkFactory;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;

/**
 * CrossQ (Bhatt et al., ICLR 2024, <a href="https://arxiv.org/abs/1902.05605">arXiv:1902.05605</a>):
 * batch normalisation instead of target networks. Batch normalisation was thought to break
 * off-policy critics because the statistics of the {@code (s, a)} batch and of the
 * {@code (s', pi(s'))} batch differ; CrossQ concatenates the two and passes them through the
 * critic in one forward pass, so the normalisation statistics cover both, then drops the target
 * network and widens the critic. At one update per environment step it matches REDQ and DroQ,
 * which need about twenty.
 *
 * <p>The critics here use {@link BatchRenorm} (as the paper does) after the input and after
 * every hidden layer. {@link #learner} gives the full recipe on {@link ModernSac};
 * {@link #naiveBatchNorm} is the ablation the paper warns about: the same normalised critics
 * fed the two batches separately, with target networks.
 */
public final class CrossQ {

    private CrossQ() {
    }

    /**
     * Critics with batch renormalisation: {@code BRN, (Dense, ReLU, BRN) x layers, Dense}.
     *
     * @param width the hidden width (the paper widens to 2048; 256 is small)
     * @param layers hidden layers
     * @return the factory
     */
    public static RlNetworkFactory critic(int width, int layers) {
        return (inputs, outputs, seed) -> {
            Model.Builder b = Model.sequential().input(inputs).add(BatchRenorm.spec(1_000));
            for (int i = 0; i < layers; i++) {
                b.dense(width, Activation.RELU).add(BatchRenorm.spec(1_000));
            }
            return b.dense(outputs).seed(seed).build();
        };
    }

    /**
     * CrossQ: joint-batch normalised critics, no target networks, UTD 1, and the paper's Adam
     * settings (learning rate 1e-3, first-moment decay 0.5).
     *
     * @param observationDims values per observation
     * @param actionDims values per action
     * @param actorWidth the actor's hidden width
     * @param criticWidth the critics' hidden width
     * @return the learner
     */
    public static ModernSac learner(int observationDims, int actionDims, int actorWidth, int criticWidth) {
        return ModernSac.of(observationDims, actionDims, RlNetworkFactory.mlp(Activation.RELU, actorWidth, actorWidth),
                critic(criticWidth, 2)).jointBatch(true).targetNetworks(false).learningRate(1e-3f).adamBeta1(0.5f);
    }

    /**
     * The ablation: the same normalised critics, but the current and next batches pass
     * separately and the next one through a Polyak target network.
     *
     * @param observationDims values per observation
     * @param actionDims values per action
     * @param actorWidth the actor's hidden width
     * @param criticWidth the critics' hidden width
     * @return the learner
     */
    public static ModernSac naiveBatchNorm(int observationDims, int actionDims, int actorWidth, int criticWidth) {
        return ModernSac.of(observationDims, actionDims, RlNetworkFactory.mlp(Activation.RELU, actorWidth, actorWidth),
                critic(criticWidth, 2)).jointBatch(false).targetNetworks(true).learningRate(1e-3f).adamBeta1(0.5f);
    }
}
