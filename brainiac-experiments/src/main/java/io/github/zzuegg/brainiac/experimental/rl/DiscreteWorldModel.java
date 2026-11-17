package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.rl.Transition;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;

/**
 * A learned environment model for discrete actions that an agent can dream in, such as
 * {@link MlpWorldModel} or {@link DiffusionWorldModel}; {@link DreamerAgent} takes any.
 */
public interface DiscreteWorldModel {

    /**
     * One training step.
     *
     * @param batch real transitions
     * @return the loss
     */
    float train(List<Transition> batch);

    /**
     * An imagined step for a batch.
     *
     * @param obs observations {@code [batch, observations]}
     * @param acts the actions
     * @return the predicted next observations, rewards and continuation probabilities
     */
    MlpWorldModel.Prediction predict(Tensor obs, int[] acts);
}
