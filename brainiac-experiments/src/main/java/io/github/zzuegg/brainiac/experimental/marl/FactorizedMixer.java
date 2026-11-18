package io.github.zzuegg.brainiac.experimental.marl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;

/**
 * How agent utilities combine into a team value in centralised-training value factorisation:
 * the extension point next to {@code ValueDecomposition.Mixer} (whose values stay as they are)
 * that {@link CompleteFactorization}'s training loop plugs mixers into. Agents act greedily on
 * their own utilities, so a mixer should keep the individual-global-max property: the joint
 * greedy action of the team value is the tuple of the agents' greedy actions.
 */
public interface FactorizedMixer {

    /**
     * The team value of the chosen joint actions.
     *
     * @param utilities every agent's utility of every action {@code [batch, agents, actions]}
     * @param chosen the one-hot joint action {@code [batch, agents, actions]}
     * @param state the global state {@code [batch, stateSize]}
     * @return {@code [batch, 1]}
     */
    Var mix(Var utilities, Tensor chosen, Tensor state);

    /** @return the trainable parameters */
    List<Var> parameters();

    /**
     * An extra loss trained with the TD loss, such as QTRAN's consistency constraints. The
     * default is none.
     *
     * @param utilities the utilities {@code [batch, agents, actions]}
     * @param chosen the one-hot joint actions taken
     * @param greedy the one-hot greedy joint actions
     * @param state the global state
     * @return the loss, or null
     */
    default Var auxiliaryLoss(Var utilities, Tensor chosen, Tensor greedy, Tensor state) {
        return null;
    }
}
