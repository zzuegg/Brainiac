package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.rl.Environment;

/**
 * An environment that reports which high-level events hold in the current state, such as
 * "at the coffee machine", for {@link RewardMachine}s that define rewards over sequences of
 * events. The label of the latest state is read after {@link #reset} or {@link #step}.
 */
public interface LabelledEnvironment extends Environment {

    /** @return the events of the current state, as a string of single-character propositions ("" for none) */
    String label();
}
