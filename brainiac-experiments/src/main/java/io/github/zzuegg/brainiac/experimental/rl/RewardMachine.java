package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.rl.Space;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * A reward machine (Toro Icarte et al., 2018/2022, <a href="https://arxiv.org/abs/2010.03950">arXiv:2010.03950</a>):
 * a finite-state machine over the event labels of a {@link LabelledEnvironment} that emits
 * the reward. Its state remembers how much of a temporally extended task is done, such as
 * "get coffee, then the mail, then go to the office", so the reward is Markov in
 * (environment state, machine state) even though it is not in the environment state alone.
 *
 * <p>{@link #environment} builds the product environment, with or without the machine state
 * in the observation; {@link RewardMachineLearner} learns on it with counterfactual
 * experience (CRM), updating every machine state from every transition.
 */
public final class RewardMachine {

    private record Edge(int from, char event, int to, float reward) {
    }

    private final int states;
    private final List<Edge> edges = new ArrayList<>();
    private final boolean[] terminal;

    /**
     * Creates a machine with the given number of states; state 0 is the initial one. Add
     * transitions with {@link #transition} and mark ends with {@link #terminal(int)}.
     *
     * @param states the number of states
     */
    public RewardMachine(int states) {
        this.states = states;
        this.terminal = new boolean[states];
    }

    /**
     * A machine for "do these events in order": each event advances one state, the last
     * pays {@code reward} and ends the task.
     *
     * @param reward the final reward
     * @param events the events in order, one character each
     * @return the machine
     */
    public static RewardMachine sequence(float reward, String events) {
        RewardMachine m = new RewardMachine(events.length() + 1);
        for (int i = 0; i < events.length(); i++) {
            m.transition(i, events.charAt(i), i + 1, i == events.length() - 1 ? reward : 0);
        }
        m.terminal(events.length());
        return m;
    }

    /**
     * @param from the state
     * @param event the event that triggers the transition
     * @param to the next state
     * @param reward the reward it emits
     * @return this machine
     */
    public RewardMachine transition(int from, char event, int to, float reward) {
        edges.add(new Edge(from, event, to, reward));
        return this;
    }

    /**
     * @param state a state that ends the task
     * @return this machine
     */
    public RewardMachine terminal(int state) {
        terminal[state] = true;
        return this;
    }

    /** @return the number of states */
    public int states() {
        return states;
    }

    /**
     * @param u a state
     * @return whether it ends the task
     */
    public boolean isTerminal(int u) {
        return terminal[u];
    }

    private Edge edge(int u, String label) {
        for (Edge e : edges) {
            if (e.from() == u && label.indexOf(e.event()) >= 0) {
                return e;
            }
        }
        return null;
    }

    /**
     * @param u the current state
     * @param label the events of the new environment state
     * @return the next machine state (unchanged when no transition matches)
     */
    public int next(int u, String label) {
        Edge e = edge(u, label);
        return e == null ? u : e.to();
    }

    /**
     * @param u the current state
     * @param label the events of the new environment state
     * @return the reward of the transition
     */
    public float reward(int u, String label) {
        Edge e = edge(u, label);
        return e == null ? 0 : e.reward();
    }

    /**
     * The product of an environment and this machine: rewards come from the machine, and
     * the episode ends when the machine reaches a terminal state.
     *
     * @param inner the labelled environment
     * @param exposeState whether to append the one-hot machine state to the observation
     *     (without it the reward is not Markov in the observation)
     * @return the product environment
     */
    public Product environment(LabelledEnvironment inner, boolean exposeState) {
        return new Product(this, inner, exposeState);
    }

    /** The product environment of a {@link RewardMachine} and a {@link LabelledEnvironment}. */
    public static final class Product implements LabelledEnvironment {

        private final RewardMachine machine;
        private final LabelledEnvironment inner;
        private final boolean expose;
        private int u;
        private float[] raw;
        private int completions;

        private Product(RewardMachine machine, LabelledEnvironment inner, boolean expose) {
            this.machine = machine;
            this.inner = Objects.requireNonNull(inner);
            this.expose = expose;
        }

        /** @return the machine */
        public RewardMachine machine() {
            return machine;
        }

        /** @return the current machine state */
        public int machineState() {
            return u;
        }

        /** @return the inner environment's latest observation */
        public float[] rawObservation() {
            return raw.clone();
        }

        /** @return how many episodes reached a terminal machine state */
        public int completions() {
            return completions;
        }

        private float[] observation() {
            if (!expose) {
                return raw.clone();
            }
            float[] o = Arrays.copyOf(raw, raw.length + machine.states());
            o[raw.length + u] = 1;
            return o;
        }

        @Override
        public Space observationSpace() {
            int n = inner.observationSpace().size() + (expose ? machine.states() : 0);
            return Space.box(n, 0, 1);
        }

        @Override
        public Space actionSpace() {
            return inner.actionSpace();
        }

        @Override
        public float[] reset(SplittableRandom random) {
            raw = inner.reset(random);
            u = machine.next(0, inner.label());
            return observation();
        }

        @Override
        public Step step(float[] action) {
            Step s = inner.step(action);
            raw = s.observation();
            String label = inner.label();
            float r = machine.reward(u, label);
            u = machine.next(u, label);
            boolean done = machine.isTerminal(u);
            if (done) {
                completions++;
            }
            return new Step(observation(), r, done || s.terminated(), !done && s.truncated());
        }

        @Override
        public String label() {
            return inner.label();
        }

        @Override
        public String render() {
            return inner.render() + " u=" + u;
        }
    }
}
