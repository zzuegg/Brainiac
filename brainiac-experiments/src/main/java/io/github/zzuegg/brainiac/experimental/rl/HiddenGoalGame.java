package io.github.zzuegg.brainiac.experimental.rl;

import java.util.SplittableRandom;

/**
 * A one-step cooperative game that needs communication: {@code n} agents must all pick the
 * same goal out of {@code k}, but only one agent, chosen at random each episode, is told which
 * goal it is. Every agent observes its one-hot id and a one-hot goal slot that is empty unless
 * it is the informed agent (with {@code fullObservation}, every agent is informed: the oracle).
 * The team reward is the fraction of agents that picked the goal, so without a channel the
 * uninformed agents can only guess (expected reward {@code (1 + (n - 1)/k) / n}).
 *
 * <p>The global state holds the goal and the informed agent, one-hot. A partially observed
 * stand-in for the predator-prey and traffic-junction tasks of CommNet and TarMAC: the
 * informed agent changes every episode, so a learned channel must also learn whom to listen
 * to.
 */
public final class HiddenGoalGame implements MultiAgentEnvironment {

    private final int n;
    private final int k;
    private final boolean full;
    private int goal;
    private int informed;

    /**
     * @param agents agents
     * @param goals goals
     * @param fullObservation whether every agent sees the goal (the oracle)
     */
    public HiddenGoalGame(int agents, int goals, boolean fullObservation) {
        this.n = agents;
        this.k = goals;
        this.full = fullObservation;
    }

    @Override
    public int agents() {
        return n;
    }

    @Override
    public int observationSize() {
        return n + k;
    }

    @Override
    public int stateSize() {
        return k + n;
    }

    @Override
    public int actions() {
        return k;
    }

    /** @return the current goal */
    public int goal() {
        return goal;
    }

    /** @return the agent that sees the goal this episode */
    public int informed() {
        return informed;
    }

    @Override
    public float[][] reset(SplittableRandom random) {
        goal = random.nextInt(k);
        informed = random.nextInt(n);
        float[][] obs = new float[n][n + k];
        for (int i = 0; i < n; i++) {
            obs[i][i] = 1;
            if (full || i == informed) {
                obs[i][n + goal] = 1;
            }
        }
        return obs;
    }

    @Override
    public float[] state() {
        float[] s = new float[k + n];
        s[goal] = 1;
        s[k + informed] = 1;
        return s;
    }

    @Override
    public Step step(int[] actions) {
        int right = 0;
        for (int a : actions) {
            right += a == goal ? 1 : 0;
        }
        return new Step(new float[n][n + k], right / (float) n, true, false);
    }

    @Override
    public String toString() {
        return "hidden goal (" + n + " agents, " + k + " goals" + (full ? ", fully observed" : "") + ")";
    }
}
