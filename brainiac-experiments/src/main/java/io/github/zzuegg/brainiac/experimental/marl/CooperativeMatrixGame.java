package io.github.zzuegg.brainiac.experimental.marl;

import io.github.zzuegg.brainiac.experimental.rl.MultiAgentEnvironment;

import java.util.SplittableRandom;

/**
 * One-step cooperative matrix games for two agents with a shared payoff, the standard tests of
 * whether value factorisation can represent non-monotonic joint values (Claus &amp; Boutilier,
 * 1998):
 *
 * <ul>
 *   <li>{@link #climbing}: {@code [[11, -30, 0], [-30, 7, 6], [0, 0, 5]]}; the optimum (11) sits
 *       next to the worst penalty, so learners settle for the safe 7 or 5;</li>
 *   <li>{@link #penalty}: {@code [[k, 0, 10], [0, 2, 0], [10, 0, k]]} with {@code k < 0}: two
 *       optimal joint actions and a penalty for miscoordinating between them.</li>
 * </ul>
 *
 * <p>Each agent observes only its one-hot id (and a constant bias); the global state is a
 * single constant.
 */
public final class CooperativeMatrixGame implements MultiAgentEnvironment {

    private final String name;
    private final float[][] payoff;

    /**
     * @param name a label
     * @param payoff the shared payoff, rows for agent 0's action and columns for agent 1's
     */
    public CooperativeMatrixGame(String name, float[][] payoff) {
        this.name = name;
        this.payoff = payoff;
    }

    /** @return the climbing game */
    public static CooperativeMatrixGame climbing() {
        return new CooperativeMatrixGame("climbing", new float[][] {{11, -30, 0}, {-30, 7, 6}, {0, 0, 5}});
    }

    /**
     * @param k the miscoordination penalty, negative
     * @return the penalty game
     */
    public static CooperativeMatrixGame penalty(float k) {
        return new CooperativeMatrixGame("penalty(" + k + ")", new float[][] {{k, 0, 10}, {0, 2, 0}, {10, 0, k}});
    }

    /** @return the payoff matrix */
    public float[][] payoff() {
        return payoff;
    }

    /** @return the best payoff */
    public float optimum() {
        float best = Float.NEGATIVE_INFINITY;
        for (float[] row : payoff) {
            for (float v : row) {
                best = Math.max(best, v);
            }
        }
        return best;
    }

    @Override
    public int agents() {
        return 2;
    }

    @Override
    public int observationSize() {
        return 3;
    }

    @Override
    public int stateSize() {
        return 1;
    }

    @Override
    public int actions() {
        return payoff.length;
    }

    private float[][] observations() {
        return new float[][] {{1, 0, 1}, {0, 1, 1}};
    }

    @Override
    public float[][] reset(SplittableRandom random) {
        return observations();
    }

    @Override
    public float[] state() {
        return new float[] {1};
    }

    @Override
    public Step step(int[] actions) {
        return new Step(observations(), payoff[actions[0]][actions[1]], true, false);
    }

    @Override
    public String toString() {
        return name;
    }
}
