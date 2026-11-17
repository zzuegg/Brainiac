package io.github.zzuegg.brainiac.experimental.rl;

import java.util.SplittableRandom;

/**
 * The two-step cooperative matrix game of the QMIX paper (Rashid et al., 2018,
 * <a href="https://arxiv.org/abs/1803.11485">arXiv:1803.11485</a>), the classic test of whether a
 * value decomposition can represent coordination.
 *
 * <p>Two agents, two actions. In the first step agent 1's action picks the second game:
 * 2A pays 7 whatever they do; 2B pays {@code [[0, 1], [1, 8]]}, 8 only if both pick action 1.
 * The optimum (8) needs a joint value that additive decompositions (independent learners,
 * VDN) cannot represent under exploratory data, so they settle for the safe 7. Each agent
 * observes the state (one-hot over start, 2A, 2B) plus its own one-hot id.
 */
public final class TwoStepGame implements MultiAgentEnvironment {

    private static final float[][] PAYOFF_B = {{0, 1}, {1, 8}};
    private int state;

    /** Creates the game. */
    public TwoStepGame() {
    }

    @Override
    public int agents() {
        return 2;
    }

    @Override
    public int observationSize() {
        return 5;
    }

    @Override
    public int stateSize() {
        return 3;
    }

    @Override
    public int actions() {
        return 2;
    }

    private float[][] observations() {
        float[][] o = new float[2][5];
        for (int i = 0; i < 2; i++) {
            o[i][state] = 1;
            o[i][3 + i] = 1;
        }
        return o;
    }

    @Override
    public float[][] reset(SplittableRandom random) {
        state = 0;
        return observations();
    }

    @Override
    public float[] state() {
        float[] s = new float[3];
        s[state] = 1;
        return s;
    }

    @Override
    public Step step(int[] actions) {
        if (state == 0) {
            state = actions[0] == 0 ? 1 : 2;
            return new Step(observations(), 0, false, false);
        }
        float r = state == 1 ? 7 : PAYOFF_B[actions[0]][actions[1]];
        return new Step(observations(), r, true, false);
    }

    @Override
    public String toString() {
        return "TwoStepGame";
    }
}
