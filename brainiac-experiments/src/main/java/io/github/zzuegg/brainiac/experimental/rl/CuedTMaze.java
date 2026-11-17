package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Space;
import java.util.SplittableRandom;

/**
 * The cued T-maze of the active-inference literature (Friston et al. 2015): the agent starts
 * at the centre; the reward is in the left or the right arm, chosen at random each episode;
 * a cue at the bottom arm tells which. Entering an arm ends the episode with +1 (reward) or
 * -1; the agent has {@code 2} moves, so it can afford to read the cue first.
 *
 * <p>Locations: 0 centre, 1 left arm, 2 right arm, 3 cue. Actions: move to location 0..3.
 * Observations (one-hot): 0 nothing, 1 cue says left, 2 cue says right, 3 reward, 4 no reward.
 * {@link #model} gives the matching generative model for an {@link ActiveInferenceAgent},
 * over hidden states {@code location * 2 + context} (context 0: reward on the left).
 */
public final class CuedTMaze implements Environment {

    /** Observation: nothing. */
    public static final int NOTHING = 0;
    /** Observation: reward. */
    public static final int REWARD = 3;
    /** Observation: no reward. */
    public static final int NO_REWARD = 4;

    private final double cueReliability;
    private int context;
    private int steps;
    private SplittableRandom random = new SplittableRandom(0);

    private CuedTMaze(double cueReliability) {
        this.cueReliability = cueReliability;
    }

    /**
     * @param cueReliability how often the cue tells the truth
     * @return the maze
     */
    public static CuedTMaze of(double cueReliability) {
        return new CuedTMaze(cueReliability);
    }

    /**
     * The generative model {@code {A, B, C, D}} of this maze as an agent would hold it.
     *
     * @param cueReliability the reliability the agent assumes for the cue
     * @param preference how strongly reward is preferred (log-probability units)
     * @return the model
     */
    public static GenerativeModel model(double cueReliability, double preference) {
        double[][] a = new double[5][8];
        double[][][] b = new double[4][8][8];
        for (int loc = 0; loc < 4; loc++) {
            for (int ctx = 0; ctx < 2; ctx++) {
                int s = loc * 2 + ctx;
                switch (loc) {
                    case 0 -> a[NOTHING][s] = 1;
                    case 3 -> {
                        a[1][s] = ctx == 0 ? cueReliability : 1 - cueReliability;
                        a[2][s] = ctx == 0 ? 1 - cueReliability : cueReliability;
                    }
                    default -> {
                        boolean good = (loc == 1) == (ctx == 0);
                        a[REWARD][s] = good ? 0.98 : 0.02;
                        a[NO_REWARD][s] = good ? 0.02 : 0.98;
                    }
                }
                for (int act = 0; act < 4; act++) {
                    int to = loc == 1 || loc == 2 ? loc : act;
                    b[act][to * 2 + ctx][s] = 1;
                }
            }
        }
        double[] c = new double[5];
        c[REWARD] = preference;
        c[NO_REWARD] = -preference;
        double[] d = new double[8];
        d[0] = 0.5;
        d[1] = 0.5;
        return new GenerativeModel(a, b, c, d);
    }

    /**
     * A discrete generative model.
     *
     * @param a the likelihood {@code A[o][s]}
     * @param b the transitions {@code B[action][s'][s]}
     * @param c preferred outcomes, unnormalized log-probabilities
     * @param d the initial state prior
     */
    public record GenerativeModel(double[][] a, double[][][] b, double[] c, double[] d) {

        /**
         * @param horizon the planning horizon
         * @param seed the seed
         * @return an agent holding this model
         */
        public ActiveInferenceAgent agent(int horizon, long seed) {
            return ActiveInferenceAgent.of(a, b, c, d, horizon, seed);
        }
    }

    private static float[] observe(int o) {
        float[] x = new float[5];
        x[o] = 1;
        return x;
    }

    @Override
    public Space observationSpace() {
        return Space.box(5, 0, 1);
    }

    @Override
    public Space actionSpace() {
        return Space.discrete(4);
    }

    /** @return this episode's context: 0 when the reward is on the left */
    public int context() {
        return context;
    }

    @Override
    public float[] reset(SplittableRandom random) {
        this.random = random;
        context = random.nextInt(2);
        steps = 0;
        return observe(NOTHING);
    }

    @Override
    public Step step(float[] action) {
        int a = Math.round(action[0]);
        steps++;
        if (a == 1 || a == 2) {
            boolean good = (a == 1) == (context == 0);
            return new Step(observe(good ? REWARD : NO_REWARD), good ? 1 : -1, true, false);
        }
        int o = NOTHING;
        if (a == 3) {
            boolean truth = random.nextDouble() < cueReliability;
            o = (context == 0) == truth ? 1 : 2;
        }
        return new Step(observe(o), 0, false, steps >= 2);
    }
}
