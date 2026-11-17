package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Space;

import java.util.SplittableRandom;

/**
 * A point mass on a track with slippery pushes: a small control task whose dynamics are
 * <em>bimodal</em>, for judging generative world models. Actions: 0 push left, 1 coast, 2
 * push right. With probability {@code slip} a push goes the other way, so the next state
 * after a push has two possible values; a model that predicts the mean lands between them,
 * where the real system never is. Observation {@code [position, velocity]}, starting at
 * {@code (-0.6, 0)}; reward {@code -|position - 0.6|} per step; episodes last 30 steps.
 */
public final class SlipperyTrack implements Environment {

    private static final double GOAL = 0.6;
    private final double slip;
    private final int horizon;
    private SplittableRandom random;
    private double x;
    private double v;
    private int t;

    /**
     * @param slip the probability that a push is reversed
     * @param horizon steps per episode
     */
    public SlipperyTrack(double slip, int horizon) {
        this.slip = slip;
        this.horizon = horizon;
    }

    @Override
    public Space observationSpace() {
        return Space.box(new float[] {-2, -1}, new float[] {2, 1});
    }

    @Override
    public Space actionSpace() {
        return Space.discrete(3);
    }

    @Override
    public float[] reset(SplittableRandom r) {
        this.random = r;
        x = -GOAL;
        v = 0;
        t = 0;
        return new float[] {(float) x, (float) v};
    }

    private static double[] advance(double x, double v, double force) {
        double nv = 0.8 * v + 0.1 * force;
        double nx = Math.max(-2, Math.min(2, x + nv));
        return new double[] {nx, nv};
    }

    @Override
    public Step step(float[] action) {
        int a = Math.round(action[0]);
        double force = a - 1;
        if (force != 0 && random.nextDouble() < slip) {
            force = -force;
        }
        double[] n = advance(x, v, force);
        x = n[0];
        v = n[1];
        t++;
        return new Step(new float[] {(float) x, (float) v}, (float) -Math.abs(x - GOAL), false, t >= horizon);
    }

    /**
     * The possible next observations of an action, for checking a model's samples.
     *
     * @param observation the observation
     * @param action the action
     * @return one next observation for a coast, two (intended push and slip) for a push
     */
    public static float[][] outcomes(float[] observation, int action) {
        double force = action - 1;
        double[] a = advance(observation[0], observation[1], force);
        if (force == 0) {
            return new float[][] {{(float) a[0], (float) a[1]}};
        }
        double[] b = advance(observation[0], observation[1], -force);
        return new float[][] {{(float) a[0], (float) a[1]}, {(float) b[0], (float) b[1]}};
    }

    @Override
    public String toString() {
        return "SlipperyTrack(slip " + slip + ")";
    }
}
