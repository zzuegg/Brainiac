package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Space;
import java.util.SplittableRandom;

/**
 * A corridor with food at the left end and water at the right end, a world for
 * {@link HomeostaticEnvironment}. Actions: 0 left, 1 right, 2 consume. Consuming at an end
 * yields one unit of that resource ({@link #lastIntake()}: {@code {food, water}} times the
 * portion). The world itself gives no reward; the observation is the one-hot position.
 * Episodes are cut off after {@code maxSteps}.
 */
public final class ForagingCorridor implements Environment {

    private final int length;
    private final int maxSteps;
    private final float portion;
    private final float[] intake = new float[2];
    private int position;
    private int steps;

    private ForagingCorridor(int length, int maxSteps, float portion) {
        this.length = length;
        this.maxSteps = maxSteps;
        this.portion = portion;
    }

    /**
     * @param length cells
     * @param maxSteps the episode length
     * @param portion what one consume action yields
     * @return the world
     */
    public static ForagingCorridor of(int length, int maxSteps, float portion) {
        return new ForagingCorridor(length, maxSteps, portion);
    }

    /** @return what the last step yielded, {@code {food, water}} */
    public float[] lastIntake() {
        return intake.clone();
    }

    private float[] observation() {
        float[] x = new float[length];
        x[position] = 1;
        return x;
    }

    @Override
    public Space observationSpace() {
        return Space.box(length, 0, 1);
    }

    @Override
    public Space actionSpace() {
        return Space.discrete(3);
    }

    @Override
    public float[] reset(SplittableRandom random) {
        position = length / 2;
        steps = 0;
        intake[0] = 0;
        intake[1] = 0;
        return observation();
    }

    @Override
    public Step step(float[] action) {
        int a = Math.round(action[0]);
        intake[0] = 0;
        intake[1] = 0;
        if (a == 0) {
            position = Math.max(0, position - 1);
        } else if (a == 1) {
            position = Math.min(length - 1, position + 1);
        } else if (position == 0) {
            intake[0] = portion;
        } else if (position == length - 1) {
            intake[1] = portion;
        }
        steps++;
        return new Step(observation(), 0, false, steps >= maxSteps);
    }
}
