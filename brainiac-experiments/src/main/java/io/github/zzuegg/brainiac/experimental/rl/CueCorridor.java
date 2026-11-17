package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Space;

import java.util.SplittableRandom;

/**
 * Bakker's T-maze (Bakker, 2001, "Reinforcement learning with long short-term memory"), a
 * partially observable memory test: a cue (left or right) is shown only on the first step;
 * the agent then walks down a corridor of blank observations to a junction, where it must
 * turn the way the cue said. The correct turn pays +1, the wrong one -0.2 (as in Bakker's
 * setup, so that even a guess beats wandering), and both end the episode. A memoryless or
 * frame-stacked agent sees nothing that tells the turn apart and succeeds half the time; a
 * recurrent agent can carry the cue. Every other step costs 0.02.
 *
 * <p>Observations (4 values): cue says left, cue says right, in the corridor, at the junction.
 * Actions: 0 forward, 1 turn left, 2 turn right (turning in the corridor wastes a step).
 */
public final class CueCorridor implements Environment {

    private final int length;
    private int position;
    private int cue;
    private int steps;
    private int successes;
    private int episodes;

    /**
     * @param length steps from the start to the junction
     */
    public CueCorridor(int length) {
        if (length < 1) {
            throw new IllegalArgumentException("length must be at least 1");
        }
        this.length = length;
    }

    @Override
    public Space observationSpace() {
        return Space.box(4, 0, 1);
    }

    @Override
    public Space actionSpace() {
        return Space.discrete(3);
    }

    @Override
    public float[] reset(SplittableRandom random) {
        position = 0;
        steps = 0;
        cue = random.nextInt(2);
        float[] o = new float[4];
        o[cue] = 1;
        return o;
    }

    private float[] observation() {
        float[] o = new float[4];
        o[position >= length ? 3 : 2] = 1;
        return o;
    }

    @Override
    public Step step(float[] action) {
        int a = Math.round(action[0]);
        steps++;
        if (position >= length && a != 0) {
            boolean right = (a - 1) == cue;
            episodes++;
            if (right) {
                successes++;
            }
            return new Step(observation(), right ? 1 : -0.2f, true, false);
        }
        if (a == 0) {
            position = Math.min(length, position + 1);
        }
        boolean timeout = steps >= 2 * length + 4;
        if (timeout) {
            episodes++;
        }
        return new Step(observation(), -0.02f, false, timeout);
    }

    /** @return the share of finished episodes that ended with the correct turn */
    public double successRate() {
        return episodes == 0 ? 0 : successes / (double) episodes;
    }

    @Override
    public String toString() {
        return "CueCorridor(" + length + ")";
    }
}
