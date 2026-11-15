package io.github.zzuegg.brainiac.rl;

import java.util.SplittableRandom;

/** Cuts episodes off after a number of steps; see {@link Environment#limit(int)}. */
final class TimeLimit implements Environment {

    private final Environment inner;
    private final int maxSteps;
    private int steps;

    TimeLimit(Environment inner, int maxSteps) {
        if (maxSteps < 1) {
            throw new IllegalArgumentException("maxSteps must be at least 1, got " + maxSteps);
        }
        this.inner = inner;
        this.maxSteps = maxSteps;
    }

    @Override
    public Space observationSpace() {
        return inner.observationSpace();
    }

    @Override
    public Space actionSpace() {
        return inner.actionSpace();
    }

    @Override
    public float[] reset(SplittableRandom random) {
        steps = 0;
        return inner.reset(random);
    }

    @Override
    public Step step(float[] action) {
        Step s = inner.step(action);
        steps++;
        return steps >= maxSteps && !s.terminated() ? new Step(s.observation(), s.reward(), false, true) : s;
    }

    @Override
    public String render() {
        return inner.render();
    }

    @Override
    public String toString() {
        return inner + " (at most " + maxSteps + " steps)";
    }
}
