package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Space;

import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Stacks the last {@code k} observations of an environment into one, the classic DQN way
 * (Mnih et al., 2015) of giving a memoryless agent a short history. Before the episode has
 * {@code k} frames the missing ones are zeros.
 */
public final class FrameStackEnvironment implements Environment {

    private final Environment inner;
    private final int frames;
    private final int size;
    private float[] stack;

    /**
     * @param inner the environment
     * @param frames how many observations to stack
     */
    public FrameStackEnvironment(Environment inner, int frames) {
        this.inner = Objects.requireNonNull(inner);
        if (frames < 1) {
            throw new IllegalArgumentException("frames must be at least 1");
        }
        this.frames = frames;
        this.size = inner.observationSpace().size();
    }

    @Override
    public Space observationSpace() {
        return Space.box(size * frames, Float.NEGATIVE_INFINITY, Float.POSITIVE_INFINITY);
    }

    @Override
    public Space actionSpace() {
        return inner.actionSpace();
    }

    @Override
    public float[] reset(SplittableRandom random) {
        stack = new float[size * frames];
        push(inner.reset(random));
        return stack.clone();
    }

    private void push(float[] o) {
        System.arraycopy(stack, size, stack, 0, size * (frames - 1));
        System.arraycopy(o, 0, stack, size * (frames - 1), size);
    }

    @Override
    public Step step(float[] action) {
        Step s = inner.step(action);
        push(s.observation());
        return new Step(stack.clone(), s.reward(), s.terminated(), s.truncated());
    }

    @Override
    public String render() {
        return inner.render();
    }

    @Override
    public String toString() {
        return inner + " x" + frames + " frames";
    }
}
