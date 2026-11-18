package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Space;

import java.util.SplittableRandom;

/**
 * A key-and-door grid world whose observation is a frame without memory: on a
 * {@code size x size} grid the agent must first step on the key, then on the door. The frame has
 * three channels (agent, key, door), flattened to {@code 3 size^2} values. The key is drawn only
 * during the first {@code reveal} steps of an episode; after that the key channel is empty
 * whether or not the key was taken, so where the key lies, and whether reaching the door pays,
 * depend on the past. A model or policy that only sees the current frame cannot tell, which is the
 * point: it separates world models with memory (transformers over frame tokens, recurrent
 * state-space models) from Markov ones.
 *
 * <p>Actions: 0 up, 1 down, 2 left, 3 right (moves off the grid are ignored). Rewards: 0.1 for
 * picking up the key, 1 for reaching the door with it (which ends the episode); the episode is
 * truncated after {@code maxSteps}. Agent, key and door start on distinct random cells.
 */
public final class KeyDoorGrid implements Environment {

    private final int size;
    private final int maxSteps;
    private final int reveal;
    private int agent;
    private int key;
    private int door;
    private boolean hasKey;
    private int steps;

    /**
     * Creates the environment.
     *
     * @param size the grid side
     * @param maxSteps the step limit
     * @param reveal the steps at the start of an episode during which the key is drawn
     */
    public KeyDoorGrid(int size, int maxSteps, int reveal) {
        this.size = size;
        this.maxSteps = maxSteps;
        this.reveal = reveal;
    }

    /** @return the observation size, {@code 3 size^2} */
    public int observationSize() {
        return 3 * size * size;
    }

    /** @return whether the agent holds the key (hidden from the observation) */
    public boolean hasKey() {
        return hasKey;
    }

    @Override
    public Space observationSpace() {
        return Space.box(observationSize(), 0, 1);
    }

    @Override
    public Space actionSpace() {
        return Space.discrete(4);
    }

    @Override
    public float[] reset(SplittableRandom random) {
        int cells = size * size;
        agent = random.nextInt(cells);
        do {
            key = random.nextInt(cells);
        } while (key == agent);
        do {
            door = random.nextInt(cells);
        } while (door == agent || door == key);
        hasKey = false;
        steps = 0;
        return observation();
    }

    private float[] observation() {
        int cells = size * size;
        float[] o = new float[3 * cells];
        o[agent] = 1;
        if (!hasKey && steps < reveal) {
            o[cells + key] = 1;
        }
        o[2 * cells + door] = 1;
        return o;
    }

    @Override
    public Step step(float[] action) {
        int a = Math.round(action[0]);
        int x = agent % size;
        int y = agent / size;
        switch (a) {
            case 0 -> y = Math.max(0, y - 1);
            case 1 -> y = Math.min(size - 1, y + 1);
            case 2 -> x = Math.max(0, x - 1);
            default -> x = Math.min(size - 1, x + 1);
        }
        agent = y * size + x;
        steps++;
        float reward = 0;
        boolean terminated = false;
        if (!hasKey && agent == key) {
            hasKey = true;
            reward = 0.1f;
        } else if (hasKey && agent == door) {
            reward = 1;
            terminated = true;
        }
        return new Step(observation(), reward, terminated, !terminated && steps >= maxSteps);
    }

    @Override
    public String toString() {
        return "KeyDoorGrid(" + size + "x" + size + ")";
    }
}
