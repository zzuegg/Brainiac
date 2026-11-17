package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.rl.Space;

import java.util.Arrays;
import java.util.SplittableRandom;

/**
 * The bit-flipping task of the HER paper (Andrychowicz et al., 2017,
 * <a href="https://arxiv.org/abs/1707.01495">arXiv:1707.01495</a>): flip {@code n} bits, one per
 * step, to match a random target pattern within {@code n} steps. The reward is {@code -1}
 * every step until the pattern matches, so random exploration almost never sees a
 * success for {@code n >= 8}: the canonical sparse-reward, goal-conditioned problem.
 * Observation: the bits followed by the goal bits.
 */
public final class BitFlip implements GoalEnvironment {

    private final int n;
    private final float[] bits;
    private final float[] goal;
    private int steps;
    private boolean over = true;

    /**
     * @param n the number of bits
     */
    public BitFlip(int n) {
        this.n = n;
        this.bits = new float[n];
        this.goal = new float[n];
    }

    @Override
    public Space observationSpace() {
        return Space.box(2 * n, 0, 1);
    }

    @Override
    public Space actionSpace() {
        return Space.discrete(n);
    }

    @Override
    public float[] reset(SplittableRandom random) {
        do {
            for (int i = 0; i < n; i++) {
                bits[i] = random.nextBoolean() ? 1 : 0;
                goal[i] = random.nextBoolean() ? 1 : 0;
            }
        } while (Arrays.equals(bits, goal));
        steps = 0;
        over = false;
        return observation();
    }

    private float[] observation() {
        float[] o = new float[2 * n];
        System.arraycopy(bits, 0, o, 0, n);
        System.arraycopy(goal, 0, o, n, n);
        return o;
    }

    @Override
    public Step step(float[] action) {
        if (over) {
            throw new IllegalStateException("the episode is over; call reset");
        }
        int a = Math.round(action[0]);
        bits[a] = 1 - bits[a];
        steps++;
        boolean done = Arrays.equals(bits, goal);
        boolean cut = !done && steps >= n;
        over = done || cut;
        return new Step(observation(), done ? 0 : -1, done, cut);
    }

    @Override
    public int goalSize() {
        return n;
    }

    @Override
    public float[] achievedGoal(float[] observation) {
        return Arrays.copyOf(observation, n);
    }

    @Override
    public float reward(float[] achieved, float[] desired) {
        return success(achieved, desired) ? 0 : -1;
    }

    @Override
    public boolean success(float[] achieved, float[] desired) {
        return Arrays.equals(achieved, desired);
    }

    @Override
    public String toString() {
        return "BitFlip(" + n + ")";
    }
}
