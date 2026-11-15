package io.github.zzuegg.brainiac.rl;

import java.util.SplittableRandom;

/**
 * Drive an underpowered car out of a valley to the flag on the right hill: it must first
 * swing back and forth to build momentum. Actions: 0 push left, 1 coast, 2 push right.
 * Reward -1 per step until the car reaches position 0.5; episodes end after
 * {@code maxSteps} (200 by default). Observation: position, velocity. The reward is sparse,
 * so plain exploration rarely finds the goal; {@link #shaped(boolean) shaping} adds a
 * bonus for height reached.
 */
public final class MountainCar implements Environment {

    private final int maxSteps;
    private boolean shaped;
    private double position;
    private double velocity;
    private int steps;
    private boolean over = true;

    /** A mountain car cut off after 200 steps. */
    public MountainCar() {
        this(200);
    }

    /**
     * @param maxSteps the most steps per episode
     */
    public MountainCar(int maxSteps) {
        this.maxSteps = maxSteps;
    }

    /**
     * @param on whether to add a reward for gaining height and speed (potential-based
     *     shaping, so the best policy does not change)
     * @return this environment
     */
    public MountainCar shaped(boolean on) {
        this.shaped = on;
        return this;
    }

    @Override
    public Space observationSpace() {
        return Space.box(new float[] {-1.2f, -0.07f}, new float[] {0.6f, 0.07f});
    }

    @Override
    public Space actionSpace() {
        return Space.discrete(3);
    }

    @Override
    public float[] reset(SplittableRandom random) {
        position = random.nextDouble(-0.6, -0.4);
        velocity = 0;
        steps = 0;
        over = false;
        return new float[] {(float) position, (float) velocity};
    }

    private double potential() {
        return 0.45 * Math.sin(3 * position) + 50 * velocity * velocity;    // height plus kinetic energy
    }

    @Override
    public Step step(float[] action) {
        if (over) {
            throw new IllegalStateException("the episode is over; call reset");
        }
        int a = Math.round(action[0]);
        if (a < 0 || a > 2) {
            throw new IllegalArgumentException("mountain car actions are 0, 1 and 2, got " + action[0]);
        }
        double before = potential();
        velocity += (a - 1) * 0.001 - 0.0025 * Math.cos(3 * position);
        velocity = Math.max(-0.07, Math.min(0.07, velocity));
        position += velocity;
        position = Math.max(-1.2, Math.min(0.6, position));
        if (position == -1.2 && velocity < 0) {
            velocity = 0;
        }
        steps++;
        boolean goal = position >= 0.5;
        boolean cut = !goal && steps >= maxSteps;
        over = goal || cut;
        float reward = -1;
        if (shaped) {
            reward += (float) (0.99 * potential() - before) * 10;
        }
        return new Step(new float[] {(float) position, (float) velocity}, reward, goal, cut);
    }

    @Override
    public String render() {
        int width = 37;
        int at = (int) Math.round((position + 1.2) / 1.8 * (width - 1));
        StringBuilder s = new StringBuilder("_".repeat(width));
        s.setCharAt((int) Math.round((0.5 + 1.2) / 1.8 * (width - 1)), 'F');
        s.setCharAt(at, 'o');
        return s + String.format("  position %+.2f velocity %+.3f", position, velocity);
    }

    @Override
    public String toString() {
        return "MountainCar" + (shaped ? " (shaped)" : "");
    }
}
