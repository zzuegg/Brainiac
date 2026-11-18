package io.github.zzuegg.brainiac.experimental.deeprl;

import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Space;

import java.util.SplittableRandom;

/**
 * The continuous mountain car (Moore, 1990; Gym's {@code MountainCarContinuous-v0}): an
 * underpowered car in a valley must rock back and forth to reach the flag at position 0.45.
 * The action is a force in {@code [-1, 1]}; every step costs {@code 0.1 a^2} and reaching the
 * flag pays 100 and ends the episode. The reward is sparse and the action cost makes "do
 * nothing" a strong local optimum, so it separates agents that explore (optimism, ensembles)
 * from ones that settle; the core {@link io.github.zzuegg.brainiac.rl.MountainCar} is the
 * discrete version.
 */
public final class ContinuousMountainCar implements Environment {

    private static final double POWER = 0.0015;
    private final int maxSteps;
    private double position;
    private double velocity;
    private int steps;
    private boolean over = true;

    /**
     * Creates the environment.
     *
     * @param maxSteps the step limit (Gym uses 999)
     */
    public ContinuousMountainCar(int maxSteps) {
        this.maxSteps = maxSteps;
    }

    @Override
    public Space observationSpace() {
        return Space.box(new float[] {-1.2f, -0.07f}, new float[] {0.6f, 0.07f});
    }

    @Override
    public Space actionSpace() {
        return Space.box(1, -1, 1);
    }

    @Override
    public float[] reset(SplittableRandom random) {
        position = random.nextDouble(-0.6, -0.4);
        velocity = 0;
        steps = 0;
        over = false;
        return observation();
    }

    private float[] observation() {
        // scaled to about [-1, 1] so networks need no input normalisation
        return new float[] {(float) ((position + 0.3) / 0.9), (float) (velocity / 0.07)};
    }

    @Override
    public Step step(float[] action) {
        if (over) {
            throw new IllegalStateException("the episode is over; call reset");
        }
        double force = Math.max(-1, Math.min(1, action[0]));
        velocity += force * POWER - 0.0025 * Math.cos(3 * position);
        velocity = Math.max(-0.07, Math.min(0.07, velocity));
        position += velocity;
        position = Math.max(-1.2, Math.min(0.6, position));
        if (position == -1.2 && velocity < 0) {
            velocity = 0;
        }
        steps++;
        boolean goal = position >= 0.45 && velocity >= 0;
        double reward = (goal ? 100 : 0) - 0.1 * force * force;
        boolean cut = !goal && steps >= maxSteps;
        over = goal || cut;
        return new Step(observation(), (float) reward, goal, cut);
    }

    @Override
    public String render() {
        return String.format("position %+.3f, velocity %+.4f", position, velocity);
    }

    @Override
    public String toString() {
        return "ContinuousMountainCar(" + maxSteps + ")";
    }
}
