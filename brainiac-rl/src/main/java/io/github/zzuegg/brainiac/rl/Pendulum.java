package io.github.zzuegg.brainiac.rl;

import java.util.SplittableRandom;

/**
 * Swing a pendulum up and hold it upright with a limited torque in {@code [-2, 2]}: a
 * continuous-action task. Observation: {@code cos(angle)}, {@code sin(angle)}, angular
 * velocity. Reward: {@code -(angle^2 + 0.1 velocity^2 + 0.001 torque^2)}, so 0 is best.
 * Episodes last {@code maxSteps} (200 by default) and are always truncated.
 */
public final class Pendulum implements Environment {

    private static final double MAX_SPEED = 8;
    private static final double MAX_TORQUE = 2;
    private static final double DT = 0.05;
    private static final double G = 10;

    private final int maxSteps;
    private double angle;
    private double speed;
    private int steps;
    private boolean over = true;

    /** A pendulum episode of 200 steps. */
    public Pendulum() {
        this(200);
    }

    /**
     * @param maxSteps the steps per episode
     */
    public Pendulum(int maxSteps) {
        this.maxSteps = maxSteps;
    }

    @Override
    public Space observationSpace() {
        return Space.box(new float[] {-1, -1, -8}, new float[] {1, 1, 8});
    }

    @Override
    public Space actionSpace() {
        return Space.box(1, -2, 2);
    }

    @Override
    public float[] reset(SplittableRandom random) {
        angle = random.nextDouble(-Math.PI, Math.PI);
        speed = random.nextDouble(-1, 1);
        steps = 0;
        over = false;
        return observation();
    }

    private float[] observation() {
        return new float[] {(float) Math.cos(angle), (float) Math.sin(angle), (float) speed};
    }

    @Override
    public Step step(float[] action) {
        if (over) {
            throw new IllegalStateException("the episode is over; call reset");
        }
        double u = Math.max(-MAX_TORQUE, Math.min(MAX_TORQUE, action[0]));
        double normalized = ((angle + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
        double cost = normalized * normalized + 0.1 * speed * speed + 0.001 * u * u;
        speed += (3 * G / 2 * Math.sin(angle) + 3.0 * u) * DT;
        speed = Math.max(-MAX_SPEED, Math.min(MAX_SPEED, speed));
        angle += speed * DT;
        steps++;
        over = steps >= maxSteps;
        return new Step(observation(), (float) -cost, false, over);
    }

    @Override
    public String render() {
        double normalized = ((angle + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
        return String.format("angle %+6.1f deg from upright, speed %+5.2f", Math.toDegrees(normalized), speed);
    }

    @Override
    public String toString() {
        return "Pendulum";
    }
}
