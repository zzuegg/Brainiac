package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Space;

import java.util.SplittableRandom;

/**
 * A parameterised one-dimensional control task: push a point mass from 0 to a target and
 * hold it there against a constant wind. The target distance and the wind are parameters,
 * so it forms a family of environments of graded difficulty, the kind open-ended methods
 * such as POET co-evolve with their agents.
 *
 * <p>Observation: {@code [target - position, velocity]}. Action: one value in
 * {@code [-1, 1]}, a force of up to {@link #MAX_FORCE}. Each step pays
 * {@code -|target - position| / (target * horizon)}, so a whole episode returns about minus
 * the average remaining distance as a share of the target (0 is perfect, -1 never moving).
 */
public final class WindyReach implements Environment {

    /** The largest force the agent can apply. */
    public static final double MAX_FORCE = 2;
    private static final double DT = 0.1;
    private static final double DRAG = 0.5;

    private final double target;
    private final double wind;
    private final int horizon;
    private double x;
    private double v;
    private int t;

    /**
     * Creates the task.
     *
     * @param target the distance to the target, positive
     * @param wind the constant wind force; negative pushes away from the target
     * @param horizon the episode length in steps
     */
    public WindyReach(double target, double wind, int horizon) {
        if (!(target > 0) || horizon < 1) {
            throw new IllegalArgumentException("target and horizon must be positive");
        }
        this.target = target;
        this.wind = wind;
        this.horizon = horizon;
    }

    /**
     * Creates the task from a parameter vector {@code [target, wind]} with a 60-step horizon.
     *
     * @param parameters the parameters
     * @return the task
     */
    public static WindyReach of(double[] parameters) {
        return new WindyReach(parameters[0], parameters[1], 60);
    }

    @Override
    public Space observationSpace() {
        return new Space.Box(new float[] {Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY},
                new float[] {Float.POSITIVE_INFINITY, Float.POSITIVE_INFINITY});
    }

    @Override
    public Space actionSpace() {
        return new Space.Box(new float[] {-1}, new float[] {1});
    }

    @Override
    public float[] reset(SplittableRandom random) {
        x = 0;
        v = 0;
        t = 0;
        return observation();
    }

    @Override
    public Step step(float[] action) {
        double a = Math.max(-1, Math.min(1, action[0]));
        v += (MAX_FORCE * a + wind - DRAG * v) * DT;
        x += v * DT;
        t++;
        float reward = (float) (-Math.abs(target - x) / (target * horizon));
        return new Step(observation(), reward, false, t >= horizon);
    }

    private float[] observation() {
        return new float[] {(float) (target - x), (float) v};
    }

    /** @return the target distance */
    public double target() {
        return target;
    }

    /** @return the wind force */
    public double wind() {
        return wind;
    }

    @Override
    public String toString() {
        return String.format("WindyReach(target %.2f, wind %.2f)", target, wind);
    }
}
