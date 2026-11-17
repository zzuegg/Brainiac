package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Space;
import java.util.Arrays;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Homeostatic reinforcement learning (Keramati and Gutkin, eLife 2014,
 * <a href="https://doi.org/10.7554/eLife.04811">10.7554/eLife.04811</a>): reward is not
 * given by the world but made by the body. The wrapper keeps internal variables (energy,
 * water, ...) that drift away from their set points every step and are pushed back by what
 * the agent takes in. The drive is the squared distance to the set points, and the reward is
 * how much a step reduced it. Overeating is as bad as starving, so a reward-maximizing agent
 * learns to keep every variable in balance. The internal deviations are appended to the
 * observation (interoception), and the episode ends when any of them leaves
 * {@code [-limit, limit]}.
 */
public final class HomeostaticEnvironment implements Environment {

    private final Environment inner;
    private final Function<Step, float[]> intake;
    private final float[] drift;
    private final float limit;
    private final float[] deviation;
    private float lastDrive;

    private HomeostaticEnvironment(Environment inner, Function<Step, float[]> intake, float[] drift, float limit) {
        this.inner = Objects.requireNonNull(inner);
        this.intake = Objects.requireNonNull(intake);
        this.drift = drift.clone();
        this.limit = limit;
        this.deviation = new float[drift.length];
    }

    /**
     * @param inner the world (its own reward is ignored)
     * @param intake what a step of the world adds to every internal variable
     * @param drift what every internal variable loses per step
     * @param limit the deviation from the set point at which the agent dies
     * @return the wrapped environment
     */
    public static HomeostaticEnvironment of(Environment inner, Function<Step, float[]> intake, float[] drift,
            float limit) {
        return new HomeostaticEnvironment(inner, intake, drift, limit);
    }

    /** @return the current deviation of every internal variable from its set point */
    public float[] deviation() {
        return deviation.clone();
    }

    private float drive() {
        float d = 0;
        for (float x : deviation) {
            d += x * x;
        }
        return d;
    }

    private float[] observe(float[] obs) {
        float[] out = new float[obs.length + deviation.length];
        System.arraycopy(obs, 0, out, 0, obs.length);
        for (int i = 0; i < deviation.length; i++) {
            out[obs.length + i] = deviation[i] / limit;
        }
        return out;
    }

    @Override
    public Space observationSpace() {
        Space.Box b = (Space.Box) inner.observationSpace();
        float[] low = new float[b.low().length + deviation.length];
        float[] high = new float[low.length];
        System.arraycopy(b.low(), 0, low, 0, b.low().length);
        System.arraycopy(b.high(), 0, high, 0, b.high().length);
        for (int i = b.low().length; i < low.length; i++) {
            low[i] = -1;
            high[i] = 1;
        }
        return Space.box(low, high);
    }

    @Override
    public Space actionSpace() {
        return inner.actionSpace();
    }

    @Override
    public float[] reset(SplittableRandom random) {
        Arrays.fill(deviation, 0);
        lastDrive = 0;
        return observe(inner.reset(random));
    }

    @Override
    public Step step(float[] action) {
        Step s = inner.step(action);
        float[] in = intake.apply(s);
        boolean dead = false;
        for (int i = 0; i < deviation.length; i++) {
            deviation[i] += in[i] - drift[i];
            dead |= Math.abs(deviation[i]) > limit;
        }
        float drive = drive();
        float reward = lastDrive - drive;
        lastDrive = drive;
        return new Step(observe(s.observation()), reward, dead || s.terminated(), !dead && s.truncated());
    }
}
