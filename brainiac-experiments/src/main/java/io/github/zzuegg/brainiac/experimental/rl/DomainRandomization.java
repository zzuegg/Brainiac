package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Space;

import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Domain randomisation: an {@link Environment} whose physics parameters (masses, lengths,
 * actuator gains) are drawn afresh at every reset from given ranges, so a policy trained on it
 * must work across the whole family rather than on one nominal simulator; deploying it on a
 * real system then only needs the real parameters to fall inside (or near) the ranges. The
 * family is any function from a parameter vector to a {@link ControlSystem}
 * ({@link ControlSystems#pendulum(double, double, double, boolean)},
 * {@link ControlSystems#cartPole(double, double, double, boolean)}).
 *
 * <p>Options: {@link #privileged} appends the true parameters to the observation (the input
 * of {@link RapidMotorAdaptation}'s base policy during training); {@link #changeAt} redraws
 * the parameters in the middle of an episode, to test online adaptation; the episode ends
 * early when the system's {@code failed} predicate holds (for example the pole falling).
 *
 * <p>Tobin et al. 2017, "Domain Randomization for Transferring Deep Neural Networks from
 * Simulation to the Real World", <a href="https://arxiv.org/abs/1703.06907">arXiv:1703.06907</a>.
 */
public final class DomainRandomization implements Environment {

    private final Function<double[], ControlSystem> family;
    private final double[] low;
    private final double[] high;
    private final int steps;
    private boolean privileged;
    private int changeStep = -1;
    private java.util.function.Predicate<double[]> failed = x -> false;
    private double failPenalty;
    private ControlSystem system;
    private double[] params;
    private double[] x;
    private int t;
    private SplittableRandom random;

    private DomainRandomization(Function<double[], ControlSystem> family, double[] low, double[] high, int steps) {
        this.family = family;
        this.low = low.clone();
        this.high = high.clone();
        this.steps = steps;
    }

    /**
     * @param family builds the system for a parameter vector
     * @param low the lower parameter bounds
     * @param high the upper parameter bounds
     * @param steps steps per episode
     * @return the randomised environment
     */
    public static DomainRandomization of(Function<double[], ControlSystem> family, double[] low, double[] high,
            int steps) {
        return new DomainRandomization(Objects.requireNonNull(family), low, high, steps);
    }

    /**
     * @param on whether observations end with the true parameters
     * @return this
     */
    public DomainRandomization privileged(boolean on) {
        this.privileged = on;
        return this;
    }

    /**
     * @param step the step at which the parameters are redrawn, or -1 for never
     * @return this
     */
    public DomainRandomization changeAt(int step) {
        this.changeStep = step;
        return this;
    }

    /**
     * @param predicate states that end the episode as a failure
     * @param penalty cost charged for every step the failure cuts off, so that failing early
     *     never pays
     * @return this
     */
    public DomainRandomization failWhen(java.util.function.Predicate<double[]> predicate, double penalty) {
        this.failed = Objects.requireNonNull(predicate);
        this.failPenalty = penalty;
        return this;
    }

    /**
     * The cost a failure charges for the steps from {@code from} to the end of the episode:
     * the part of the failure cost that falls in a window starting at {@code from}, for an
     * episode that failed before it.
     *
     * @param from the first step of the window
     * @return the (positive) cost
     */
    public double failureCost(int from) {
        return failPenalty * Math.max(0, steps - from);
    }

    /** @return the step within the episode */
    public int time() {
        return t;
    }

    /** @return the current parameters (a copy) */
    public double[] parameters() {
        return params.clone();
    }

    /** @return the current system */
    public ControlSystem system() {
        return system;
    }

    /** @return the current state (a copy) */
    public double[] state() {
        return x.clone();
    }

    private void draw() {
        params = new double[low.length];
        for (int i = 0; i < low.length; i++) {
            params[i] = low[i] == high[i] ? low[i] : random.nextDouble(low[i], high[i]);
        }
        system = family.apply(params);
    }

    @Override
    public Space observationSpace() {
        int n = family.apply(low).stateSize() + (privileged ? low.length : 0);
        float[] lo = new float[n];
        float[] hi = new float[n];
        java.util.Arrays.fill(lo, -1e6f);
        java.util.Arrays.fill(hi, 1e6f);
        return Space.box(lo, hi);
    }

    @Override
    public Space actionSpace() {
        ControlSystem s = family.apply(low);
        return Space.box(ControlSystem.toFloat(s.actionLow()), ControlSystem.toFloat(s.actionHigh()));
    }

    @Override
    public float[] reset(SplittableRandom r) {
        this.random = r;
        draw();
        x = system.start(r);
        t = 0;
        return observation();
    }

    private float[] observation() {
        double[] o = x;
        if (privileged) {
            o = new double[x.length + params.length];
            System.arraycopy(x, 0, o, 0, x.length);
            System.arraycopy(params, 0, o, x.length, params.length);
        }
        return ControlSystem.toFloat(o);
    }

    @Override
    public Step step(float[] action) {
        double[] u = new double[action.length];
        for (int i = 0; i < u.length; i++) {
            u[i] = action[i];
        }
        double c = system.cost(x, u);
        x = system.step(x, u);
        t++;
        if (t == changeStep) {
            draw();
        }
        boolean fail = failed.test(x);
        if (fail) {
            c += failPenalty * (steps - t);
        }
        return new Step(observation(), (float) -c, fail, !fail && t >= steps);
    }

    @Override
    public String toString() {
        return "randomised " + family.apply(low);
    }
}
