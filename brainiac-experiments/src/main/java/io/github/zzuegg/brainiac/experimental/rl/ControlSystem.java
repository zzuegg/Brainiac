package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Space;

import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * A known discrete-time dynamical system with a cost, the "model" that trajectory optimisers
 * and model-based planners use: {@code x' = f(x, u)} and a running cost {@code l(x, u)} plus a
 * final cost {@code l_f(x)}. Derivatives default to central finite differences, which is what
 * iLQR and DDP need from a black-box simulator; a differentiable model can override them with
 * exact autograd Jacobians ({@link ControlSystems#massSpring}).
 *
 * <p>{@link #asEnvironment} wraps the same system as a core {@link Environment} (reward = minus
 * the running cost), so planners and model-free learners are compared on identical physics.
 */
public interface ControlSystem {

    /** A receding-horizon controller for a {@link ControlSystem}. */
    interface Planner {
        /**
         * @param state the current state
         * @return the action to apply
         */
        double[] act(double[] state);

        /** Forgets any warm start, for a new episode. */
        default void reset() {
        }
    }

    /**
     * The outcome of running a planner for one episode.
     *
     * @param cost the summed running cost plus the final cost
     * @param success whether the final state solves the task
     * @param millisPerStep planning time per control step
     * @param finalState the last state
     */
    record Episode(double cost, boolean success, double millisPerStep, double[] finalState) {
    }

    /**
     * Runs a planner in closed loop on this system.
     *
     * @param planner the controller
     * @param start the start state
     * @param steps control steps
     * @return what happened
     */
    default Episode run(Planner planner, double[] start, int steps) {
        planner.reset();
        double[] x = start.clone();
        double total = 0;
        long nanos = 0;
        for (int t = 0; t < steps; t++) {
            long t0 = System.nanoTime();
            double[] u = planner.act(x);
            nanos += System.nanoTime() - t0;
            total += cost(x, u);
            x = step(x, u);
        }
        total += finalCost(x);
        return new Episode(total, success(x), nanos / 1e6 / steps, x);
    }

    /** @return the state dimension */
    int stateSize();

    /** @return the action dimension */
    int actionSize();

    /** @return the lower action bounds */
    double[] actionLow();

    /** @return the upper action bounds */
    double[] actionHigh();

    /**
     * @param x a state
     * @param u an action (clipped to the bounds by the system)
     * @return the next state
     */
    double[] step(double[] x, double[] u);

    /**
     * @param x a state
     * @param u an action
     * @return the running cost of acting
     */
    double cost(double[] x, double[] u);

    /**
     * @param x the last state
     * @return the final cost
     */
    double finalCost(double[] x);

    /**
     * @param random the random source
     * @return a start state
     */
    double[] start(SplittableRandom random);

    /**
     * @param x a final state
     * @return whether it counts as solving the task
     */
    boolean success(double[] x);

    /**
     * The Jacobians of the dynamics, by central differences.
     *
     * @param x a state
     * @param u an action
     * @return {@code {A = df/dx [n][n], B = df/du [n][m]}}
     */
    default double[][][] jacobians(double[] x, double[] u) {
        int n = stateSize();
        int m = actionSize();
        double[][] a = new double[n][n];
        double[][] b = new double[n][m];
        double h = 1e-5;
        for (int j = 0; j < n; j++) {
            double[] xp = x.clone();
            double[] xm = x.clone();
            xp[j] += h;
            xm[j] -= h;
            double[] fp = step(xp, u);
            double[] fm = step(xm, u);
            for (int i = 0; i < n; i++) {
                a[i][j] = (fp[i] - fm[i]) / (2 * h);
            }
        }
        for (int j = 0; j < m; j++) {
            double[] up = u.clone();
            double[] um = u.clone();
            up[j] += h;
            um[j] -= h;
            double[] fp = step(x, up);
            double[] fm = step(x, um);
            for (int i = 0; i < n; i++) {
                b[i][j] = (fp[i] - fm[i]) / (2 * h);
            }
        }
        return new double[][][] {a, b};
    }

    /**
     * Gradient and Hessian of a scalar function of a vector, by central differences.
     *
     * @param f the function
     * @param x the point
     * @return {@code {gradient (row 0), Hessian rows 1..n}} as an {@code [n + 1][n]} array
     */
    static double[][] secondOrder(Function<double[], Double> f, double[] x) {
        int n = x.length;
        double h = 1e-4;
        double[][] out = new double[n + 1][n];
        double f0 = f.apply(x);
        for (int i = 0; i < n; i++) {
            double[] p = x.clone();
            double[] m = x.clone();
            p[i] += h;
            m[i] -= h;
            double fp = f.apply(p);
            double fm = f.apply(m);
            out[0][i] = (fp - fm) / (2 * h);
            out[1 + i][i] = (fp - 2 * f0 + fm) / (h * h);
            for (int j = 0; j < i; j++) {
                double[] pp = x.clone();
                double[] pm = x.clone();
                double[] mp = x.clone();
                double[] mm = x.clone();
                pp[i] += h;
                pp[j] += h;
                pm[i] += h;
                pm[j] -= h;
                mp[i] -= h;
                mp[j] += h;
                mm[i] -= h;
                mm[j] -= h;
                double v = (f.apply(pp) - f.apply(pm) - f.apply(mp) + f.apply(mm)) / (4 * h * h);
                out[1 + i][j] = v;
                out[1 + j][i] = v;
            }
        }
        return out;
    }

    /**
     * Wraps the system as an environment with a time limit; the reward is minus the running
     * cost, and the observation is the state.
     *
     * @param steps steps per episode
     * @return the environment
     */
    default Environment asEnvironment(int steps) {
        ControlSystem system = this;
        return new Environment() {
            private double[] x;
            private int t;

            @Override
            public Space observationSpace() {
                float[] low = new float[stateSize()];
                float[] high = new float[stateSize()];
                java.util.Arrays.fill(low, -1e6f);
                java.util.Arrays.fill(high, 1e6f);
                return Space.box(low, high);
            }

            @Override
            public Space actionSpace() {
                float[] low = new float[actionSize()];
                float[] high = new float[actionSize()];
                for (int i = 0; i < low.length; i++) {
                    low[i] = (float) actionLow()[i];
                    high[i] = (float) actionHigh()[i];
                }
                return Space.box(low, high);
            }

            @Override
            public float[] reset(SplittableRandom random) {
                x = start(random);
                t = 0;
                return toFloat(x);
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
                return new Step(toFloat(x), (float) -c, false, t >= steps);
            }

            @Override
            public String toString() {
                return system.toString();
            }
        };
    }

    /**
     * @param x doubles
     * @return the same as floats
     */
    static float[] toFloat(double[] x) {
        float[] f = new float[x.length];
        for (int i = 0; i < f.length; i++) {
            f[i] = (float) x[i];
        }
        return f;
    }
}
