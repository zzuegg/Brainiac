package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;

/**
 * Ordinary differential equation solvers over {@link Var} states, so gradients flow through
 * the solution ("discretize-then-optimize"). Three methods:
 *
 * <ul>
 *   <li>{@link #euler(int)}: fixed-step forward Euler, first order;</li>
 *   <li>{@link #rk4(int)}: fixed-step classic Runge-Kutta, fourth order;</li>
 *   <li>{@link #dopri5(float, float)}: adaptive Dormand-Prince 5(4) with an embedded error
 *       estimate, step-size control from relative and absolute tolerances, and FSAL.</li>
 * </ul>
 *
 * <p>Integration may run backwards ({@code t1 < t0}). A solver instance remembers how many
 * function evaluations and steps its last call used ({@link #evaluations()},
 * {@link #acceptedSteps()}, {@link #rejectedSteps()}): adaptive compute is a feature of
 * continuous-depth models. For constant-memory gradients see {@link NeuralOde}'s adjoint mode.
 *
 * <p>References: Dormand and Prince (1980); Chen et al., "Neural Ordinary Differential
 * Equations", <a href="https://arxiv.org/abs/1806.07366">arXiv:1806.07366</a>.
 */
public final class OdeSolver {

    /** The right-hand side {@code dz/dt = f(t, z)}. */
    @FunctionalInterface
    public interface Dynamics {
        /**
         * Evaluates the derivative.
         *
         * @param t the time
         * @param z the state
         * @return {@code dz/dt}, the shape of {@code z}
         */
        Var apply(float t, Var z);
    }

    /** The integration method. */
    public enum Method {
        /** Forward Euler. */
        EULER,
        /** Classic fourth-order Runge-Kutta. */
        RK4,
        /** Adaptive Dormand-Prince 5(4). */
        DOPRI5
    }

    // Dormand-Prince tableau
    private static final double[] C = {0, 1.0 / 5, 3.0 / 10, 4.0 / 5, 8.0 / 9, 1, 1};
    private static final double[][] A = {
        {},
        {1.0 / 5},
        {3.0 / 40, 9.0 / 40},
        {44.0 / 45, -56.0 / 15, 32.0 / 9},
        {19372.0 / 6561, -25360.0 / 2187, 64448.0 / 6561, -212.0 / 729},
        {9017.0 / 3168, -355.0 / 33, 46732.0 / 5247, 49.0 / 176, -5103.0 / 18656},
        {35.0 / 384, 0, 500.0 / 1113, 125.0 / 192, -2187.0 / 6784, 11.0 / 84}};
    private static final double[] B5 = {35.0 / 384, 0, 500.0 / 1113, 125.0 / 192, -2187.0 / 6784, 11.0 / 84, 0};
    private static final double[] B4 = {5179.0 / 57600, 0, 7571.0 / 16695, 393.0 / 640, -92097.0 / 339200,
        187.0 / 2100, 1.0 / 40};

    private final Method method;
    private final int steps;
    private final float rtol;
    private final float atol;
    private int evaluations;
    private int accepted;
    private int rejected;

    private OdeSolver(Method method, int steps, float rtol, float atol) {
        this.method = method;
        this.steps = steps;
        this.rtol = rtol;
        this.atol = atol;
    }

    /**
     * A fixed-step Euler solver.
     *
     * @param steps steps per integration
     * @return the solver
     */
    public static OdeSolver euler(int steps) {
        return new OdeSolver(Method.EULER, requirePositive(steps), 0, 0);
    }

    /**
     * A fixed-step fourth-order Runge-Kutta solver.
     *
     * @param steps steps per integration
     * @return the solver
     */
    public static OdeSolver rk4(int steps) {
        return new OdeSolver(Method.RK4, requirePositive(steps), 0, 0);
    }

    /**
     * An adaptive Dormand-Prince 5(4) solver.
     *
     * @param rtol the relative tolerance
     * @param atol the absolute tolerance
     * @return the solver
     */
    public static OdeSolver dopri5(float rtol, float atol) {
        if (!(rtol > 0) || !(atol > 0)) {
            throw new IllegalArgumentException("tolerances must be positive");
        }
        return new OdeSolver(Method.DOPRI5, 0, rtol, atol);
    }

    private static int requirePositive(int steps) {
        if (steps < 1) {
            throw new IllegalArgumentException("steps must be positive, got " + steps);
        }
        return steps;
    }

    /** @return the method */
    public Method method() {
        return method;
    }

    /** @return the fixed number of steps (0 for the adaptive method) */
    public int steps() {
        return steps;
    }

    /** @return function evaluations of the last {@link #integrate} call */
    public int evaluations() {
        return evaluations;
    }

    /** @return accepted steps of the last call */
    public int acceptedSteps() {
        return accepted;
    }

    /** @return rejected steps of the last call (adaptive only) */
    public int rejectedSteps() {
        return rejected;
    }

    /**
     * Integrates {@code dz/dt = f(t, z)} from {@code t0} to {@code t1}.
     *
     * @param f the dynamics
     * @param z0 the initial state
     * @param t0 the start time
     * @param t1 the end time, may be before {@code t0}
     * @return the state at {@code t1}
     */
    public Var integrate(Dynamics f, Var z0, float t0, float t1) {
        evaluations = 0;
        accepted = 0;
        rejected = 0;
        if (t0 == t1) {
            return z0;
        }
        return switch (method) {
            case EULER, RK4 -> fixed(f, z0, t0, t1);
            case DOPRI5 -> adaptive(f, z0, t0, t1);
        };
    }

    private Var eval(Dynamics f, double t, Var z) {
        evaluations++;
        return f.apply((float) t, z);
    }

    private Var fixed(Dynamics f, Var z, float t0, float t1) {
        double h = (t1 - (double) t0) / steps;
        for (int i = 0; i < steps; i++) {
            double t = t0 + i * h;
            float hf = (float) h;
            if (method == Method.EULER) {
                z = z.add(eval(f, t, z).mul(hf));
            } else {
                Var k1 = eval(f, t, z);
                Var k2 = eval(f, t + h / 2, z.add(k1.mul(hf / 2)));
                Var k3 = eval(f, t + h / 2, z.add(k2.mul(hf / 2)));
                Var k4 = eval(f, t + h, z.add(k3.mul(hf)));
                z = z.add(k1.add(k2.mul(2f)).add(k3.mul(2f)).add(k4).mul(hf / 6));
            }
            accepted++;
        }
        return z;
    }

    private Var adaptive(Dynamics f, Var z, float t0, float t1) {
        double direction = Math.signum(t1 - t0);
        double span = Math.abs(t1 - (double) t0);
        double t = t0;
        double h = span / 10 * direction;
        Var k1 = eval(f, t, z);
        while ((t1 - t) * direction > 1e-9 * span) {
            if (Math.abs(h) > Math.abs(t1 - t)) {
                h = t1 - t;
            }
            Var[] k = new Var[7];
            k[0] = k1;
            for (int s = 1; s < 7; s++) {
                Var acc = z;
                for (int j = 0; j < s; j++) {
                    if (A[s][j] != 0) {
                        acc = acc.add(k[j].mul((float) (h * A[s][j])));
                    }
                }
                k[s] = eval(f, t + C[s] * h, acc);
            }
            Var z5 = z;
            Var errVec = null;
            for (int s = 0; s < 7; s++) {
                if (B5[s] != 0) {
                    z5 = z5.add(k[s].mul((float) (h * B5[s])));
                }
                Var e = k[s].mul((float) (h * (B5[s] - B4[s])));
                errVec = errVec == null ? e : errVec.add(e);
            }
            // error norm: RMS of err / (atol + rtol * max(|z|, |z5|))
            float[] ev = errVec.value().toArray();
            float[] a = z.value().toArray();
            float[] b = z5.value().toArray();
            double sum = 0;
            for (int i = 0; i < ev.length; i++) {
                double scale = atol + rtol * Math.max(Math.abs(a[i]), Math.abs(b[i]));
                sum += (ev[i] / scale) * (ev[i] / scale);
            }
            double err = Math.sqrt(sum / ev.length);
            if (err <= 1) {
                t += h;
                z = z5;
                k1 = k[6];   // first same as last
                accepted++;
            } else {
                rejected++;
            }
            double factor = err == 0 ? 5 : Math.min(5, Math.max(0.2, 0.9 * Math.pow(err, -0.2)));
            h *= factor;
            if (accepted + rejected > 100000) {
                throw new IllegalStateException("dopri5 needed more than 100000 steps; the problem may be stiff");
            }
        }
        return z;
    }
}
