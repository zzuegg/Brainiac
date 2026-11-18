package io.github.zzuegg.brainiac.experimental.sciml;

/**
 * Symplectic integrators for Hamiltonian systems {@code q' = dH/dp, p' = -dH/dq}. They
 * preserve phase-space volume and keep the energy error bounded over long times instead of
 * letting it drift, which is why learned Hamiltonians are rolled out with them.
 *
 * <ul>
 *   <li>{@link #leapfrog}: the explicit kick-drift-kick scheme for separable
 *       {@code H = T(p) + V(q)};
 *   <li>{@link #stormerVerlet}: the generalised (implicit) Stormer-Verlet scheme for any
 *       {@code H(q, p)}, such as a learned one, with the implicit stages solved by fixed-point
 *       iteration.
 * </ul>
 *
 * <p>Particle simulators use the same idea as velocity Verlet
 * ({@link MechanicalSystems#particleTrajectory}).
 */
public final class SymplecticIntegrator {

    private SymplecticIntegrator() {
    }

    /** The gradient of a Hamiltonian. */
    @FunctionalInterface
    public interface Gradient {

        /**
         * @param q positions
         * @param p momenta
         * @return {@code {dH/dq, dH/dp}}
         */
        double[][] at(double[] q, double[] p);
    }

    /**
     * One leapfrog step for a separable Hamiltonian.
     *
     * @param h the gradient (its {@code dH/dq} must not depend on {@code p}, nor {@code dH/dp}
     *     on {@code q})
     * @param q positions
     * @param p momenta
     * @param dt the step
     * @return {@code {q', p'}}
     */
    public static double[][] leapfrog(Gradient h, double[] q, double[] p, double dt) {
        int d = q.length;
        double[] dq = h.at(q, p)[0];
        double[] ph = new double[d];
        for (int i = 0; i < d; i++) {
            ph[i] = p[i] - 0.5 * dt * dq[i];
        }
        double[] dp = h.at(q, ph)[1];
        double[] q1 = new double[d];
        for (int i = 0; i < d; i++) {
            q1[i] = q[i] + dt * dp[i];
        }
        double[] dq1 = h.at(q1, ph)[0];
        double[] p1 = new double[d];
        for (int i = 0; i < d; i++) {
            p1[i] = ph[i] - 0.5 * dt * dq1[i];
        }
        return new double[][] {q1, p1};
    }

    /**
     * One generalised Stormer-Verlet step for any Hamiltonian:
     *
     * <pre>
     * p_half = p - dt/2 dH/dq(q, p_half)                      (implicit)
     * q'     = q + dt/2 (dH/dp(q, p_half) + dH/dp(q', p_half))   (implicit)
     * p'     = p_half - dt/2 dH/dq(q', p_half)
     * </pre>
     *
     * @param h the gradient
     * @param q positions
     * @param p momenta
     * @param dt the step
     * @param iterations fixed-point iterations per implicit stage, for example 5
     * @return {@code {q', p'}}
     */
    public static double[][] stormerVerlet(Gradient h, double[] q, double[] p, double dt, int iterations) {
        int d = q.length;
        double[] ph = p.clone();
        for (int it = 0; it < iterations; it++) {
            double[] g = h.at(q, ph)[0];
            for (int i = 0; i < d; i++) {
                ph[i] = p[i] - 0.5 * dt * g[i];
            }
        }
        double[] v0 = h.at(q, ph)[1];
        double[] q1 = new double[d];
        for (int i = 0; i < d; i++) {
            q1[i] = q[i] + dt * v0[i];
        }
        for (int it = 0; it < iterations; it++) {
            double[] v1 = h.at(q1, ph)[1];
            for (int i = 0; i < d; i++) {
                q1[i] = q[i] + 0.5 * dt * (v0[i] + v1[i]);
            }
        }
        double[] g1 = h.at(q1, ph)[0];
        double[] p1 = new double[d];
        for (int i = 0; i < d; i++) {
            p1[i] = ph[i] - 0.5 * dt * g1[i];
        }
        return new double[][] {q1, p1};
    }

    /**
     * A rollout with {@link #stormerVerlet}.
     *
     * @param h the gradient
     * @param x0 the initial state {@code (q, p)} concatenated
     * @param dt the step
     * @param steps the steps
     * @return states {@code [steps + 1][2d]}
     */
    public static double[][] rollout(Gradient h, double[] x0, double dt, int steps) {
        int d = x0.length / 2;
        double[][] out = new double[steps + 1][];
        out[0] = x0.clone();
        double[] q = java.util.Arrays.copyOfRange(x0, 0, d);
        double[] p = java.util.Arrays.copyOfRange(x0, d, 2 * d);
        for (int t = 1; t <= steps; t++) {
            double[][] next = stormerVerlet(h, q, p, dt, 4);
            q = next[0];
            p = next[1];
            double[] x = new double[2 * d];
            System.arraycopy(q, 0, x, 0, d);
            System.arraycopy(p, 0, x, d, d);
            out[t] = x;
        }
        return out;
    }
}
