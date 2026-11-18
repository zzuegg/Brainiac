package io.github.zzuegg.brainiac.experimental.sciml;

import java.util.SplittableRandom;
import java.util.function.ToDoubleFunction;
import java.util.function.UnaryOperator;

/**
 * Ground-truth dynamical systems for learned simulators and equation discovery, each with
 * its right-hand side, its energy where one is conserved, and accurate integration:
 * mass-spring, pendulum, double pendulum, damped oscillator, Van der Pol, Lotka-Volterra,
 * Lorenz-63 and Lorenz-96, chains of particles on springs inside walls, and Lennard-Jones
 * clusters (energies and forces).
 *
 * <p>{@link #trajectory} integrates with classical RK4 split into many substeps, accurate to
 * far below any learned model's error; {@link SymplecticIntegrator} gives the
 * structure-preserving alternative for the Hamiltonian ones.
 */
public final class MechanicalSystems {

    private MechanicalSystems() {
    }

    /**
     * An autonomous ODE {@code x' = f(x)}.
     *
     * @param name a short name
     * @param dimension the state size
     * @param rhs the right-hand side
     * @param energy the conserved energy, or {@code null} if there is none
     */
    public record Ode(String name, int dimension, UnaryOperator<double[]> rhs, ToDoubleFunction<double[]> energy) {

        /**
         * @param x the state
         * @return {@code f(x)}
         */
        public double[] derivative(double[] x) {
            return rhs.apply(x);
        }
    }

    /**
     * A unit-mass harmonic oscillator, state {@code (q, p)}, {@code H = p^2/2 + k q^2/2}.
     *
     * @param k the spring constant
     * @return the system
     */
    public static Ode massSpring(double k) {
        return new Ode("mass-spring", 2, x -> new double[] {x[1], -k * x[0]},
                x -> 0.5 * x[1] * x[1] + 0.5 * k * x[0] * x[0]);
    }

    /**
     * A pendulum, state {@code (q, p)}, {@code H = p^2/2 + g (1 - cos q)}.
     *
     * @param g gravity over length
     * @return the system
     */
    public static Ode pendulum(double g) {
        return new Ode("pendulum", 2, x -> new double[] {x[1], -g * Math.sin(x[0])},
                x -> 0.5 * x[1] * x[1] + g * (1 - Math.cos(x[0])));
    }

    /**
     * A damped oscillator {@code q'' = -k q - c q'}, state {@code (q, v)}; not conservative.
     * Its energy function is the undamped one, which decays.
     *
     * @param k the spring constant
     * @param c the damping
     * @return the system
     */
    public static Ode dampedOscillator(double k, double c) {
        return new Ode("damped", 2, x -> new double[] {x[1], -k * x[0] - c * x[1]},
                x -> 0.5 * x[1] * x[1] + 0.5 * k * x[0] * x[0]);
    }

    /**
     * The double pendulum with unit masses and lengths, state {@code (t1, t2, w1, w2)}
     * (angles and angular velocities).
     *
     * @param g gravity
     * @return the system
     */
    public static Ode doublePendulum(double g) {
        return new Ode("double-pendulum", 4, x -> {
            double t1 = x[0];
            double t2 = x[1];
            double w1 = x[2];
            double w2 = x[3];
            double d = t1 - t2;
            double den = 3 - Math.cos(2 * d);
            double a1 = (-3 * g * Math.sin(t1) - g * Math.sin(t1 - 2 * t2)
                    - 2 * Math.sin(d) * (w2 * w2 + w1 * w1 * Math.cos(d))) / den;
            double a2 = (2 * Math.sin(d) * (2 * w1 * w1 + 2 * g * Math.cos(t1) + w2 * w2 * Math.cos(d))) / den;
            return new double[] {w1, w2, a1, a2};
        }, x -> {
            double t1 = x[0];
            double t2 = x[1];
            double w1 = x[2];
            double w2 = x[3];
            double kinetic = w1 * w1 + 0.5 * w2 * w2 + w1 * w2 * Math.cos(t1 - t2);
            double potential = -2 * g * Math.cos(t1) - g * Math.cos(t2);
            return kinetic + potential;
        });
    }

    /**
     * The Van der Pol oscillator {@code x'' = mu (1 - x^2) x' - x}, state {@code (x, x')}.
     *
     * @param mu the nonlinearity
     * @return the system
     */
    public static Ode vanDerPol(double mu) {
        return new Ode("van-der-pol", 2, x -> new double[] {x[1], mu * (1 - x[0] * x[0]) * x[1] - x[0]}, null);
    }

    /**
     * Lotka-Volterra predator-prey: {@code x' = a x - b x y}, {@code y' = d x y - c y}; the
     * conserved quantity is {@code d x - c ln x + b y - a ln y}.
     *
     * @param a prey growth
     * @param b predation
     * @param c predator death
     * @param d predator growth
     * @return the system
     */
    public static Ode lotkaVolterra(double a, double b, double c, double d) {
        return new Ode("lotka-volterra", 2, x -> new double[] {a * x[0] - b * x[0] * x[1], d * x[0] * x[1] - c * x[1]},
                x -> d * x[0] - c * Math.log(x[0]) + b * x[1] - a * Math.log(x[1]));
    }

    /**
     * The Lorenz-63 system.
     *
     * @param sigma usually 10
     * @param rho usually 28
     * @param beta usually 8/3
     * @return the system
     */
    public static Ode lorenz63(double sigma, double rho, double beta) {
        return new Ode("lorenz-63", 3, x -> new double[] {
            sigma * (x[1] - x[0]), x[0] * (rho - x[2]) - x[1], x[0] * x[1] - beta * x[2]}, null);
    }

    /**
     * The Lorenz-96 system {@code x_i' = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F}, periodic in
     * {@code i}.
     *
     * @param n the variables, for example 40
     * @param forcing usually 8
     * @return the system
     */
    public static Ode lorenz96(int n, double forcing) {
        return new Ode("lorenz-96", n, x -> {
            double[] d = new double[n];
            for (int i = 0; i < n; i++) {
                d[i] = (x[(i + 1) % n] - x[(i + n - 2) % n]) * x[(i + n - 1) % n] - x[i] + forcing;
            }
            return d;
        }, null);
    }

    /**
     * One classical RK4 step.
     *
     * @param f the right-hand side
     * @param x the state
     * @param dt the step
     * @return the next state
     */
    public static double[] rk4(UnaryOperator<double[]> f, double[] x, double dt) {
        int n = x.length;
        double[] k1 = f.apply(x);
        double[] y = new double[n];
        for (int i = 0; i < n; i++) {
            y[i] = x[i] + 0.5 * dt * k1[i];
        }
        double[] k2 = f.apply(y);
        for (int i = 0; i < n; i++) {
            y[i] = x[i] + 0.5 * dt * k2[i];
        }
        double[] k3 = f.apply(y);
        for (int i = 0; i < n; i++) {
            y[i] = x[i] + dt * k3[i];
        }
        double[] k4 = f.apply(y);
        double[] out = new double[n];
        for (int i = 0; i < n; i++) {
            out[i] = x[i] + dt / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
        }
        return out;
    }

    /**
     * An accurate trajectory: RK4 with {@code substeps} substeps per sample interval.
     *
     * @param ode the system
     * @param x0 the initial state
     * @param dt the sample interval
     * @param steps the samples after the initial one
     * @param substeps RK4 steps per interval, for example 10
     * @return {@code [steps + 1][dimension]}
     */
    public static double[][] trajectory(Ode ode, double[] x0, double dt, int steps, int substeps) {
        double[][] out = new double[steps + 1][];
        out[0] = x0.clone();
        double h = dt / substeps;
        for (int t = 1; t <= steps; t++) {
            double[] x = out[t - 1];
            for (int s = 0; s < substeps; s++) {
                x = rk4(ode.rhs(), x, h);
            }
            out[t] = x;
        }
        return out;
    }

    /**
     * Adds Gaussian noise of a relative scale to a trajectory.
     *
     * @param trajectory the clean states
     * @param level noise standard deviation as a fraction of each variable's standard deviation
     * @param random the random source
     * @return a noisy copy
     */
    public static double[][] noisy(double[][] trajectory, double level, SplittableRandom random) {
        int d = trajectory[0].length;
        double[] std = new double[d];
        for (int j = 0; j < d; j++) {
            double m = 0;
            double s = 0;
            for (double[] x : trajectory) {
                m += x[j];
            }
            m /= trajectory.length;
            for (double[] x : trajectory) {
                s += (x[j] - m) * (x[j] - m);
            }
            std[j] = Math.sqrt(s / trajectory.length);
        }
        double[][] out = new double[trajectory.length][d];
        for (int t = 0; t < trajectory.length; t++) {
            for (int j = 0; j < d; j++) {
                out[t][j] = trajectory[t][j] + level * std[j] * random.nextGaussian();
            }
        }
        return out;
    }

    // ------------------------------------------------------------------ particle systems

    /**
     * Particles in the unit square joined in a chain by springs, pushed back by stiff walls:
     * the {@code n}-particle system for graph network simulators.
     *
     * @param stiffness the spring constant
     * @param rest the spring rest length
     * @param wall the wall stiffness (a quadratic penalty outside {@code [0, 1]^2})
     * @return accelerations {@code [n][2]} of unit-mass particles at positions {@code [n][2]}
     */
    public static UnaryOperator<double[][]> springChain(double stiffness, double rest, double wall) {
        return x -> {
            int n = x.length;
            double[][] a = new double[n][2];
            for (int i = 0; i + 1 < n; i++) {
                double dx = x[i + 1][0] - x[i][0];
                double dy = x[i + 1][1] - x[i][1];
                double r = Math.sqrt(dx * dx + dy * dy) + 1e-12;
                double f = stiffness * (r - rest) / r;
                a[i][0] += f * dx;
                a[i][1] += f * dy;
                a[i + 1][0] -= f * dx;
                a[i + 1][1] -= f * dy;
            }
            for (int i = 0; i < n; i++) {
                for (int c = 0; c < 2; c++) {
                    if (x[i][c] < 0) {
                        a[i][c] -= wall * x[i][c];
                    } else if (x[i][c] > 1) {
                        a[i][c] -= wall * (x[i][c] - 1);
                    }
                }
            }
            return a;
        };
    }

    /**
     * Velocity-Verlet integration of unit-mass particles.
     *
     * @param acceleration accelerations from positions
     * @param x0 initial positions {@code [n][d]}
     * @param v0 initial velocities {@code [n][d]}
     * @param dt the step
     * @param steps the steps
     * @param substeps Verlet substeps per step
     * @return positions {@code [steps + 1][n][d]}
     */
    public static double[][][] particleTrajectory(UnaryOperator<double[][]> acceleration, double[][] x0,
            double[][] v0, double dt, int steps, int substeps) {
        int n = x0.length;
        int d = x0[0].length;
        double[][][] out = new double[steps + 1][][];
        double[][] x = copy(x0);
        double[][] v = copy(v0);
        out[0] = copy(x);
        double h = dt / substeps;
        double[][] a = acceleration.apply(x);
        for (int t = 1; t <= steps; t++) {
            for (int s = 0; s < substeps; s++) {
                for (int i = 0; i < n; i++) {
                    for (int c = 0; c < d; c++) {
                        v[i][c] += 0.5 * h * a[i][c];
                        x[i][c] += h * v[i][c];
                    }
                }
                a = acceleration.apply(x);
                for (int i = 0; i < n; i++) {
                    for (int c = 0; c < d; c++) {
                        v[i][c] += 0.5 * h * a[i][c];
                    }
                }
            }
            out[t] = copy(x);
        }
        return out;
    }

    /**
     * A random spring chain inside the unit square, stretched or compressed so it moves.
     *
     * @param n the particles
     * @param spacing the mean link length (links vary in {@code [0.6, 1.4]} of it)
     * @param random the random source
     * @return initial positions {@code [n][2]}
     */
    public static double[][] randomChain(int n, double spacing, SplittableRandom random) {
        double[][] x = new double[n][2];
        double angle = random.nextDouble(2 * Math.PI);
        double[] p = {0.3 + 0.4 * random.nextDouble(), 0.3 + 0.4 * random.nextDouble()};
        for (int i = 0; i < n; i++) {
            angle += random.nextDouble(-0.8, 0.8);
            double step = spacing * (0.6 + 0.8 * random.nextDouble());
            p[0] = Math.clamp(p[0] + step * Math.cos(angle), 0.05, 0.95);
            p[1] = Math.clamp(p[1] + step * Math.sin(angle), 0.05, 0.95);
            x[i][0] = p[0];
            x[i][1] = p[1];
        }
        return x;
    }

    /**
     * The Lennard-Jones energy {@code sum_{i<j} 4 (r^-12 - r^-6)} with unit well depth and
     * size.
     *
     * @param x positions {@code [n][3]}
     * @return the energy
     */
    public static double lennardJonesEnergy(double[][] x) {
        double e = 0;
        for (int i = 0; i < x.length; i++) {
            for (int j = i + 1; j < x.length; j++) {
                double r2 = dist2(x[i], x[j]);
                double s6 = 1 / (r2 * r2 * r2);
                e += 4 * (s6 * s6 - s6);
            }
        }
        return e;
    }

    /**
     * The Lennard-Jones forces {@code -dE/dx}.
     *
     * @param x positions {@code [n][3]}
     * @return forces {@code [n][3]}
     */
    public static double[][] lennardJonesForces(double[][] x) {
        int n = x.length;
        int d = x[0].length;
        double[][] f = new double[n][d];
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                double r2 = dist2(x[i], x[j]);
                double s6 = 1 / (r2 * r2 * r2);
                double coef = 24 * (2 * s6 * s6 - s6) / r2;   // -dE/dr / r
                for (int c = 0; c < d; c++) {
                    double dc = x[i][c] - x[j][c];
                    f[i][c] += coef * dc;
                    f[j][c] -= coef * dc;
                }
            }
        }
        return f;
    }

    /**
     * A random Lennard-Jones cluster: atoms near a jittered cubic lattice at spacing close to
     * the pair minimum {@code 2^(1/6)}, so energies are finite and forces moderate.
     *
     * @param atoms the atoms
     * @param jitter the position noise, for example 0.1
     * @param random the random source
     * @return positions {@code [atoms][3]}
     */
    public static double[][] lennardJonesCluster(int atoms, double jitter, SplittableRandom random) {
        double spacing = Math.pow(2, 1.0 / 6);
        int side = (int) Math.ceil(Math.cbrt(atoms));
        double[][] x = new double[atoms][3];
        for (int i = 0; i < atoms; i++) {
            x[i][0] = spacing * (i % side) + jitter * random.nextGaussian();
            x[i][1] = spacing * ((i / side) % side) + jitter * random.nextGaussian();
            x[i][2] = spacing * (i / (side * side)) + jitter * random.nextGaussian();
        }
        return x;
    }

    private static double dist2(double[] a, double[] b) {
        double s = 0;
        for (int c = 0; c < a.length; c++) {
            s += (a[c] - b[c]) * (a[c] - b[c]);
        }
        return s;
    }

    static double[][] copy(double[][] x) {
        double[][] y = new double[x.length][];
        for (int i = 0; i < x.length; i++) {
            y[i] = x[i].clone();
        }
        return y;
    }
}
