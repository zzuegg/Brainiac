package io.github.zzuegg.brainiac.experimental.rl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * Known-model versions of classic control tasks as {@link ControlSystem}s, with physical
 * parameters exposed so they can be randomised ({@link DomainRandomization}):
 * <ul>
 *   <li>{@link #pendulum}: the core {@code Pendulum} physics (gravity 10, time step 0.05,
 *       torque limit 2, speed limit 8) for a rod of mass {@code m} and length {@code l}; the
 *       state is {@code (angle from upright, angular speed)}.</li>
 *   <li>{@link #cartPole}: the core {@code CartPole} physics (time step 0.02) with a
 *       continuous force in {@code [-10, 10]}; the state is {@code (x, x', angle, angle')}.</li>
 *   <li>{@link #pointMass}: a 2-D point mass that must reach {@code (1, 1)} from
 *       {@code (-1, -1)} around a disc obstacle that sits on the straight line, a non-convex
 *       cost where local trajectory optimisers stall.</li>
 *   <li>{@link #massSpring}: the differentiable flexible arm of {@link DifferentiableSim}, with
 *       exact autograd Jacobians.</li>
 * </ul>
 * Costs are smooth (for example {@code 2 (1 - cos angle)} instead of the wrapped angle
 * squared) so that iLQR and DDP can take their derivatives.
 */
public final class ControlSystems {

    private ControlSystems() {
    }

    private static double clip(double v, double lo, double hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    private static double wrap(double angle) {
        return ((angle + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
    }

    /**
     * The same system with a Gaussian start state {@code N(mean, diag(std^2))}, for example the
     * start distribution of {@link Pilco}.
     *
     * @param system the system
     * @param mean the start mean
     * @param std the per-dimension standard deviations
     * @return the system with the new start distribution
     */
    public static ControlSystem withStart(ControlSystem system, double[] mean, double[] std) {
        double[] mu = mean.clone();
        double[] sd = std.clone();
        return new ControlSystem() {
            @Override
            public int stateSize() {
                return system.stateSize();
            }

            @Override
            public int actionSize() {
                return system.actionSize();
            }

            @Override
            public double[] actionLow() {
                return system.actionLow();
            }

            @Override
            public double[] actionHigh() {
                return system.actionHigh();
            }

            @Override
            public double[] step(double[] x, double[] u) {
                return system.step(x, u);
            }

            @Override
            public double[][][] jacobians(double[] x, double[] u) {
                return system.jacobians(x, u);
            }

            @Override
            public double cost(double[] x, double[] u) {
                return system.cost(x, u);
            }

            @Override
            public double finalCost(double[] x) {
                return system.finalCost(x);
            }

            @Override
            public double[] start(SplittableRandom random) {
                double[] x = new double[mu.length];
                for (int i = 0; i < x.length; i++) {
                    x[i] = mu[i] + sd[i] * random.nextGaussian();
                }
                return x;
            }

            @Override
            public boolean success(double[] x) {
                return system.success(x);
            }

            @Override
            public String toString() {
                return system + " (Gaussian start)";
            }
        };
    }

    /**
     * Removes the action bounds by squashing: the wrapped system takes unbounded actions
     * {@code v} and applies {@code u = mid + half * tanh(v)} inside the original box. This is the
     * standard way to give a gradient-based trajectory optimiser such as iLQR torque limits it
     * can see, instead of clipping its steps after the fact.
     *
     * @param system the bounded system
     * @return the squashed system
     */
    public static ControlSystem squashed(ControlSystem system) {
        double[] lo = system.actionLow();
        double[] hi = system.actionHigh();
        return new ControlSystem() {
            private double[] squash(double[] v) {
                double[] u = new double[v.length];
                for (int i = 0; i < u.length; i++) {
                    u[i] = (lo[i] + hi[i]) / 2 + (hi[i] - lo[i]) / 2 * Math.tanh(v[i]);
                }
                return u;
            }

            @Override
            public int stateSize() {
                return system.stateSize();
            }

            @Override
            public int actionSize() {
                return system.actionSize();
            }

            @Override
            public double[] actionLow() {
                double[] out = new double[lo.length];
                java.util.Arrays.fill(out, -5);
                return out;
            }

            @Override
            public double[] actionHigh() {
                double[] out = new double[hi.length];
                java.util.Arrays.fill(out, 5);
                return out;
            }

            @Override
            public double[] step(double[] x, double[] v) {
                return system.step(x, squash(v));
            }

            @Override
            public double cost(double[] x, double[] v) {
                return system.cost(x, squash(v));
            }

            @Override
            public double finalCost(double[] x) {
                return system.finalCost(x);
            }

            @Override
            public double[] start(SplittableRandom random) {
                return system.start(random);
            }

            @Override
            public boolean success(double[] x) {
                return system.success(x);
            }

            @Override
            public String toString() {
                return system + ", squashed";
            }
        };
    }

    /**
     * A pendulum swing-up.
     *
     * @param mass the rod mass (1 in the core environment)
     * @param length the rod length (1 in the core environment)
     * @param gain the actuator gain (1 nominal)
     * @param swingUp start hanging down (otherwise uniformly random, as the core environment)
     * @return the system
     */
    public static ControlSystem pendulum(double mass, double length, double gain, boolean swingUp) {
        return new ControlSystem() {
            @Override
            public int stateSize() {
                return 2;
            }

            @Override
            public int actionSize() {
                return 1;
            }

            @Override
            public double[] actionLow() {
                return new double[] {-2};
            }

            @Override
            public double[] actionHigh() {
                return new double[] {2};
            }

            @Override
            public double[] step(double[] x, double[] u) {
                double torque = gain * clip(u[0], -2, 2);
                double speed = x[1] + (3 * 10 / (2 * length) * Math.sin(x[0]) + 3 * torque / (mass * length * length))
                        * 0.05;
                speed = clip(speed, -8, 8);
                return new double[] {x[0] + speed * 0.05, speed};
            }

            @Override
            public double cost(double[] x, double[] u) {
                return 2 * (1 - Math.cos(x[0])) + 0.1 * x[1] * x[1] + 0.001 * u[0] * u[0];
            }

            @Override
            public double finalCost(double[] x) {
                return 0;
            }

            @Override
            public double[] start(SplittableRandom random) {
                return swingUp ? new double[] {Math.PI + random.nextDouble(-0.1, 0.1), random.nextDouble(-0.1, 0.1)}
                        : new double[] {random.nextDouble(-Math.PI, Math.PI), random.nextDouble(-1, 1)};
            }

            @Override
            public boolean success(double[] x) {
                return Math.abs(wrap(x[0])) < 0.3 && Math.abs(x[1]) < 1.5;
            }

            @Override
            public String toString() {
                return String.format("pendulum (m %.2f, l %.2f, gain %.2f)", mass, length, gain);
            }
        };
    }

    /**
     * The nominal pendulum swing-up of the core environment.
     *
     * @return the system
     */
    public static ControlSystem pendulum() {
        return pendulum(1, 1, 1, true);
    }

    /**
     * A cart-pole with a continuous force.
     *
     * @param poleMass the pole mass (0.1 nominal)
     * @param halfLength the pole half-length (0.5 nominal)
     * @param gain the actuator gain (1 nominal)
     * @param swingUp start hanging down (otherwise near upright, the balancing task)
     * @return the system
     */
    public static ControlSystem cartPole(double poleMass, double halfLength, double gain, boolean swingUp) {
        return new ControlSystem() {
            @Override
            public int stateSize() {
                return 4;
            }

            @Override
            public int actionSize() {
                return 1;
            }

            @Override
            public double[] actionLow() {
                return new double[] {-10};
            }

            @Override
            public double[] actionHigh() {
                return new double[] {10};
            }

            @Override
            public double[] step(double[] s, double[] u) {
                double force = gain * clip(u[0], -10, 10);
                double total = 1.0 + poleMass;
                double moment = poleMass * halfLength;
                double cos = Math.cos(s[2]);
                double sin = Math.sin(s[2]);
                double temp = (force + moment * s[3] * s[3] * sin) / total;
                double thetaAcc = (9.8 * sin - cos * temp) / (halfLength * (4.0 / 3.0 - poleMass * cos * cos / total));
                double xAcc = temp - moment * thetaAcc * cos / total;
                double dt = 0.02;
                return new double[] {s[0] + dt * s[1], s[1] + dt * xAcc, s[2] + dt * s[3], s[3] + dt * thetaAcc};
            }

            @Override
            public double cost(double[] s, double[] u) {
                return 2 * (1 - Math.cos(s[2])) + 0.1 * s[0] * s[0] + 0.01 * (s[1] * s[1] + s[3] * s[3])
                        + 1e-4 * u[0] * u[0];
            }

            @Override
            public double finalCost(double[] s) {
                return 0;
            }

            @Override
            public double[] start(SplittableRandom random) {
                double[] s = new double[4];
                for (int i = 0; i < 4; i++) {
                    s[i] = random.nextDouble(-0.05, 0.05);
                }
                if (swingUp) {
                    s[2] += Math.PI;
                }
                return s;
            }

            @Override
            public boolean success(double[] s) {
                return Math.abs(wrap(s[2])) < 0.2 && Math.abs(s[0]) < 2.4;
            }

            @Override
            public String toString() {
                return String.format("cart-pole (pole mass %.2f, half-length %.2f, gain %.2f)", poleMass, halfLength,
                        gain);
            }
        };
    }

    /**
     * A point mass reaching {@code (1, 1)} from {@code (-1, -1)} around a disc obstacle of
     * radius 0.5 at the origin: {@code v <- 0.9 v + 0.2 u}, {@code p <- p + 0.1 v}, with a
     * steep penalty inside the disc, a small action cost and a final cost
     * {@code 10 |p - goal|^2 + |v|^2}.
     *
     * @return the system
     */
    public static ControlSystem pointMass() {
        return new ControlSystem() {
            @Override
            public int stateSize() {
                return 4;
            }

            @Override
            public int actionSize() {
                return 2;
            }

            @Override
            public double[] actionLow() {
                return new double[] {-1, -1};
            }

            @Override
            public double[] actionHigh() {
                return new double[] {1, 1};
            }

            @Override
            public double[] step(double[] s, double[] u) {
                double vx = 0.9 * s[2] + 0.2 * clip(u[0], -1, 1);
                double vy = 0.9 * s[3] + 0.2 * clip(u[1], -1, 1);
                return new double[] {s[0] + 0.1 * vx, s[1] + 0.1 * vy, vx, vy};
            }

            @Override
            public double cost(double[] s, double[] u) {
                double inside = Math.max(0, 0.5 - Math.hypot(s[0], s[1]));
                return 0.01 * (u[0] * u[0] + u[1] * u[1]) + 200 * inside * inside
                        + 0.1 * ((s[0] - 1) * (s[0] - 1) + (s[1] - 1) * (s[1] - 1));
            }

            @Override
            public double finalCost(double[] s) {
                return 10 * ((s[0] - 1) * (s[0] - 1) + (s[1] - 1) * (s[1] - 1)) + s[2] * s[2] + s[3] * s[3];
            }

            @Override
            public double[] start(SplittableRandom random) {
                return new double[] {-1 + random.nextDouble(-0.01, 0.01), -1 + random.nextDouble(-0.01, 0.01), 0, 0};
            }

            @Override
            public boolean success(double[] s) {
                return Math.hypot(s[0] - 1, s[1] - 1) < 0.2;
            }

            @Override
            public String toString() {
                return "point mass around an obstacle";
            }
        };
    }

    /**
     * The mass-spring reach of {@link DifferentiableSim} for one goal, with Jacobians from
     * autograd through the simulator's {@link Var} step. The running cost is a small action
     * penalty, the final cost the simulator's {@code |ball - goal|^2 + 0.1 |ball velocity|^2},
     * and success means the ball ends within 0.1 of the goal.
     *
     * @param sim the simulator
     * @param goalX the goal's x
     * @param goalY the goal's y
     * @return the system (actions are raw forces, squashed by the simulator)
     */
    public static ControlSystem massSpring(DifferentiableSim sim, double goalX, double goalY) {
        return new ControlSystem() {
            @Override
            public int stateSize() {
                return 8;
            }

            @Override
            public int actionSize() {
                return 2;
            }

            @Override
            public double[] actionLow() {
                return new double[] {-3, -3};
            }

            @Override
            public double[] actionHigh() {
                return new double[] {3, 3};
            }

            private Var row(double[] v) {
                return Var.param(Tensor.of(Shape.of(1, v.length), ControlSystem.toFloat(v)));
            }

            @Override
            public double[] step(double[] x, double[] u) {
                float[] next = sim.step(Var.of(Tensor.of(Shape.of(1, 8), ControlSystem.toFloat(x))),
                        Var.of(Tensor.of(Shape.of(1, 2), ControlSystem.toFloat(u)))).value().toArray();
                double[] out = new double[8];
                for (int i = 0; i < 8; i++) {
                    out[i] = next[i];
                }
                return out;
            }

            @Override
            public double[][][] jacobians(double[] x, double[] u) {
                Var xv = row(x);
                Var uv = row(u);
                Var y = sim.step(xv, uv);
                double[][] a = new double[8][8];
                double[][] b = new double[8][2];
                for (int i = 0; i < 8; i++) {
                    List<Var> g = Var.gradients(y.slice(1, i, i + 1).sum(), List.of(xv, uv), false);
                    float[] gx = g.get(0).value().toArray();
                    float[] gu = g.get(1).value().toArray();
                    for (int j = 0; j < 8; j++) {
                        a[i][j] = gx[j];
                    }
                    b[i][0] = gu[0];
                    b[i][1] = gu[1];
                }
                return new double[][][] {a, b};
            }

            @Override
            public double cost(double[] x, double[] u) {
                return 1e-3 * (u[0] * u[0] + u[1] * u[1]);
            }

            @Override
            public double finalCost(double[] x) {
                // DifferentiableSim#cost, in double precision for clean finite differences
                double dx = x[4] - goalX;
                double dy = x[5] - goalY;
                return dx * dx + dy * dy + 0.1 * (x[6] * x[6] + x[7] * x[7]);
            }

            @Override
            public double[] start(SplittableRandom random) {
                return new double[8];
            }

            @Override
            public boolean success(double[] x) {
                return Math.hypot(x[4] - goalX, x[5] - goalY) < 0.1;
            }

            @Override
            public String toString() {
                return "mass-spring reach";
            }
        };
    }
}
