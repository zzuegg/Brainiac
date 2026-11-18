package io.github.zzuegg.brainiac.experiments.classic;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ml.MlMetrics;
import io.github.zzuegg.brainiac.experimental.ml.StateSpaceFilter;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Bayesian filters ({@link StateSpaceFilter}) against a GRU trained on the same trajectories.
 * <ul>
 *   <li>{@link #linearTracking}: a constant-velocity target observed in noisy position. The
 *       Kalman filter and RTS smoother with the true model against a causal GRU trained to
 *       output the true position, and the raw observations.</li>
 *   <li>{@link #pendulum}: a noisy pendulum observed through {@code sin(theta)} only, which
 *       cannot tell {@code theta} from {@code pi - theta}. The extended Kalman filter and a
 *       bootstrap particle filter against the GRU.</li>
 * </ul>
 * Reported: RMSE of the tracked quantity and the Gaussian NLL of the truth under each method's
 * predictive distribution (the GRU's variance is its training residual variance).
 */
public final class TrackingFilters {

    private TrackingFilters() {
    }

    /**
     * Trajectories: true states and observations.
     *
     * @param states {@code [trajectories][T]} the tracked quantity
     * @param observations {@code [trajectories][T]} one observation per step
     */
    public record Trajectories(double[][] states, double[][] observations) {
    }

    private static final double DT = 0.1;

    /**
     * Constant-velocity trajectories.
     *
     * @param n the number of trajectories
     * @param steps steps per trajectory
     * @param seed the seed
     * @return positions and noisy position readings
     */
    public static Trajectories constantVelocity(int n, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] pos = new double[n][steps];
        double[][] obs = new double[n][steps];
        for (int i = 0; i < n; i++) {
            double p = r.nextGaussian();
            double v = r.nextGaussian();
            for (int t = 0; t < steps; t++) {
                if (t > 0) {
                    p += v * DT + 0.005 * r.nextGaussian();
                    v += 0.1 * r.nextGaussian();
                }
                pos[i][t] = p;
                obs[i][t] = p + 0.5 * r.nextGaussian();
            }
        }
        return new Trajectories(pos, obs);
    }

    /** @return the constant-velocity model the trajectories are drawn from */
    public static StateSpaceFilter.LinearModel constantVelocityModel() {
        return new StateSpaceFilter.LinearModel(new double[][] {{1, DT}, {0, 1}},
                new double[][] {{0.005 * 0.005, 0}, {0, 0.01}}, new double[][] {{1, 0}}, new double[][] {{0.25}},
                new double[] {0, 0}, new double[][] {{1, 0}, {0, 1}});
    }

    /**
     * Pendulum trajectories observed through {@code sin(theta)}.
     *
     * @param n the number of trajectories
     * @param steps steps per trajectory
     * @param seed the seed
     * @return angles and noisy {@code sin(theta)} readings
     */
    public static Trajectories pendulumTrajectories(int n, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] th = new double[n][steps];
        double[][] obs = new double[n][steps];
        for (int i = 0; i < n; i++) {
            double[] x = {r.nextDouble(-2, 2), r.nextGaussian() * 0.2};
            for (int t = 0; t < steps; t++) {
                if (t > 0) {
                    x = pendulumStep(x);
                    x[1] += 0.05 * r.nextGaussian();
                }
                th[i][t] = x[0];
                obs[i][t] = Math.sin(x[0]) + 0.1 * r.nextGaussian();
            }
        }
        return new Trajectories(th, obs);
    }

    private static double[] pendulumStep(double[] x) {
        return new double[] {x[0] + DT * x[1], x[1] - DT * 9.81 * Math.sin(x[0])};
    }

    /**
     * Kalman filter, smoother and GRU on constant-velocity tracking.
     *
     * @param seed the seed
     * @return method to {RMSE, NLL}
     */
    public static Map<String, double[]> linearTracking(long seed) {
        Trajectories train = constantVelocity(200, 60, seed);
        Trajectories test = constantVelocity(40, 60, seed + 1);
        Map<String, double[]> out = new LinkedHashMap<>();
        StateSpaceFilter.LinearModel model = constantVelocityModel();
        Accumulator kf = new Accumulator();
        Accumulator rts = new Accumulator();
        Accumulator raw = new Accumulator();
        for (int i = 0; i < test.states().length; i++) {
            double[][] ys = column(test.observations()[i]);
            StateSpaceFilter.Estimate f = StateSpaceFilter.kalman(model, ys, false);
            StateSpaceFilter.Estimate s = StateSpaceFilter.kalman(model, ys, true);
            kf.add(f.mean(0), f.variance(0), test.states()[i]);
            rts.add(s.mean(0), s.variance(0), test.states()[i]);
            double[] var = new double[ys.length];
            java.util.Arrays.fill(var, 0.25);
            raw.add(test.observations()[i], var, test.states()[i]);
        }
        out.put("raw observations", raw.result());
        out.put("Kalman filter", kf.result());
        out.put("RTS smoother", rts.result());
        out.put("GRU (causal)", gru(train, test, seed));
        return out;
    }

    /**
     * EKF, particle filter and GRU on the pendulum.
     *
     * @param particles particles in the filter
     * @param seed the seed
     * @return method to {RMSE of theta, NLL}
     */
    public static Map<String, double[]> pendulum(int particles, long seed) {
        Trajectories train = pendulumTrajectories(200, 60, seed);
        Trajectories test = pendulumTrajectories(30, 60, seed + 1);
        Map<String, double[]> out = new LinkedHashMap<>();
        double[][] q = {{1e-6, 0}, {0, 0.05 * 0.05}};
        double[][] r = {{0.01}};
        StateSpaceFilter.Differentiable f = new StateSpaceFilter.Differentiable() {
            @Override
            public double[] apply(double[] x) {
                return pendulumStep(x);
            }

            @Override
            public double[][] jacobian(double[] x) {
                return new double[][] {{1, DT}, {-DT * 9.81 * Math.cos(x[0]), 1}};
            }
        };
        StateSpaceFilter.Differentiable h = new StateSpaceFilter.Differentiable() {
            @Override
            public double[] apply(double[] x) {
                return new double[] {Math.sin(x[0])};
            }

            @Override
            public double[][] jacobian(double[] x) {
                return new double[][] {{Math.cos(x[0]), 0}};
            }
        };
        Accumulator ekf = new Accumulator();
        Accumulator pf = new Accumulator();
        for (int i = 0; i < test.states().length; i++) {
            double[][] ys = column(test.observations()[i]);
            StateSpaceFilter.Estimate e = StateSpaceFilter.extendedKalman(f, q, h, r, new double[] {0, 0},
                    new double[][] {{1.33, 0}, {0, 0.04}}, ys);
            ekf.add(e.mean(0), e.variance(0), test.states()[i]);
            StateSpaceFilter.Estimate p = StateSpaceFilter.particle(
                    rnd -> new double[] {rnd.nextDouble(-2, 2), 0.2 * rnd.nextGaussian()},
                    (x, rnd) -> {
                        double[] n = pendulumStep(x);
                        n[1] += 0.05 * rnd.nextGaussian();
                        return n;
                    },
                    (y, x) -> {
                        double d = y[0] - Math.sin(x[0]);
                        return -0.5 * d * d / 0.01;
                    }, ys, particles, seed + i);
            pf.add(p.mean(0), p.variance(0), test.states()[i]);
        }
        out.put("EKF", ekf.result());
        out.put("particle filter", pf.result());
        out.put("GRU (causal)", gru(train, test, seed));
        return out;
    }

    private static double[][] column(double[] v) {
        double[][] c = new double[v.length][1];
        for (int t = 0; t < v.length; t++) {
            c[t][0] = v[t];
        }
        return c;
    }

    /** RMSE and Gaussian NLL accumulated over trajectories. */
    private static final class Accumulator {
        private double se;
        private double nll;
        private int n;

        void add(double[] mean, double[] var, double[] truth) {
            for (int t = 0; t < truth.length; t++) {
                se += (mean[t] - truth[t]) * (mean[t] - truth[t]);
            }
            double[] v = new double[var.length];
            for (int t = 0; t < v.length; t++) {
                v[t] = Math.max(var[t], 1e-6);
            }
            nll += MlMetrics.gaussianNll(mean, v, truth) * truth.length;
            n += truth.length;
        }

        double[] result() {
            return new double[] {Math.sqrt(se / n), nll / n};
        }
    }

    /** A causal GRU mapping the observation sequence to the state sequence. */
    private static double[] gru(Trajectories train, Trajectories test, long seed) {
        int steps = train.states()[0].length;
        Model net = Model.sequential().input(steps, 1).gru(32, true).dense(1).seed(seed).build();
        Optimizer opt = Optimizer.adam(0.01f);
        SplittableRandom r = new SplittableRandom(seed);
        int batch = 32;
        for (int s = 0; s < 300; s++) {
            float[] x = new float[batch * steps];
            float[] y = new float[batch * steps];
            for (int b = 0; b < batch; b++) {
                int i = r.nextInt(train.states().length);
                for (int t = 0; t < steps; t++) {
                    x[b * steps + t] = (float) train.observations()[i][t];
                    y[b * steps + t] = (float) train.states()[i][t];
                }
            }
            net.forward(Var.of(Tensor.of(Shape.of(batch, steps, 1), x)), true).sub(Tensor.of(Shape.of(batch, steps, 1),
                    y)).square().mean().backward();
            opt.step(net.parameters());
        }
        double residual = mse(net, train);
        Accumulator acc = new Accumulator();
        double[][] pred = predict(net, test);
        for (int i = 0; i < pred.length; i++) {
            double[] var = new double[steps];
            java.util.Arrays.fill(var, residual);
            acc.add(pred[i], var, test.states()[i]);
        }
        return acc.result();
    }

    private static double[][] predict(Model net, Trajectories data) {
        int n = data.states().length;
        int steps = data.states()[0].length;
        float[] x = new float[n * steps];
        for (int i = 0; i < n; i++) {
            for (int t = 0; t < steps; t++) {
                x[i * steps + t] = (float) data.observations()[i][t];
            }
        }
        float[] p = net.forward(Var.of(Tensor.of(Shape.of(n, steps, 1), x)), false).value().toArray();
        double[][] out = new double[n][steps];
        for (int i = 0; i < n; i++) {
            for (int t = 0; t < steps; t++) {
                out[i][t] = p[i * steps + t];
            }
        }
        return out;
    }

    private static double mse(Model net, Trajectories data) {
        double[][] p = predict(net, data);
        double s = 0;
        int c = 0;
        for (int i = 0; i < p.length; i++) {
            for (int t = 0; t < p[i].length; t++) {
                s += (p[i][t] - data.states()[i][t]) * (p[i][t] - data.states()[i][t]);
                c++;
            }
        }
        return s / c;
    }
}
