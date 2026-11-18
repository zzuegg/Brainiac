package io.github.zzuegg.brainiac.experiments.sciml;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ml.StateSpaceFilter;
import io.github.zzuegg.brainiac.experimental.sciml.EnsembleKalmanFilter;
import io.github.zzuegg.brainiac.experimental.sciml.MechanicalSystems;
import io.github.zzuegg.brainiac.experimental.sciml.ScoreBasedAssimilation;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.UnaryOperator;

/**
 * Data assimilation (wave 8, item 8.3.16):
 *
 * <ul>
 *   <li>Lorenz-96 ({@code n = 40, F = 8}) observing every fourth variable: a localised,
 *       inflated stochastic {@link EnsembleKalmanFilter} against the same filter without
 *       localisation, the extended Kalman filter and the bootstrap particle filter of
 *       {@link StateSpaceFilter}, and a free run of the model.</li>
 *   <li>Lorenz-63 observing {@code x} only: {@link ScoreBasedAssimilation} (a diffusion prior
 *       over short trajectory windows, conditioned on the observations at sampling time)
 *       against the EnKF over the same windows.</li>
 * </ul>
 */
public final class DataAssimilation {

    private DataAssimilation() {
    }

    /**
     * A filter row.
     *
     * @param method the method
     * @param rmse root mean squared error of the state estimate over the scored cycles
     * @param spreadRatio mean ensemble spread over the RMSE (1 is well calibrated; NaN where
     *     not applicable)
     * @param millis wall time
     */
    public record FilterRow(String method, double rmse, double spreadRatio, double millis) {
    }

    private static UnaryOperator<double[]> step(MechanicalSystems.Ode ode, double dt, int substeps) {
        return x -> {
            double[] y = x;
            for (int s = 0; s < substeps; s++) {
                y = MechanicalSystems.rk4(ode.rhs(), y, dt / substeps);
            }
            return y;
        };
    }

    private static double rmse(double[][] estimate, double[][] truth, int from) {
        double s = 0;
        int c = 0;
        for (int t = from; t < truth.length; t++) {
            double e = 0;
            for (int i = 0; i < truth[t].length; i++) {
                e += (estimate[t][i] - truth[t][i]) * (estimate[t][i] - truth[t][i]);
            }
            s += Math.sqrt(e / truth[t].length);
            c++;
        }
        return s / c;
    }

    private static double mean(double[] v, int from) {
        double s = 0;
        for (int i = from; i < v.length; i++) {
            s += v[i];
        }
        return s / (v.length - from);
    }

    /**
     * 8.3.16, Lorenz-96: EnKF against EKF, particle filter and a free run.
     *
     * @param cycles assimilation cycles (interval 0.05)
     * @param members ensemble members
     * @param particles particle-filter particles
     * @param seed the seed
     * @return rows for the localised EnKF, the EnKF without localisation, the EKF, the
     *     particle filter and the free run
     */
    public static List<FilterRow> lorenz96(int cycles, int members, int particles, long seed) {
        int n = 40;
        double noise = 1.0;
        MechanicalSystems.Ode ode = MechanicalSystems.lorenz96(n, 8);
        UnaryOperator<double[]> model = step(ode, 0.05, 2);
        SplittableRandom r = new SplittableRandom(seed);
        double[] x = new double[n];
        for (int i = 0; i < n; i++) {
            x[i] = 8 + r.nextGaussian();
        }
        for (int t = 0; t < 500; t++) {
            x = model.apply(x);   // spin-up onto the attractor
        }
        double[][] truth = new double[cycles][];
        truth[0] = x;
        for (int t = 1; t < cycles; t++) {
            truth[t] = model.apply(truth[t - 1]);
        }
        int[] observed = new int[n / 4];
        for (int j = 0; j < observed.length; j++) {
            observed[j] = 4 * j;
        }
        double[][] ys = new double[cycles][observed.length];
        for (int t = 0; t < cycles; t++) {
            for (int j = 0; j < observed.length; j++) {
                ys[t][j] = truth[t][observed[j]] + noise * r.nextGaussian();
            }
        }
        // the prior: a climatological draw near the truth (a state 200 steps later on the attractor)
        double[] background = truth[0].clone();
        for (int t = 0; t < 200; t++) {
            background = model.apply(background);
        }
        double[] prior = background;
        int from = cycles / 4;
        List<FilterRow> rows = new ArrayList<>();
        for (double loc : new double[] {4, 0}) {
            long t0 = System.nanoTime();
            double[][] ens = new double[members][n];
            SplittableRandom er = new SplittableRandom(seed + 1);
            for (int k = 0; k < members; k++) {
                for (int i = 0; i < n; i++) {
                    ens[k][i] = prior[i] + 2 * er.nextGaussian();
                }
            }
            EnsembleKalmanFilter enkf = EnsembleKalmanFilter.of(model, observed, noise, 1.05, loc, true);
            EnsembleKalmanFilter.Estimate est = enkf.run(ens, ys, seed + 2);
            double e = rmse(est.means(), truth, from);
            rows.add(new FilterRow(loc > 0 ? "EnKF (" + members + ", localised)" : "EnKF (" + members
                    + ", no localisation)", e, mean(est.spreads(), from) / e, (System.nanoTime() - t0) / 1e6));
        }
        // EKF with a finite-difference Jacobian of the step
        long t0 = System.nanoTime();
        StateSpaceFilter.Differentiable f = new StateSpaceFilter.Differentiable() {
            @Override
            public double[] apply(double[] v) {
                return model.apply(v);
            }

            @Override
            public double[][] jacobian(double[] v) {
                double h = 1e-5;
                double[][] j = new double[n][n];
                for (int c = 0; c < n; c++) {
                    double[] p = v.clone();
                    double[] m = v.clone();
                    p[c] += h;
                    m[c] -= h;
                    double[] fp = model.apply(p);
                    double[] fm = model.apply(m);
                    for (int row = 0; row < n; row++) {
                        j[row][c] = (fp[row] - fm[row]) / (2 * h);
                    }
                }
                return j;
            }
        };
        StateSpaceFilter.Differentiable h = new StateSpaceFilter.Differentiable() {
            @Override
            public double[] apply(double[] v) {
                double[] y = new double[observed.length];
                for (int j = 0; j < observed.length; j++) {
                    y[j] = v[observed[j]];
                }
                return y;
            }

            @Override
            public double[][] jacobian(double[] v) {
                double[][] j = new double[observed.length][n];
                for (int k = 0; k < observed.length; k++) {
                    j[k][observed[k]] = 1;
                }
                return j;
            }
        };
        double[][] q = diagonal(n, 0.05);
        double[][] rr = diagonal(observed.length, noise * noise);
        StateSpaceFilter.Estimate ekf = StateSpaceFilter.extendedKalman(f, q, h, rr, prior, diagonal(n, 4), ys);
        double ekfErr = rmse(ekf.means(), truth, from);
        double ekfSpread = 0;
        for (int t = from; t < cycles; t++) {
            double tr = 0;
            for (int i = 0; i < n; i++) {
                tr += ekf.covariances()[t][i][i];
            }
            ekfSpread += Math.sqrt(tr / n) / (cycles - from);
        }
        rows.add(new FilterRow("EKF", ekfErr, ekfSpread / ekfErr, (System.nanoTime() - t0) / 1e6));
        // bootstrap particle filter, with a little process noise so particles can move
        t0 = System.nanoTime();
        StateSpaceFilter.Estimate pf = StateSpaceFilter.particle(rnd -> {
            double[] p = prior.clone();
            for (int i = 0; i < n; i++) {
                p[i] += 2 * rnd.nextGaussian();
            }
            return p;
        }, (v, rnd) -> {
            double[] next = model.apply(v);
            for (int i = 0; i < n; i++) {
                next[i] += 0.2 * rnd.nextGaussian();
            }
            return next;
        }, (y, v) -> {
            double s = 0;
            for (int j = 0; j < observed.length; j++) {
                double d = y[j] - v[observed[j]];
                s += d * d;
            }
            return -0.5 * s / (noise * noise);
        }, ys, particles, seed + 3);
        double pfErr = rmse(pf.means(), truth, from);
        double pfSpread = 0;
        for (int t = from; t < cycles; t++) {
            double tr = 0;
            for (int i = 0; i < n; i++) {
                tr += pf.covariances()[t][i][i];
            }
            pfSpread += Math.sqrt(Math.max(0, tr) / n) / (cycles - from);
        }
        rows.add(new FilterRow("particle filter (" + particles + ")", pfErr, pfSpread / pfErr,
                (System.nanoTime() - t0) / 1e6));
        // free run from the prior
        double[][] free = new double[cycles][];
        free[0] = prior;
        for (int t = 1; t < cycles; t++) {
            free[t] = model.apply(free[t - 1]);
        }
        rows.add(new FilterRow("free run (no assimilation)", rmse(free, truth, from), Double.NaN, 0));
        return rows;
    }

    private static double[][] diagonal(int n, double v) {
        double[][] d = new double[n][n];
        for (int i = 0; i < n; i++) {
            d[i][i] = v;
        }
        return d;
    }

    /**
     * 8.3.16, Lorenz-63: score-based data assimilation against the EnKF, observing {@code x}
     * with noise at every step of size 0.05.
     *
     * @param window the SDA window length
     * @param trainSteps denoiser training steps
     * @param windows test windows
     * @param samples posterior samples per window
     * @param seed the seed
     * @return rows for SDA (posterior mean, a smoother), the EnKF filter means on the same
     *     windows, and a free run
     */
    public static List<FilterRow> lorenz63(int window, int trainSteps, int windows, int samples, long seed) {
        MechanicalSystems.Ode ode = MechanicalSystems.lorenz63(10, 28, 8.0 / 3);
        double dt = 0.05;
        UnaryOperator<double[]> model = step(ode, dt, 5);
        double noise = 2.0;
        SplittableRandom r = new SplittableRandom(seed);
        double[] x = {1, 1, 20};
        for (int t = 0; t < 200; t++) {
            x = model.apply(x);
        }
        double[][] train = new double[4000][];
        train[0] = x;
        for (int t = 1; t < train.length; t++) {
            train[t] = model.apply(train[t - 1]);
        }
        long t0 = System.nanoTime();
        ScoreBasedAssimilation sda = ScoreBasedAssimilation.of(train, window, 64, 2, 100, seed);
        Optimizer opt = Optimizer.adam(2e-3f);
        for (int s = 0; s < trainSteps; s++) {
            Var loss = sda.loss(sda.windows(train, 64, r), r);
            loss.backward();
            opt.step(sda.parameters());
        }
        double trainMillis = (System.nanoTime() - t0) / 1e6;
        // a separate test trajectory, observed in x
        int length = windows * window;
        double[][] truth = new double[length][];
        truth[0] = model.apply(train[train.length - 1]);
        for (int t = 0; t < 300; t++) {
            truth[0] = model.apply(truth[0]);
        }
        for (int t = 1; t < length; t++) {
            truth[t] = model.apply(truth[t - 1]);
        }
        double[][] ys = new double[length][1];
        for (int t = 0; t < length; t++) {
            ys[t][0] = truth[t][0] + noise * r.nextGaussian();
        }
        boolean[][] observed = new boolean[window][3];
        for (boolean[] o : observed) {
            o[0] = true;
        }
        t0 = System.nanoTime();
        double[][][] obs = new double[windows * samples][window][3];
        for (int w = 0; w < windows; w++) {
            for (int k = 0; k < samples; k++) {
                for (int t = 0; t < window; t++) {
                    obs[w * samples + k][t][0] = ys[w * window + t][0];
                }
            }
        }
        double[][][] post = sda.sample(obs, observed, noise, 0.5, 50, new SplittableRandom(seed + 9));
        double[][] sdaMean = new double[length][3];
        double[] sdaSpread = new double[length];
        for (int w = 0; w < windows; w++) {
            for (int t = 0; t < window; t++) {
                int idx = w * window + t;
                for (int i = 0; i < 3; i++) {
                    double m = 0;
                    for (int k = 0; k < samples; k++) {
                        m += post[w * samples + k][t][i] / samples;
                    }
                    double v = 0;
                    for (int k = 0; k < samples; k++) {
                        v += Math.pow(post[w * samples + k][t][i] - m, 2) / Math.max(1, samples - 1);
                    }
                    sdaMean[idx][i] = m;
                    sdaSpread[idx] += v / 3;
                }
                sdaSpread[idx] = Math.sqrt(sdaSpread[idx]);
            }
        }
        double sdaMillis = trainMillis + (System.nanoTime() - t0) / 1e6;
        List<FilterRow> rows = new ArrayList<>();
        double sdaErr = rmse(sdaMean, truth, 0);
        rows.add(new FilterRow("SDA (window " + window + ", " + samples + " samples, smoother)", sdaErr,
                mean(sdaSpread, 0) / sdaErr, sdaMillis));
        // EnKF from a climatological ensemble
        t0 = System.nanoTime();
        int members = 20;
        double[][] ens = new double[members][];
        SplittableRandom er = new SplittableRandom(seed + 4);
        for (int k = 0; k < members; k++) {
            ens[k] = train[er.nextInt(train.length)].clone();
        }
        EnsembleKalmanFilter enkf = EnsembleKalmanFilter.of(model, new int[] {0}, noise, 1.02, 0, false);
        EnsembleKalmanFilter.Estimate est = enkf.run(ens, ys, seed + 5);
        int from = Math.min(length / 4, 20);
        double enkfErr = rmse(est.means(), truth, from);
        rows.add(new FilterRow("EnKF (20, filter)", enkfErr, mean(est.spreads(), from) / enkfErr,
                (System.nanoTime() - t0) / 1e6));
        // SDA scored on the same cycles as the EnKF, for a like-for-like number
        rows.add(new FilterRow("SDA on the EnKF's scored cycles", rmse(sdaMean, truth, from), Double.NaN, 0));
        double[][] free = new double[length][];
        free[0] = train[1234];
        for (int t = 1; t < length; t++) {
            free[t] = model.apply(free[t - 1]);
        }
        rows.add(new FilterRow("free run (no assimilation)", rmse(free, truth, 0), Double.NaN, 0));
        return rows;
    }
}
