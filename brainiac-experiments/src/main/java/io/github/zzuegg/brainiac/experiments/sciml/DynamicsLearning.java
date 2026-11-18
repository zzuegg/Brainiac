package io.github.zzuegg.brainiac.experiments.sciml;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.evolve.SymbolicRegression;
import io.github.zzuegg.brainiac.experimental.ml.EntropicTransport;
import io.github.zzuegg.brainiac.experimental.nn.EchoStateReservoir;
import io.github.zzuegg.brainiac.experimental.nn.EquivariantGraphLayer;
import io.github.zzuegg.brainiac.experimental.nn.NeuralOde;
import io.github.zzuegg.brainiac.experimental.nn.OdeSolver;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experimental.sciml.DynamicModeDecomposition;
import io.github.zzuegg.brainiac.experimental.sciml.GraphNetworkSimulator;
import io.github.zzuegg.brainiac.experimental.sciml.HamiltonianNetwork;
import io.github.zzuegg.brainiac.experimental.sciml.KoopmanAutoencoder;
import io.github.zzuegg.brainiac.experimental.sciml.LagrangianNetwork;
import io.github.zzuegg.brainiac.experimental.sciml.LatentSde;
import io.github.zzuegg.brainiac.experimental.sciml.MechanicalSystems;
import io.github.zzuegg.brainiac.experimental.sciml.NeuralSde;
import io.github.zzuegg.brainiac.experimental.sciml.PdeProblems;
import io.github.zzuegg.brainiac.experimental.sciml.SdeSolver;
import io.github.zzuegg.brainiac.experimental.sciml.Sindy;
import io.github.zzuegg.brainiac.experimental.sciml.SymplecticIntegrator;
import io.github.zzuegg.brainiac.experimental.sciml.UniversalDifferentialEquation;
import io.github.zzuegg.brainiac.experiments.exotic.ReservoirForecast;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.UnaryOperator;

/**
 * Learned dynamics (wave 8, items 8.3.8-8.3.13), each against the named baselines on
 * {@link MechanicalSystems} ground truth:
 *
 * <ul>
 *   <li>{@link #hamiltonian} (8.3.8): Hamiltonian and Lagrangian networks against an MLP
 *       derivative model rolled out with Euler and with RK4; rollout error and energy drift
 *       over 1,000 steps.
 *   <li>{@link #sindy} (8.3.9): SINDy against genetic-programming symbolic regression and an
 *       MLP derivative model; structure recovery, coefficient error and time.
 *   <li>{@link #koopman} (8.3.10): DMD, EDMD and a Koopman autoencoder against an echo state
 *       network, NG-RC and an MLP one-step model; forecast error and valid prediction time.
 *   <li>{@link #ude} (8.3.11): a universal differential equation against a pure neural ODE and
 *       the known physics alone at several training-set sizes, then SINDy on the learned term.
 *   <li>{@link #stochastic} (8.3.12): a neural SDE against a deterministic neural ODE drift and
 *       a quantile MLP on Ornstein-Uhlenbeck and double-well data, plus a latent SDE on
 *       irregularly sampled double-well series.
 *   <li>{@link #particles} (8.3.13): a graph network simulator with and without input noise,
 *       EGNN and a fixed-size MLP on spring chains, rolled out at the training size and at
 *       twice the particles.
 * </ul>
 */
public final class DynamicsLearning {

    private DynamicsLearning() {
    }

    // ------------------------------------------------------------------ shared helpers

    static Tensor tensor(double[][] x) {
        return HamiltonianNetwork.tensor(x);
    }

    static double[][] rows(Tensor t) {
        int n = t.shape().dim(0);
        int k = t.shape().dim(1);
        float[] f = t.toArray();
        double[][] out = new double[n][k];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < k; j++) {
                out[i][j] = f[i * k + j];
            }
        }
        return out;
    }

    static double mse(double[][] a, double[][] b) {
        double s = 0;
        int count = 0;
        for (int t = 0; t < Math.min(a.length, b.length); t++) {
            for (int j = 0; j < a[t].length; j++) {
                double d = a[t][j] - b[t][j];
                s += d * d;
                count++;
            }
        }
        return s / count;
    }

    static void fit(List<Var> params, java.util.function.Supplier<Var> loss, int steps, float lr) {
        Optimizer opt = Optimizer.adam(lr);
        for (int s = 0; s < steps; s++) {
            Var l = loss.get();
            l.backward();
            opt.step(params);
        }
    }

    // ------------------------------------------------------------------ 8.3.8

    /**
     * A conservative-dynamics row.
     *
     * @param model the model and integrator
     * @param rolloutMse mean squared error of the rollout against the truth
     * @param energyDrift relative change of the true energy over the rollout (for a damped
     *     system: relative difference from the true final energy)
     */
    public record EnergyRow(String model, double rolloutMse, double energyDrift) {
    }

    /**
     * 8.3.8: Hamiltonian and Lagrangian networks against MLP derivative models.
     *
     * @param ode the system (one degree of freedom, state {@code (q, p)} with unit mass)
     * @param samples training states
     * @param noise derivative noise standard deviation
     * @param steps training steps
     * @param rollout rollout steps of size 0.1
     * @param seed the seed
     * @return rows for HNN (Stormer-Verlet), LNN (RK4), MLP (Euler) and MLP (RK4)
     */
    public static List<EnergyRow> hamiltonian(MechanicalSystems.Ode ode, int samples, double noise, int steps,
            int rollout, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] x = new double[samples][];
        double[][] dx = new double[samples][];
        for (int i = 0; i < samples; i++) {
            x[i] = new double[] {r.nextDouble(-2, 2), r.nextDouble(-2, 2)};
            double[] d = ode.derivative(x[i]);
            dx[i] = new double[] {d[0] + noise * r.nextGaussian(), d[1] + noise * r.nextGaussian()};
        }
        Tensor xt = tensor(x);
        Tensor dt = tensor(dx);
        double h = 0.1;
        double[] x0 = {1.0, 0.0};
        double[][] truth = MechanicalSystems.trajectory(ode, x0, h, rollout, 10);
        List<EnergyRow> rows = new ArrayList<>();
        HamiltonianNetwork hnn = HamiltonianNetwork.of(1, 64, 2, seed);
        fit(hnn.parameters(), () -> hnn.loss(xt, dt), steps, 5e-3f);
        rows.add(energyRow("HNN + Stormer-Verlet", SymplecticIntegrator.rollout(hnn.gradient(), x0, h, rollout), truth,
                ode));
        LagrangianNetwork lnn = LagrangianNetwork.of(1, 64, 2, seed);
        double[][] acc = new double[samples][];
        for (int i = 0; i < samples; i++) {
            acc[i] = new double[] {dx[i][1]};
        }
        Tensor at = tensor(acc);
        fit(lnn.parameters(), () -> lnn.loss(xt, at), steps, 5e-3f);
        rows.add(energyRow("LNN + RK4", rk4Rollout(lnn.vectorField(), x0, h, rollout), truth, ode));
        SimpleMlp mlp = SimpleMlp.tanh(seed, 2, 64, 64, 2);
        fit(mlp.parameters(), () -> mlp.forward(Var.of(xt)).sub(dt).square().mean(), steps, 5e-3f);
        UnaryOperator<double[]> field = v -> rows(mlp.predict(tensor(new double[][] {v})))[0];
        rows.add(energyRow("MLP + Euler", eulerRollout(field, x0, h, rollout), truth, ode));
        rows.add(energyRow("MLP + RK4", rk4Rollout(field, x0, h, rollout), truth, ode));
        return rows;
    }

    static double[][] rk4Rollout(UnaryOperator<double[]> f, double[] x0, double h, int steps) {
        double[][] out = new double[steps + 1][];
        out[0] = x0.clone();
        for (int t = 1; t <= steps; t++) {
            out[t] = MechanicalSystems.rk4(f, out[t - 1], h);
        }
        return out;
    }

    static double[][] eulerRollout(UnaryOperator<double[]> f, double[] x0, double h, int steps) {
        double[][] out = new double[steps + 1][];
        out[0] = x0.clone();
        for (int t = 1; t <= steps; t++) {
            double[] d = f.apply(out[t - 1]);
            out[t] = new double[x0.length];
            for (int i = 0; i < x0.length; i++) {
                out[t][i] = out[t - 1][i] + h * d[i];
            }
        }
        return out;
    }

    private static EnergyRow energyRow(String name, double[][] pred, double[][] truth, MechanicalSystems.Ode ode) {
        double e0 = ode.energy().applyAsDouble(truth[0]);
        double eTrue = ode.energy().applyAsDouble(truth[truth.length - 1]);
        double ePred = ode.energy().applyAsDouble(pred[pred.length - 1]);
        double drift = Math.abs(ePred - eTrue) / Math.abs(e0);
        return new EnergyRow(name, mse(pred, truth), Double.isFinite(drift) ? drift : Double.MAX_VALUE);
    }

    // ------------------------------------------------------------------ 8.3.9

    /**
     * An equation-discovery row.
     *
     * @param method the method
     * @param structure share of library terms correctly classified as present or absent (NaN
     *     where not applicable)
     * @param coefficientError relative coefficient error on the true support, or relative
     *     derivative error for black-box models
     * @param millis fit time
     * @param equations the discovered equations, if any
     */
    public record DiscoveryRow(String method, double structure, double coefficientError, double millis,
            String equations) {
    }

    /** A system with its true SINDy coefficients over a degree-2 polynomial library. */
    private record Truth(MechanicalSystems.Ode ode, String[] names, double[] x0, double dt, int steps,
            double threshold, double[][] coefficients) {
    }

    private static Truth truth(String system) {
        return switch (system) {
            case "lorenz" -> {
                Sindy.Library lib = Sindy.polynomial(new String[] {"x", "y", "z"}, 2, false);
                double[][] c = new double[lib.names().length][3];
                set(c, lib, "x", 0, -10);
                set(c, lib, "y", 0, 10);
                set(c, lib, "x", 1, 28);
                set(c, lib, "y", 1, -1);
                set(c, lib, "x*z", 1, -1);
                set(c, lib, "x*y", 2, 1);
                set(c, lib, "z", 2, -8.0 / 3);
                yield new Truth(MechanicalSystems.lorenz63(10, 28, 8.0 / 3), new String[] {"x", "y", "z"},
                        new double[] {-8, 7, 27}, 0.005, 2000, 0.25, c);
            }
            case "lotka-volterra" -> {
                Sindy.Library lib = Sindy.polynomial(new String[] {"x", "y"}, 2, false);
                double[][] c = new double[lib.names().length][2];
                set(c, lib, "x", 0, 1.0);
                set(c, lib, "x*y", 0, -0.5);
                set(c, lib, "y", 1, -1.0);
                set(c, lib, "x*y", 1, 0.25);
                yield new Truth(MechanicalSystems.lotkaVolterra(1.0, 0.5, 1.0, 0.25), new String[] {"x", "y"},
                        new double[] {3, 1}, 0.01, 2000, 0.05, c);
            }
            default -> {
                Sindy.Library lib = Sindy.polynomial(new String[] {"x", "v"}, 2, false);
                double[][] c = new double[lib.names().length][2];
                set(c, lib, "v", 0, 1);
                set(c, lib, "x", 1, -1);
                set(c, lib, "v", 1, -0.2);
                yield new Truth(MechanicalSystems.dampedOscillator(1, 0.2), new String[] {"x", "v"},
                        new double[] {2, 0}, 0.01, 2000, 0.05, c);
            }
        };
    }

    private static void set(double[][] c, Sindy.Library lib, String term, int eq, double v) {
        c[java.util.Arrays.asList(lib.names()).indexOf(term)][eq] = v;
    }

    /**
     * 8.3.9: SINDy on one system at one noise level, with genetic programming and an MLP as
     * baselines.
     *
     * @param system {@code "lorenz"}, {@code "lotka-volterra"} or {@code "damped"}
     * @param noise noise level relative to each variable's standard deviation
     * @param baselines whether to run the GP and MLP baselines
     * @param seed the seed
     * @return rows for SINDy and optionally GP and MLP
     */
    public static List<DiscoveryRow> sindy(String system, double noise, boolean baselines, long seed) {
        Truth t = truth(system);
        double[][] clean = MechanicalSystems.trajectory(t.ode(), t.x0(), t.dt(), t.steps(), 5);
        double[][] x = MechanicalSystems.noisy(clean, noise, new SplittableRandom(seed));
        List<DiscoveryRow> rows = new ArrayList<>();
        long start = System.nanoTime();
        double[][] dx = noise > 0 ? Sindy.derivatives(x, t.dt(), 15, 3) : Sindy.derivatives(x, t.dt(), 5, 4);
        double[][] xs = noise > 0 ? smooth(x, t.dt()) : x;
        Sindy.Library lib = Sindy.polynomial(t.names(), 2, false);
        Sindy model = Sindy.fit(xs, dx, lib, t.threshold());
        double ms = (System.nanoTime() - start) / 1e6;
        double[] score = model.score(t.coefficients());
        rows.add(new DiscoveryRow("SINDy", score[0], score[1], ms, model.equations(t.names()).strip()));
        if (!baselines) {
            return rows;
        }
        // true derivatives at the clean states, for the black-box scores
        double[][] trueDx = new double[clean.length][];
        for (int i = 0; i < clean.length; i++) {
            trueDx[i] = t.ode().derivative(clean[i]);
        }
        // genetic programming, one equation at a time, on a subsample
        start = System.nanoTime();
        int sub = 200;
        double[][] xsSub = new double[sub][];
        double[][] dxSub = new double[sub][];
        for (int i = 0; i < sub; i++) {
            xsSub[i] = xs[i * (xs.length / sub)];
            dxSub[i] = dx[i * (xs.length / sub)];
        }
        double num = 0;
        double den = 0;
        StringBuilder eqs = new StringBuilder();
        for (int j = 0; j < t.names().length; j++) {
            double[] y = new double[sub];
            for (int i = 0; i < sub; i++) {
                y[i] = dxSub[i][j];
            }
            SymbolicRegression gp = new SymbolicRegression(SymbolicRegression.Selection.TOURNAMENT, 200, 40, 15, 1e-3);
            SymbolicRegression.Fit f = gp.fit(xsSub, y, seed + j);
            eqs.append(t.names()[j]).append("' = ").append(SymbolicRegression.simplify(f.best())).append("; ");
            for (int i = 0; i < sub; i++) {
                double p = f.best().eval(xsSub[i]);
                double tr = trueDx[i * (xs.length / sub)][j];
                num += (p - tr) * (p - tr);
                den += tr * tr;
            }
        }
        rows.add(new DiscoveryRow("genetic programming", Double.NaN, Math.sqrt(num / den),
                (System.nanoTime() - start) / 1e6, eqs.toString().strip()));
        // MLP derivative model (no closed form)
        start = System.nanoTime();
        double[] scale = new double[t.names().length];
        for (double[] v : xs) {
            for (int j = 0; j < scale.length; j++) {
                scale[j] = Math.max(scale[j], Math.abs(v[j]));
            }
        }
        double[][] xn = new double[xs.length][scale.length];
        for (int i = 0; i < xs.length; i++) {
            for (int j = 0; j < scale.length; j++) {
                xn[i][j] = xs[i][j] / scale[j];
            }
        }
        double dscale = 0;
        for (double[] v : dx) {
            for (double d : v) {
                dscale = Math.max(dscale, Math.abs(d));
            }
        }
        double ds = dscale;
        double[][] dn = new double[dx.length][];
        for (int i = 0; i < dx.length; i++) {
            dn[i] = dx[i].clone();
            for (int j = 0; j < dn[i].length; j++) {
                dn[i][j] /= ds;
            }
        }
        SimpleMlp mlp = SimpleMlp.tanh(seed, scale.length, 64, 64, scale.length);
        Tensor xin = tensor(xn);
        Tensor yout = tensor(dn);
        fit(mlp.parameters(), () -> mlp.forward(Var.of(xin)).sub(yout).square().mean(), 300, 1e-2f);
        double[][] pred = rows(mlp.predict(xin));
        num = 0;
        den = 0;
        for (int i = 0; i < pred.length; i++) {
            for (int j = 0; j < scale.length; j++) {
                double d = pred[i][j] * ds - trueDx[i][j];
                num += d * d;
                den += trueDx[i][j] * trueDx[i][j];
            }
        }
        rows.add(new DiscoveryRow("MLP derivative model", Double.NaN, Math.sqrt(num / den),
                (System.nanoTime() - start) / 1e6, ""));
        return rows;
    }

    /** A centred moving average of width 9, to denoise states before building the library. */
    private static double[][] smooth(double[][] x, double dt) {
        int n = x.length;
        int d = x[0].length;
        double[][] out = new double[n][d];
        for (int i = 0; i < n; i++) {
            int lo = Math.max(0, i - 4);
            int hi = Math.min(n - 1, i + 4);
            for (int k = lo; k <= hi; k++) {
                for (int j = 0; j < d; j++) {
                    out[i][j] += x[k][j] / (hi - lo + 1);
                }
            }
        }
        return out;
    }

    // ------------------------------------------------------------------ 8.3.10

    /**
     * A forecasting row.
     *
     * @param model the model
     * @param forecastError mean squared error over the horizon, normalised by the series variance
     * @param validSteps steps until the normalised error first exceeds 0.3
     */
    public record ForecastRow(String model, double forecastError, int validSteps) {
    }

    /**
     * 8.3.10, linear field: DMD on a 1-D advection-diffusion field, where it should be exact.
     *
     * @param points grid points
     * @param snapshots training snapshots
     * @param horizon forecast steps
     * @return the DMD row
     */
    public static ForecastRow dmdAdvectionDiffusion(int points, int snapshots, int horizon) {
        double[] u0 = PdeProblems.randomField(points, 6, 1, new SplittableRandom(3));
        double dt = 0.05;
        int total = snapshots + horizon;
        double[][] adv = PdeProblems.solve(PdeProblems.Equation.ADVECTION, u0, 0, 2 * Math.PI, 1, total * dt,
                total + 1, dt).u();
        double[][] field = new double[total + 1][];
        for (int t = 0; t <= total; t++) {
            field[t] = PdeProblems.solve(PdeProblems.Equation.HEAT, adv[t], 0, 2 * Math.PI, 0.02, t * dt + 1e-12, 2, dt)
                    .last();
        }
        DynamicModeDecomposition dmd = DynamicModeDecomposition.dmd(java.util.Arrays.copyOfRange(field, 0,
                snapshots + 1), points);
        double[][] f = dmd.forecast(field[snapshots], horizon);
        return forecastRow("DMD", java.util.Arrays.copyOfRange(f, 1, f.length), field, snapshots + 1);
    }

    private static ForecastRow forecastRow(String name, double[][] predicted, double[][] series, int start) {
        int d = series[0].length;
        double var = 0;
        double[] mean = new double[d];
        for (double[] v : series) {
            for (int j = 0; j < d; j++) {
                mean[j] += v[j] / series.length;
            }
        }
        for (double[] v : series) {
            for (int j = 0; j < d; j++) {
                var += (v[j] - mean[j]) * (v[j] - mean[j]) / (series.length * d);
            }
        }
        double err = 0;
        int valid = predicted.length;
        for (int t = 0; t < predicted.length; t++) {
            double e = 0;
            for (int j = 0; j < d; j++) {
                double diff = predicted[t][j] - series[start + t][j];
                e += diff * diff / d;
            }
            err += e / predicted.length;
            if (valid == predicted.length && !(Math.sqrt(e / var) < 0.3)) {
                valid = t;
            }
        }
        return new ForecastRow(name, Double.isFinite(err / var) ? err / var : Double.MAX_VALUE, valid);
    }

    /**
     * 8.3.10: Koopman methods against reservoir computers and an MLP on one system.
     *
     * @param ode the system
     * @param x0 the initial state
     * @param dt the sample interval
     * @param train training samples
     * @param horizon forecast steps
     * @param steps Koopman autoencoder and MLP training steps
     * @param seed the seed
     * @return rows for DMD, EDMD, Koopman AE, ESN, NG-RC and the MLP
     */
    public static List<ForecastRow> koopman(MechanicalSystems.Ode ode, double[] x0, double dt, int train, int horizon,
            int steps, long seed) {
        double[][] series = MechanicalSystems.trajectory(ode, x0, dt, train + horizon, 10);
        int d = x0.length;
        double[][] trainPart = java.util.Arrays.copyOfRange(series, 0, train);
        double[] start = series[train - 1];
        List<ForecastRow> rows = new ArrayList<>();
        DynamicModeDecomposition dmd = DynamicModeDecomposition.dmd(trainPart, d);
        rows.add(forecastRow("DMD", tail(dmd.forecast(start, horizon)), series, train));
        Sindy.Library lib = Sindy.polynomial(names(d), 3, false);
        int constant = 0;
        UnaryOperator<double[]> dict = x -> {
            double[] f = lib.features().apply(x);
            // state first, then the other observables
            double[] out = new double[f.length - 1];
            System.arraycopy(f, 1, out, 0, f.length - 1);
            return out;
        };
        DynamicModeDecomposition edmd = DynamicModeDecomposition.edmd(trainPart, dict, 1e-8 + constant);
        rows.add(forecastRow("EDMD (cubic)", tail(edmd.forecast(start, horizon)), series, train));
        // Koopman autoencoder on normalised states
        double[] scale = new double[d];
        for (double[] v : trainPart) {
            for (int j = 0; j < d; j++) {
                scale[j] = Math.max(scale[j], Math.abs(v[j]));
            }
        }
        double[][] norm = new double[series.length][d];
        for (int t = 0; t < series.length; t++) {
            for (int j = 0; j < d; j++) {
                norm[t][j] = series[t][j] / scale[j];
            }
        }
        KoopmanAutoencoder kae = KoopmanAutoencoder.of(d, 6, 48, seed);
        SplittableRandom r = new SplittableRandom(seed);
        int window = 8;
        fit(kae.parameters(), () -> kae.loss(windows(norm, train, window, 32, r), 0.5f), steps, 3e-3f);
        double[][] kf = kae.forecast(norm[train - 1], horizon);
        for (double[] v : kf) {
            for (int j = 0; j < d; j++) {
                v[j] *= scale[j];
            }
        }
        rows.add(forecastRow("Koopman autoencoder", tail(kf), series, train));
        EchoStateReservoir esn = EchoStateReservoir.of(d, 200, 0.9, 0.5, 0.6, 0.05, seed);
        rows.add(forecastRow("ESN", ReservoirForecast.echoState(series, train, horizon, esn, 50, 1e-6).predicted(),
                series, train));
        rows.add(forecastRow("NG-RC", ReservoirForecast.nextGeneration(series, train, horizon, 2, 1, 1e-2).predicted(),
                series, train));
        SimpleMlp mlp = SimpleMlp.tanh(seed, d, 64, 64, d);
        double[][] a = java.util.Arrays.copyOfRange(norm, 0, train - 1);
        double[][] b = java.util.Arrays.copyOfRange(norm, 1, train);
        Tensor at = tensor(a);
        Tensor bt = tensor(b);
        fit(mlp.parameters(), () -> mlp.forward(Var.of(at)).add(at).sub(bt).square().mean(), steps, 3e-3f);
        double[][] mf = new double[horizon][];
        double[] cur = norm[train - 1];
        for (int t = 0; t < horizon; t++) {
            double[] dlt = rows(mlp.predict(tensor(new double[][] {cur})))[0];
            double[] next = new double[d];
            for (int j = 0; j < d; j++) {
                next[j] = cur[j] + dlt[j];
            }
            cur = next;
            mf[t] = new double[d];
            for (int j = 0; j < d; j++) {
                mf[t][j] = next[j] * scale[j];
            }
        }
        rows.add(forecastRow("MLP one-step", mf, series, train));
        return rows;
    }

    private static double[][] tail(double[][] forecast) {
        return java.util.Arrays.copyOfRange(forecast, 1, forecast.length);
    }

    private static String[] names(int d) {
        String[] n = new String[d];
        for (int i = 0; i < d; i++) {
            n[i] = "x" + i;
        }
        return n;
    }

    private static Tensor windows(double[][] series, int limit, int window, int count, SplittableRandom r) {
        int d = series[0].length;
        float[] f = new float[count * (window + 1) * d];
        for (int c = 0; c < count; c++) {
            int s = r.nextInt(limit - window);
            for (int t = 0; t <= window; t++) {
                for (int j = 0; j < d; j++) {
                    f[(c * (window + 1) + t) * d + j] = (float) series[s + t][j];
                }
            }
        }
        return Tensor.of(Shape.of(count, window + 1, d), f);
    }

    /**
     * The dominant oscillation frequency (radians per unit time) from discrete-time
     * eigenvalues.
     *
     * @param eigenvalues {@code [k][2]}
     * @param dt the sample interval
     * @return the frequency of the eigenvalue with the largest modulus and a non-zero angle
     */
    public static double dominantFrequency(double[][] eigenvalues, double dt) {
        double best = -1;
        double freq = 0;
        for (double[] e : eigenvalues) {
            double mod = Math.hypot(e[0], e[1]);
            double ang = Math.abs(Math.atan2(e[1], e[0]));
            if (ang > 1e-6 && mod > best) {
                best = mod;
                freq = ang / dt;
            }
        }
        return freq;
    }

    // ------------------------------------------------------------------ 8.3.11

    /**
     * A data-efficiency row.
     *
     * @param model the model
     * @param points training points
     * @param extrapolationError mean squared error beyond the training window
     */
    public record UdeRow(String model, int points, double extrapolationError) {
    }

    /**
     * The Lotka-Volterra system of the UDE paper ({@code alpha = 1.3, beta = 0.9, gamma = 0.8,
     * delta = 1.8}): the known part is the linear growth and decay, the interaction is unknown.
     *
     * @return the system
     */
    public static MechanicalSystems.Ode udeLotkaVolterra() {
        return MechanicalSystems.lotkaVolterra(1.3, 0.9, 1.8, 0.8);
    }

    /**
     * 8.3.11: UDE against a pure neural ODE and the known physics alone.
     *
     * @param points training points on {@code t in [0, 3]}
     * @param steps training steps
     * @param seed the seed
     * @param recovered receives the SINDy recovery of the UDE's learned term, if not {@code null}
     * @return rows for the UDE, the UDE with its SINDy-recovered term, the physics alone and
     *     the neural ODE
     */
    public static List<UdeRow> ude(int points, int steps, long seed, StringBuilder recovered) {
        MechanicalSystems.Ode lv = udeLotkaVolterra();
        double dt = 3.0 / (points - 1);
        double[] x0 = {0.44249296, 4.6280594};
        double[][] train = MechanicalSystems.trajectory(lv, x0, dt, points - 1, 20);
        double testDt = 0.1;
        double[][] full = MechanicalSystems.trajectory(lv, x0, testDt, 60, 10);
        double[][] beyond = java.util.Arrays.copyOfRange(full, 31, 61);
        UniversalDifferentialEquation.KnownPhysics known = (x, p) -> Var.concat(1,
                x.slice(1, 0, 1).mul(p.get(0)), x.slice(1, 1, 2).mul(p.get(1)).neg());
        int sub = Math.max(4, (int) Math.ceil(dt / 0.05));
        List<UdeRow> rows = new ArrayList<>();
        UniversalDifferentialEquation ude = UniversalDifferentialEquation.of(2, known, new float[] {1.3f, 1.8f}, 16,
                OdeSolver.rk4(sub), (float) dt, 5f, seed);
        fit(ude.parameters(), () -> ude.loss(train), steps, 1e-2f);
        rows.add(new UdeRow("UDE (known linear terms + NN)", points, extrapolate(ude, train[points - 1], dt, full)));
        UniversalDifferentialEquation physics = UniversalDifferentialEquation.of(2, known, new float[] {1.3f, 1.8f}, 0,
                OdeSolver.rk4(sub), (float) dt, 1f, seed);
        fit(physics.parameters(), () -> physics.loss(train), steps, 1e-2f);
        rows.add(new UdeRow("known physics only", points, extrapolate(physics, train[points - 1], dt, full)));
        NeuralOde node = new NeuralOde(2, 32, OdeSolver.rk4(sub), (float) dt, NeuralOde.Gradient.DIRECT,
                new SplittableRandom(seed));
        double[][] a = java.util.Arrays.copyOfRange(train, 0, points - 1);
        double[][] b = java.util.Arrays.copyOfRange(train, 1, points);
        Tensor at = tensor(a);
        Tensor bt = tensor(b);
        fit(node.parameters(), () -> node.forward(Var.of(at)).sub(bt).square().mean(), steps, 1e-2f);
        double[][] np = new double[beyond.length][];
        Var cur = Var.of(tensor(new double[][] {train[points - 1]}));
        // roll at the training step, then sample at the test times by linear interpolation
        List<double[]> coarse = new ArrayList<>();
        coarse.add(train[points - 1]);
        double t = 3;
        while (t < 6 + dt) {
            Var c = cur;
            cur = Var.noGrad(() -> node.forward(c));
            coarse.add(rows(cur.value())[0]);
            t += dt;
        }
        for (int i = 0; i < beyond.length; i++) {
            double time = 3.1 + 0.1 * i - 3;
            int k = (int) Math.floor(time / dt);
            double w = time / dt - k;
            double[] p0 = coarse.get(Math.min(k, coarse.size() - 1));
            double[] p1 = coarse.get(Math.min(k + 1, coarse.size() - 1));
            np[i] = new double[] {p0[0] * (1 - w) + p1[0] * w, p0[1] * (1 - w) + p1[1] * w};
        }
        rows.add(new UdeRow("pure neural ODE", points, finiteMse(np, beyond)));
        // the UDE paper's last step: sparse regression on the learned term, then extrapolate
        // with the recovered closed-form model
        double[][] states = MechanicalSystems.trajectory(lv, x0, 0.05, 60, 5);
        double[][] term = ude.networkTerm(states);
        Sindy.Library lib = Sindy.polynomial(new String[] {"x", "y"}, 2, false);
        double[][] xi = Sindy.stlsq(lib.evaluate(states), term, 0.1, 1e-8, 10);
        double[] p = ude.physicalParameters();
        UnaryOperator<double[]> closed = v -> {
            double[] theta = lib.evaluate(new double[][] {v})[0];
            double[] d = {p[0] * v[0], -p[1] * v[1]};
            for (int i = 0; i < theta.length; i++) {
                d[0] += xi[i][0] * theta[i];
                d[1] += xi[i][1] * theta[i];
            }
            return d;
        };
        double[][] sym = new double[beyond.length][];
        double[] state = train[points - 1].clone();
        for (int i = 0; i < beyond.length; i++) {
            for (int k = 0; k < 10; k++) {
                state = MechanicalSystems.rk4(closed, state, testDt / 10);
            }
            sym[i] = state.clone();
        }
        rows.add(1, new UdeRow("UDE + SINDy on the learned term", points, finiteMse(sym, beyond)));
        if (recovered != null) {
            recovered.append(String.format("[%d points] alpha %.3f delta %.3f; NN term:", points, p[0], p[1]));
            for (int j = 0; j < 2; j++) {
                recovered.append(j == 0 ? " x' +=" : ", y' +=");
                for (int i = 0; i < xi.length; i++) {
                    if (xi[i][j] != 0) {
                        recovered.append(String.format(" %+.3f %s", xi[i][j], lib.names()[i]));
                    }
                }
            }
            recovered.append("; ");
        }
        return rows;
    }

    private static double finiteMse(double[][] a, double[][] b) {
        double m = mse(a, b);
        return Double.isFinite(m) ? m : Double.MAX_VALUE;
    }

    private static double extrapolate(UniversalDifferentialEquation model, double[] start, double dt, double[][] full) {
        UniversalDifferentialEquation fine = model;
        double[][] coarse = fine.rollout(start, (int) Math.ceil(3 / dt) + 1);
        double[][] pred = new double[30][];
        for (int i = 0; i < 30; i++) {
            double time = 0.1 * (i + 1);
            int k = (int) Math.floor(time / dt);
            double w = time / dt - k;
            double[] p0 = coarse[Math.min(k, coarse.length - 1)];
            double[] p1 = coarse[Math.min(k + 1, coarse.length - 1)];
            pred[i] = new double[] {p0[0] * (1 - w) + p1[0] * w, p0[1] * (1 - w) + p1[1] * w};
        }
        return finiteMse(pred, java.util.Arrays.copyOfRange(full, 31, 61));
    }

    // ------------------------------------------------------------------ 8.3.12

    /**
     * A stochastic-dynamics row.
     *
     * @param model the model
     * @param nll negative log-likelihood per test transition (NaN where not defined)
     * @param wasserstein the 2-Wasserstein distance between simulated and true marginals
     *     (entropic estimate)
     * @param switchingRate well switches per unit time (NaN for a single-well process)
     */
    public record SdeRow(String model, double nll, double wasserstein, double switchingRate) {
    }

    /**
     * 8.3.12: neural SDE against a deterministic drift and a quantile MLP on one process.
     *
     * @param doubleWell whether to use the double well {@code dX = (X - X^3) dt + 0.5 dW}
     *     instead of Ornstein-Uhlenbeck {@code dX = -X dt + 0.5 dW}
     * @param length the training path length in steps of 0.05
     * @param steps training steps
     * @param seed the seed
     * @return rows for the truth, the neural SDE, the neural ODE drift and the quantile MLP
     */
    public static List<SdeRow> stochastic(boolean doubleWell, int length, int steps, long seed) {
        double dt = 0.05;
        double sigma = doubleWell ? 0.5 : 0.5;
        UnaryOperator<double[]> drift = doubleWell ? x -> new double[] {x[0] - x[0] * x[0] * x[0]}
                : x -> new double[] {-x[0]};
        UnaryOperator<double[]> diff = x -> new double[] {sigma};
        SplittableRandom r = new SplittableRandom(seed);
        double[][] path = finePath(drift, diff, length, dt, r);
        double[][] test = finePath(drift, diff, length, dt, r);
        List<SdeRow> rows = new ArrayList<>();
        double[] trueMarginal = column(test);
        rows.add(new SdeRow("true process (fresh path)", Double.NaN, w2(column(finePath(drift, diff, length, dt, r)),
                trueMarginal), doubleWell ? switches(column(path), dt) : Double.NaN));
        Tensor x = tensor(java.util.Arrays.copyOfRange(path, 0, path.length - 1));
        Tensor y = tensor(java.util.Arrays.copyOfRange(path, 1, path.length));
        Tensor xt = tensor(java.util.Arrays.copyOfRange(test, 0, test.length - 1));
        Tensor yt = tensor(java.util.Arrays.copyOfRange(test, 1, test.length));
        NeuralSde sde = NeuralSde.of(1, 32, seed);
        fit(sde.parameters(), () -> sde.nll(x, y, (float) dt), steps, 1e-2f);
        double[] sim = column(sde.simulate(new double[] {path[0][0]}, dt, length, new SplittableRandom(seed + 7)));
        rows.add(new SdeRow("neural SDE", Var.noGrad(() -> sde.nll(xt, yt, (float) dt)).value().item(),
                w2(sim, trueMarginal), doubleWell ? switches(sim, dt) : Double.NaN));
        // deterministic neural ODE drift: fit (x' - x) / dt, Gaussian NLL with the fitted residual variance
        SimpleMlp ode = SimpleMlp.tanh(seed, 1, 32, 32, 1);
        Tensor vel = y.sub(x).div((float) dt);
        fit(ode.parameters(), () -> ode.forward(Var.of(x)).sub(vel).square().mean(), steps, 1e-2f);
        MlpSnapshotAccess snap = new MlpSnapshotAccess(ode);
        double resVar = 0;
        float[] px = x.toArray();
        float[] py = y.toArray();
        for (int i = 0; i < px.length; i++) {
            double m = px[i] + dt * snap.apply(px[i]);
            resVar += (py[i] - m) * (py[i] - m) / px.length;
        }
        double nll = 0;
        float[] tx = xt.toArray();
        float[] ty = yt.toArray();
        for (int i = 0; i < tx.length; i++) {
            double m = tx[i] + dt * snap.apply(tx[i]);
            nll += 0.5 * ((ty[i] - m) * (ty[i] - m) / resVar + Math.log(2 * Math.PI * resVar)) / tx.length;
        }
        double[] det = new double[length + 1];
        det[0] = path[0][0];
        for (int t = 1; t <= length; t++) {
            det[t] = det[t - 1] + dt * snap.apply(det[t - 1]);
        }
        rows.add(new SdeRow("neural ODE drift (deterministic)", nll, w2(det, trueMarginal),
                doubleWell ? switches(det, dt) : Double.NaN));
        rows.add(quantileMlp(path, test, trueMarginal, length, dt, doubleWell, steps, seed));
        return rows;
    }

    /** A tiny evaluator of a scalar MLP built on the public SimpleMlp API. */
    private static final class MlpSnapshotAccess {
        private final SimpleMlp net;

        MlpSnapshotAccess(SimpleMlp net) {
            this.net = net;
        }

        double apply(double x) {
            return net.predict(Tensor.of(Shape.of(1, 1), (float) x)).item();
        }
    }

    private static SdeRow quantileMlp(double[][] path, double[][] test, double[] trueMarginal, int length, double dt,
            boolean doubleWell, int steps, long seed) {
        int q = 9;
        float[] levels = new float[q];
        for (int i = 0; i < q; i++) {
            levels[i] = (i + 1f) / (q + 1);
        }
        SimpleMlp net = SimpleMlp.tanh(seed, 1, 32, 32, q);
        Tensor x = tensor(java.util.Arrays.copyOfRange(path, 0, path.length - 1));
        Tensor y = tensor(java.util.Arrays.copyOfRange(path, 1, path.length));
        Tensor tau = Tensor.of(Shape.of(1, q), levels);
        fit(net.parameters(), () -> {
            Var pred = net.forward(Var.of(x)).add(x);
            Var err = Var.of(y).sub(pred);    // [n, q]
            return err.mul(tau).maximum(err.mul(Var.of(tau.sub(1f)))).mean();
        }, steps, 1e-2f);
        SplittableRandom r = new SplittableRandom(seed + 11);
        double[] sim = new double[length + 1];
        sim[0] = path[0][0];
        for (int t = 1; t <= length; t++) {
            float[] qs = net.predict(Tensor.of(Shape.of(1, 1), (float) sim[t - 1])).toArray();
            double[] sorted = new double[q];
            for (int i = 0; i < q; i++) {
                sorted[i] = qs[i] + sim[t - 1];
            }
            java.util.Arrays.sort(sorted);
            double u = r.nextDouble() * (q + 1) - 1;   // position among the levels
            int lo = (int) Math.floor(u);
            double w = u - lo;
            double a = sorted[Math.clamp(lo, 0, q - 1)];
            double b = sorted[Math.clamp(lo + 1, 0, q - 1)];
            sim[t] = a * (1 - w) + b * w;
        }
        return new SdeRow("quantile MLP (sampled)", Double.NaN, w2(sim, trueMarginal),
                doubleWell ? switches(sim, dt) : Double.NaN);
    }

    private static double[][] finePath(UnaryOperator<double[]> drift, UnaryOperator<double[]> diff, int length,
            double dt, SplittableRandom r) {
        int sub = 10;
        double[][] fine = SdeSolver.simulate(drift, diff, new double[] {r.nextGaussian() * 0.5}, dt / sub,
                length * sub, true, r);
        double[][] out = new double[length + 1][];
        for (int t = 0; t <= length; t++) {
            out[t] = fine[t * sub];
        }
        return out;
    }

    private static double[] column(double[][] x) {
        double[] c = new double[x.length];
        for (int i = 0; i < x.length; i++) {
            c[i] = x[i][0];
        }
        return c;
    }

    /**
     * Well switches per unit time, counting a switch when the path crosses from below -0.5 to
     * above 0.5 or back (a hysteresis band that ignores jitter around 0).
     *
     * @param x the path
     * @param dt the step
     * @return the switching rate
     */
    public static double switches(double[] x, double dt) {
        int state = 0;
        int count = 0;
        for (double v : x) {
            int s = v > 0.5 ? 1 : v < -0.5 ? -1 : 0;
            if (s != 0 && state != 0 && s != state) {
                count++;
            }
            if (s != 0) {
                state = s;
            }
        }
        return count / (x.length * dt);
    }

    /**
     * The 2-Wasserstein distance between two 1-D empirical marginals, from the entropic
     * transport plan between 200-point subsamples.
     *
     * @param a samples
     * @param b samples
     * @return the distance
     */
    public static double w2(double[] a, double[] b) {
        double[][] x = subsample(a, 200);
        double[][] y = subsample(b, 200);
        return Math.sqrt(Math.max(0, EntropicTransport.transport(x, y, 0.01, 300).transportCost()));
    }

    private static double[][] subsample(double[] a, int n) {
        double[][] out = new double[n][1];
        for (int i = 0; i < n; i++) {
            out[i][0] = a[(int) ((long) i * a.length / n)];
        }
        return out;
    }

    /**
     * 8.3.12, latent SDE: fit to irregularly sampled double-well series and compare the
     * generated marginal and switching rate with the truth.
     *
     * @param series training series
     * @param steps training steps
     * @param seed the seed
     * @return {@code {W2 of generated vs true marginal, generated switching rate, true rate,
     *     final loss}}
     */
    public static double[] latentSde(int series, int steps, long seed) {
        double dt = 0.1;
        int length = 40;
        SplittableRandom r = new SplittableRandom(seed);
        UnaryOperator<double[]> drift = x -> new double[] {x[0] - x[0] * x[0] * x[0]};
        UnaryOperator<double[]> diff = x -> new double[] {0.6};
        double[][] values = new double[series][];
        boolean[][] observed = new boolean[series][length + 1];
        List<Double> all = new ArrayList<>();
        for (int i = 0; i < series; i++) {
            double[][] p = finePath(drift, diff, length, dt, r);
            values[i] = new double[length + 1];
            for (int t = 0; t <= length; t++) {
                values[i][t] = p[t][0] + 0.05 * r.nextGaussian();
                observed[i][t] = t == 0 || r.nextDouble() < 0.4;
                all.add(p[t][0]);
            }
        }
        LatentSde model = LatentSde.of(2, 32, seed);
        Optimizer opt = Optimizer.adam(1e-2f);
        double loss = Double.NaN;
        for (int s = 0; s < steps; s++) {
            Var l = model.loss(new LatentSde.Batch(values, observed), (float) dt, r);
            loss = l.value().item();
            l.backward();
            opt.step(model.parameters());
        }
        double[][] gen = model.sample(series, dt, length, new SplittableRandom(seed + 5));
        double[] g = new double[series * (length + 1)];
        double rate = 0;
        for (int i = 0; i < series; i++) {
            System.arraycopy(gen[i], 0, g, i * (length + 1), length + 1);
            rate += switches(gen[i], dt) / series;
        }
        double trueRate = 0;
        double[] truth = new double[all.size()];
        for (int i = 0; i < truth.length; i++) {
            truth[i] = all.get(i);
        }
        for (int i = 0; i < series; i++) {
            trueRate += switches(java.util.Arrays.copyOfRange(truth, i * (length + 1), (i + 1) * (length + 1)), dt)
                    / series;
        }
        return new double[] {w2(g, truth), rate, trueRate, loss};
    }

    // ------------------------------------------------------------------ 8.3.13

    /**
     * A particle-simulation row.
     *
     * @param model the model
     * @param trainSizeMse rollout MSE at the training particle count
     * @param doubleSizeMse rollout MSE at twice the particles (NaN if the model cannot run)
     */
    public record ParticleRow(String model, double trainSizeMse, double doubleSizeMse) {
    }

    /** Spring rest length, the same for every chain length so the local physics generalises. */
    private static final double SPACING = 0.07;

    private static double[][][] chainTrajectory(int n, int steps, SplittableRandom r) {
        UnaryOperator<double[][]> acc = MechanicalSystems.springChain(30, SPACING, 300);
        double[][] x0 = MechanicalSystems.randomChain(n, SPACING, r);
        double[][] v0 = new double[n][2];
        for (double[] v : v0) {
            v[0] = 0.3 * r.nextGaussian();
            v[1] = 0.3 * r.nextGaussian();
        }
        // one time unit = 0.02, five Verlet substeps
        double[][][] t = MechanicalSystems.particleTrajectory(acc, x0, v0, 0.02, steps, 5);
        return t;
    }

    private static double particleMse(double[][][] pred, double[][][] truth, int offset) {
        double s = 0;
        int c = 0;
        for (int t = 0; t < pred.length; t++) {
            for (int i = 0; i < pred[t].length; i++) {
                for (int k = 0; k < 2; k++) {
                    double d = pred[t][i][k] - truth[offset + t][i][k];
                    s += d * d;
                    c++;
                }
            }
        }
        double m = s / c;
        return Double.isFinite(m) ? m : Double.MAX_VALUE;
    }

    /**
     * 8.3.13: graph network simulators on spring chains.
     *
     * @param n training particle count
     * @param trajectories training trajectories
     * @param iterations training iterations per model
     * @param rollout rollout steps
     * @param seed the seed
     * @return rows for GNS with noise, GNS without noise, EGNN and the MLP
     */
    public static List<ParticleRow> particles(int n, int trajectories, int iterations, int rollout, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int history = 2;
        int length = 60;
        List<double[][][]> train = new ArrayList<>();
        for (int i = 0; i < trajectories; i++) {
            train.add(chainTrajectory(n, length, r));
        }
        double[][][] testSmall = chainTrajectory(n, rollout + history, new SplittableRandom(seed + 100));
        double[][][] testLarge = chainTrajectory(2 * n, rollout + history, new SplittableRandom(seed + 200));
        double scale = 0.01;
        List<ParticleRow> rows = new ArrayList<>();
        for (double noise : new double[] {3e-4, 0}) {
            GraphNetworkSimulator gns = GraphNetworkSimulator.of(history, 32, 3, 0.15, scale, ACCELERATION_SCALE,
                    seed);
            Optimizer opt = Optimizer.adam(2e-3f);
            SplittableRandom nr = new SplittableRandom(seed + 1);
            for (int it = 0; it < iterations; it++) {
                List<double[][][]> batch = new ArrayList<>();
                for (int k = 0; k < 4; k++) {
                    double[][][] tr = train.get(nr.nextInt(trajectories));
                    int s = nr.nextInt(length - history - 1);
                    batch.add(java.util.Arrays.copyOfRange(tr, s, s + history + 2));
                }
                Var loss = gns.loss(batch, noise, nr);
                loss.backward();
                opt.step(gns.parameters());
            }
            double[][][] ps = gns.rollout(java.util.Arrays.copyOfRange(testSmall, 0, history + 1), rollout);
            double[][][] pl = gns.rollout(java.util.Arrays.copyOfRange(testLarge, 0, history + 1), rollout);
            rows.add(new ParticleRow(noise > 0 ? "GNS (input noise)" : "GNS (no noise)",
                    particleMse(ps, testSmall, history + 1), particleMse(pl, testLarge, history + 1)));
        }
        rows.add(egnn(train, testSmall, testLarge, history, iterations, rollout, 0.1, seed));
        rows.add(fixedMlp(train, testSmall, n, history, iterations, rollout, scale, seed));
        return rows;
    }

    private static final double ACCELERATION_SCALE = 5e-4;

    /** EGNN baseline: coordinates are the ballistic guess, features the velocity and wall distances. */
    private static ParticleRow egnn(List<double[][][]> train, double[][][] testSmall, double[][][] testLarge,
            int history, int iterations, int rollout, double scale, long seed) {
        int features = 5;
        EquivariantGraphLayer l1 = EquivariantGraphLayer.of(2, features, 32, seed);
        EquivariantGraphLayer l2 = EquivariantGraphLayer.of(2, features, 32, seed + 1);
        List<Var> params = new ArrayList<>(l1.parameters());
        params.addAll(l2.parameters());
        Optimizer opt = Optimizer.adam(2e-3f);
        SplittableRandom r = new SplittableRandom(seed + 2);
        int length = train.get(0).length;
        for (int it = 0; it < iterations; it++) {
            double[][][] tr = train.get(r.nextInt(train.size()));
            int s = r.nextInt(length - history - 1) + history;
            Tensor in = egnnInput(tr[s - 1], tr[s], scale);
            int n = tr[s].length;
            Tensor target = egnnTarget(tr[s + 1], tr[s], scale);
            Var out = l2.forward(l1.forward(Var.of(in))).slice(2, 0, 2).reshape(n, 2);
            // coordinates are in units of 0.1 (so squared distances stay O(1)); weight the error
            // so the per-step acceleration is O(1) as for the GNS
            Var loss = out.sub(target).mul((float) (scale / ACCELERATION_SCALE)).square().mean();
            loss.backward();
            opt.step(params);
        }
        return new ParticleRow("EGNN", particleMse(egnnRollout(l1, l2, testSmall, history, rollout, scale),
                testSmall, history + 1), particleMse(egnnRollout(l1, l2, testLarge, history, rollout, scale),
                testLarge, history + 1));
    }

    private static Tensor egnnInput(double[][] prev, double[][] cur, double scale) {
        int n = cur.length;
        float[] f = new float[n * 7];
        for (int i = 0; i < n; i++) {
            // coordinates: the ballistic guess x + v, in units of the scale
            for (int k = 0; k < 2; k++) {
                f[i * 7 + k] = (float) ((2 * cur[i][k] - prev[i][k]) / scale);
            }
            f[i * 7 + 2] = (float) (Math.hypot(cur[i][0] - prev[i][0], cur[i][1] - prev[i][1]) / scale);
            f[i * 7 + 3] = (float) Math.clamp(cur[i][0] / 0.15, -1, 1);
            f[i * 7 + 4] = (float) Math.clamp((1 - cur[i][0]) / 0.15, -1, 1);
            f[i * 7 + 5] = (float) Math.clamp(cur[i][1] / 0.15, -1, 1);
            f[i * 7 + 6] = (float) Math.clamp((1 - cur[i][1]) / 0.15, -1, 1);
        }
        return Tensor.of(Shape.of(1, n, 7), f);
    }

    private static Tensor egnnTarget(double[][] next, double[][] cur, double scale) {
        int n = cur.length;
        float[] f = new float[n * 2];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < 2; k++) {
                f[i * 2 + k] = (float) (next[i][k] / scale);
            }
        }
        return Tensor.of(Shape.of(n, 2), f);
    }

    private static double[][][] egnnRollout(EquivariantGraphLayer l1, EquivariantGraphLayer l2, double[][][] start,
            int history, int steps, double scale) {
        double[][] prev = start[history - 1];
        double[][] cur = start[history];
        double[][][] out = new double[steps][][];
        for (int t = 0; t < steps; t++) {
            Tensor in = egnnInput(prev, cur, scale);
            int n = cur.length;
            float[] v = Var.noGrad(() -> l2.forward(l1.forward(Var.of(in))).slice(2, 0, 2)).value().toArray();
            double[][] next = new double[n][2];
            for (int i = 0; i < n; i++) {
                next[i][0] = v[2 * i] * scale;
                next[i][1] = v[2 * i + 1] * scale;
            }
            out[t] = next;
            prev = cur;
            cur = next;
        }
        return out;
    }

    private static ParticleRow fixedMlp(List<double[][][]> train, double[][][] testSmall, int n, int history,
            int iterations, int rollout, double scale, long seed) {
        SimpleMlp mlp = SimpleMlp.tanh(seed, n * 2 * (history + 1), 128, 128, n * 2);
        Optimizer opt = Optimizer.adam(2e-3f);
        SplittableRandom r = new SplittableRandom(seed + 3);
        int length = train.get(0).length;
        for (int it = 0; it < iterations; it++) {
            double[][] in = new double[8][];
            double[][] out = new double[8][];
            for (int b = 0; b < 8; b++) {
                double[][][] tr = train.get(r.nextInt(train.size()));
                int s = r.nextInt(length - history - 1) + history;
                in[b] = mlpInput(tr, s, history);
                out[b] = new double[n * 2];
                for (int i = 0; i < n; i++) {
                    for (int k = 0; k < 2; k++) {
                        out[b][2 * i + k] = (tr[s + 1][i][k] - 2 * tr[s][i][k] + tr[s - 1][i][k])
                                / ACCELERATION_SCALE;
                    }
                }
            }
            Tensor x = tensor(in);
            Tensor y = tensor(out);
            Var loss = mlp.forward(Var.of(x)).sub(y).square().mean();
            loss.backward();
            opt.step(mlp.parameters());
        }
        List<double[][]> frames = new ArrayList<>(List.of(java.util.Arrays.copyOfRange(testSmall, 0, history + 1)));
        double[][][] pred = new double[rollout][][];
        for (int t = 0; t < rollout; t++) {
            double[][][] window = frames.subList(frames.size() - history - 1, frames.size()).toArray(new double[0][][]);
            double[] a = rows(mlp.predict(tensor(new double[][] {mlpInput(window, history, history)})))[0];
            double[][] next = new double[n][2];
            for (int i = 0; i < n; i++) {
                for (int k = 0; k < 2; k++) {
                    next[i][k] = 2 * window[history][i][k] - window[history - 1][i][k]
                            + a[2 * i + k] * ACCELERATION_SCALE;
                }
            }
            frames.add(next);
            pred[t] = next;
        }
        return new ParticleRow("MLP (fixed n)", particleMse(pred, testSmall, history + 1), Double.NaN);
    }

    private static double[] mlpInput(double[][][] tr, int s, int history) {
        int n = tr[s].length;
        double[] in = new double[n * 2 * (history + 1)];
        int k = 0;
        for (int f = s - history; f <= s; f++) {
            for (int i = 0; i < n; i++) {
                in[k++] = tr[f][i][0];
                in[k++] = tr[f][i][1];
            }
        }
        return in;
    }
}
