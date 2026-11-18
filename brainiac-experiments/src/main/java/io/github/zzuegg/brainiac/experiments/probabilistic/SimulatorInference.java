package io.github.zzuegg.brainiac.experiments.probabilistic;

import io.github.zzuegg.brainiac.experimental.ml.LogDensity;
import io.github.zzuegg.brainiac.experimental.ml.McmcDiagnostics;
import io.github.zzuegg.brainiac.experimental.ml.SimulationBasedInference;
import io.github.zzuegg.brainiac.experimental.ml.SimulationBasedInference.Prior;
import io.github.zzuegg.brainiac.experimental.ml.SimulationBasedInference.Simulator;
import io.github.zzuegg.brainiac.experimental.sbi.ConditionalDensity;
import io.github.zzuegg.brainiac.experimental.sbi.FlowMatchingPosterior;
import io.github.zzuegg.brainiac.experimental.sbi.NeuralLikelihood;
import io.github.zzuegg.brainiac.experimental.sbi.NeuralRatio;
import io.github.zzuegg.brainiac.experimental.sbi.SequentialPosterior;
import io.github.zzuegg.brainiac.experimental.train.HamiltonianMonteCarlo;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Likelihood-free inference on three simulators, against reference posteriors:
 * <ul>
 *   <li>{@link #gaussianLinear}: {@code x = A theta + N(0, 0.3^2)}, {@code theta ~ N(0, I_2)},
 *       with an analytic posterior; NUTS on the (here tractable) likelihood is the
 *       likelihood-based baseline;</li>
 *   <li>{@link #twoMoons}: the two-moons simulator of Greenberg et al. 2019 at
 *       {@code x_o = (0, 0)}, whose bimodal crescent posterior is computed by importance
 *       resampling with the closed-form likelihood;</li>
 *   <li>{@link #lotkaVolterra}: predator-prey dynamics with four log-normal rate parameters,
 *       observed as noisy log populations at ten times; scored by the error of the posterior
 *       mean against the true parameters.</li>
 * </ul>
 * Methods: rejection ABC, SMC-ABC and neural posterior estimation (an amortised mixture
 * density network) at equal simulation budgets. C2ST is the k-NN classifier accuracy between
 * reference and estimated samples (0.5 is perfect).
 */
public final class SimulatorInference {

    private SimulatorInference() {
    }

    private static Prior gaussianPrior(double[] mean, double sd) {
        return new Prior() {
            @Override
            public double[] sample(SplittableRandom random) {
                double[] t = new double[mean.length];
                for (int i = 0; i < t.length; i++) {
                    t[i] = mean[i] + sd * random.nextGaussian();
                }
                return t;
            }

            @Override
            public double logDensity(double[] theta) {
                double s = 0;
                for (int i = 0; i < theta.length; i++) {
                    s += -0.5 * Math.pow((theta[i] - mean[i]) / sd, 2);
                }
                return s;
            }
        };
    }

    private static Map<String, double[]> compare(Prior prior, Simulator sim, double[] observed, double[][] reference,
            int[] budgets, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        for (int b : budgets) {
            SplittableRandom r = new SplittableRandom(seed + b);
            double[][] rej = SimulationBasedInference.rejectionAbc(prior, sim, observed, b, Math.max(100, b / 20), r)
                    .samples();
            double[][] smc = SimulationBasedInference.smcAbc(prior, sim, observed, 200, b, 0.5, r).samples();
            double[][] npe = SimulationBasedInference.neuralPosterior(prior, sim, b, 4, 32, 800, seed + b)
                    .sample(observed, 500, r);
            out.put("rejection ABC n=" + b, new double[] {SimulationBasedInference.c2st(reference, rej)});
            out.put("SMC-ABC n=" + b, new double[] {SimulationBasedInference.c2st(reference, smc)});
            out.put("NPE n=" + b, new double[] {SimulationBasedInference.c2st(reference, npe)});
        }
        return out;
    }

    /**
     * The Gaussian-linear model.
     *
     * @param budgets the simulation budgets
     * @param seed the seed
     * @return method and budget to {@code {C2ST}}
     */
    public static Map<String, double[]> gaussianLinear(int[] budgets, long seed) {
        return gaussianLinear(budgets, seed, false);
    }

    /**
     * Wave 13 rows on the Gaussian-linear model: neural likelihood (13.5.1), contrastive ratio
     * (13.5.2), flow-matching posterior (13.5.3) and TSNPE (13.5.4) at one budget.
     *
     * @param budget the simulation budget
     * @param seed the seed
     * @return method to {@code {C2ST}}
     */
    public static Map<String, double[]> gaussianLinearNeural(int budget, long seed) {
        return gaussianLinear(new int[] {budget}, seed, true);
    }

    private static Map<String, double[]> gaussianLinear(int[] budgets, long seed, boolean neural) {
        double[][] a = {{1, 0.5}, {-0.3, 1}, {0.8, 0.8}};
        double s = 0.3;
        Prior prior = gaussianPrior(new double[2], 1);
        Simulator sim = (t, r) -> {
            double[] x = new double[3];
            for (int i = 0; i < 3; i++) {
                x[i] = a[i][0] * t[0] + a[i][1] * t[1] + s * r.nextGaussian();
            }
            return x;
        };
        SplittableRandom r = new SplittableRandom(seed);
        double[] observed = sim.simulate(new double[] {0.7, -0.4}, r);
        // posterior precision P = I + A^T A / s^2, mean P^-1 A^T x / s^2
        double p00 = 1;
        double p01 = 0;
        double p11 = 1;
        double b0 = 0;
        double b1 = 0;
        for (int i = 0; i < 3; i++) {
            p00 += a[i][0] * a[i][0] / (s * s);
            p01 += a[i][0] * a[i][1] / (s * s);
            p11 += a[i][1] * a[i][1] / (s * s);
            b0 += a[i][0] * observed[i] / (s * s);
            b1 += a[i][1] * observed[i] / (s * s);
        }
        double det = p00 * p11 - p01 * p01;
        double m0 = (p11 * b0 - p01 * b1) / det;
        double m1 = (-p01 * b0 + p00 * b1) / det;
        double c00 = p11 / det;
        double c01 = -p01 / det;
        double c11 = p00 / det;
        double l00 = Math.sqrt(c00);
        double l10 = c01 / l00;
        double l11 = Math.sqrt(c11 - l10 * l10);
        double[][] reference = new double[500][];
        for (int i = 0; i < 500; i++) {
            double z0 = r.nextGaussian();
            double z1 = r.nextGaussian();
            reference[i] = new double[] {m0 + l00 * z0, m1 + l10 * z0 + l11 * z1};
        }
        if (neural) {
            return neuralRows(prior, sim, observed, reference, budgets[0], seed);
        }
        Map<String, double[]> out = compare(prior, sim, observed, reference, budgets, seed);
        LogDensity post = (t, g) -> {
            double lp = -0.5 * (t[0] * t[0] + t[1] * t[1]);
            if (g != null) {
                g[0] = -t[0];
                g[1] = -t[1];
            }
            for (int i = 0; i < 3; i++) {
                double res = observed[i] - a[i][0] * t[0] - a[i][1] * t[1];
                lp += -0.5 * res * res / (s * s);
                if (g != null) {
                    g[0] += res * a[i][0] / (s * s);
                    g[1] += res * a[i][1] / (s * s);
                }
            }
            return lp;
        };
        double[][] nuts = HamiltonianMonteCarlo.nuts(post, new double[2], 200, 1000, 0.8, 8, r).samples();
        double[][] thinned = new double[500][];
        for (int i = 0; i < 500; i++) {
            thinned[i] = nuts[2 * i];
        }
        out.put("NUTS (exact likelihood)", new double[] {SimulationBasedInference.c2st(reference, thinned)});
        return out;
    }

    /**
     * The two-moons simulator.
     *
     * @param budgets the simulation budgets
     * @param seed the seed
     * @return method and budget to {@code {C2ST}}
     */
    public static Map<String, double[]> twoMoons(int[] budgets, long seed) {
        return twoMoons(budgets, seed, false);
    }

    /**
     * Wave 13 rows on two moons: neural likelihood (13.5.1), contrastive ratio (13.5.2),
     * flow-matching posterior (13.5.3) and TSNPE (13.5.4) at one budget.
     *
     * @param budget the simulation budget
     * @param seed the seed
     * @return method to {@code {C2ST}}
     */
    public static Map<String, double[]> twoMoonsNeural(int budget, long seed) {
        return twoMoons(new int[] {budget}, seed, true);
    }

    private static Map<String, double[]> twoMoons(int[] budgets, long seed, boolean neural) {
        Prior prior = new Prior() {
            @Override
            public double[] sample(SplittableRandom random) {
                return new double[] {-1 + 2 * random.nextDouble(), -1 + 2 * random.nextDouble()};
            }

            @Override
            public double logDensity(double[] t) {
                return Math.abs(t[0]) <= 1 && Math.abs(t[1]) <= 1 ? 0 : Double.NEGATIVE_INFINITY;
            }
        };
        Simulator sim = (t, r) -> {
            double ang = Math.PI * (r.nextDouble() - 0.5);
            double rad = 0.1 + 0.01 * r.nextGaussian();
            return new double[] {rad * Math.cos(ang) + 0.25 - Math.abs(t[0] + t[1]) / Math.sqrt(2),
                rad * Math.sin(ang) + (-t[0] + t[1]) / Math.sqrt(2)};
        };
        double[] observed = {0, 0};
        // reference: importance resampling of prior draws with the closed-form likelihood
        SplittableRandom r = new SplittableRandom(seed);
        int pool = 200_000;
        double[][] thetas = new double[pool][];
        double[] lw = new double[pool];
        for (int i = 0; i < pool; i++) {
            double[] t = prior.sample(r);
            thetas[i] = t;
            double px = observed[0] - 0.25 + Math.abs(t[0] + t[1]) / Math.sqrt(2);
            double py = observed[1] - (-t[0] + t[1]) / Math.sqrt(2);
            double rad = Math.hypot(px, py);
            double ang = Math.atan2(py, px);
            lw[i] = Math.abs(ang) <= Math.PI / 2 && rad > 0
                    ? -0.5 * Math.pow((rad - 0.1) / 0.01, 2) - Math.log(rad) : Double.NEGATIVE_INFINITY;
        }
        int[] pick = io.github.zzuegg.brainiac.experimental.ml.ProbabilisticProgram.systematic(lw, r);
        double[][] reference = new double[500][];
        for (int i = 0; i < 500; i++) {
            reference[i] = thetas[pick[i * (pool / 500)]].clone();
            reference[i][0] += 1e-3 * r.nextGaussian();   // break ties between resampled copies
            reference[i][1] += 1e-3 * r.nextGaussian();
        }
        if (neural) {
            return neuralRows(prior, sim, observed, reference, budgets[0], seed);
        }
        return compare(prior, sim, observed, reference, budgets, seed);
    }

    private static final double[] LV_TRUE = {Math.log(0.8), Math.log(0.045), Math.log(0.95), Math.log(0.055)};

    private static double[] lotkaVolterraSimulate(double[] logRates, SplittableRandom r) {
        double a = Math.exp(logRates[0]);
        double b = Math.exp(logRates[1]);
        double c = Math.exp(logRates[2]);
        double d = Math.exp(logRates[3]);
        double x = 30;
        double y = 1;
        double dt = 0.05;
        double[] out = new double[20];
        int k = 0;
        for (int step = 1; step <= 400; step++) {
            double[] k1 = lv(x, y, a, b, c, d);
            double[] k2 = lv(x + 0.5 * dt * k1[0], y + 0.5 * dt * k1[1], a, b, c, d);
            double[] k3 = lv(x + 0.5 * dt * k2[0], y + 0.5 * dt * k2[1], a, b, c, d);
            double[] k4 = lv(x + dt * k3[0], y + dt * k3[1], a, b, c, d);
            x = Math.max(1e-6, x + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]));
            y = Math.max(1e-6, y + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]));
            if (step % 40 == 0 && k < 10) {
                out[2 * k] = Math.log(Math.min(x, 1e6)) + 0.1 * r.nextGaussian();
                out[2 * k + 1] = Math.log(Math.min(y, 1e6)) + 0.1 * r.nextGaussian();
                k++;
            }
        }
        return out;
    }

    private static double[] lv(double x, double y, double a, double b, double c, double d) {
        return new double[] {a * x - b * x * y, -c * y + d * x * y};
    }

    /**
     * Lotka-Volterra with log-rates {@code ~ N((-0.125, -3, -0.125, -3), 0.5^2)}.
     *
     * @param budget the simulation budget
     * @param seed the seed
     * @return method to {@code {RMSE of the posterior mean log-rates, mean posterior std}}
     */
    public static Map<String, double[]> lotkaVolterra(int budget, long seed) {
        Prior prior = gaussianPrior(new double[] {-0.125, -3, -0.125, -3}, 0.5);
        Simulator sim = SimulatorInference::lotkaVolterraSimulate;
        SplittableRandom r = new SplittableRandom(seed);
        double[] observed = sim.simulate(LV_TRUE, r);
        Map<String, double[]> out = new LinkedHashMap<>();
        out.put("rejection ABC", score(SimulationBasedInference.rejectionAbc(prior, sim, observed, budget,
                Math.max(100, budget / 20), r).samples()));
        out.put("SMC-ABC", score(SimulationBasedInference.smcAbc(prior, sim, observed, 200, budget, 0.5, r).samples()));
        out.put("NPE", score(SimulationBasedInference.neuralPosterior(prior, sim, budget, 4, 32, 800, seed)
                .sample(observed, 500, r)));
        return out;
    }

    /**
     * Wave 13 rows on Lotka-Volterra: sequential NPE (APT with the atomic loss, TSNPE; 13.5.4)
     * and sequential neural likelihood (13.5.1) over {@code rounds} rounds, against amortised
     * NPE at the same total budget.
     *
     * @param rounds the rounds
     * @param perRound simulations per round
     * @param seed the seed
     * @return method to {@code {RMSE of the posterior mean log-rates, mean posterior std}}
     */
    public static Map<String, double[]> lotkaVolterraSequential(int rounds, int perRound, long seed) {
        Prior prior = gaussianPrior(new double[] {-0.125, -3, -0.125, -3}, 0.5);
        Simulator sim = SimulatorInference::lotkaVolterraSimulate;
        SplittableRandom r = new SplittableRandom(seed);
        double[] observed = sim.simulate(LV_TRUE, r);
        Map<String, double[]> out = new LinkedHashMap<>();
        ConditionalDensity.Config cfg = new ConditionalDensity.Config(3, 32, 300, 128, 3e-3);
        for (SequentialPosterior.Method m : SequentialPosterior.Method.values()) {
            out.put(m.name(), score(SequentialPosterior.run(m, prior, sim, observed, rounds, perRound, cfg, 10, 1e-3,
                    400, seed).samples()));
        }
        out.put("NPE (amortised)", score(SimulationBasedInference.neuralPosterior(prior, sim, rounds * perRound, 3, 32,
                300 * rounds, seed).sample(observed, 400, r)));
        return out;
    }

    private static Map<String, double[]> neuralRows(Prior prior, Simulator sim, double[] observed,
            double[][] reference, int budget, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        SplittableRandom r = new SplittableRandom(seed + 13);
        ConditionalDensity.Config cfg = new ConditionalDensity.Config(4, 32, 500, 128, 3e-3);
        NeuralLikelihood nle = NeuralLikelihood.train(prior, sim, budget, ConditionalDensity.Kind.MDN, cfg, seed);
        out.put("NLE n=" + budget, new double[] {SimulationBasedInference.c2st(reference,
                nle.posterior(prior, new double[][] {observed}, 300, r))});
        NeuralRatio nre = NeuralRatio.train(NeuralRatio.Variant.NRE_C, prior, sim, budget, 5, 0, 32, 500, seed);
        out.put("NRE-C n=" + budget, new double[] {SimulationBasedInference.c2st(reference,
                nre.posteriorSir(prior, observed, 300, 4000, r))});
        FlowMatchingPosterior fm = FlowMatchingPosterior.fmpe(prior, sim, budget, 64, 600, seed);
        out.put("FMPE n=" + budget, new double[] {SimulationBasedInference.c2st(reference,
                fm.sample(observed, 300, 20, r))});
        out.put("TSNPE 2x" + budget / 2, new double[] {SimulationBasedInference.c2st(reference,
                SequentialPosterior.run(SequentialPosterior.Method.TSNPE, prior, sim, observed, 2, budget / 2, cfg, 10,
                        1e-3, 300, seed).samples())});
        return out;
    }

    private static double[] score(double[][] samples) {
        double[] m = McmcDiagnostics.mean(samples);
        double err = 0;
        double sd = 0;
        for (int j = 0; j < 4; j++) {
            err += (m[j] - LV_TRUE[j]) * (m[j] - LV_TRUE[j]) / 4;
            double v = 0;
            for (double[] s : samples) {
                v += (s[j] - m[j]) * (s[j] - m[j]) / samples.length;
            }
            sd += Math.sqrt(v) / 4;
        }
        return new double[] {Math.sqrt(err), sd};
    }
}
