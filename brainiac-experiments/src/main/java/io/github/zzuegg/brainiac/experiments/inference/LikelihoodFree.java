package io.github.zzuegg.brainiac.experiments.inference;

import io.github.zzuegg.brainiac.experimental.data.SimulatorSuite;
import io.github.zzuegg.brainiac.experimental.data.SimulatorSuite.Problem;
import io.github.zzuegg.brainiac.experimental.estimators.MutualInformation;
import io.github.zzuegg.brainiac.experimental.ml.SimulationBasedInference;
import io.github.zzuegg.brainiac.experimental.ml.SimulationBasedInference.Simulator;
import io.github.zzuegg.brainiac.experimental.sbi.ConditionalDensity;
import io.github.zzuegg.brainiac.experimental.sbi.FlowMatchingPosterior;
import io.github.zzuegg.brainiac.experimental.sbi.NeuralLikelihood;
import io.github.zzuegg.brainiac.experimental.sbi.NeuralRatio;
import io.github.zzuegg.brainiac.experimental.sbi.PosteriorDiagnostics;
import io.github.zzuegg.brainiac.experimental.sbi.SequentialPosterior;
import io.github.zzuegg.brainiac.experimental.sbi.SummaryStatistics;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.UnaryOperator;

/**
 * Likelihood-free inference on {@link SimulatorSuite} tasks against reference posteriors, at
 * equal simulation budgets. Each method returns a table of {@code method -> metrics}; C2ST is
 * {@link SimulationBasedInference#c2st} (0.5 is perfect).
 */
public final class LikelihoodFree {

    private LikelihoodFree() {
    }

    static final double[] SLCP_THETA = {0.7, -2.9, -1.0, -0.9, 0.6};

    static double[][] iidObservations(Problem p, double[] theta, int k, SplittableRandom r) {
        double[][] out = new double[k][];
        for (int i = 0; i < k; i++) {
            out[i] = p.simulator().simulate(theta, r);
        }
        return out;
    }

    /** A simulator returning {@code k} concatenated i.i.d. draws, for NPE on trial sets. */
    static Simulator stacked(Simulator sim, int k) {
        return (t, r) -> {
            double[] first = sim.simulate(t, r);
            double[] out = new double[first.length * k];
            System.arraycopy(first, 0, out, 0, first.length);
            for (int i = 1; i < k; i++) {
                System.arraycopy(sim.simulate(t, r), 0, out, i * first.length, first.length);
            }
            return out;
        };
    }

    static double[] flatten(double[][] rows) {
        int d = rows[0].length;
        double[] out = new double[rows.length * d];
        for (int i = 0; i < rows.length; i++) {
            System.arraycopy(rows[i], 0, out, i * d, d);
        }
        return out;
    }

    /** Analytic posterior samples of {@code SimulatorSuite.gaussianLinear(d)} given k trials. */
    static double[][] gaussianIidReference(double[][] xs, int n, SplittableRandom r) {
        int d = xs[0].length;
        double prec = 1 + xs.length / 0.25;
        double[][] out = new double[n][d];
        for (int j = 0; j < d; j++) {
            double s = 0;
            for (double[] x : xs) {
                s += x[j] / 0.25;
            }
            for (int i = 0; i < n; i++) {
                out[i][j] = s / prec + r.nextGaussian() / Math.sqrt(prec);
            }
        }
        return out;
    }

    /**
     * Neural likelihood estimation on SLCP (13.5.1): NLE with an MDN and with a coupling flow,
     * against MDN NPE and SMC-ABC at the same budget.
     *
     * @param budget simulations per method
     * @param seed the seed
     * @return rows {@code {C2ST}}
     */
    public static Map<String, double[]> neuralLikelihood(int budget, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        SplittableRandom r = new SplittableRandom(seed);
        Problem slcp = SimulatorSuite.slcp(0);
        double[] xo = slcp.simulator().simulate(SLCP_THETA, r);
        double[][] ref = slcp.reference().sample(xo, SLCP_THETA, 300, r);
        ConditionalDensity.Config cfg = new ConditionalDensity.Config(4, 48, 600, 128, 3e-3);
        NeuralLikelihood nle = NeuralLikelihood.train(slcp.prior(), slcp.simulator(), budget,
                ConditionalDensity.Kind.MDN, cfg, seed);
        out.put("NLE-MDN", new double[] {SimulationBasedInference.c2st(ref,
                nle.posterior(slcp.prior(), new double[][] {xo}, 300, r))});
        NeuralLikelihood flow = NeuralLikelihood.train(slcp.prior(), slcp.simulator(), budget,
                ConditionalDensity.Kind.COUPLING_FLOW, cfg, seed);
        out.put("NLE-flow", new double[] {SimulationBasedInference.c2st(ref,
                flow.posterior(slcp.prior(), new double[][] {xo}, 300, r))});
        double[][] npe = SimulationBasedInference.neuralPosterior(slcp.prior(), slcp.simulator(), budget, 4, 48, 600,
                seed).sample(xo, 300, r);
        out.put("NPE", new double[] {SimulationBasedInference.c2st(ref, npe)});
        double[][] smc = SimulationBasedInference.smcAbc(slcp.prior(), slcp.simulator(), xo, 200, budget, 0.5, r)
                .samples();
        out.put("SMC-ABC", new double[] {SimulationBasedInference.c2st(ref, smc)});
        return out;
    }

    /**
     * I.i.d. trials on the 2-D Gaussian-linear task (13.5.1 and 13.5.3): one NLE and one NPSE
     * (score composition) serve every trial count, while MDN NPE is retrained on stacked trials
     * for each count.
     *
     * @param budget simulations per method (and per NPE retraining)
     * @param trials the trial counts
     * @param seed the seed
     * @return rows {@code {C2ST}} against the analytic posterior
     */
    public static Map<String, double[]> iidTrials(int budget, int[] trials, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        SplittableRandom r = new SplittableRandom(seed);
        Problem lin = SimulatorSuite.gaussianLinear(2);
        NeuralLikelihood nle = NeuralLikelihood.train(lin.prior(), lin.simulator(), budget,
                ConditionalDensity.Kind.MDN, new ConditionalDensity.Config(2, 32, 400, 128, 3e-3), seed + 1);
        FlowMatchingPosterior score = FlowMatchingPosterior.npse(lin.prior(), lin.simulator(), budget, 48, 800, seed);
        double[] tl = {0.8, -0.5};
        for (int k : trials) {
            double[][] xs = iidObservations(lin, tl, k, r);
            double[][] reference = gaussianIidReference(xs, 300, r);
            out.put("NLE k=" + k, new double[] {SimulationBasedInference.c2st(reference,
                    nle.posterior(lin.prior(), xs, 300, r))});
            out.put("NPSE composed k=" + k, new double[] {SimulationBasedInference.c2st(reference,
                    score.sampleComposed(xs, 300, 100, r))});
            double[][] npeK = SimulationBasedInference.neuralPosterior(lin.prior(), stacked(lin.simulator(), k),
                    budget, 2, 32, 400, seed + k).sample(flatten(xs), 300, r);
            out.put("NPE stacked k=" + k, new double[] {SimulationBasedInference.c2st(reference, npeK)});
        }
        return out;
    }

    /**
     * Neural ratio estimation (13.5.2) on the 2-D Gaussian-linear task: C2ST at one observation
     * against the analytic posterior, and expected coverage of 90% and 50% credible regions over
     * {@code draws} prior-predictive checks, for NRE-A, NRE-C, BNRE and MDN NPE at one budget.
     * Ratio posteriors are drawn by importance resampling of 2000 prior draws.
     *
     * @param budget simulations per method
     * @param draws coverage checks
     * @param seed the seed
     * @return rows {@code {C2ST, coverage at 0.9, coverage at 0.5}}
     */
    public static Map<String, double[]> ratios(int budget, int draws, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        SplittableRandom r = new SplittableRandom(seed);
        Problem lin = SimulatorSuite.gaussianLinear(2);
        double[] xo = lin.simulator().simulate(new double[] {0.8, -0.5}, r);
        double[][] ref = lin.reference().sample(xo, null, 300, r);
        Map<String, PosteriorDiagnostics.PosteriorSampler> methods = new LinkedHashMap<>();
        for (NeuralRatio.Variant v : NeuralRatio.Variant.values()) {
            NeuralRatio nr = NeuralRatio.train(v, lin.prior(), lin.simulator(), budget, 5, 100, 32, 500, seed);
            methods.put(v.name(), (x, n, rr) -> nr.posteriorSir(lin.prior(), x, n, 2000, rr));
        }
        SimulationBasedInference.NeuralPosterior npe = SimulationBasedInference.neuralPosterior(lin.prior(),
                lin.simulator(), budget, 2, 32, 500, seed);
        methods.put("NPE", npe::sample);
        for (Map.Entry<String, PosteriorDiagnostics.PosteriorSampler> e : methods.entrySet()) {
            double c2st = SimulationBasedInference.c2st(ref, e.getValue().sample(xo, 300, r));
            double[][][] checks = PosteriorDiagnostics.simulateChecks(lin.prior(), lin.simulator(), e.getValue(), draws,
                    200, seed + 5);
            PosteriorDiagnostics.Coverage cov = PosteriorDiagnostics.expectedCoverage(checks[0], checks[2], 200);
            out.put(e.getKey(), new double[] {c2st, cov.coverage()[17], cov.coverage()[9]});
        }
        return out;
    }

    /**
     * Sequential posterior estimation on SLCP (13.5.4): APT with the atomic loss, TSNPE and SNL
     * over {@code rounds} rounds, against amortised MDN NPE and SMC-ABC with the same total
     * simulation budget.
     *
     * @param rounds the rounds
     * @param perRound simulations per round
     * @param seed the seed
     * @return rows {@code {C2ST, leakage outside the prior}}
     */
    public static Map<String, double[]> sequential(int rounds, int perRound, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        SplittableRandom r = new SplittableRandom(seed);
        Problem slcp = SimulatorSuite.slcp(0);
        double[] xo = slcp.simulator().simulate(SLCP_THETA, r);
        double[][] ref = slcp.reference().sample(xo, SLCP_THETA, 300, r);
        int total = rounds * perRound;
        ConditionalDensity.Config cfg = new ConditionalDensity.Config(5, 48, 300, 128, 3e-3);
        for (SequentialPosterior.Method m : SequentialPosterior.Method.values()) {
            SequentialPosterior.Result res = SequentialPosterior.run(m, slcp.prior(), slcp.simulator(), xo, rounds,
                    perRound, cfg, 10, 1e-3, 300, seed);
            out.put(m.name(), new double[] {SimulationBasedInference.c2st(ref, res.samples()), res.leakage()});
        }
        NeuralLikelihood snl = NeuralLikelihood.sequential(slcp.prior(), slcp.simulator(), new double[][] {xo}, rounds,
                perRound, ConditionalDensity.Kind.MDN, cfg, seed);
        out.put("SNL", new double[] {SimulationBasedInference.c2st(ref,
                snl.posterior(slcp.prior(), new double[][] {xo}, 300, r)), 0});
        double[][] npe = SimulationBasedInference.neuralPosterior(slcp.prior(), slcp.simulator(), total, 5, 48,
                300 * rounds, seed).sample(xo, 300, r);
        out.put("NPE (amortised)", new double[] {SimulationBasedInference.c2st(ref, npe), 0});
        double[][] smc = SimulationBasedInference.smcAbc(slcp.prior(), slcp.simulator(), xo, 200, total, 0.5, r)
                .samples();
        out.put("SMC-ABC", new double[] {SimulationBasedInference.c2st(ref, smc), 0});
        return out;
    }

    private static double[] scoreMeans(double[][][] posteriors, double[][] truth, double[] width) {
        double err = 0;
        double sd = 0;
        int d = width.length;
        for (int i = 0; i < truth.length; i++) {
            double[][] s = posteriors[i];
            for (int j = 0; j < d; j++) {
                double m = 0;
                for (double[] v : s) {
                    m += v[j] / s.length;
                }
                double var = 0;
                for (double[] v : s) {
                    var += (v[j] - m) * (v[j] - m) / s.length;
                }
                err += Math.pow((m - truth[i][j]) / width[j], 2) / (d * truth.length);
                sd += Math.sqrt(var) / width[j] / (d * truth.length);
            }
        }
        return new double[] {Math.sqrt(err), sd};
    }

    private static double[][] apply(UnaryOperator<double[]> f, double[][] rows) {
        double[][] out = new double[rows.length][];
        for (int i = 0; i < rows.length; i++) {
            out[i] = f.apply(rows[i]);
        }
        return out;
    }

    /**
     * ABC summary statistics (13.5.5) on the Ricker model from raw counts: a pool of
     * {@code budget} prior simulations serves every method and {@code tests} observations.
     * Rejection ABC with Wood's hand summaries, with regression adjustment, semi-automatic
     * summaries (linear on hand summaries; MLP on raw log counts), InfoNCE summary networks on
     * raw log counts, against MDN NPE on raw log counts with the same budget and the prior.
     * Errors are in units of the prior width.
     *
     * @param budget pool size
     * @param tests test observations
     * @param seed the seed
     * @return rows {@code {RMSE of the posterior mean, mean posterior sd, KSG mutual information
     *     between summaries and parameters (0 when not applicable)}}
     */
    public static Map<String, double[]> summaries(int budget, int tests, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        SplittableRandom r = new SplittableRandom(seed);
        Problem ricker = SimulatorSuite.ricker(false);
        double[] width = {2, 0.5, 10};
        double[][] t = new double[budget][];
        double[][] raw = new double[budget][];
        for (int i = 0; i < budget; i++) {
            t[i] = ricker.prior().sample(r);
            raw[i] = logCounts(ricker.simulator().simulate(t[i], r));
        }
        double[][] hand = new double[budget][];
        for (int i = 0; i < budget; i++) {
            hand[i] = SimulatorSuite.rickerSummaries(expm1(raw[i]));
        }
        double[][] truth = new double[tests][];
        double[][] obsRaw = new double[tests][];
        for (int i = 0; i < tests; i++) {
            truth[i] = ricker.prior().sample(r);
            obsRaw[i] = logCounts(ricker.simulator().simulate(truth[i], r));
        }
        int keep = budget / 20;
        UnaryOperator<double[]> handF = x -> SimulatorSuite.rickerSummaries(expm1(x));
        UnaryOperator<double[]> semiLin = SummaryStatistics.semiAutomatic(SummaryStatistics.Regression.LINEAR, t, hand,
                seed);
        UnaryOperator<double[]> semiLinRaw = x -> semiLin.apply(handF.apply(x));
        UnaryOperator<double[]> semiMlp = SummaryStatistics.semiAutomatic(SummaryStatistics.Regression.MLP, t, raw,
                seed);
        UnaryOperator<double[]> info = SummaryStatistics.infomax(t, raw, 3, 600, seed);
        Map<String, UnaryOperator<double[]>> sums = new LinkedHashMap<>();
        sums.put("hand", handF);
        sums.put("semi-auto linear(hand)", semiLinRaw);
        sums.put("semi-auto MLP(raw)", semiMlp);
        sums.put("infomax(raw)", info);
        double[][][] prior = new double[tests][][];
        for (int i = 0; i < tests; i++) {
            prior[i] = new double[keep][];
            for (int k = 0; k < keep; k++) {
                prior[i][k] = ricker.prior().sample(r);
            }
        }
        out.put("prior", new double[] {scoreMeans(prior, truth, width)[0], scoreMeans(prior, truth, width)[1], 0});
        for (Map.Entry<String, UnaryOperator<double[]>> e : sums.entrySet()) {
            double[][] pool = apply(e.getValue(), raw);
            double mi = MutualInformation.ksg(pool, t, 5);
            double[][][] rej = new double[tests][][];
            double[][][] adj = new double[tests][][];
            for (int i = 0; i < tests; i++) {
                double[] so = e.getValue().apply(obsRaw[i]);
                rej[i] = SummaryStatistics.rejection(t, pool, so, keep);
                adj[i] = SummaryStatistics.regressionAdjusted(t, pool, so, keep, r);
            }
            double[] a = scoreMeans(rej, truth, width);
            double[] b = scoreMeans(adj, truth, width);
            out.put("ABC " + e.getKey(), new double[] {a[0], a[1], mi});
            out.put("ABC+adjust " + e.getKey(), new double[] {b[0], b[1], mi});
        }
        Simulator logSim = (th, rr) -> logCounts(ricker.simulator().simulate(th, rr));
        SimulationBasedInference.NeuralPosterior npe = SimulationBasedInference.neuralPosterior(ricker.prior(), logSim,
                budget, 3, 48, 800, seed);
        double[][][] np = new double[tests][][];
        for (int i = 0; i < tests; i++) {
            np[i] = npe.sample(obsRaw[i], keep, r);
        }
        double[] n = scoreMeans(np, truth, width);
        out.put("NPE(raw)", new double[] {n[0], n[1], 0});
        return out;
    }

    private static double[] logCounts(double[] y) {
        double[] out = new double[y.length];
        for (int i = 0; i < y.length; i++) {
            out[i] = Math.log1p(y[i]);
        }
        return out;
    }

    private static double[] expm1(double[] y) {
        double[] out = new double[y.length];
        for (int i = 0; i < y.length; i++) {
            out[i] = Math.rint(Math.expm1(y[i]));
        }
        return out;
    }

    /**
     * Flow-matching posterior estimation (13.5.3): FMPE against MDN NPE on the Gaussian mixture
     * and SLCP at one budget.
     *
     * @param budget simulations per method
     * @param iterations training steps for both
     * @param seed the seed
     * @return rows {@code {C2ST}}
     */
    public static Map<String, double[]> flows(int budget, int iterations, long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        SplittableRandom r = new SplittableRandom(seed);
        for (Problem p : new Problem[] {SimulatorSuite.gaussianMixture(), SimulatorSuite.slcp(0)}) {
            double[] theta = p.dimension() == 2 ? new double[] {1.5, -2.0} : SLCP_THETA;
            double[] xo = p.simulator().simulate(theta, r);
            double[][] ref = p.reference().sample(xo, theta, 300, r);
            FlowMatchingPosterior fm = FlowMatchingPosterior.fmpe(p.prior(), p.simulator(), budget, 64, iterations,
                    seed);
            out.put("FMPE " + p.name(), new double[] {SimulationBasedInference.c2st(ref, fm.sample(xo, 300, 20, r))});
            double[][] npe = SimulationBasedInference.neuralPosterior(p.prior(), p.simulator(), budget, 4, 48,
                    iterations, seed).sample(xo, 300, r);
            out.put("NPE " + p.name(), new double[] {SimulationBasedInference.c2st(ref, npe)});
        }
        return out;
    }
}
