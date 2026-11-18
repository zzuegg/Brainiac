package io.github.zzuegg.brainiac.experiments.inference;

import io.github.zzuegg.brainiac.experimental.data.SimulatorSuite;
import io.github.zzuegg.brainiac.experimental.data.SimulatorSuite.DesignProblem;
import io.github.zzuegg.brainiac.experimental.design.AdaptiveDesign;
import io.github.zzuegg.brainiac.experimental.design.DesignOptimiser;
import io.github.zzuegg.brainiac.experimental.design.InformationGain;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.ToDoubleFunction;

/**
 * Bayesian experimental design: EIG estimators against the closed form and static design
 * optimisers (13.5.8); deep adaptive design, Step-DAD and classical sequential baselines on
 * location finding (13.5.9).
 */
public final class ExperimentalDesign {

    private ExperimentalDesign() {
    }

    /**
     * EIG estimators on the linear-Gaussian design at about {@code 10^4} likelihood
     * evaluations each, against the closed form.
     *
     * @param replicates replicates per estimator
     * @param seed the seed
     * @return rows {@code {bias, sd}} in nats; the "exact" row holds {@code {EIG, 0}}
     */
    public static Map<String, double[]> estimators(int replicates, long seed) {
        DesignProblem p = SimulatorSuite.linearGaussianDesign();
        double[] xi = {0.5, 0.5};
        double exact = SimulatorSuite.linearGaussianEig(xi);
        ToDoubleFunction<double[]> prior = t -> -0.5 * t[0] * t[0] - 0.5 * t[1] * t[1] / 4 - Math.log(2 * Math.PI * 2);
        String[] names = {"NMC 100x100", "PCE L=99", "NCE L=99", "PCE L=999 (10x cost)", "NCE L=999 (10x cost)",
            "posterior bound", "marginal bound"};
        double[][] v = new double[names.length][replicates];
        for (int rep = 0; rep < replicates; rep++) {
            SplittableRandom r = new SplittableRandom(seed + rep);
            v[0][rep] = InformationGain.nmc(p, xi, 100, 100, r);
            v[1][rep] = InformationGain.pce(p, xi, 100, 99, r);
            v[2][rep] = InformationGain.nce(p, xi, 100, 99, r);
            v[3][rep] = InformationGain.pce(p, xi, 100, 999, r);
            v[4][rep] = InformationGain.nce(p, xi, 100, 999, r);
            v[5][rep] = InformationGain.posterior(p, prior, xi, 5000, 1, r);
            v[6][rep] = InformationGain.marginal(p, xi, 5000, 3, r);
        }
        Map<String, double[]> out = new LinkedHashMap<>();
        out.put("exact", new double[] {exact, 0});
        for (int m = 0; m < names.length; m++) {
            out.put(names[m], biasSd(v[m], exact));
        }
        double[] mine = new double[3];
        for (int rep = 0; rep < mine.length; rep++) {
            mine[rep] = InformationGain.mine(p, xi, 2000, 300, seed + 100 + rep);
        }
        out.put("MINE (simulations only)", biasSd(mine, exact));
        return out;
    }

    private static double[] biasSd(double[] v, double exact) {
        double m = 0;
        for (double x : v) {
            m += x / v.length;
        }
        double s = 0;
        for (double x : v) {
            s += (x - m) * (x - m) / Math.max(1, v.length - 1);
        }
        return new double[] {m - exact, Math.sqrt(s)};
    }

    /**
     * Static design optimisation: each optimiser's design, scored by a high-precision PCE
     * (location finding, CES) or NMC (death process) estimate.
     *
     * @param seed the seed
     * @return rows {@code {EIG of the chosen design, objective evaluations}}
     */
    public static Map<String, double[]> staticDesigns(long seed) {
        Map<String, double[]> out = new LinkedHashMap<>();
        DesignProblem loc = SimulatorSuite.locationFinding(1);
        ToDoubleFunction<double[]> noisy = x -> InformationGain.pce(loc, x, 60, 60, new SplittableRandom(seed));
        ToDoubleFunction<double[]> score = x -> InformationGain.pce(loc, x, 3000, 300, new SplittableRandom(seed + 1));
        double[] lo = loc.designLow();
        double[] hi = loc.designHigh();
        put(out, "location: random design", score, DesignOptimiser.random(x -> 0, lo, hi, 1, seed));
        put(out, "location: best of 20 random", score, DesignOptimiser.random(noisy, lo, hi, 20, seed));
        put(out, "location: grid 5x5", score, DesignOptimiser.grid(noisy, lo, hi, 5));
        put(out, "location: GP-EI BayesOpt", score, DesignOptimiser.bayesOpt(noisy, lo, hi, 20, seed));
        put(out, "location: CMA-ES", score, DesignOptimiser.cmaEs(noisy, lo, hi, 10, seed));
        put(out, "location: PCE ascent", score, DesignOptimiser.pceAscent(loc, new double[] {2.5, -2.5}, 40, 0.05, 60,
                60, seed));
        put(out, "location: fine grid 21x21 (reference)", score, DesignOptimiser.grid(x -> InformationGain.pce(loc, x,
                400, 100, new SplittableRandom(seed + 2)), lo, hi, 21));

        DesignProblem death = SimulatorSuite.deathProcess();
        ToDoubleFunction<double[]> dNoisy = x -> InformationGain.nmc(death, x, 100, 100, new SplittableRandom(seed));
        ToDoubleFunction<double[]> dScore = x -> InformationGain.nmc(death, x, 2000, 500,
                new SplittableRandom(seed + 1));
        put(out, "death: random design", dScore, DesignOptimiser.random(x -> 0, death.designLow(), death.designHigh(),
                1, seed));
        put(out, "death: grid 9", dScore, DesignOptimiser.grid(dNoisy, death.designLow(), death.designHigh(), 9));
        put(out, "death: PCE ascent", dScore, DesignOptimiser.pceAscent(death, new double[] {3}, 40, 0.05, 100, 100,
                seed));

        DesignProblem ces = SimulatorSuite.ces();
        ToDoubleFunction<double[]> cNoisy = x -> InformationGain.pce(ces, x, 60, 60, new SplittableRandom(seed));
        ToDoubleFunction<double[]> cScore = x -> InformationGain.pce(ces, x, 1000, 300, new SplittableRandom(seed + 1));
        put(out, "CES: random design", cScore, DesignOptimiser.random(x -> 0, ces.designLow(), ces.designHigh(), 1,
                seed));
        put(out, "CES: best of 20 random", cScore, DesignOptimiser.random(cNoisy, ces.designLow(), ces.designHigh(),
                20, seed));
        put(out, "CES: CMA-ES", cScore, DesignOptimiser.cmaEs(cNoisy, ces.designLow(), ces.designHigh(), 12, seed));
        return out;
    }

    private static void put(Map<String, double[]> out, String name, ToDoubleFunction<double[]> score,
            DesignOptimiser.Result r) {
        out.put(name, new double[] {score.applyAsDouble(r.design()), r.evaluations()});
    }

    /**
     * Adaptive design on location finding with one source: trained DAD and static policies,
     * Step-DAD, myopic re-optimisation, uncertainty sampling and random designs, on test
     * instances from the prior and from a shifted prior.
     *
     * @param horizon {@code T}
     * @param iterations training steps of the policies
     * @param tests test instances per setting
     * @param seed the seed
     * @return rows {@code {sPCE (prior), RMSE (prior), sPCE (shifted), RMSE (shifted), ms per
     *     decision}}
     */
    public static Map<String, double[]> adaptive(int horizon, int iterations, int tests, long seed) {
        AdaptiveDesign dad = AdaptiveDesign.train(1, horizon, true, iterations, 64, 127, seed);
        AdaptiveDesign stat = AdaptiveDesign.train(1, horizon, false, iterations, 64, 127, seed + 1);
        AdaptiveDesign.Settings settings = new AdaptiveDesign.Settings(3000, 7, 30, 100, horizon / 2, 25);
        DesignProblem loc = SimulatorSuite.locationFinding(1);
        Map<String, double[]> out = new LinkedHashMap<>();
        for (AdaptiveDesign.Strategy s : AdaptiveDesign.Strategy.values()) {
            double[] row = new double[5];
            for (int shifted = 0; shifted < 2; shifted++) {
                SplittableRandom r = new SplittableRandom(seed + 31 * shifted + 5);
                for (int i = 0; i < tests; i++) {
                    double[] theta = {r.nextGaussian() + 2.0 * shifted, r.nextGaussian() - 2.0 * shifted};
                    SplittableRandom rr = new SplittableRandom(seed + 1000L * shifted + i);
                    AdaptiveDesign.Rollout ro = AdaptiveDesign.run(s, s == AdaptiveDesign.Strategy.STATIC ? stat : dad,
                            theta, horizon, settings, rr);
                    row[2 * shifted] += InformationGain.sequentialPce(loc, theta, ro.designs(), ro.ys(), 5000, rr)
                            / tests;
                    double[] mean = AdaptiveDesign.posteriorMean(ro, 1, 5000, rr);
                    double se = (mean[0] - theta[0]) * (mean[0] - theta[0])
                            + (mean[1] - theta[1]) * (mean[1] - theta[1]);
                    row[2 * shifted + 1] += Math.sqrt(se / 2) / tests;
                    row[4] += ro.secondsPerStep() * 1000 / (2 * tests);
                }
            }
            out.put(s.name(), row);
        }
        return out;
    }
}
