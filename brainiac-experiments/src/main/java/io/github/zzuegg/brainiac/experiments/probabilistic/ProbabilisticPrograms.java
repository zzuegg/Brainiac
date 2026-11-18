package io.github.zzuegg.brainiac.experiments.probabilistic;

import io.github.zzuegg.brainiac.experimental.ml.FactorGraph;
import io.github.zzuegg.brainiac.experimental.ml.FlowVariationalInference;
import io.github.zzuegg.brainiac.experimental.ml.HiddenMarkovModel;
import io.github.zzuegg.brainiac.experimental.ml.McmcDiagnostics;
import io.github.zzuegg.brainiac.experimental.ml.ProbabilisticProgram;
import io.github.zzuegg.brainiac.experimental.ml.ProbabilisticProgram.Bernoulli;
import io.github.zzuegg.brainiac.experimental.ml.ProbabilisticProgram.Beta;
import io.github.zzuegg.brainiac.experimental.ml.ProbabilisticProgram.Categorical;
import io.github.zzuegg.brainiac.experimental.ml.ProbabilisticProgram.Model;
import io.github.zzuegg.brainiac.experimental.ml.ProbabilisticProgram.Normal;
import io.github.zzuegg.brainiac.experimental.ml.ProbabilisticProgram.Trace;
import io.github.zzuegg.brainiac.experimental.ml.ProbabilisticProgram.Weighted;
import io.github.zzuegg.brainiac.experimental.train.HamiltonianMonteCarlo;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Four models written as {@link ProbabilisticProgram}s, each inferred by every engine that
 * applies and compared with the exact answer from closed form or from the repo's exact
 * inference blocks: Bayesian linear regression (Gaussian posterior), a coin-bias change point
 * (enumeration with beta-binomial marginals), a Gaussian HMM written as a program
 * ({@link HiddenMarkovModel} forward-backward) and the sprinkler network ({@link FactorGraph}
 * variable elimination). Every entry is {@code {posterior error, ESS per second}}.
 */
public final class ProbabilisticPrograms {

    private ProbabilisticPrograms() {
    }

    private static double[] row(double error, double ess, long startNanos) {
        double seconds = Math.max(1e-6, (System.nanoTime() - startNanos) / 1e9);
        return new double[] {error, ess / seconds};
    }

    // ------------------------------------------------------------ linear regression

    /**
     * Bayesian linear regression {@code y = a + b x + N(0, 0.5^2)}, {@code a, b ~ N(0, 1)}, 20
     * points. Error: the Euclidean distance of the posterior mean of {@code (a, b)} from the
     * exact one.
     *
     * @param seed the seed
     * @return engine to {@code {error, ESS/s}}
     */
    public static Map<String, double[]> linearRegression(long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = 20;
        double sigma = 0.5;
        double[] xs = new double[n];
        double[] ys = new double[n];
        for (int i = 0; i < n; i++) {
            xs[i] = -2 + 4 * r.nextDouble();
            ys[i] = 0.5 - 1.2 * xs[i] + sigma * r.nextGaussian();
        }
        Model model = c -> {
            double a = c.sample("a", new Normal(0, 1));
            double b = c.sample("b", new Normal(0, 1));
            for (int i = 0; i < n; i++) {
                c.observe(new Normal(a + b * xs[i], sigma), ys[i]);
            }
        };
        // exact posterior: precision I + X^T X / s^2
        double s00 = 1;
        double s01 = 0;
        double s11 = 1;
        double t0 = 0;
        double t1 = 0;
        for (int i = 0; i < n; i++) {
            s00 += 1 / (sigma * sigma);
            s01 += xs[i] / (sigma * sigma);
            s11 += xs[i] * xs[i] / (sigma * sigma);
            t0 += ys[i] / (sigma * sigma);
            t1 += xs[i] * ys[i] / (sigma * sigma);
        }
        double det = s00 * s11 - s01 * s01;
        double[] exact = {(s11 * t0 - s01 * t1) / det, (-s01 * t0 + s00 * t1) / det};
        Map<String, double[]> out = new LinkedHashMap<>();
        long t = System.nanoTime();
        Weighted is = ProbabilisticProgram.importance(model, 4000, seed);
        out.put("likelihood weighting", row(dist(is.mean("a"), is.mean("b"), exact), is.ess(), t));
        t = System.nanoTime();
        List<Trace> mh = ProbabilisticProgram.traceMh(model, 4000, 500, 0.1, seed);
        out.put("trace MH", row(dist(mean(mh, "a"), mean(mh, "b"), exact), chainEss(mh, "a"), t));
        t = System.nanoTime();
        Weighted smc = ProbabilisticProgram.smc(model, 300, 5, 0.1, seed);
        out.put("SMC", row(dist(smc.mean("a"), smc.mean("b"), exact), smc.ess(), t));
        t = System.nanoTime();
        HamiltonianMonteCarlo.Result nuts = HamiltonianMonteCarlo.nuts(ProbabilisticProgram.logDensity(model, "a", "b"),
                new double[2], 200, 500, 0.8, 8, new SplittableRandom(seed));
        double[] m = McmcDiagnostics.mean(nuts.samples());
        out.put("NUTS", row(dist(m[0], m[1], exact), McmcDiagnostics.minEffectiveSampleSize(nuts.samples()), t));
        t = System.nanoTime();
        FlowVariationalInference advi = FlowVariationalInference.create(FlowVariationalInference.Family.MEAN_FIELD, 2,
                0, seed);
        advi.fit(ProbabilisticProgram.logDensity(model, "a", "b"), 300, 16, 0.05f, seed);
        double[] q = McmcDiagnostics.mean(advi.draw(2000, seed));
        out.put("ADVI", row(dist(q[0], q[1], exact), 2000, t));
        return out;
    }

    private static double dist(double a, double b, double[] exact) {
        return Math.hypot(a - exact[0], b - exact[1]);
    }

    private static double mean(List<Trace> traces, String name) {
        double s = 0;
        for (Trace t : traces) {
            s += t.get(name);
        }
        return s / traces.size();
    }

    private static double chainEss(List<Trace> traces, String name) {
        double[] c = new double[traces.size()];
        for (int i = 0; i < c.length; i++) {
            c[i] = traces.get(i).get(name);
        }
        return McmcDiagnostics.effectiveSampleSize(c);
    }

    // ------------------------------------------------------------ change point

    /**
     * A coin whose bias jumps at an unknown flip: {@code tau} uniform on 1..39, biases
     * {@code Beta(1, 1)}, 40 flips with the true change at 15. Error: the absolute error of
     * the posterior mean of {@code tau} against exact enumeration.
     *
     * @param seed the seed
     * @return engine to {@code {error, ESS/s}}
     */
    public static Map<String, double[]> changePoint(long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = 40;
        double[] flips = new double[n];
        for (int i = 0; i < n; i++) {
            flips[i] = r.nextDouble() < (i < 15 ? 0.2 : 0.8) ? 1 : 0;
        }
        double[] uniform = new double[n - 1];
        java.util.Arrays.fill(uniform, 1.0 / (n - 1));
        Model model = c -> {
            int tau = 1 + (int) c.sample("tau", new Categorical(uniform));
            double p1 = c.sample("p1", new Beta(1, 1));
            double p2 = c.sample("p2", new Beta(1, 1));
            for (int i = 0; i < n; i++) {
                c.observe(new Bernoulli(i < tau ? p1 : p2), flips[i]);
            }
        };
        double[] logPost = new double[n - 1];
        for (int k = 0; k < n - 1; k++) {
            int tau = k + 1;
            int h1 = 0;
            int h2 = 0;
            for (int i = 0; i < n; i++) {
                if (i < tau) {
                    h1 += (int) flips[i];
                } else {
                    h2 += (int) flips[i];
                }
            }
            logPost[k] = ProbabilisticProgram.logBeta(1 + h1, 1 + tau - h1)
                    + ProbabilisticProgram.logBeta(1 + h2, 1 + (n - tau) - h2);
        }
        double max = java.util.Arrays.stream(logPost).max().orElse(0);
        double z = 0;
        double exact = 0;
        for (int k = 0; k < n - 1; k++) {
            double w = Math.exp(logPost[k] - max);
            z += w;
            exact += w * k;
        }
        exact /= z;
        Map<String, double[]> out = new LinkedHashMap<>();
        long t = System.nanoTime();
        Weighted is = ProbabilisticProgram.importance(model, 5000, seed);
        out.put("likelihood weighting", row(Math.abs(is.mean("tau") - exact), is.ess(), t));
        t = System.nanoTime();
        List<Trace> mh = ProbabilisticProgram.traceMh(model, 5000, 500, 0, seed);
        out.put("trace MH", row(Math.abs(mean(mh, "tau") - exact), chainEss(mh, "tau"), t));
        t = System.nanoTime();
        Weighted smc = ProbabilisticProgram.smc(model, 300, 2, 0, seed);
        out.put("SMC", row(Math.abs(smc.mean("tau") - exact), smc.ess(), t));
        return out;
    }

    // ------------------------------------------------------------ HMM

    /**
     * A three-state Gaussian HMM written as a program (states {@code z_t} sampled from the
     * transition row, observations conditioned on {@code N(mean_z, 0.7^2)}), 15 steps. Error:
     * the mean absolute error of the state marginals against forward-backward; the extra entry
     * {@code "<engine> log Z error"} compares the evidence estimate with the exact likelihood.
     *
     * @param seed the seed
     * @return engine to {@code {error, ESS/s}}
     */
    public static Map<String, double[]> hmm(long seed) {
        double[] start = {0.6, 0.3, 0.1};
        double[][] trans = {{0.8, 0.15, 0.05}, {0.1, 0.8, 0.1}, {0.05, 0.15, 0.8}};
        double[] means = {-2, 0, 2};
        double sd = 0.7;
        HiddenMarkovModel exact = HiddenMarkovModel.gaussian(start, trans, means, new double[] {sd * sd, sd * sd,
            sd * sd});
        int len = 15;
        double[] obs = exact.sample(len, new SplittableRandom(seed))[0];
        double[][] post = exact.posteriors(obs);
        double logZ = exact.logLikelihood(obs);
        Model model = c -> {
            int s = (int) c.sample("z0", new Categorical(start));
            c.observe(new Normal(means[s], sd), obs[0]);
            for (int t = 1; t < len; t++) {
                s = (int) c.sample("z" + t, new Categorical(trans[s]));
                c.observe(new Normal(means[s], sd), obs[t]);
            }
        };
        Map<String, double[]> out = new LinkedHashMap<>();
        long t = System.nanoTime();
        Weighted is = ProbabilisticProgram.importance(model, 5000, seed);
        out.put("likelihood weighting", row(marginalError(is, post), is.ess(), t));
        out.put("likelihood weighting log Z error", new double[] {Math.abs(is.logEvidence() - logZ), 0});
        t = System.nanoTime();
        List<Trace> mh = ProbabilisticProgram.traceMh(model, 5000, 500, 0, seed);
        double[] lw = new double[mh.size()];
        out.put("trace MH", row(marginalError(new Weighted(mh, lw, 0), post), chainEss(mh, "z7"), t));
        t = System.nanoTime();
        Weighted smc = ProbabilisticProgram.smc(model, 500, 0, 0, seed);
        out.put("SMC", row(marginalError(smc, post), smc.ess(), t));
        out.put("SMC log Z error", new double[] {Math.abs(smc.logEvidence() - logZ), 0});
        return out;
    }

    private static double marginalError(Weighted w, double[][] post) {
        double err = 0;
        for (int t = 0; t < post.length; t++) {
            for (int k = 0; k < post[t].length; k++) {
                String name = "z" + t;
                int kk = k;
                err += Math.abs(w.expectation(tr -> tr.get(name) == kk ? 1 : 0) - post[t][k]);
            }
        }
        return err / (post.length * post[0].length);
    }

    // ------------------------------------------------------------ sprinkler

    /**
     * The sprinkler network (cloudy, sprinkler, rain, wet grass) observed with wet grass.
     * Error: the absolute error of {@code P(rain | wet)} against variable elimination.
     *
     * @param seed the seed
     * @return engine to {@code {error, ESS/s}}
     */
    public static Map<String, double[]> sprinkler(long seed) {
        double[] pWet = {0.0, 0.9, 0.9, 0.99};   // index 2 * sprinkler + rain
        Model model = c -> {
            int cloudy = (int) c.sample("cloudy", new Bernoulli(0.5));
            int sprinkler = (int) c.sample("sprinkler", new Bernoulli(cloudy == 1 ? 0.1 : 0.5));
            int rain = (int) c.sample("rain", new Bernoulli(cloudy == 1 ? 0.8 : 0.2));
            c.observe(new Bernoulli(Math.max(pWet[2 * sprinkler + rain], 1e-12)), 1);
        };
        FactorGraph g = new FactorGraph();
        int cl = g.variable(2);
        int sp = g.variable(2);
        int ra = g.variable(2);
        int we = g.variable(2);
        g.factor(new int[] {cl}, new double[] {0.5, 0.5});
        g.conditional(sp, new int[] {cl}, new double[] {0.5, 0.5, 0.9, 0.1});
        g.conditional(ra, new int[] {cl}, new double[] {0.8, 0.2, 0.2, 0.8});
        g.conditional(we, new int[] {sp, ra}, new double[] {1, 0, 0.1, 0.9, 0.1, 0.9, 0.01, 0.99});
        g.observe(we, 1);
        double exact = g.exactMarginal(ra)[1];
        Map<String, double[]> out = new LinkedHashMap<>();
        long t = System.nanoTime();
        Weighted is = ProbabilisticProgram.importance(model, 5000, seed);
        out.put("likelihood weighting", row(Math.abs(is.mean("rain") - exact), is.ess(), t));
        t = System.nanoTime();
        List<Trace> mh = ProbabilisticProgram.traceMh(model, 20000, 500, 0, seed);
        out.put("trace MH", row(Math.abs(mean(mh, "rain") - exact), chainEss(mh, "rain"), t));
        t = System.nanoTime();
        Weighted smc = ProbabilisticProgram.smc(model, 2000, 0, 0, seed);
        out.put("SMC", row(Math.abs(smc.mean("rain") - exact), smc.ess(), t));
        return out;
    }
}
