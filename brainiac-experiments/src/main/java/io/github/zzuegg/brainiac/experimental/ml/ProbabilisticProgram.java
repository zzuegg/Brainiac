package io.github.zzuegg.brainiac.experimental.ml;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * A tiny trace-based probabilistic programming language embedded in Java (Wingate et al. 2011,
 * <a href="https://proceedings.mlr.press/v15/wingate11a.html">Lightweight implementations of
 * probabilistic programming languages</a>; van de Meent et al. 2018,
 * <a href="https://arxiv.org/abs/1809.10756">arXiv:1809.10756</a>).
 *
 * <p>A {@link Model} is a lambda that calls {@link Context#sample} for latent choices (each
 * with a unique name, its "address") and {@link Context#observe} to condition on data.
 * Inference engines run the program many times and handle those two calls differently:
 * <ul>
 *   <li>{@link #importance} (likelihood weighting) samples every latent from its prior and
 *       weights the run by the likelihood of its observations;</li>
 *   <li>{@link #traceMh} (lightweight Metropolis-Hastings) changes one site of the current
 *       trace, re-runs the program replaying every other site, and accepts by the
 *       Metropolis-Hastings ratio with the fresh and stale sites accounted for;</li>
 *   <li>{@link #smc} (sequential Monte Carlo) advances a particle population one observation
 *       at a time, weights by that observation and resamples when the ESS drops, which also
 *       estimates the evidence;</li>
 *   <li>{@link #logDensity} turns a program with a fixed set of continuous sites into a
 *       {@link LogDensity} in unconstrained coordinates, so Hamiltonian Monte Carlo, NUTS and
 *       {@link FlowVariationalInference ADVI or flow VI} run on it unchanged.</li>
 * </ul>
 */
public final class ProbabilisticProgram {

    private ProbabilisticProgram() {
    }

    /** Where a distribution's values live, which picks the transform to unconstrained space. */
    public enum Support {
        /** All reals: identity. */
        REAL,
        /** Positive reals: log. */
        POSITIVE,
        /** The unit interval: logit. */
        UNIT,
        /** Integers: not transformable, only for importance, MH and SMC. */
        DISCRETE
    }

    /** A primitive distribution over one number. */
    public interface Distribution {
        /**
         * Draws a value.
         *
         * @param random the random source
         * @return a sample
         */
        double sample(SplittableRandom random);

        /**
         * The log probability (density or mass) of a value.
         *
         * @param x the value
         * @return {@code log p(x)}, negative infinity outside the support
         */
        double logProb(double x);

        /**
         * The support.
         *
         * @return the support
         */
        Support support();
    }

    /**
     * A normal distribution.
     *
     * @param mean the mean
     * @param std the standard deviation
     */
    public record Normal(double mean, double std) implements Distribution {
        @Override
        public double sample(SplittableRandom random) {
            return mean + std * random.nextGaussian();
        }

        @Override
        public double logProb(double x) {
            double z = (x - mean) / std;
            return -0.5 * z * z - Math.log(std) - 0.5 * Math.log(2 * Math.PI);
        }

        @Override
        public Support support() {
            return Support.REAL;
        }
    }

    /**
     * A beta distribution.
     *
     * @param a the first shape
     * @param b the second shape
     */
    public record Beta(double a, double b) implements Distribution {
        @Override
        public double sample(SplittableRandom random) {
            double x = Gamma.draw(a, random);
            double y = Gamma.draw(b, random);
            return x / (x + y);
        }

        @Override
        public double logProb(double x) {
            if (x <= 0 || x >= 1) {
                return Double.NEGATIVE_INFINITY;
            }
            return (a - 1) * Math.log(x) + (b - 1) * Math.log1p(-x) - logBeta(a, b);
        }

        @Override
        public Support support() {
            return Support.UNIT;
        }
    }

    /**
     * A gamma distribution.
     *
     * @param shape the shape
     * @param rate the rate
     */
    public record Gamma(double shape, double rate) implements Distribution {
        @Override
        public double sample(SplittableRandom random) {
            return draw(shape, random) / rate;
        }

        @Override
        public double logProb(double x) {
            if (x <= 0) {
                return Double.NEGATIVE_INFINITY;
            }
            return shape * Math.log(rate) + (shape - 1) * Math.log(x) - rate * x - logGamma(shape);
        }

        @Override
        public Support support() {
            return Support.POSITIVE;
        }

        /** Marsaglia-Tsang gamma(shape, 1) draws. */
        static double draw(double shape, SplittableRandom r) {
            if (shape < 1) {
                return draw(shape + 1, r) * Math.pow(r.nextDouble(), 1 / shape);
            }
            double d = shape - 1.0 / 3;
            double c = 1 / Math.sqrt(9 * d);
            while (true) {
                double x = r.nextGaussian();
                double v = 1 + c * x;
                if (v <= 0) {
                    continue;
                }
                v = v * v * v;
                double u = r.nextDouble();
                if (Math.log(u) < 0.5 * x * x + d - d * v + d * Math.log(v)) {
                    return d * v;
                }
            }
        }
    }

    /**
     * A Bernoulli distribution over {0, 1}.
     *
     * @param p the probability of 1
     */
    public record Bernoulli(double p) implements Distribution {
        @Override
        public double sample(SplittableRandom random) {
            return random.nextDouble() < p ? 1 : 0;
        }

        @Override
        public double logProb(double x) {
            return x == 1 ? Math.log(p) : x == 0 ? Math.log1p(-p) : Double.NEGATIVE_INFINITY;
        }

        @Override
        public Support support() {
            return Support.DISCRETE;
        }
    }

    /**
     * A categorical distribution over {0, ..., k-1}.
     *
     * @param probabilities the probabilities, summing to 1
     */
    public record Categorical(double[] probabilities) implements Distribution {
        @Override
        public double sample(SplittableRandom random) {
            double u = random.nextDouble();
            for (int i = 0; i < probabilities.length; i++) {
                u -= probabilities[i];
                if (u < 0) {
                    return i;
                }
            }
            return probabilities.length - 1;
        }

        @Override
        public double logProb(double x) {
            int i = (int) x;
            return i == x && i >= 0 && i < probabilities.length ? Math.log(probabilities[i]) : Double.NEGATIVE_INFINITY;
        }

        @Override
        public Support support() {
            return Support.DISCRETE;
        }
    }

    /** A probabilistic program. */
    @FunctionalInterface
    public interface Model {
        /**
         * Runs the program once.
         *
         * @param context handles the {@code sample} and {@code observe} calls
         */
        void run(Context context);
    }

    /** What a running program can do. */
    public interface Context {
        /**
         * Draws (or replays) a latent choice.
         *
         * @param name the unique address of this choice
         * @param distribution its prior
         * @return the value
         */
        double sample(String name, Distribution distribution);

        /**
         * Conditions on an observed value.
         *
         * @param distribution its likelihood
         * @param value the observed value
         */
        void observe(Distribution distribution, double value);
    }

    /**
     * One execution of a program.
     *
     * @param values the latent values by address, in execution order
     * @param siteLogProbs the prior log probability of each latent
     * @param logLikelihood the summed log likelihood of the observations
     */
    public record Trace(Map<String, Double> values, Map<String, Double> siteLogProbs, double logLikelihood) {
        /**
         * The summed prior log probability.
         *
         * @return {@code log p(x)}
         */
        public double logPrior() {
            double s = 0;
            for (double v : siteLogProbs.values()) {
                s += v;
            }
            return s;
        }

        /**
         * The log joint.
         *
         * @return {@code log p(x, y)}
         */
        public double logJoint() {
            return logPrior() + logLikelihood;
        }

        /**
         * One latent's value.
         *
         * @param name the address
         * @return its value, or NaN when the run did not visit it
         */
        public double get(String name) {
            Double v = values.get(name);
            return v == null ? Double.NaN : v;
        }
    }

    /**
     * Weighted traces from importance sampling or SMC.
     *
     * @param traces the traces
     * @param logWeights their normalised-or-not log weights
     * @param logEvidence the estimate of {@code log p(y)}
     */
    public record Weighted(List<Trace> traces, double[] logWeights, double logEvidence) {
        /**
         * The weighted posterior mean of a function of the trace.
         *
         * @param f the function
         * @return {@code E[f]}
         */
        public double expectation(java.util.function.ToDoubleFunction<Trace> f) {
            double max = Arrays.stream(logWeights).max().orElse(0);
            double s = 0;
            double z = 0;
            for (int i = 0; i < logWeights.length; i++) {
                double w = Math.exp(logWeights[i] - max);
                s += w * f.applyAsDouble(traces.get(i));
                z += w;
            }
            return s / z;
        }

        /**
         * The weighted posterior mean of one latent.
         *
         * @param name the address
         * @return its posterior mean
         */
        public double mean(String name) {
            return expectation(t -> t.get(name));
        }

        /**
         * The importance effective sample size.
         *
         * @return the ESS
         */
        public double ess() {
            return McmcDiagnostics.importanceEss(logWeights);
        }
    }

    /** Thrown to stop a run at an observation (SMC). */
    private static final class Stop extends RuntimeException {
        private static final long serialVersionUID = 1L;

        Stop() {
            super(null, null, false, false);
        }
    }

    /** Runs a program with replayed values, fresh prior draws and an optional stop. */
    private static final class Execution implements Context {
        final Map<String, Double> replay;
        final SplittableRandom random;
        final int stopAfter;
        final Map<String, Double> values = new LinkedHashMap<>();
        final Map<String, Double> logProbs = new LinkedHashMap<>();
        final List<Double> observationLogLiks = new ArrayList<>();
        final List<String> fresh = new ArrayList<>();
        double logLikelihood;

        Execution(Map<String, Double> replay, SplittableRandom random, int stopAfter) {
            this.replay = replay;
            this.random = random;
            this.stopAfter = stopAfter;
        }

        @Override
        public double sample(String name, Distribution d) {
            if (values.containsKey(name)) {
                throw new IllegalStateException("address used twice: " + name);
            }
            Double v = replay == null ? null : replay.get(name);
            if (v == null) {
                v = d.sample(random);
                fresh.add(name);
            }
            values.put(name, v);
            logProbs.put(name, d.logProb(v));
            return v;
        }

        @Override
        public void observe(Distribution d, double value) {
            double l = d.logProb(value);
            observationLogLiks.add(l);
            logLikelihood += l;
            if (stopAfter > 0 && observationLogLiks.size() >= stopAfter) {
                throw new Stop();
            }
        }

        Trace trace() {
            return new Trace(values, logProbs, logLikelihood);
        }
    }

    private static Execution execute(Model model, Map<String, Double> replay, SplittableRandom random, int stopAfter) {
        Execution e = new Execution(replay, random, stopAfter);
        try {
            model.run(e);
        } catch (Stop stop) {
            // the run reached the requested observation
        }
        return e;
    }

    /**
     * Runs the program once from the prior.
     *
     * @param model the program
     * @param random the random source
     * @return the trace
     */
    public static Trace prior(Model model, SplittableRandom random) {
        return execute(model, null, random, 0).trace();
    }

    /**
     * Likelihood weighting: {@code n} prior runs weighted by their likelihood.
     *
     * @param model the program
     * @param n the number of runs
     * @param seed the seed
     * @return the weighted traces
     */
    public static Weighted importance(Model model, int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<Trace> traces = new ArrayList<>();
        double[] lw = new double[n];
        for (int i = 0; i < n; i++) {
            Trace t = prior(model, r);
            traces.add(t);
            lw[i] = t.logLikelihood();
        }
        return new Weighted(traces, lw, McmcDiagnostics.logMeanExp(lw));
    }

    /**
     * Lightweight single-site Metropolis-Hastings. Each step picks one visited site uniformly;
     * a real-valued site moves by a Gaussian random walk of scale {@code driftScale} half of the
     * time, otherwise (and always for other supports) it is redrawn from its prior. The program
     * is re-run replaying all other sites; sites the new run visits for the first time are drawn
     * from their priors, and the acceptance ratio corrects for them and for sites it no longer
     * visits.
     *
     * @param model the program
     * @param iterations the kept steps
     * @param burnIn the discarded steps
     * @param driftScale the random-walk scale for real sites (0 disables it)
     * @param seed the seed
     * @return the kept traces, one per step
     */
    public static List<Trace> traceMh(Model model, int iterations, int burnIn, double driftScale, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Execution current = execute(model, null, r, 0);
        List<Trace> out = new ArrayList<>();
        for (int it = 0; it < iterations + burnIn; it++) {
            current = mhStep(model, current, r, driftScale, 0);
            if (it >= burnIn) {
                out.add(current.trace());
            }
        }
        return out;
    }

    /** One single-site MH move on a (possibly partial, stopped at an observation) run. */
    private static Execution mhStep(Model model, Execution current, SplittableRandom r, double driftScale,
            int stopAfter) {
        List<String> sites = new ArrayList<>(current.values.keySet());
        if (sites.isEmpty()) {
            return current;
        }
        String site = sites.get(r.nextInt(sites.size()));
        Map<String, Double> replay = new LinkedHashMap<>(current.values);
        replay.remove(site);
        boolean drift = false;
        double old = current.values.get(site);
        if (driftScale > 0 && r.nextBoolean() && isReal(model, current, site)) {
            replay.put(site, old + driftScale * r.nextGaussian());
            drift = true;
        }
        Execution next = execute(model, replay, r, stopAfter);
        double logAlpha = next.trace().logJoint() - current.trace().logJoint()
                + Math.log(sites.size()) - Math.log(Math.max(1, next.values.size()));
        for (String f : next.fresh) {
            if (!(drift && f.equals(site))) {
                logAlpha -= next.logProbs.get(f);
            }
        }
        if (!drift && next.logProbs.containsKey(site)) {
            logAlpha += current.logProbs.get(site);
        }
        for (Map.Entry<String, Double> e : current.logProbs.entrySet()) {
            if (!next.values.containsKey(e.getKey())) {
                logAlpha += e.getValue();
            }
        }
        if (Double.isFinite(next.trace().logJoint()) && Math.log(r.nextDouble()) < logAlpha) {
            return next;
        }
        return current;
    }

    private static boolean isReal(Model model, Execution current, String site) {
        Support[] found = new Support[1];
        Context probe = new Context() {
            @Override
            public double sample(String name, Distribution d) {
                if (name.equals(site)) {
                    found[0] = d.support();
                }
                return current.values.getOrDefault(name, 0.0);
            }

            @Override
            public void observe(Distribution d, double value) {
            }
        };
        try {
            model.run(probe);
        } catch (RuntimeException ex) {
            return false;
        }
        return found[0] == Support.REAL;
    }

    /**
     * Sequential Monte Carlo with resampling at observations: every particle is advanced to the
     * next observation (re-running the program, replaying its sites and drawing new ones from
     * the prior), weighted by that observation's likelihood, and the population is resampled
     * systematically when its ESS falls below half. After each resampling every particle takes
     * {@code moves} single-site MH steps on the partial program (resample-move, Gilks and
     * Berzuini 2001), which rejuvenates continuous sites that resampling only copies. Assumes
     * every run makes the same number of observations.
     *
     * @param model the program
     * @param particles the population size
     * @param moves MH rejuvenation steps per particle after a resampling, 0 for plain SMC
     * @param driftScale the random-walk scale of those moves for real sites
     * @param seed the seed
     * @return the final weighted particles and the evidence estimate
     */
    public static Weighted smc(Model model, int particles, int moves, double driftScale, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int observations = execute(model, null, r, 0).observationLogLiks.size();
        List<Map<String, Double>> states = new ArrayList<>();
        for (int i = 0; i < particles; i++) {
            states.add(Map.of());
        }
        double[] lw = new double[particles];
        double logZ = 0;
        List<Trace> traces = new ArrayList<>();
        for (int k = 1; k <= Math.max(1, observations); k++) {
            traces.clear();
            for (int i = 0; i < particles; i++) {
                Execution e = execute(model, states.get(i), r, observations == 0 ? 0 : k);
                states.set(i, e.values);
                if (observations > 0) {
                    lw[i] += e.observationLogLiks.get(k - 1);
                }
                traces.add(e.trace());
            }
            double ess = McmcDiagnostics.importanceEss(lw);
            if (ess < particles / 2.0 && k < observations) {
                logZ += McmcDiagnostics.logMeanExp(lw);
                int[] pick = systematic(lw, r);
                List<Map<String, Double>> next = new ArrayList<>();
                for (int i : pick) {
                    Map<String, Double> state = states.get(i);
                    if (moves > 0) {
                        Execution e = execute(model, state, r, k);
                        for (int m = 0; m < moves; m++) {
                            e = mhStep(model, e, r, driftScale, k);
                        }
                        state = e.values;
                    }
                    next.add(state);
                }
                states = next;
                Arrays.fill(lw, 0);
            }
        }
        logZ += McmcDiagnostics.logMeanExp(lw);
        return new Weighted(List.copyOf(traces), lw.clone(), logZ);
    }

    /**
     * Systematic resampling.
     *
     * @param logWeights the log weights
     * @param random the random source
     * @return the chosen indices
     */
    public static int[] systematic(double[] logWeights, SplittableRandom random) {
        int n = logWeights.length;
        double max = Arrays.stream(logWeights).max().orElse(0);
        double[] c = new double[n];
        double s = 0;
        for (int i = 0; i < n; i++) {
            s += Math.exp(logWeights[i] - max);
            c[i] = s;
        }
        int[] out = new int[n];
        double u = random.nextDouble() / n;
        int j = 0;
        for (int i = 0; i < n; i++) {
            double target = (u + (double) i / n) * s;
            while (j < n - 1 && c[j] < target) {
                j++;
            }
            out[i] = j;
        }
        return out;
    }

    /**
     * A program with a fixed set of continuous sites as a log density in unconstrained
     * coordinates: each site is mapped to the reals by its support's transform (identity, log
     * or logit) and the log-Jacobian is added. The gradient is by central differences.
     *
     * @param model the program
     * @param names the continuous sites, in coordinate order
     * @return the log joint in unconstrained space
     */
    public static LogDensity logDensity(Model model, String... names) {
        String[] order = names.clone();
        return LogDensity.numerical(u -> {
            Map<String, Double> replay = new LinkedHashMap<>();
            double[] jac = new double[1];
            Support[] supports = supports(model, order);
            for (int i = 0; i < order.length; i++) {
                replay.put(order[i], constrain(supports[i], u[i], jac));
            }
            Execution e = execute(model, replay, new SplittableRandom(0), 0);
            if (!e.fresh.isEmpty()) {
                throw new IllegalStateException("sites not covered by the coordinates: " + e.fresh);
            }
            return e.trace().logJoint() + jac[0];
        }, 1e-5);
    }

    /**
     * Maps unconstrained coordinates back to site values.
     *
     * @param model the program
     * @param u the unconstrained point
     * @param names the sites, in coordinate order
     * @return the site values
     */
    public static double[] constrain(Model model, double[] u, String... names) {
        Support[] s = supports(model, names);
        double[] x = new double[u.length];
        double[] jac = new double[1];
        for (int i = 0; i < u.length; i++) {
            x[i] = constrain(s[i], u[i], jac);
        }
        return x;
    }

    private static double constrain(Support s, double u, double[] logJac) {
        return switch (s) {
            case REAL -> u;
            case POSITIVE -> {
                logJac[0] += u;
                yield Math.exp(u);
            }
            case UNIT -> {
                double x = 1 / (1 + Math.exp(-u));
                logJac[0] += Math.log(x) + Math.log1p(-x);
                yield x;
            }
            case DISCRETE -> throw new IllegalArgumentException("discrete sites have no unconstrained form");
        };
    }

    private static Support[] supports(Model model, String[] names) {
        Map<String, Support> found = new LinkedHashMap<>();
        model.run(new Context() {
            @Override
            public double sample(String name, Distribution d) {
                found.put(name, d.support());
                return switch (d.support()) {
                    case POSITIVE -> 1;
                    case UNIT -> 0.5;
                    default -> 0;
                };
            }

            @Override
            public void observe(Distribution d, double value) {
            }
        });
        Support[] s = new Support[names.length];
        for (int i = 0; i < names.length; i++) {
            s[i] = found.get(names[i]);
            if (s[i] == null) {
                throw new IllegalArgumentException("unknown site " + names[i]);
            }
        }
        return s;
    }

    /**
     * {@code log Gamma(x)} by the Lanczos approximation.
     *
     * @param x a positive number
     * @return {@code log Gamma(x)}
     */
    public static double logGamma(double x) {
        if (x < 0.5) {
            return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - logGamma(1 - x);
        }
        double[] g = {0.99999999999980993, 676.5203681218851, -1259.1392167224028, 771.32342877765313,
            -176.61502916214059, 12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6,
            1.5056327351493116e-7};
        x -= 1;
        double a = g[0];
        double t = x + 7.5;
        for (int i = 1; i < 9; i++) {
            a += g[i] / (x + i);
        }
        return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
    }

    /**
     * {@code log B(a, b)}.
     *
     * @param a the first argument
     * @param b the second argument
     * @return the log beta function
     */
    public static double logBeta(double a, double b) {
        return logGamma(a) + logGamma(b) - logGamma(a + b);
    }
}
