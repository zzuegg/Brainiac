package io.github.zzuegg.brainiac.experimental.deeprl;

import io.github.zzuegg.brainiac.rl.Environment;
import io.github.zzuegg.brainiac.rl.Learner;
import io.github.zzuegg.brainiac.rl.Rl;
import io.github.zzuegg.brainiac.rl.RlListener;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.LongFunction;
import java.util.function.Supplier;
import java.util.function.ToLongFunction;

/**
 * Learning curves at equal environment steps, with the statistics that make small-seed RL
 * comparisons honest (Agarwal et al., "Deep Reinforcement Learning at the Edge of the
 * Statistical Precipice", NeurIPS 2021, <a href="https://arxiv.org/abs/2108.13264">arXiv:2108.13264</a>):
 * every learner is trained once per seed up to the largest budget and evaluated greedily
 * whenever its step count first passes a checkpoint; the scores per checkpoint are summarised
 * by the interquartile mean (IQM) with a stratified bootstrap confidence interval, and pairs of
 * learners by the probability of improvement. Gradient updates (when the learner reports them)
 * and wall-clock seconds are recorded beside the returns, so equal environment steps can be
 * compared with unequal compute.
 *
 * <pre>{@code
 * LearningCurveHarness.Report r = LearningCurveHarness.of(Pendulum::new)
 *         .budgets(1_000, 2_000).seeds(5).evalEpisodes(3)
 *         .add("SAC", seed -> Sac.of(3, 1, nets).seed(seed))
 *         .run();
 * double iqm = r.curve("SAC").iqm(1);
 * }</pre>
 */
public final class LearningCurveHarness {

    private final Supplier<? extends Environment> environments;
    private final Map<String, Entry<?>> learners = new LinkedHashMap<>();
    private long[] budgets = {1_000};
    private long[] seeds = {1, 2, 3, 4, 5};
    private int evalEpisodes = 3;

    private record Entry<L extends Learner>(LongFunction<L> factory, ToLongFunction<? super L> updates) {
        long count(Learner l) {
            @SuppressWarnings("unchecked")
            L typed = (L) l;
            return updates == null ? -1 : updates.applyAsLong(typed);
        }
    }

    private LearningCurveHarness(Supplier<? extends Environment> environments) {
        this.environments = environments;
    }

    /**
     * @param environments builds a fresh environment (one for training and one for evaluation
     *     per run)
     * @return the harness
     */
    public static LearningCurveHarness of(Supplier<? extends Environment> environments) {
        return new LearningCurveHarness(Objects.requireNonNull(environments, "environments"));
    }

    /**
     * @param checkpoints environment steps at which every learner is evaluated, increasing
     * @return this harness
     */
    public LearningCurveHarness budgets(long... checkpoints) {
        if (checkpoints.length == 0) {
            throw new IllegalArgumentException("at least one budget");
        }
        long[] b = checkpoints.clone();
        Arrays.sort(b);
        this.budgets = b;
        return this;
    }

    /**
     * @param n the number of seeds, 1 to {@code n}
     * @return this harness
     */
    public LearningCurveHarness seeds(int n) {
        long[] s = new long[n];
        for (int i = 0; i < n; i++) {
            s[i] = i + 1;
        }
        this.seeds = s;
        return this;
    }

    /**
     * @param episodes greedy evaluation episodes per checkpoint
     * @return this harness
     */
    public LearningCurveHarness evalEpisodes(int episodes) {
        this.evalEpisodes = Math.max(1, episodes);
        return this;
    }

    /**
     * Adds a learner that does not report its gradient updates.
     *
     * @param name the row name
     * @param factory builds the learner for a seed
     * @return this harness
     */
    public LearningCurveHarness add(String name, LongFunction<? extends Learner> factory) {
        learners.put(name, new Entry<Learner>(factory::apply, null));
        return this;
    }

    /**
     * Adds a learner and a way to read its gradient-update count.
     *
     * @param <L> the learner type
     * @param name the row name
     * @param factory builds the learner for a seed
     * @param updates reads the gradient updates taken after training
     * @return this harness
     */
    public <L extends Learner> LearningCurveHarness add(String name, LongFunction<L> factory,
            ToLongFunction<? super L> updates) {
        learners.put(name, new Entry<>(factory, updates));
        return this;
    }

    /**
     * Trains and evaluates every learner on every seed.
     *
     * @return the curves
     */
    public Report run() {
        Map<String, Curve> out = new LinkedHashMap<>();
        for (Map.Entry<String, Entry<?>> e : learners.entrySet()) {
            double[][] scores = new double[seeds.length][budgets.length];
            long[] updates = new long[seeds.length];
            double[] seconds = new double[seeds.length];
            for (int s = 0; s < seeds.length; s++) {
                long seed = seeds[s];
                Learner learner = e.getValue().factory().apply(seed);
                Environment train = environments.get();
                Environment eval = environments.get();
                double[] row = scores[s];
                Arrays.fill(row, Double.NaN);
                int[] next = {0};
                long evalSeed = 10_000 + seed;
                RlListener listener = (episode, result) -> {
                    while (next[0] < budgets.length && result.totalSteps() >= budgets[next[0]]) {
                        row[next[0]++] = Rl.evaluate(learner, eval, evalEpisodes, evalSeed);
                    }
                };
                long t0 = System.nanoTime();
                learner.fit(train, budgets[budgets.length - 1], listener);
                while (next[0] < budgets.length) {
                    row[next[0]++] = Rl.evaluate(learner, eval, evalEpisodes, evalSeed);
                }
                seconds[s] = (System.nanoTime() - t0) / 1e9;
                updates[s] = e.getValue().count(learner);
            }
            out.put(e.getKey(), new Curve(e.getKey(), budgets.clone(), scores, updates, seconds));
        }
        return new Report(out);
    }

    /**
     * One learner's scores.
     *
     * @param name the row name
     * @param budgets the checkpoints
     * @param scores {@code [seed][checkpoint]} greedy returns
     * @param updates gradient updates per seed after the largest budget, -1 where unknown
     * @param seconds wall-clock seconds per seed, evaluation included
     */
    public record Curve(String name, long[] budgets, double[][] scores, long[] updates, double[] seconds) {

        /**
         * @param checkpoint the checkpoint index
         * @return the scores of every seed at it
         */
        public double[] at(int checkpoint) {
            double[] x = new double[scores.length];
            for (int i = 0; i < x.length; i++) {
                x[i] = scores[i][checkpoint];
            }
            return x;
        }

        /**
         * @param checkpoint the checkpoint index
         * @return the interquartile mean over seeds
         */
        public double iqm(int checkpoint) {
            return LearningCurveHarness.iqm(at(checkpoint));
        }

        /** @return the interquartile mean at the last checkpoint */
        public double finalIqm() {
            return iqm(budgets.length - 1);
        }

        /** @return the mean gradient updates per environment step at the largest budget, or -1 */
        public double updatesPerStep() {
            if (updates.length == 0 || updates[0] < 0) {
                return -1;
            }
            return Arrays.stream(updates).average().orElse(0) / budgets[budgets.length - 1];
        }

        /** @return the mean wall-clock seconds per seed */
        public double meanSeconds() {
            return Arrays.stream(seconds).average().orElse(0);
        }

        /**
         * @param checkpoint the checkpoint index
         * @param level the confidence level, such as 0.95
         * @param seed the bootstrap seed
         * @return the bootstrap interval of the IQM, {@code {low, high}}
         */
        public double[] interval(int checkpoint, double level, long seed) {
            return stratifiedBootstrapIqm(new double[][] {at(checkpoint)}, 2_000, level, seed);
        }
    }

    /**
     * Every learner's curve.
     *
     * @param curves by row name, in insertion order
     */
    public record Report(Map<String, Curve> curves) {

        /**
         * @param name the row name
         * @return its curve
         */
        public Curve curve(String name) {
            Curve c = curves.get(name);
            if (c == null) {
                throw new IllegalArgumentException("no learner " + name);
            }
            return c;
        }

        /**
         * Kendall's tau between the rankings of the learners at two checkpoints (by IQM): 1 when
         * the budget does not change the order, -1 when it reverses it.
         *
         * @param a the first checkpoint
         * @param b the second checkpoint
         * @return tau
         */
        public double rankStability(int a, int b) {
            List<Curve> list = new ArrayList<>(curves.values());
            double[] x = new double[list.size()];
            double[] y = new double[list.size()];
            for (int i = 0; i < x.length; i++) {
                x[i] = list.get(i).iqm(a);
                y[i] = list.get(i).iqm(b);
            }
            return kendallTau(x, y);
        }

        /**
         * One line per learner: IQM and 95% interval at every checkpoint, updates per step and
         * seconds per seed.
         *
         * @return the table
         */
        public String table() {
            StringBuilder s = new StringBuilder();
            for (Curve c : curves.values()) {
                s.append(String.format("%-28s", c.name()));
                for (int k = 0; k < c.budgets().length; k++) {
                    double[] ci = c.interval(k, 0.95, 7);
                    s.append(String.format(" | %6d: %8.1f [%7.1f, %7.1f]", c.budgets()[k], c.iqm(k), ci[0], ci[1]));
                }
                s.append(String.format(" | upd/step %.2f | %.1fs%n", c.updatesPerStep(), c.meanSeconds()));
            }
            return s.toString();
        }
    }

    /**
     * The interquartile mean: the mean of the middle half after dropping the lowest and highest
     * quarter (with {@code floor(n/4)} values dropped on each side, as SciPy's
     * {@code trim_mean(x, 0.25)}); NaN values are ignored.
     *
     * @param x the scores
     * @return the IQM
     */
    public static double iqm(double[] x) {
        double[] v = Arrays.stream(x).filter(d -> !Double.isNaN(d)).sorted().toArray();
        if (v.length == 0) {
            return Double.NaN;
        }
        int cut = v.length / 4;
        double sum = 0;
        for (int i = cut; i < v.length - cut; i++) {
            sum += v[i];
        }
        return sum / (v.length - 2 * cut);
    }

    /**
     * The stratified bootstrap confidence interval of the IQM over several tasks: runs are
     * resampled with replacement within each task, and the IQM is taken over the pooled scores.
     *
     * @param scoresPerTask {@code [task][run]} scores, normalised to a common scale
     * @param resamples bootstrap resamples
     * @param level the confidence level
     * @param seed the seed
     * @return {@code {low, high}}
     */
    public static double[] stratifiedBootstrapIqm(double[][] scoresPerTask, int resamples, double level, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        int total = 0;
        for (double[] t : scoresPerTask) {
            total += t.length;
        }
        double[] stats = new double[resamples];
        double[] pooled = new double[total];
        for (int r = 0; r < resamples; r++) {
            int k = 0;
            for (double[] t : scoresPerTask) {
                for (int i = 0; i < t.length; i++) {
                    pooled[k++] = t[random.nextInt(t.length)];
                }
            }
            stats[r] = iqm(pooled);
        }
        Arrays.sort(stats);
        double tail = (1 - level) / 2;
        int lo = (int) Math.floor(tail * (resamples - 1));
        int hi = (int) Math.ceil((1 - tail) * (resamples - 1));
        return new double[] {stats[lo], stats[hi]};
    }

    /**
     * The probability of improvement of X over Y, averaged over tasks: within each task, the
     * Mann-Whitney statistic {@code P(x > y) + P(x = y) / 2} over all pairs of runs.
     *
     * @param x {@code [task][run]} scores of X
     * @param y {@code [task][run]} scores of Y
     * @return the probability, 0.5 when the two are indistinguishable
     */
    public static double probabilityOfImprovement(double[][] x, double[][] y) {
        double sum = 0;
        for (int t = 0; t < x.length; t++) {
            double wins = 0;
            for (double a : x[t]) {
                for (double b : y[t]) {
                    wins += a > b ? 1 : a == b ? 0.5 : 0;
                }
            }
            sum += wins / ((double) x[t].length * y[t].length);
        }
        return sum / x.length;
    }

    /**
     * Kendall's tau-a between two score vectors over the same items.
     *
     * @param x the first scores
     * @param y the second scores
     * @return tau in {@code [-1, 1]}
     */
    public static double kendallTau(double[] x, double[] y) {
        int n = x.length;
        if (n < 2) {
            return 1;
        }
        double s = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                s += Math.signum(x[i] - x[j]) * Math.signum(y[i] - y[j]);
            }
        }
        return s / (n * (n - 1) / 2.0);
    }
}
