package io.github.zzuegg.brainiac.train;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Hyperparameter search: random search, Hyperband (Li et al., 2018), which gives many
 * configurations a small budget and the best of them more, and a tree-structured Parzen
 * estimator (Bergstra et al., 2011) that samples where good results were found.
 *
 * <p>The objective trains with a configuration for a budget (epochs, steps, ...) and returns a
 * score to minimize, such as a validation loss.
 */
public final class HyperparameterSearch {

    /** One hyperparameter's range. */
    public sealed interface Range permits Uniform, LogUniform, IntRange, Choice {

        /**
         * @param random the source
         * @return a value drawn from the range
         */
        Object sample(SplittableRandom random);
    }

    /**
     * Doubles drawn uniformly.
     *
     * @param low the smallest value
     * @param high the largest value
     */
    public record Uniform(double low, double high) implements Range {
        @Override
        public Object sample(SplittableRandom random) {
            return low + random.nextDouble() * (high - low);
        }
    }

    /**
     * Positive doubles drawn uniformly in their logarithm, for learning rates and decays.
     *
     * @param low the smallest value
     * @param high the largest value
     */
    public record LogUniform(double low, double high) implements Range {
        /** Checks the bounds. */
        public LogUniform {
            if (!(low > 0 && high >= low)) {
                throw new IllegalArgumentException("log-uniform needs 0 < low <= high, got " + low + ", " + high);
            }
        }

        @Override
        public Object sample(SplittableRandom random) {
            return Math.exp(Math.log(low) + random.nextDouble() * (Math.log(high) - Math.log(low)));
        }
    }

    /**
     * Integers drawn uniformly, both ends included.
     *
     * @param low the smallest value
     * @param high the largest value
     */
    public record IntRange(int low, int high) implements Range {
        @Override
        public Object sample(SplittableRandom random) {
            return random.nextInt(low, high + 1);
        }
    }

    /**
     * One of several values.
     *
     * @param options the values
     */
    public record Choice(List<?> options) implements Range {
        @Override
        public Object sample(SplittableRandom random) {
            return options.get(random.nextInt(options.size()));
        }
    }

    /** Trains with a configuration for a budget and scores it; lower is better. */
    @FunctionalInterface
    public interface Objective {

        /**
         * @param config the hyperparameters by name
         * @param budget how much to train: epochs, steps or another unit, at least 1
         * @return the score, lower is better
         */
        double evaluate(Map<String, Object> config, int budget);
    }

    /**
     * One evaluation.
     *
     * @param config the hyperparameters
     * @param budget the budget it ran with
     * @param score its score
     */
    public record Trial(Map<String, Object> config, int budget, double score) {
    }

    /**
     * Every trial run and the best one.
     *
     * @param trials all trials in the order they ran
     * @param best the lowest-scoring trial at the largest budget any trial had
     */
    public record Result(List<Trial> trials, Trial best) {
    }

    private final Map<String, Range> space;
    private final SplittableRandom random;

    /**
     * @param space each hyperparameter's range, by name
     * @param seed seeds the draws
     */
    public HyperparameterSearch(Map<String, Range> space, long seed) {
        this.space = new LinkedHashMap<>(Objects.requireNonNull(space, "space"));
        this.random = new SplittableRandom(seed);
    }

    /** @return a configuration drawn from the space */
    public Map<String, Object> sample() {
        Map<String, Object> c = new LinkedHashMap<>();
        space.forEach((k, r) -> c.put(k, r.sample(random)));
        return c;
    }

    /**
     * Random search: {@code trials} configurations, each with the full budget.
     *
     * @param objective trains and scores
     * @param trials how many configurations
     * @param budget the budget of each
     * @return the trials and the best
     */
    public Result random(Objective objective, int trials, int budget) {
        List<Trial> all = new ArrayList<>();
        for (int i = 0; i < trials; i++) {
            Map<String, Object> c = sample();
            all.add(new Trial(c, budget, objective.evaluate(c, budget)));
        }
        return result(all);
    }

    /**
     * Hyperband: brackets of successive halving that trade the number of configurations
     * against the budget each starts with. In each round of a bracket the best
     * {@code 1 / eta} go on with {@code eta} times the budget. Each evaluation trains from
     * scratch with its budget.
     *
     * @param objective trains and scores
     * @param maxBudget the largest budget one configuration gets
     * @param eta the reduction factor, at least 2 (3 is usual)
     * @return the trials and the best
     */
    public Result hyperband(Objective objective, int maxBudget, int eta) {
        if (eta < 2 || maxBudget < 1) {
            throw new IllegalArgumentException("hyperband needs eta >= 2 and a budget >= 1");
        }
        int sMax = (int) Math.floor(Math.log(maxBudget) / Math.log(eta) + 1e-9);
        List<Trial> all = new ArrayList<>();
        for (int s = sMax; s >= 0; s--) {
            int n = (int) Math.ceil((double) (sMax + 1) / (s + 1) * Math.pow(eta, s));
            List<Map<String, Object>> configs = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                configs.add(sample());
            }
            for (int round = 0; round <= s && !configs.isEmpty(); round++) {
                int budget = Math.max(1, (int) Math.round(maxBudget * Math.pow(eta, round - s)));
                List<Trial> scored = new ArrayList<>();
                for (Map<String, Object> c : configs) {
                    Trial t = new Trial(c, budget, objective.evaluate(c, budget));
                    scored.add(t);
                    all.add(t);
                }
                scored.sort(Comparator.comparingDouble(Trial::score));
                int keep = Math.max(1, scored.size() / eta);
                configs = new ArrayList<>();
                for (int i = 0; i < keep; i++) {
                    configs.add(scored.get(i).config());
                }
            }
        }
        return result(all);
    }

    /**
     * A tree-structured Parzen estimator: after {@code startup} random trials, each new
     * configuration is the one of {@code candidates} draws that best separates the good
     * quarter of the trials so far from the rest, comparing each hyperparameter's density
     * under the two groups.
     *
     * @param objective trains and scores
     * @param trials how many configurations in all
     * @param budget the budget of each
     * @param startup random trials first
     * @param candidates draws compared per trial
     * @return the trials and the best
     */
    public Result tpe(Objective objective, int trials, int budget, int startup, int candidates) {
        List<Trial> all = new ArrayList<>();
        for (int i = 0; i < trials; i++) {
            Map<String, Object> c;
            if (i < startup || all.size() < 4) {
                c = sample();
            } else {
                List<Trial> sorted = new ArrayList<>(all);
                sorted.sort(Comparator.comparingDouble(Trial::score));
                int good = Math.max(1, sorted.size() / 4);
                List<Trial> best = sorted.subList(0, good);
                List<Trial> rest = sorted.subList(good, sorted.size());
                c = null;
                double bestRatio = Double.NEGATIVE_INFINITY;
                for (int k = 0; k < candidates; k++) {
                    Map<String, Object> cand = k % 2 == 0 ? sample() : perturb(best.get(random.nextInt(good)).config());
                    double ratio = 0;
                    for (Map.Entry<String, Range> e : space.entrySet()) {
                        ratio += Math.log(density(e.getValue(), e.getKey(), cand, best))
                                - Math.log(density(e.getValue(), e.getKey(), cand, rest));
                    }
                    if (ratio > bestRatio) {
                        bestRatio = ratio;
                        c = cand;
                    }
                }
            }
            all.add(new Trial(c, budget, objective.evaluate(c, budget)));
        }
        return result(all);
    }

    /** A good configuration nudged: numbers moved by a tenth of their range, choices kept. */
    private Map<String, Object> perturb(Map<String, Object> config) {
        Map<String, Object> c = new LinkedHashMap<>();
        for (Map.Entry<String, Range> e : space.entrySet()) {
            Object v = config.get(e.getKey());
            Object moved = switch (e.getValue()) {
                case Uniform u -> Math.clamp((double) v + random.nextGaussian() * 0.1 * (u.high() - u.low()), u.low(),
                        u.high());
                case LogUniform u -> Math.exp(Math.clamp(Math.log((double) v)
                        + random.nextGaussian() * 0.1 * (Math.log(u.high()) - Math.log(u.low())), Math.log(u.low()),
                        Math.log(u.high())));
                case IntRange r -> Math.clamp(Math.round((int) v + random.nextGaussian() * 0.1
                        * (r.high() - r.low() + 1)), r.low(), r.high());
                case Choice ch -> random.nextDouble() < 0.8 ? v : ch.sample(random);
            };
            c.put(e.getKey(), moved);
        }
        return c;
    }

    /** A Parzen density of one hyperparameter's value under some trials, smoothed by a uniform prior. */
    private static double density(Range range, String key, Map<String, Object> config, List<Trial> trials) {
        Object v = config.get(key);
        if (range instanceof Choice ch) {
            long same = trials.stream().filter(t -> Objects.equals(t.config().get(key), v)).count();
            return (same + 1.0) / (trials.size() + ch.options().size());
        }
        double lo;
        double hi;
        java.util.function.ToDoubleFunction<Object> f;
        switch (range) {
            case LogUniform u -> {
                lo = Math.log(u.low());
                hi = Math.log(u.high());
                f = o -> Math.log(((Number) o).doubleValue());
            }
            case Uniform u -> {
                lo = u.low();
                hi = u.high();
                f = o -> ((Number) o).doubleValue();
            }
            case IntRange r -> {
                lo = r.low();
                hi = r.high() + 1;
                f = o -> ((Number) o).doubleValue();
            }
            default -> throw new IllegalStateException();
        }
        double width = Math.max(hi - lo, 1e-12);
        double bandwidth = width / Math.max(2, Math.sqrt(trials.size()) * 2);
        double x = f.applyAsDouble(v);
        double sum = 1 / width;   // the prior
        for (Trial t : trials) {
            double z = (x - f.applyAsDouble(t.config().get(key))) / bandwidth;
            sum += Math.exp(-0.5 * z * z) / (bandwidth * Math.sqrt(2 * Math.PI));
        }
        return sum / (trials.size() + 1);
    }

    private static Result result(List<Trial> all) {
        int top = all.stream().mapToInt(Trial::budget).max().orElse(0);
        Trial best = all.stream().filter(t -> t.budget() == top).min(Comparator.comparingDouble(Trial::score))
                .orElse(null);
        return new Result(List.copyOf(all), best);
    }
}
