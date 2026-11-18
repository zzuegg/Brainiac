package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.train.HyperparameterSearch;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Supplier;

/**
 * Population-based training: a hyperparameter search that runs inside one training run. A
 * population of models trains in parallel, each with its own hyperparameters; every round
 * the worst share of the population <b>exploits</b> (copies the weights and hyperparameters
 * of a randomly chosen member of the best share) and <b>explores</b> (perturbs the copied
 * hyperparameters by a factor, or occasionally resamples them). The result is not one
 * configuration but a <em>schedule</em>, such as a learning rate that falls as training
 * progresses, found at the cost of training the population once.
 *
 * <p>It uses core {@link HyperparameterSearch} ranges and reports its evaluations as
 * {@link HyperparameterSearch.Trial}s, so it compares directly with random search, Hyperband
 * and TPE (which train every configuration from scratch). Numeric ranges are perturbed
 * multiplicatively ({@link HyperparameterSearch.LogUniform}) or additively by a share of the
 * range, and clamped to it; choices are resampled with the resample probability.
 *
 * <p>Paper: Jaderberg et al., "Population Based Training of Neural Networks" (2017),
 * <a href="https://arxiv.org/abs/1711.09846">arXiv:1711.09846</a>.
 */
public final class PopulationBasedTraining {

    /** One member's model and optimizer state, trained a few steps at a time. */
    public interface Member {

        /**
         * Trains further with the given hyperparameters.
         *
         * @param config the hyperparameters by name
         * @param steps how many steps
         * @return the member's score after training, lower is better
         */
        double train(Map<String, Object> config, int steps);

        /**
         * Copies the member's weights and optimizer state, for exploitation.
         *
         * @return an independent copy
         */
        Member copy();
    }

    /**
     * The outcome.
     *
     * @param best the best member's final score
     * @param schedule the best member's hyperparameters in each round (its inherited history)
     * @param trials every evaluation, with the cumulative budget of that member as its budget
     * @param steps the total training steps spent over the population
     */
    public record Result(double best, List<Map<String, Object>> schedule, List<HyperparameterSearch.Trial> trials,
            long steps) {
    }

    private final Map<String, HyperparameterSearch.Range> space;
    private final SplittableRandom random;
    private double truncation = 0.25;
    private double perturbation = 0.2;
    private double resample = 0.25;

    /**
     * Creates the search.
     *
     * @param space each hyperparameter's range, by name
     * @param seed seeds the draws
     */
    public PopulationBasedTraining(Map<String, HyperparameterSearch.Range> space, long seed) {
        this.space = new LinkedHashMap<>(Objects.requireNonNull(space, "space"));
        this.random = new SplittableRandom(seed);
    }

    /**
     * Sets the exploit and explore settings.
     *
     * @param truncation the share of the population that is replaced each round (and the
     *     share it copies from), such as 0.25
     * @param perturbation the perturbation factor: numbers are multiplied by {@code 1 +- p}
     * @param resample the chance a hyperparameter is redrawn from its range instead
     * @return this
     */
    public PopulationBasedTraining settings(double truncation, double perturbation, double resample) {
        this.truncation = truncation;
        this.perturbation = perturbation;
        this.resample = resample;
        return this;
    }

    private Map<String, Object> sample() {
        Map<String, Object> c = new LinkedHashMap<>();
        for (Map.Entry<String, HyperparameterSearch.Range> e : space.entrySet()) {
            c.put(e.getKey(), e.getValue().sample(random));
        }
        return c;
    }

    private Map<String, Object> explore(Map<String, Object> parent) {
        Map<String, Object> c = new LinkedHashMap<>();
        for (Map.Entry<String, HyperparameterSearch.Range> e : space.entrySet()) {
            HyperparameterSearch.Range range = e.getValue();
            Object v = parent.get(e.getKey());
            if (random.nextDouble() < resample) {
                c.put(e.getKey(), range.sample(random));
                continue;
            }
            double f = random.nextBoolean() ? 1 + perturbation : 1 - perturbation;
            c.put(e.getKey(), switch (range) {
                case HyperparameterSearch.LogUniform r -> clamp((double) v * f, r.low(), r.high());
                case HyperparameterSearch.Uniform r ->
                        clamp((double) v + (f - 1) * (r.high() - r.low()), r.low(), r.high());
                case HyperparameterSearch.IntRange r ->
                        (int) clamp(Math.round((int) v * f), r.low(), r.high());
                case HyperparameterSearch.Choice r -> v;
            });
        }
        return c;
    }

    private static double clamp(double v, double low, double high) {
        return Math.max(low, Math.min(high, v));
    }

    /**
     * Runs PBT.
     *
     * @param init creates a fresh member
     * @param population the population size
     * @param rounds exploit/explore rounds
     * @param stepsPerRound training steps per member per round
     * @return the result
     */
    public Result run(Supplier<Member> init, int population, int rounds, int stepsPerRound) {
        List<Member> members = new ArrayList<>();
        List<Map<String, Object>> configs = new ArrayList<>();
        List<List<Map<String, Object>>> histories = new ArrayList<>();
        for (int i = 0; i < population; i++) {
            members.add(init.get());
            Map<String, Object> c = sample();
            configs.add(c);
            histories.add(new ArrayList<>());
        }
        List<HyperparameterSearch.Trial> trials = new ArrayList<>();
        double[] scores = new double[population];
        long steps = 0;
        for (int round = 0; round < rounds; round++) {
            for (int i = 0; i < population; i++) {
                scores[i] = members.get(i).train(configs.get(i), stepsPerRound);
                histories.get(i).add(configs.get(i));
                trials.add(new HyperparameterSearch.Trial(configs.get(i), (round + 1) * stepsPerRound, scores[i]));
                steps += stepsPerRound;
            }
            if (round == rounds - 1) {
                break;
            }
            List<Integer> order = new ArrayList<>();
            for (int i = 0; i < population; i++) {
                order.add(i);
            }
            order.sort(Comparator.comparingDouble(i -> scores[i]));
            int cut = Math.max(1, (int) Math.round(truncation * population));
            List<Integer> top = order.subList(0, cut);
            List<Integer> bottom = new ArrayList<>(order.subList(population - cut, population));
            Collections.reverse(bottom);
            for (int loser : bottom) {
                int parent = top.get(random.nextInt(top.size()));
                members.set(loser, members.get(parent).copy());
                configs.set(loser, explore(configs.get(parent)));
                histories.set(loser, new ArrayList<>(histories.get(parent)));
                scores[loser] = scores[parent];
            }
        }
        int best = 0;
        for (int i = 1; i < population; i++) {
            if (scores[i] < scores[best]) {
                best = i;
            }
        }
        return new Result(scores[best], List.copyOf(histories.get(best)), trials, steps);
    }
}
