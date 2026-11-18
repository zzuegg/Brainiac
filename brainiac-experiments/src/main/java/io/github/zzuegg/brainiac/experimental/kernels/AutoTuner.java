package io.github.zzuegg.brainiac.experimental.kernels;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.function.ToDoubleFunction;

/**
 * Searches a {@link ScheduleSpace} with a budget of real measurements. Four strategies are
 * provided:
 * <ul>
 * <li>{@link #random}: uniform sampling;</li>
 * <li>{@link #grid}: an evenly strided sweep of the enumeration;</li>
 * <li>{@link #evolution}: regularized evolution (tournament, mutate, age out);</li>
 * <li>{@link #costModelGuided}: AutoTVM/Ansor-style search. A {@link LearnedCostModel} is
 *     refitted on everything measured so far. Simulated annealing over the model's predictions
 *     (from the best measured schedules and random restarts) then proposes the next batch, and
 *     only that batch is measured.</li>
 * </ul>
 * Each strategy returns the best-so-far throughput after every measurement.
 */
public final class AutoTuner {

    private final ScheduleSpace space;
    private final ToDoubleFunction<ScheduleSpace.Schedule> measure;

    /**
     * The outcome of one search.
     *
     * @param best the best schedule
     * @param bestGflops its throughput
     * @param trace best-so-far GFLOP/s after each measurement
     * @param model the cost model, for the guided search (otherwise null)
     */
    public record Result(ScheduleSpace.Schedule best, double bestGflops, double[] trace, LearnedCostModel model) {

        /**
         * @param target a throughput
         * @return the number of measurements needed to reach it, or -1
         */
        public int measurementsTo(double target) {
            for (int i = 0; i < trace.length; i++) {
                if (trace[i] >= target) {
                    return i + 1;
                }
            }
            return -1;
        }
    }

    /**
     * @param space the space
     * @param measure GFLOP/s of a schedule (callers usually memoize it)
     */
    public AutoTuner(ScheduleSpace space, ToDoubleFunction<ScheduleSpace.Schedule> measure) {
        this.space = space;
        this.measure = measure;
    }

    private final class Run {
        final double[] trace;
        final List<ScheduleSpace.Schedule> seen = new ArrayList<>();
        final List<Double> scores = new ArrayList<>();
        final Set<ScheduleSpace.Schedule> set = new HashSet<>();
        ScheduleSpace.Schedule best;
        double bestScore = Double.NEGATIVE_INFINITY;

        Run(int budget) {
            trace = new double[budget];
        }

        boolean full() {
            return seen.size() >= trace.length;
        }

        double take(ScheduleSpace.Schedule s) {
            double g = measure.applyAsDouble(s);
            if (g > bestScore) {
                bestScore = g;
                best = s;
            }
            trace[seen.size()] = bestScore;
            seen.add(s);
            scores.add(g);
            set.add(s);
            return g;
        }

        Result result(LearnedCostModel model) {
            return new Result(best, bestScore, trace, model);
        }
    }

    /**
     * @param budget measurements
     * @param seed the seed
     * @return the result
     */
    public Result random(int budget, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Run run = new Run(budget);
        while (!run.full()) {
            ScheduleSpace.Schedule s = space.random(r);
            if (!run.set.contains(s)) {
                run.take(s);
            }
        }
        return run.result(null);
    }

    /**
     * @param budget measurements
     * @return the result of measuring every {@code size / budget}-th schedule
     */
    public Result grid(int budget) {
        Run run = new Run(budget);
        double stride = space.size() / (double) budget;
        for (int i = 0; i < budget; i++) {
            run.take(space.get((int) (i * stride + stride / 2)));
        }
        return run.result(null);
    }

    /**
     * Regularized evolution (Real et al. 2019): a population of {@code population} is aged
     * out first-in-first-out, and parents are the best of a random tournament.
     *
     * @param budget measurements
     * @param population population size
     * @param seed the seed
     * @return the result
     */
    public Result evolution(int budget, int population, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Run run = new Run(budget);
        List<ScheduleSpace.Schedule> pop = new ArrayList<>();
        List<Double> fit = new ArrayList<>();
        while (!run.full() && pop.size() < population) {
            ScheduleSpace.Schedule s = space.random(r);
            if (!run.set.contains(s)) {
                pop.add(s);
                fit.add(run.take(s));
            }
        }
        int guard = 0;
        while (!run.full() && guard++ < budget * 50) {
            int parent = -1;
            for (int t = 0; t < 3; t++) {
                int c = r.nextInt(pop.size());
                if (parent < 0 || fit.get(c) > fit.get(parent)) {
                    parent = c;
                }
            }
            ScheduleSpace.Schedule child = space.mutate(pop.get(parent), r);
            if (run.set.contains(child)) {
                continue;
            }
            pop.add(child);
            fit.add(run.take(child));
            pop.remove(0);
            fit.remove(0);
        }
        return run.result(null);
    }

    /**
     * Cost-model-guided search.
     *
     * @param budget measurements
     * @param initial random measurements before the first fit
     * @param batch measurements per round
     * @param seed the seed
     * @return the result, with the final cost model
     */
    public Result costModelGuided(int budget, int initial, int batch, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Run run = new Run(budget);
        LearnedCostModel model = new LearnedCostModel(space);
        while (!run.full() && run.seen.size() < initial) {
            ScheduleSpace.Schedule s = space.random(r);
            if (!run.set.contains(s)) {
                model.add(s, run.take(s));
            }
        }
        while (!run.full()) {
            model.fit();
            List<ScheduleSpace.Schedule> candidates = anneal(model, run, r);
            candidates.sort(Comparator.comparingDouble(model::predict).reversed());
            int taken = 0;
            for (ScheduleSpace.Schedule s : candidates) {
                if (run.full() || taken >= batch) {
                    break;
                }
                if (!run.set.contains(s)) {
                    model.add(s, run.take(s));
                    taken++;
                }
            }
            if (taken == 0) {
                ScheduleSpace.Schedule s = space.random(r);
                if (!run.set.contains(s)) {
                    model.add(s, run.take(s));
                }
            }
        }
        model.fit();
        return run.result(model);
    }

    /** Simulated annealing on the model from the best measured and random starts. */
    private List<ScheduleSpace.Schedule> anneal(LearnedCostModel model, Run run, SplittableRandom r) {
        List<Integer> order = new ArrayList<>();
        for (int i = 0; i < run.seen.size(); i++) {
            order.add(i);
        }
        order.sort((p, q) -> Double.compare(run.scores.get(q), run.scores.get(p)));
        Set<ScheduleSpace.Schedule> out = new HashSet<>();
        for (int chain = 0; chain < 16; chain++) {
            ScheduleSpace.Schedule s = chain < 8 && chain < order.size() ? run.seen.get(order.get(chain))
                    : space.random(r);
            double e = model.predict(s);
            double temperature = 1.0;
            for (int step = 0; step < 40; step++) {
                ScheduleSpace.Schedule next = space.mutate(s, r);
                double en = model.predict(next);
                if (en >= e || r.nextDouble() < Math.exp((en - e) / temperature)) {
                    s = next;
                    e = en;
                }
                temperature *= 0.9;
                if (!run.set.contains(s)) {
                    out.add(s);
                }
            }
        }
        return new ArrayList<>(out);
    }

    @Override
    public String toString() {
        return "AutoTuner(" + space + ")";
    }
}
