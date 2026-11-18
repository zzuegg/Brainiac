package io.github.zzuegg.brainiac.experiments.evolution;

import io.github.zzuegg.brainiac.experimental.evolve.CmaEs;
import io.github.zzuegg.brainiac.experimental.evolve.MapElites;
import io.github.zzuegg.brainiac.experimental.evolve.Nsga2;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Multi-objective search on the ZDT test problems (Zitzler, Deb and Thiele 2000): NSGA-II
 * (Deb et al. 2002) against weighted-sum scalarisation solved by CMA-ES, MAP-Elites with the
 * first objective as its one-dimensional descriptor, and random search, at equal evaluation
 * budgets. Every method is scored twice by the hypervolume up to the reference point
 * {@code (1.1, 1.1)} (the true fronts reach about 0.87 for ZDT1, 0.54 for ZDT2 and 1.33 for
 * ZDT3): on the archive of everything it evaluated, and on the set it returns (NSGA-II's final
 * population, the optimum of each of the 10 weightings, MAP-Elites' elites, random search's
 * whole archive). The spread is the range of {@code f1} covered by the archive's
 * non-dominated points (1 on the true fronts, about 0.85 for ZDT3's disconnected front).
 */
public final class MultiObjectiveSearch {

    private MultiObjectiveSearch() {
    }

    /** The ZDT problems, 30 decision variables in {@code [0, 1]}. */
    public enum Problem {
        /** Convex front {@code f2 = 1 - sqrt(f1)}. */
        ZDT1,
        /** Non-convex front {@code f2 = 1 - f1^2}: weighted sums only find its two ends. */
        ZDT2,
        /** Disconnected front. */
        ZDT3;

        /**
         * @param x the decision vector
         * @return {@code (f1, f2)}
         */
        public double[] objectives(double[] x) {
            double f1 = x[0];
            double s = 0;
            for (int i = 1; i < x.length; i++) {
                s += x[i];
            }
            double g = 1 + 9 * s / (x.length - 1);
            double h = switch (this) {
                case ZDT1 -> 1 - Math.sqrt(f1 / g);
                case ZDT2 -> 1 - (f1 / g) * (f1 / g);
                case ZDT3 -> 1 - Math.sqrt(f1 / g) - f1 / g * Math.sin(10 * Math.PI * f1);
            };
            return new double[] {f1, g * h};
        }
    }

    private static final double[] REFERENCE = {1.1, 1.1};
    private static final int DIMS = 30;

    /**
     * One method's result, means over seeds.
     *
     * @param method the method
     * @param hypervolume hypervolume of the archive after 10%, 25%, 50% and 100% of the budget
     * @param spread the range of {@code f1} covered by the final non-dominated set
     * @param returnedHypervolume hypervolume of the returned set
     */
    public record Score(String method, double[] hypervolume, double spread, double returnedHypervolume) {

        /** @return the final hypervolume */
        public double finalHypervolume() {
            return hypervolume[hypervolume.length - 1];
        }
    }

    /**
     * Runs every method on a problem.
     *
     * @param problem the problem
     * @param budget evaluations per run
     * @param seeds seeds
     * @return method name to score
     */
    public static Map<String, Score> compare(Problem problem, int budget, int seeds) {
        Map<String, Score> out = new LinkedHashMap<>();
        for (String m : List.of("random", "weighted-sum CMA-ES", "MAP-Elites", "NSGA-II")) {
            double[] hv = new double[4];
            double spread = 0;
            double returned = 0;
            for (long seed = 1; seed <= seeds; seed++) {
                List<double[]> result = new ArrayList<>();
                List<double[]> archive = run(m, problem, budget, seed, result);
                returned += Nsga2.hypervolume2d(result, REFERENCE) / seeds;
                int[] checkpoints = {budget / 10, budget / 4, budget / 2, archive.size()};
                for (int k = 0; k < 4; k++) {
                    hv[k] += Nsga2.hypervolume2d(archive.subList(0, Math.min(checkpoints[k], archive.size())),
                            REFERENCE) / seeds;
                }
                spread += spread(archive) / seeds;
            }
            out.put(m, new Score(m, hv, spread, returned));
        }
        return out;
    }

    /** The range of f1 over the non-dominated points inside the reference box (a 2-D sweep). */
    private static double spread(List<double[]> archive) {
        List<double[]> p = new ArrayList<>(archive);
        p.sort(java.util.Comparator.comparingDouble((double[] q) -> q[0]).thenComparingDouble(q -> q[1]));
        double best = REFERENCE[1];
        double lo = Double.POSITIVE_INFINITY;
        double hi = Double.NEGATIVE_INFINITY;
        for (double[] q : p) {
            if (q[1] < best && q[0] < REFERENCE[0]) {
                best = q[1];
                lo = Math.min(lo, q[0]);
                hi = Math.max(hi, q[0]);
            }
        }
        return hi > lo ? hi - lo : 0;
    }

    private static double[] clip(double[] x) {
        double[] y = x.clone();
        for (int i = 0; i < y.length; i++) {
            y[i] = Math.max(0, Math.min(1, y[i]));
        }
        return y;
    }

    /**
     * @param result receives the objective vectors of the set the method returns
     * @return every objective vector the method evaluated, in order
     */
    private static List<double[]> run(String method, Problem problem, int budget, long seed, List<double[]> result) {
        List<double[]> archive = new ArrayList<>();
        Function<double[], double[]> f = x -> {
            double[] o = problem.objectives(clip(x));
            if (archive.size() < budget) {
                archive.add(o);
            }
            return o;
        };
        SplittableRandom r = new SplittableRandom(seed);
        double[] lo = new double[DIMS];
        double[] hi = new double[DIMS];
        java.util.Arrays.fill(hi, 1);
        switch (method) {
            case "random" -> {
                for (int e = 0; e < budget; e++) {
                    double[] x = new double[DIMS];
                    for (int i = 0; i < DIMS; i++) {
                        x[i] = r.nextDouble();
                    }
                    f.apply(x);
                }
                result.addAll(archive);
            }
            case "weighted-sum CMA-ES" -> {
                int weights = 10;
                for (int k = 0; k < weights; k++) {
                    double w = k / (double) (weights - 1);
                    double[] start = new double[DIMS];
                    for (int i = 0; i < DIMS; i++) {
                        start[i] = r.nextDouble();
                    }
                    double[] best = CmaEs.of(start, 0.3).seed(seed * 31 + k).minimize(x -> {
                        double[] o = f.apply(x);
                        return w * o[0] + (1 - w) * o[1];
                    }, budget / weights);
                    result.add(problem.objectives(clip(best)));
                }
            }
            case "MAP-Elites" -> MapElites.<double[]>of(rr -> {
                double[] x = new double[DIMS];
                for (int i = 0; i < DIMS; i++) {
                    x[i] = rr.nextDouble();
                }
                return x;
            }, (x, rr) -> clip(MapElites.gaussian(0.1).apply(x, rr)), x -> {
                double[] o = f.apply(x);
                return new MapElites.Evaluation(-o[1], new double[] {o[0]});
            }).grid(new int[] {50}, new double[] {0}, new double[] {1}).seed(seed).run(100, budget - 100).elites()
                    .forEach(e -> result.add(new double[] {e.descriptor()[0], -e.fitness()}));
            case "NSGA-II" -> Nsga2.of(lo, hi, f).population(100).seed(seed).run(budget / 100 - 1).population()
                    .forEach(s -> result.add(s.objectives()));
            default -> throw new IllegalArgumentException(method);
        }
        return archive;
    }
}
