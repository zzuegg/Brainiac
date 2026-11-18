package io.github.zzuegg.brainiac.experimental.evolve;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * NSGA-II, the elitist non-dominated sorting genetic algorithm for multi-objective
 * minimisation over a box. Each generation merges parents and offspring, sorts the merged
 * population into Pareto fronts by fast non-dominated sorting, and fills the next population
 * front by front, breaking the tie in the last front that fits by crowding distance (the
 * perimeter of the cuboid spanned by a point's neighbours on the front, infinite at the
 * extremes), so the population spreads along the front. Parents are picked by binary
 * tournaments on (rank, crowding); offspring come from simulated binary crossover and
 * polynomial mutation, the operators of the original paper (with wider default distribution
 * indices, 2 and 5, than its 20 and 20, which on ZDT1 at a 10k-evaluation budget had barely
 * left the random region).
 *
 * <p>{@link #hypervolume2d} measures a two-objective front: the area it dominates up to a
 * reference point (larger is better, and only a front closer to and wider along the true front
 * increases it).
 *
 * <p>Deb, Pratap, Agarwal and Meyarivan 2002, "A Fast and Elitist Multiobjective Genetic
 * Algorithm: NSGA-II", IEEE Transactions on Evolutionary Computation 6(2).
 */
public final class Nsga2 {

    /**
     * A member of the population.
     *
     * @param x the decision vector
     * @param objectives its objective values (minimised)
     */
    public record Solution(double[] x, double[] objectives) {
    }

    private final double[] low;
    private final double[] high;
    private final Function<double[], double[]> objectives;
    private int size = 100;
    private double crossoverEta = 2;
    private double mutationEta = 5;
    private long seed = 1;
    private int evaluations;
    private List<Solution> population = new ArrayList<>();

    private Nsga2(double[] low, double[] high, Function<double[], double[]> objectives) {
        this.low = low.clone();
        this.high = high.clone();
        this.objectives = objectives;
    }

    /**
     * @param low the lower bounds of the decision space
     * @param high the upper bounds
     * @param objectives the objective vector of a point (all minimised)
     * @return the search, to configure
     */
    public static Nsga2 of(double[] low, double[] high, Function<double[], double[]> objectives) {
        Objects.requireNonNull(objectives, "objectives");
        if (low.length == 0 || low.length != high.length) {
            throw new IllegalArgumentException("need matching, non-empty bounds");
        }
        return new Nsga2(low, high, objectives);
    }

    /**
     * @param members the population size (even; default 100)
     * @return this
     */
    public Nsga2 population(int members) {
        if (members < 4 || members % 2 != 0) {
            throw new IllegalArgumentException("the population must be even and at least 4");
        }
        this.size = members;
        return this;
    }

    /**
     * @param crossover the SBX distribution index (default 2; larger keeps children closer to
     *     their parents)
     * @param mutation the polynomial mutation index (default 5)
     * @return this
     */
    public Nsga2 distributionIndices(double crossover, double mutation) {
        this.crossoverEta = crossover;
        this.mutationEta = mutation;
        return this;
    }

    /**
     * @param s the seed
     * @return this
     */
    public Nsga2 seed(long s) {
        this.seed = s;
        return this;
    }

    /** @return objective evaluations used */
    public int evaluations() {
        return evaluations;
    }

    /** @return the current population */
    public List<Solution> population() {
        return List.copyOf(population);
    }

    /** @return the non-dominated members of the current population */
    public List<Solution> front() {
        double[][] objs = population.stream().map(Solution::objectives).toArray(double[][]::new);
        List<Solution> out = new ArrayList<>();
        for (int i : nonDominatedSort(objs).get(0)) {
            out.add(population.get(i));
        }
        return out;
    }

    private Solution evaluate(double[] x) {
        evaluations++;
        return new Solution(x, objectives.apply(x));
    }

    /**
     * Runs generations (the first call also creates the random initial population).
     *
     * @param generations generations
     * @return this
     */
    public Nsga2 run(int generations) {
        SplittableRandom r = new SplittableRandom(seed + evaluations);
        int n = low.length;
        if (population.isEmpty()) {
            for (int i = 0; i < size; i++) {
                double[] x = new double[n];
                for (int j = 0; j < n; j++) {
                    x[j] = r.nextDouble(low[j], high[j]);
                }
                population.add(evaluate(x));
            }
        }
        for (int g = 0; g < generations; g++) {
            double[][] objs = population.stream().map(Solution::objectives).toArray(double[][]::new);
            int[] rank = new int[objs.length];
            double[] crowd = new double[objs.length];
            List<List<Integer>> fronts = nonDominatedSort(objs);
            for (int f = 0; f < fronts.size(); f++) {
                double[] cd = crowding(objs, fronts.get(f));
                for (int k = 0; k < fronts.get(f).size(); k++) {
                    rank[fronts.get(f).get(k)] = f;
                    crowd[fronts.get(f).get(k)] = cd[k];
                }
            }
            List<Solution> merged = new ArrayList<>(population);
            while (merged.size() < 2 * size) {
                double[] a = population.get(tournament(rank, crowd, r)).x();
                double[] b = population.get(tournament(rank, crowd, r)).x();
                double[][] kids = crossover(a, b, r);
                for (double[] kid : kids) {
                    mutate(kid, r);
                    merged.add(evaluate(kid));
                }
            }
            population = select(merged);
        }
        return this;
    }

    private int tournament(int[] rank, double[] crowd, SplittableRandom r) {
        int a = r.nextInt(rank.length);
        int b = r.nextInt(rank.length);
        if (rank[a] != rank[b]) {
            return rank[a] < rank[b] ? a : b;
        }
        return crowd[a] >= crowd[b] ? a : b;
    }

    private double[][] crossover(double[] a, double[] b, SplittableRandom r) {
        double[] c1 = a.clone();
        double[] c2 = b.clone();
        if (r.nextDouble() < 0.9) {
            for (int j = 0; j < a.length; j++) {
                if (r.nextDouble() < 0.5) {
                    double u = r.nextDouble();
                    double beta = u <= 0.5 ? Math.pow(2 * u, 1 / (crossoverEta + 1))
                            : Math.pow(1 / (2 * (1 - u)), 1 / (crossoverEta + 1));
                    c1[j] = clip(0.5 * ((1 + beta) * a[j] + (1 - beta) * b[j]), j);
                    c2[j] = clip(0.5 * ((1 - beta) * a[j] + (1 + beta) * b[j]), j);
                }
            }
        }
        return new double[][] {c1, c2};
    }

    private void mutate(double[] x, SplittableRandom r) {
        double p = 1.0 / x.length;
        for (int j = 0; j < x.length; j++) {
            if (r.nextDouble() < p) {
                double u = r.nextDouble();
                double delta = u < 0.5 ? Math.pow(2 * u, 1 / (mutationEta + 1)) - 1
                        : 1 - Math.pow(2 * (1 - u), 1 / (mutationEta + 1));
                x[j] = clip(x[j] + delta * (high[j] - low[j]), j);
            }
        }
    }

    private double clip(double v, int j) {
        return Math.max(low[j], Math.min(high[j], v));
    }

    private List<Solution> select(List<Solution> merged) {
        double[][] objs = merged.stream().map(Solution::objectives).toArray(double[][]::new);
        List<Solution> next = new ArrayList<>();
        for (List<Integer> front : nonDominatedSort(objs)) {
            if (next.size() + front.size() <= size) {
                for (int i : front) {
                    next.add(merged.get(i));
                }
            } else {
                double[] cd = crowding(objs, front);
                Integer[] order = new Integer[front.size()];
                for (int k = 0; k < order.length; k++) {
                    order[k] = k;
                }
                Arrays.sort(order, Comparator.comparingDouble((Integer k) -> -cd[k]));
                for (int k = 0; next.size() < size; k++) {
                    next.add(merged.get(front.get(order[k])));
                }
            }
            if (next.size() >= size) {
                break;
            }
        }
        return next;
    }

    /**
     * @param a objectives
     * @param b objectives
     * @return whether {@code a} Pareto-dominates {@code b} (no worse everywhere, better somewhere)
     */
    public static boolean dominates(double[] a, double[] b) {
        boolean better = false;
        for (int k = 0; k < a.length; k++) {
            if (a[k] > b[k]) {
                return false;
            }
            better |= a[k] < b[k];
        }
        return better;
    }

    /**
     * Fast non-dominated sorting.
     *
     * @param objs objective vectors
     * @return the fronts, best first, as indices
     */
    public static List<List<Integer>> nonDominatedSort(double[][] objs) {
        int n = objs.length;
        List<List<Integer>> dominated = new ArrayList<>();
        int[] count = new int[n];
        List<List<Integer>> fronts = new ArrayList<>();
        List<Integer> first = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            dominated.add(new ArrayList<>());
            for (int j = 0; j < n; j++) {
                if (dominates(objs[i], objs[j])) {
                    dominated.get(i).add(j);
                } else if (dominates(objs[j], objs[i])) {
                    count[i]++;
                }
            }
            if (count[i] == 0) {
                first.add(i);
            }
        }
        List<Integer> current = first;
        while (!current.isEmpty()) {
            fronts.add(current);
            List<Integer> next = new ArrayList<>();
            for (int i : current) {
                for (int j : dominated.get(i)) {
                    if (--count[j] == 0) {
                        next.add(j);
                    }
                }
            }
            current = next;
        }
        return fronts;
    }

    /**
     * Crowding distances of one front.
     *
     * @param objs all objective vectors
     * @param front the front's indices
     * @return the distance of each front member, in the front's order
     */
    public static double[] crowding(double[][] objs, List<Integer> front) {
        int k = front.size();
        double[] d = new double[k];
        if (k <= 2) {
            Arrays.fill(d, Double.POSITIVE_INFINITY);
            return d;
        }
        int m = objs[front.get(0)].length;
        for (int o = 0; o < m; o++) {
            final int obj = o;
            Integer[] order = new Integer[k];
            for (int i = 0; i < k; i++) {
                order[i] = i;
            }
            Arrays.sort(order, Comparator.comparingDouble((Integer i) -> objs[front.get(i)][obj]));
            double lo = objs[front.get(order[0])][o];
            double hi = objs[front.get(order[k - 1])][o];
            d[order[0]] = Double.POSITIVE_INFINITY;
            d[order[k - 1]] = Double.POSITIVE_INFINITY;
            if (hi > lo) {
                for (int i = 1; i + 1 < k; i++) {
                    d[order[i]] += (objs[front.get(order[i + 1])][o] - objs[front.get(order[i - 1])][o]) / (hi - lo);
                }
            }
        }
        return d;
    }

    /**
     * The exact two-objective hypervolume: the area dominated by the points and bounded by the
     * reference point (points not dominating the reference are ignored).
     *
     * @param points objective vectors {@code (f1, f2)}, minimised
     * @param reference the reference point
     * @return the area
     */
    public static double hypervolume2d(List<double[]> points, double[] reference) {
        List<double[]> p = new ArrayList<>();
        for (double[] q : points) {
            if (q[0] < reference[0] && q[1] < reference[1]) {
                p.add(q);
            }
        }
        p.sort(Comparator.comparingDouble((double[] q) -> q[0]));
        double area = 0;
        double bestF2 = reference[1];
        // sweep by increasing f1: each point that lowers the best f2 so far adds the horizontal
        // strip between the old and new best f2, from its f1 to the reference
        for (double[] q : p) {
            if (q[1] < bestF2) {
                area += (reference[0] - q[0]) * (bestF2 - q[1]);
                bestF2 = q[1];
            }
        }
        return area;
    }
}
