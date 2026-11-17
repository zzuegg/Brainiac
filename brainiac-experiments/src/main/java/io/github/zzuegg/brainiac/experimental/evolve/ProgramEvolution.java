package io.github.zzuegg.brainiac.experimental.evolve;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Deque;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * AlphaEvolve-style program evolution (Novikov et al., "AlphaEvolve: A coding agent for
 * scientific and algorithmic discovery", 2025, building on FunSearch, Romera-Paredes et al.,
 * <a href="https://doi.org/10.1038/s41586-023-06924-6">Nature 2023</a>): a language model
 * proposes edits to programs and an automatic evaluator scores them, inside a
 * quality-diversity archive. Each step samples a parent from the archive together with a few
 * high-scoring <em>inspirations</em>, asks the {@link Proposer} (the model) for a child given
 * both, evaluates it, and lets it into the {@link MapElites} grid if it beats the elite of its
 * behaviour cell. The archive keeps different kinds of solutions (short and long, fast and
 * accurate) alive, so the proposer always has varied material to recombine.
 *
 * <p>{@link #regularizedEvolution} is the baseline search of AutoML-Zero (Real et al.,
 * <a href="https://arxiv.org/abs/2003.03384">2003.03384</a>): a population under tournament
 * selection with aging, where children see only their parent.
 *
 * @param <P> the program type
 */
public final class ProgramEvolution<P> {

    /**
     * A program with its evaluation.
     *
     * @param program the program
     * @param fitness its score, higher is better
     * @param <P> the program type
     */
    public record Scored<P>(P program, double fitness) {
    }

    /**
     * Proposes a child program, as the language model does from a prompt holding the parent
     * and the inspirations with their scores.
     *
     * @param <P> the program type
     */
    @FunctionalInterface
    public interface Proposer<P> {
        /**
         * @param parent the program to improve
         * @param inspirations other good programs from the archive, best first
         * @param random the random source
         * @return the proposed child
         */
        P propose(Scored<P> parent, List<Scored<P>> inspirations, SplittableRandom random);
    }

    private final MapElites<P> map;
    private int inspirations = 3;
    private SplittableRandom random = new SplittableRandom(0);
    private final List<Double> history = new ArrayList<>();
    private Proposer<P> proposer;

    private ProgramEvolution(P seed, Proposer<P> proposer, Function<P, MapElites.Evaluation> evaluator, int[] bins,
            double[] low, double[] high) {
        Objects.requireNonNull(evaluator);
        this.proposer = Objects.requireNonNull(proposer);
        this.map = MapElites.of(r -> seed, this::mutate, evaluator).grid(bins, low, high);
    }

    /**
     * @param seed the initial program
     * @param proposer the model that writes children
     * @param evaluator scores a program and places it in the behaviour grid
     * @param bins cells per behaviour dimension
     * @param low the lowest descriptor value per dimension
     * @param high the highest descriptor value per dimension
     * @param <P> the program type
     * @return the search
     */
    public static <P> ProgramEvolution<P> of(P seed, Proposer<P> proposer, Function<P, MapElites.Evaluation> evaluator,
            int[] bins, double[] low, double[] high) {
        return new ProgramEvolution<>(seed, proposer, evaluator, bins, low, high);
    }

    /**
     * @param k inspirations per prompt (3 by default)
     * @return this search
     */
    public ProgramEvolution<P> inspirations(int k) {
        this.inspirations = k;
        return this;
    }

    /**
     * @param s the seed
     * @return this search
     */
    public ProgramEvolution<P> seed(long s) {
        this.random = new SplittableRandom(s);
        map.seed(s);
        return this;
    }

    private P mutate(P parent, SplittableRandom r) {
        List<MapElites.Elite<P>> elites = new ArrayList<>(map.elites());
        elites.sort(Comparator.comparingDouble(MapElites.Elite<P>::fitness).reversed());
        List<Scored<P>> pool = new ArrayList<>();
        int top = Math.max(1, (elites.size() + 1) / 2);   // inspirations come from the better half
        for (int i = 0; i < inspirations && !elites.isEmpty(); i++) {
            MapElites.Elite<P> e = elites.get(random.nextInt(top));
            pool.add(new Scored<>(e.genome(), e.fitness()));
        }
        pool.sort(Comparator.comparingDouble(Scored<P>::fitness).reversed());
        double parentFitness = elites.stream().filter(e -> e.genome() == parent).mapToDouble(MapElites.Elite::fitness)
                .findFirst().orElse(Double.NaN);
        return proposer.propose(new Scored<>(parent, parentFitness), pool, r);
    }

    /**
     * Runs evaluations.
     *
     * @param evaluations how many children to evaluate
     * @return the best fitness after each evaluation (the seed's evaluation included first)
     */
    public double[] run(int evaluations) {
        if (map.evaluations() == 0) {
            map.run(1, 0);
            history.add(map.best() == null ? Double.NEGATIVE_INFINITY : map.best().fitness());
        }
        for (int i = 0; i < evaluations; i++) {
            map.run(0, 1);
            history.add(map.best().fitness());
        }
        return history.stream().mapToDouble(Double::doubleValue).toArray();
    }

    /** @return the archive */
    public MapElites<P> archive() {
        return map;
    }

    /** @return the best program and its fitness */
    public Scored<P> best() {
        MapElites.Elite<P> b = map.best();
        return new Scored<>(b.genome(), b.fitness());
    }

    /**
     * @param p replaces the proposer, for example to switch models mid-run
     * @return this search
     */
    public ProgramEvolution<P> proposer(Proposer<P> p) {
        this.proposer = Objects.requireNonNull(p);
        return this;
    }

    /**
     * AutoML-Zero's regularized evolution: tournament selection from a population, the
     * child replaces the oldest member.
     *
     * @param seed the initial program (the population starts as copies of it)
     * @param mutate makes a child from one parent
     * @param fitness scores a program
     * @param population the population size
     * @param tournament the tournament size
     * @param evaluations how many children to evaluate
     * @param random the random source
     * @param <P> the program type
     * @return the best fitness after each evaluation (the seed's evaluation first)
     */
    public static <P> double[] regularizedEvolution(P seed, BiFunction<P, SplittableRandom, P> mutate,
            Function<P, Double> fitness, int population, int tournament, int evaluations, SplittableRandom random) {
        Deque<Scored<P>> pop = new ArrayDeque<>();
        Scored<P> first = new Scored<>(seed, fitness.apply(seed));
        double best = first.fitness();
        double[] history = new double[evaluations + 1];
        history[0] = best;
        for (int i = 0; i < population; i++) {
            pop.addLast(first);
        }
        List<Scored<P>> view = new ArrayList<>();
        for (int e = 1; e <= evaluations; e++) {
            view.clear();
            view.addAll(pop);
            Scored<P> parent = null;
            for (int t = 0; t < tournament; t++) {
                Scored<P> c = view.get(random.nextInt(view.size()));
                if (parent == null || c.fitness() > parent.fitness()) {
                    parent = c;
                }
            }
            P child = mutate.apply(Objects.requireNonNull(parent).program(), random);
            Scored<P> s = new Scored<>(child, fitness.apply(child));
            pop.addLast(s);
            pop.removeFirst();
            best = Math.max(best, s.fitness());
            history[e] = best;
        }
        return history;
    }
}
