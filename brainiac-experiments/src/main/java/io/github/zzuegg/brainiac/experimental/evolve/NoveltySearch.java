package io.github.zzuegg.brainiac.experimental.evolve;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.function.Predicate;

/**
 * Novelty search: rewards behaving <em>differently</em> instead of scoring well. Each
 * candidate is described by a behaviour vector (say, where a robot ended up); its novelty is
 * the mean distance to its {@code k} nearest neighbours among the population and an archive
 * of past novel behaviours ({@link BehaviourArchive}). On deceptive problems, where
 * following the objective leads into a dead end, searching for novelty alone finds the goal
 * far more often.
 *
 * <p>Uses a (mu + lambda) loop: each generation every parent gets a mutated child, and the
 * best half of parents and children by score survives. The score blends the ranks of
 * novelty and fitness: {@link #noveltyWeight(double) weight} 1 is pure novelty search, 0 a
 * plain objective-driven evolutionary algorithm (useful as the baseline), and in between
 * the two are traded off.
 *
 * <p>Lehman and Stanley 2011, "Abandoning Objectives: Evolution Through the Search for
 * Novelty Alone", <a href="https://doi.org/10.1162/EVCO_a_00025">Evolutionary Computation
 * 19(2)</a>.
 *
 * @param <G> the genome type
 */
public final class NoveltySearch<G> {

    /**
     * What evaluating a genome gives.
     *
     * @param fitness the objective (higher is better), only used for tracking and blending
     * @param behaviour the behaviour descriptor
     */
    public record Evaluation(double fitness, double[] behaviour) {
    }

    /**
     * A scored genome.
     *
     * @param genome the genome
     * @param fitness its objective
     * @param behaviour its behaviour descriptor
     * @param novelty its novelty when last scored
     * @param <G> the genome type
     */
    public record Individual<G>(G genome, double fitness, double[] behaviour, double novelty) {
    }

    /**
     * One generation's summary.
     *
     * @param index the generation
     * @param bestFitness the best fitness seen so far
     * @param meanNovelty the mean novelty of the survivors
     * @param archiveSize the archive's size
     */
    public record Generation(int index, double bestFitness, double meanNovelty, int archiveSize) {
    }

    private final Function<SplittableRandom, G> create;
    private final BiFunction<G, SplittableRandom, G> mutate;
    private final Function<G, Evaluation> evaluate;
    private int populationSize = 50;
    private int k = 15;
    private double threshold = -1;
    private double noveltyWeight = 1;
    private double addChance = 0.02;
    private long seed = 1;

    private SplittableRandom random;
    private List<Individual<G>> population;
    private BehaviourArchive archive;
    private Individual<G> best;
    private int generation;
    private final List<Generation> history = new ArrayList<>();

    private NoveltySearch(Function<SplittableRandom, G> create, BiFunction<G, SplittableRandom, G> mutate,
            Function<G, Evaluation> evaluate) {
        this.create = Objects.requireNonNull(create);
        this.mutate = Objects.requireNonNull(mutate);
        this.evaluate = Objects.requireNonNull(evaluate);
    }

    /**
     * Sets up a search.
     *
     * @param create makes a random genome
     * @param mutate makes a mutated copy of a genome
     * @param evaluate scores a genome and describes its behaviour
     * @param <G> the genome type
     * @return the search, to configure
     */
    public static <G> NoveltySearch<G> of(Function<SplittableRandom, G> create,
            BiFunction<G, SplittableRandom, G> mutate, Function<G, Evaluation> evaluate) {
        return new NoveltySearch<>(create, mutate, evaluate);
    }

    /**
     * Sets the population size.
     *
     * @param size the number of survivors per generation
     * @return this
     */
    public NoveltySearch<G> population(int size) {
        this.populationSize = size;
        return this;
    }

    /**
     * Sets the number of neighbours novelty is measured against.
     *
     * @param neighbours k
     * @return this
     */
    public NoveltySearch<G> neighbours(int neighbours) {
        this.k = neighbours;
        return this;
    }

    /**
     * Sets the initial novelty an individual needs to enter the archive; it then adapts so
     * the archive grows steadily. Negative (the default) picks it from the first generation.
     *
     * @param t the threshold
     * @return this
     */
    public NoveltySearch<G> archiveThreshold(double t) {
        this.threshold = t;
        return this;
    }

    /**
     * Blends novelty and fitness: 1 is pure novelty search, 0 pure objective search.
     *
     * @param w the weight of novelty
     * @return this
     */
    public NoveltySearch<G> noveltyWeight(double w) {
        if (w < 0 || w > 1) {
            throw new IllegalArgumentException("weight must be in [0, 1]");
        }
        this.noveltyWeight = w;
        return this;
    }

    /**
     * Sets the seed.
     *
     * @param s the seed
     * @return this
     */
    public NoveltySearch<G> seed(long s) {
        this.seed = s;
        return this;
    }

    /** @return the individual with the best fitness seen */
    public Individual<G> best() {
        return best;
    }

    /** @return the archive of novel behaviours */
    public BehaviourArchive archive() {
        return archive;
    }

    /** @return the current survivors */
    public List<Individual<G>> population() {
        return List.copyOf(population);
    }

    /** @return the generations so far */
    public List<Generation> history() {
        return List.copyOf(history);
    }

    /**
     * Runs until the generations run out or an individual satisfies {@code done}.
     *
     * @param generations the maximum number of generations
     * @param done a stop condition, such as reaching a goal
     * @return the best individual by fitness
     */
    public Individual<G> run(int generations, Predicate<Individual<G>> done) {
        for (int g = 0; g < generations; g++) {
            step();
            if (done.test(best)) {
                break;
            }
        }
        return best;
    }

    /**
     * Runs one generation.
     *
     * @return its summary
     */
    public Generation step() {
        List<Individual<G>> pool = new ArrayList<>();
        if (population == null) {
            random = new SplittableRandom(seed);
            archive = null;
            for (int i = 0; i < 2 * populationSize; i++) {
                pool.add(score(create.apply(random)));
            }
        } else {
            pool.addAll(population);
            for (Individual<G> parent : population) {
                pool.add(score(mutate.apply(parent.genome(), random)));
            }
        }
        if (archive == null) {
            archive = new BehaviourArchive(pool.getFirst().behaviour().length);
        }
        List<double[]> behaviours = pool.stream().map(Individual::behaviour).toList();
        List<Individual<G>> scored = new ArrayList<>();
        for (Individual<G> ind : pool) {
            scored.add(new Individual<>(ind.genome(), ind.fitness(), ind.behaviour(),
                    archive.novelty(ind.behaviour(), k, behaviours)));
        }
        if (threshold < 0) {
            threshold = scored.stream().mapToDouble(Individual::novelty).average().orElse(1);
        }
        int added = 0;
        int start = population == null ? 0 : population.size();   // only children may enter
        for (int i = start; i < scored.size(); i++) {
            Individual<G> ind = scored.get(i);
            if (ind.novelty() > threshold || random.nextDouble() < addChance) {
                archive.add(ind.behaviour());
                added++;
            }
        }
        threshold *= added > 4 ? 1.2 : added == 0 ? 0.95 : 1;
        double[] score = blend(scored);
        Integer[] order = new Integer[scored.size()];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        java.util.Arrays.sort(order, Comparator.comparingDouble(i -> -score[i]));
        population = new ArrayList<>();
        for (int i = 0; i < populationSize; i++) {
            population.add(scored.get(order[i]));
        }
        double mean = population.stream().mapToDouble(Individual::novelty).average().orElse(0);
        Generation g = new Generation(generation++, best.fitness(), mean, archive.size());
        history.add(g);
        return g;
    }

    private Individual<G> score(G genome) {
        Evaluation e = evaluate.apply(genome);
        Individual<G> ind = new Individual<>(genome, e.fitness(), e.behaviour().clone(), 0);
        if (best == null || ind.fitness() > best.fitness()) {
            best = ind;
        }
        return ind;
    }

    private double[] blend(List<Individual<G>> scored) {
        double[] nov = EvolutionStrategy.centredRanks(scored.stream().mapToDouble(Individual::novelty).toArray());
        double[] fit = EvolutionStrategy.centredRanks(scored.stream().mapToDouble(Individual::fitness).toArray());
        double[] out = new double[nov.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = noveltyWeight * nov[i] + (1 - noveltyWeight) * fit[i]
                    + 1e-9 * (noveltyWeight > 0 ? scored.get(i).novelty() : scored.get(i).fitness());
        }
        return out;
    }
}
