package io.github.zzuegg.brainiac.experimental.evolve;

import io.github.zzuegg.brainiac.data.BytePairTokenizer;
import io.github.zzuegg.brainiac.data.Tokenizer;
import io.github.zzuegg.brainiac.experimental.data.TokenizerGenome;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.ToDoubleFunction;

/**
 * Evolves byte-pair tokenizers: the vocabulary size and the merges themselves are genes
 * ({@link TokenizerGenome}), and a tokenizer's cost is how hard text is to model through it
 * (for example {@code TokenizerScores.bigramBitsPerCharacter} of a small model, or a
 * compression proxy). Greedy BPE picks merges by raw pair frequency and a size by hand;
 * evolution searches both against the score that matters.
 *
 * <p>The first generation consists of truncated and perturbed copies of a greedily trained
 * tokenizer; then elitism, tournament selection, crossover and mutation. Costs are
 * minimised. Background: "Tokenization Is More Than Compression" (Schmidt et al. 2024,
 * <a href="https://arxiv.org/abs/2402.18376">arXiv:2402.18376</a>).
 */
public final class TokenizerSearch {

    /**
     * A scored tokenizer.
     *
     * @param genome its genes
     * @param cost its cost (lower is better)
     * @param vocabulary its vocabulary size
     */
    public record Scored(TokenizerGenome genome, double cost, int vocabulary) {
    }

    private final String trainText;
    private final ToDoubleFunction<Tokenizer> cost;
    private int population = 12;
    private int elites = 2;
    private int maxMerges = 200;
    private long seed = 1;
    private final Map<TokenizerGenome, Double> cache = new HashMap<>();
    private final List<Scored> history = new ArrayList<>();

    private TokenizerSearch(String trainText, ToDoubleFunction<Tokenizer> cost) {
        this.trainText = Objects.requireNonNull(trainText);
        this.cost = Objects.requireNonNull(cost);
    }

    /**
     * Sets up a search.
     *
     * @param trainText text the seed tokenizer is trained on and guided mutations count pairs in
     * @param cost the cost of a tokenizer, lower is better
     * @return the search, to configure
     */
    public static TokenizerSearch of(String trainText, ToDoubleFunction<Tokenizer> cost) {
        return new TokenizerSearch(trainText, cost);
    }

    /**
     * Sets the population size, elites and the seed tokenizer's merge count.
     *
     * @param size the population
     * @param keep elites carried over
     * @param merges merges of the greedy seed tokenizer
     * @return this
     */
    public TokenizerSearch population(int size, int keep, int merges) {
        this.population = size;
        this.elites = keep;
        this.maxMerges = merges;
        return this;
    }

    /**
     * Sets the seed.
     *
     * @param s the seed
     * @return this
     */
    public TokenizerSearch seed(long s) {
        this.seed = s;
        return this;
    }

    /** @return the best tokenizer of every generation */
    public List<Scored> history() {
        return List.copyOf(history);
    }

    /**
     * The cost of a genome (cached).
     *
     * @param genome the genome
     * @return its cost
     */
    public double cost(TokenizerGenome genome) {
        return cache.computeIfAbsent(genome, g -> cost.applyAsDouble(g.toTokenizer()));
    }

    /**
     * Runs the search.
     *
     * @param generations the number of generations
     * @return the best tokenizer found
     */
    public Scored run(int generations) {
        SplittableRandom random = new SplittableRandom(seed);
        TokenizerGenome full = TokenizerGenome.of(BytePairTokenizer.train(trainText, 256 + maxMerges));
        String sample = trainText.length() > 4000 ? trainText.substring(0, 4000) : trainText;
        List<TokenizerGenome> pop = new ArrayList<>();
        for (int i = 0; i < population; i++) {
            TokenizerGenome g = new TokenizerGenome(full.merges(), random.nextInt(full.merges().size() + 1));
            int k = random.nextInt(3);
            for (int j = 0; j < k; j++) {
                g = g.mutate(sample, random);
            }
            pop.add(g);
        }
        Scored best = null;
        for (int gen = 0; gen < generations; gen++) {
            List<Scored> scored = new ArrayList<>();
            for (TokenizerGenome g : pop) {
                scored.add(new Scored(g, cost(g), g.vocabularySize()));
            }
            scored.sort(Comparator.comparingDouble(Scored::cost));
            if (best == null || scored.getFirst().cost() < best.cost()) {
                best = scored.getFirst();
            }
            history.add(best);
            List<TokenizerGenome> next = new ArrayList<>();
            for (int i = 0; i < elites; i++) {
                next.add(scored.get(i).genome());
            }
            while (next.size() < population) {
                TokenizerGenome a = tournament(scored, random);
                TokenizerGenome child = random.nextDouble() < 0.3 ? a.cross(tournament(scored, random), random) : a;
                int k = 1 + random.nextInt(2);
                for (int j = 0; j < k; j++) {
                    child = child.mutate(sample, random);
                }
                next.add(child);
            }
            pop = next;
        }
        return best;
    }

    private static TokenizerGenome tournament(List<Scored> pop, SplittableRandom random) {
        Scored best = null;
        for (int i = 0; i < 3; i++) {
            Scored s = pop.get(random.nextInt(pop.size()));
            if (best == null || s.cost() < best.cost()) {
                best = s;
            }
        }
        return best.genome();
    }
}
