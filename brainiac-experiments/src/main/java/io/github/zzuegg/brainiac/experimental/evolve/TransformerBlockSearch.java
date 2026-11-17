package io.github.zzuegg.brainiac.experimental.evolve;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.data.Batch;
import io.github.zzuegg.brainiac.experimental.nn.GrowableTransformer;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Function;
import java.util.function.Supplier;

/**
 * Evolves transformer designs ({@link TransformerBlockGenome}): heads and key/value heads,
 * gating, normalization placement, depth and per-neuron FFN activations. Each generation
 * the whole population is trained <em>as one</em>: every training step feeds the same batch
 * to all candidates, adds their losses and takes one optimizer step over all their
 * parameters. Candidates share no parameters, so the gradient of the sum is each
 * candidate's own gradient and an element-wise optimizer (Adam, SGD) trains each exactly as
 * it would alone, but with one loop, one backward pass and one optimizer call for the
 * whole population (on a GPU, one batch of kernels). After the training budget candidates
 * are scored by their loss on fixed validation batches, plus an optional size penalty.
 *
 * <p>Selection keeps the elites and fills up with tournament winners' children (uniform
 * crossover, then a mutation). Designs seen before are not retrained.
 *
 * <p>So et al. 2019, "The Evolved Transformer", <a href="https://arxiv.org/abs/1901.11117">arXiv:1901.11117</a>;
 * gated attention: Qiu et al. 2025, <a href="https://arxiv.org/abs/2505.06708">arXiv:2505.06708</a>.
 */
public final class TransformerBlockSearch {

    /**
     * A scored design.
     *
     * @param genome the design
     * @param validationLoss its loss after training
     * @param parameters its parameter count
     * @param fitness the score (higher is better)
     */
    public record Candidate(TransformerBlockGenome genome, double validationLoss, long parameters, double fitness) {
    }

    /**
     * One generation.
     *
     * @param index the generation
     * @param best the best candidate so far
     * @param medianLoss the median validation loss of the generation
     * @param trained candidates trained this generation
     */
    public record Generation(int index, Candidate best, double medianLoss, int trained) {
    }

    private final int vocabulary;
    private final int length;
    private final int model;
    private final Function<SplittableRandom, Batch> data;
    private int population = 8;
    private int elites = 2;
    private int steps = 60;
    private int validationBatches = 4;
    private double sizePenalty;
    private Supplier<Optimizer> optimizer = () -> Optimizer.adam(5e-3f);
    private long seed = 1;
    private final Map<TransformerBlockGenome, Candidate> seen = new HashMap<>();
    private final List<Generation> history = new ArrayList<>();

    private TransformerBlockSearch(int vocabulary, int length, int model, Function<SplittableRandom, Batch> data) {
        this.vocabulary = vocabulary;
        this.length = length;
        this.model = model;
        this.data = Objects.requireNonNull(data);
    }

    /**
     * Sets up a search on a next-token task.
     *
     * @param vocabulary the number of token ids
     * @param length the sequence length
     * @param model the residual width of every design
     * @param data draws a batch of ids {@code [batch, time]} and next-token targets
     * @return the search, to configure
     */
    public static TransformerBlockSearch of(int vocabulary, int length, int model,
            Function<SplittableRandom, Batch> data) {
        return new TransformerBlockSearch(vocabulary, length, model, data);
    }

    /**
     * Sets the population and elite counts.
     *
     * @param size candidates per generation
     * @param keep elites carried over unchanged
     * @return this
     */
    public TransformerBlockSearch population(int size, int keep) {
        this.population = size;
        this.elites = keep;
        return this;
    }

    /**
     * Sets the training budget per candidate.
     *
     * @param trainingSteps optimizer steps
     * @return this
     */
    public TransformerBlockSearch steps(int trainingSteps) {
        this.steps = trainingSteps;
        return this;
    }

    /**
     * Subtracts {@code penalty * parameters} from the fitness, to favour small designs.
     *
     * @param penalty the cost of one parameter
     * @return this
     */
    public TransformerBlockSearch sizePenalty(double penalty) {
        this.sizePenalty = penalty;
        return this;
    }

    /**
     * Sets the optimizer; it must be element-wise (Adam, SGD) so candidates stay independent.
     *
     * @param factory makes an optimizer
     * @return this
     */
    public TransformerBlockSearch optimizer(Supplier<Optimizer> factory) {
        this.optimizer = factory;
        return this;
    }

    /**
     * Sets the seed.
     *
     * @param s the seed
     * @return this
     */
    public TransformerBlockSearch seed(long s) {
        this.seed = s;
        return this;
    }

    /** @return the generations so far */
    public List<Generation> history() {
        return List.copyOf(history);
    }

    /**
     * Trains several designs together as one population and scores them.
     *
     * @param genomes the designs
     * @param trainSeed the seed for weights and batches
     * @return one candidate per design, in order
     */
    public List<Candidate> trainTogether(List<TransformerBlockGenome> genomes, long trainSeed) {
        List<GrowableTransformer> models = new ArrayList<>();
        List<Var> all = new ArrayList<>();
        for (int i = 0; i < genomes.size(); i++) {
            TransformerBlockGenome g = genomes.get(i);
            GrowableTransformer m = GrowableTransformer.of(vocabulary, length, g.config(model), g.depth(),
                    new SplittableRandom(trainSeed * 1_000_003L + g.hashCode()));
            models.add(m);
            all.addAll(m.parameters());
        }
        Optimizer opt = optimizer.get();
        SplittableRandom batches = new SplittableRandom(trainSeed);
        for (int s = 0; s < steps; s++) {
            Batch b = data.apply(batches);
            Var input = Var.of(b.features());
            Var total = null;
            for (GrowableTransformer m : models) {
                Var l = GrowableTransformer.loss(m.forward(input), b.targets());
                total = total == null ? l : total.add(l);
            }
            total.backward();
            opt.step(all);
        }
        SplittableRandom valid = new SplittableRandom(seed ^ 0x5DEECE66DL);
        List<Batch> vb = new ArrayList<>();
        for (int i = 0; i < validationBatches; i++) {
            vb.add(data.apply(valid));
        }
        List<Candidate> out = new ArrayList<>();
        for (int i = 0; i < genomes.size(); i++) {
            double loss = 0;
            for (Batch b : vb) {
                loss += GrowableTransformer.loss(models.get(i).forward(Var.of(b.features())), b.targets()).value()
                        .item();
            }
            loss /= vb.size();
            if (!Double.isFinite(loss)) {
                loss = 1e9;
            }
            long params = models.get(i).parameterCount();
            out.add(new Candidate(genomes.get(i), loss, params, -loss - sizePenalty * params));
        }
        return out;
    }

    /**
     * Runs the search.
     *
     * @param generations the number of generations
     * @param start designs to seed the first generation with (the rest are random)
     * @return the final population, best first
     */
    public List<Candidate> run(int generations, List<TransformerBlockGenome> start) {
        SplittableRandom random = new SplittableRandom(seed);
        List<TransformerBlockGenome> next = new ArrayList<>(start);
        while (next.size() < population) {
            next.add(TransformerBlockGenome.random(random));
        }
        List<Candidate> pop = List.of();
        for (int gen = 0; gen < generations; gen++) {
            List<TransformerBlockGenome> fresh = next.stream().filter(g -> !seen.containsKey(g)).distinct().toList();
            if (!fresh.isEmpty()) {
                for (Candidate c : trainTogether(fresh, seed)) {
                    seen.put(c.genome(), c);
                }
            }
            pop = next.stream().map(seen::get).distinct()
                    .sorted(Comparator.comparingDouble(c -> -c.fitness())).toList();
            double median = pop.get(pop.size() / 2).validationLoss();
            Candidate best = seen.values().stream().max(Comparator.comparingDouble(Candidate::fitness)).orElseThrow();
            history.add(new Generation(gen, best, median, fresh.size()));
            if (gen == generations - 1) {
                break;
            }
            next = new ArrayList<>();
            for (int i = 0; i < Math.min(elites, pop.size()); i++) {
                next.add(pop.get(i).genome());
            }
            while (next.size() < population) {
                TransformerBlockGenome a = tournament(pop, random);
                TransformerBlockGenome b = tournament(pop, random);
                TransformerBlockGenome child = random.nextDouble() < 0.5 ? a.cross(b, random) : a;
                next.add(child.mutate(random));
            }
        }
        return pop;
    }

    private static TransformerBlockGenome tournament(List<Candidate> pop, SplittableRandom random) {
        Candidate best = null;
        for (int i = 0; i < 3; i++) {
            Candidate c = pop.get(random.nextInt(pop.size()));
            if (best == null || c.fitness() > best.fitness()) {
                best = c;
            }
        }
        return best.genome();
    }
}
