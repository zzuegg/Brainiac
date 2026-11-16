package io.github.zzuegg.brainiac.evolve;

import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.train.Benchmark;
import io.github.zzuegg.brainiac.train.Benchmark.Task;
import io.github.zzuegg.brainiac.train.Evaluation;
import io.github.zzuegg.brainiac.train.Optimizer;
import io.github.zzuegg.brainiac.train.Trainer;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.function.Consumer;
import java.util.function.Supplier;

/**
 * Searches for a better network design with a genetic algorithm.
 *
 * <p>The search starts from a design known to work (the template), optionally with fewer
 * neurons. Each generation, every new candidate is built, trained for a short budget, and
 * scored on a validation split of the training data; the best few are trained again with a
 * second seed so luck counts less. The next generation keeps the best unchanged (elitism)
 * and fills up with children of tournament winners, made by {@link Crossover} and the
 * enabled {@link Mutation}s. Which genes may change is exactly the set of mutations you
 * enable.
 *
 * <pre>{@code
 * Evolution.Result r = Evolution.of(Tasks.spiral())
 *         .template(Genome.dense(2, Activation.RELU, 64, 64))
 *         .neuronScale(0.75)                       // start from 3/4 of the hidden neurons
 *         .mutations("connections,neurons,activations")
 *         .population(20).generations(10)
 *         .epochs(40)                              // training budget per candidate
 *         .run();
 * System.out.println(r.best().genome().describe());
 * }</pre>
 *
 * <p>Runs are reproducible: the same settings and seed give the same evolution, also when
 * candidates are trained on several {@linkplain #threads threads}.
 */
public final class Evolution {

    /**
     * One scored design.
     *
     * @param genome the design
     * @param fitness its fitness (averaged over its evaluations)
     * @param score how it did in its first evaluation
     * @param evaluations how many seeds it was trained with
     */
    public record Individual(Genome genome, double fitness, Fitness.Score score, int evaluations) {
    }

    /**
     * What happened in one generation.
     *
     * @param index the generation, from 0
     * @param best its best individual
     * @param meanFitness the mean fitness of the population
     * @param trained how many candidates were trained (designs seen before are not)
     * @param seconds the generation's wall time
     */
    public record Generation(int index, Individual best, double meanFitness, int trained, double seconds) {
        @Override
        public String toString() {
            Fitness.Score s = best.score();
            return String.format("generation %3d  best %.4f  mean %.4f  metric %.4f  %,d params  %d neurons"
                            + "  %,d connections  trained %d in %.1f s",
                    index, best.fitness(), meanFitness, s.metric(), s.parameters(), best.genome().neurons(),
                    s.connections(), trained, seconds);
        }
    }

    /**
     * The outcome of a run.
     *
     * @param history one entry per generation
     * @param population the last generation, best first
     * @param template the starting design
     */
    public record Result(List<Generation> history, List<Individual> population, Genome template) {
        /** @return the best individual found */
        public Individual best() {
            return population.getFirst();
        }

        /** @return a table of the generations */
        public String table() {
            StringBuilder sb = new StringBuilder();
            history.forEach(g -> sb.append(g).append('\n'));
            return sb.toString();
        }
    }

    private final Task task;
    private Genome template;
    private double neuronScale = 1;
    private Consumer<Model.Builder> frontEnd;
    private Activation head;
    private int population = 20;
    private int generations = 10;
    private int elites = 2;
    private int tournament = 3;
    private double crossoverRate = 0.5;
    private Crossover crossover = Crossover.UNIFORM;
    private final List<Mutation> mutations = new ArrayList<>();
    private final List<String> mutationNames = new ArrayList<>();
    private Fitness fitness = Fitness.metric();
    private int epochs = -1;
    private double validation = 0.2;
    private int rescore = 3;
    private long seed = 1;
    private int threads = 1;
    private int maxNeurons = -1;
    private Device device;
    private Consumer<Trainer> trainerSettings = t -> { };
    private Consumer<Generation> listener = g -> { };

    private Evolution(Task task) {
        this.task = Objects.requireNonNull(task, "task");
    }

    /**
     * Starts setting up a search for a task.
     *
     * @param task the task: its training data is split into training and validation data
     *     for the search, its test data is left alone
     * @return the search, to configure
     */
    public static Evolution of(Task task) {
        return new Evolution(task);
    }

    /**
     * Sets the design to start from, such as {@link Genome#dense} for a known working
     * multilayer perceptron. Required.
     *
     * @param genome the design
     * @return this
     */
    public Evolution template(Genome genome) {
        this.template = Objects.requireNonNull(genome, "genome");
        return this;
    }

    /**
     * Starts from a fraction of the template's neurons per region, chosen at random, for
     * example 0.75 to look for a smaller design. Default 1.
     *
     * @param scale the fraction, in (0, 1]
     * @return this
     */
    public Evolution neuronScale(double scale) {
        if (!(scale > 0 && scale <= 1)) {
            throw new IllegalArgumentException("scale must be in (0, 1], got " + scale);
        }
        this.neuronScale = scale;
        return this;
    }

    /**
     * Fixed layers between the input and the evolved brain, such as a few convolutions for
     * images. Their output is flattened, and the template's input count must match it.
     *
     * @param layers adds the layers to a model builder
     * @return this
     */
    public Evolution frontEnd(Consumer<Model.Builder> layers) {
        this.frontEnd = layers;
        return this;
    }

    /**
     * The output layer's activation. By default softmax for classification with several
     * classes, sigmoid for one output, and identity for regression.
     *
     * @param activation the activation
     * @return this
     */
    public Evolution head(Activation activation) {
        this.head = activation;
        return this;
    }

    /**
     * @param size candidates per generation; default 20
     * @return this
     */
    public Evolution population(int size) {
        if (size < 2) {
            throw new IllegalArgumentException("population must be at least 2, got " + size);
        }
        this.population = size;
        return this;
    }

    /**
     * @param count generations, including the first; default 10
     * @return this
     */
    public Evolution generations(int count) {
        if (count < 1) {
            throw new IllegalArgumentException("generations must be at least 1, got " + count);
        }
        this.generations = count;
        return this;
    }

    /**
     * @param count the best candidates copied unchanged into the next generation; default 2
     * @return this
     */
    public Evolution elites(int count) {
        this.elites = Math.max(0, count);
        return this;
    }

    /**
     * @param size candidates per tournament when picking a parent; default 3
     * @return this
     */
    public Evolution tournament(int size) {
        this.tournament = Math.max(1, size);
        return this;
    }

    /**
     * How children are made from two parents.
     *
     * @param crossover the crossover; {@link Crossover#UNIFORM} by default
     * @param rate the fraction of children made by crossover; the rest are mutated copies
     *     of one parent. Default 0.5
     * @return this
     */
    public Evolution crossover(Crossover crossover, double rate) {
        this.crossover = Objects.requireNonNull(crossover, "crossover");
        this.crossoverRate = rate;
        return this;
    }

    /**
     * Enables mutations: these, and only these, genes change.
     *
     * @param list the mutations
     * @return this
     */
    public Evolution mutations(Mutation... list) {
        mutations.addAll(List.of(list));
        return this;
    }

    /**
     * Enables built-in mutations by name with their default settings; see
     * {@link Mutations#byName}.
     *
     * @param names comma-separated, e.g. "connections,neurons,activations"
     * @return this
     */
    public Evolution mutations(String names) {
        for (String n : names.split(",")) {
            if (!n.isBlank()) {
                Mutations.byName(n.trim(), 1);   // fails now for unknown names
                mutationNames.add(n.trim());
            }
        }
        return this;
    }

    /**
     * @param limit the most neurons per region for the built-in "neurons" and "regions"
     *     mutations enabled by name; default twice the template's largest region
     * @return this
     */
    public Evolution maxNeurons(int limit) {
        this.maxNeurons = limit;
        return this;
    }

    private int maxNeurons() {
        if (maxNeurons > 0) {
            return maxNeurons;
        }
        if (template == null) {
            return 256;
        }
        return 2 * template.regions().stream().mapToInt(r -> r.neurons().size()).max().orElse(64);
    }

    /**
     * @param fitness how to score a candidate; default {@link Fitness#metric()}
     * @return this
     */
    public Evolution fitness(Fitness fitness) {
        this.fitness = Objects.requireNonNull(fitness, "fitness");
        return this;
    }

    /**
     * Prefers smaller designs: subtracts {@code penalty} per parameter from the fitness.
     *
     * @param penalty per parameter, e.g. {@code 0.01 / 1000}
     * @return this
     */
    public Evolution sizePenalty(double penalty) {
        this.fitness = fitness.minusPerParameter(penalty);
        return this;
    }

    /**
     * @param count training epochs per candidate; default the task's own
     * @return this
     */
    public Evolution epochs(int count) {
        this.epochs = count;
        return this;
    }

    /**
     * @param fraction the part of the training data held out to score candidates; default
     *     0.2. With 0, candidates are scored on the data they were trained on, which is what
     *     tiny tasks such as XOR need; that is also used when the training data has fewer
     *     than 50 samples.
     * @return this
     */
    public Evolution validation(double fraction) {
        if (!(fraction >= 0 && fraction < 1)) {
            throw new IllegalArgumentException("fraction must be in [0, 1), got " + fraction);
        }
        this.validation = fraction;
        return this;
    }

    /**
     * @param count the best new candidates of each generation trained again with a second
     *     seed, their fitness averaged; default 3, 0 turns it off
     * @return this
     */
    public Evolution rescore(int count) {
        this.rescore = Math.max(0, count);
        return this;
    }

    /**
     * @param seed the seed of the whole search: the same seed gives the same evolution
     * @return this
     */
    public Evolution seed(long seed) {
        this.seed = seed;
        return this;
    }

    /**
     * @param count candidates trained at the same time; default 1. Results do not depend on it.
     * @return this
     */
    public Evolution threads(int count) {
        this.threads = Math.max(1, count);
        return this;
    }

    /**
     * @param device where candidates are trained; default the default device
     * @return this
     */
    public Evolution device(Device device) {
        this.device = device;
        return this;
    }

    /**
     * @param settings applied to every candidate's trainer after the task's defaults, such
     *     as a batch size
     * @return this
     */
    public Evolution trainer(Consumer<Trainer> settings) {
        this.trainerSettings = Objects.requireNonNull(settings, "settings");
        return this;
    }

    /**
     * @param listener called after every generation, e.g. to print progress
     * @return this
     */
    public Evolution onGeneration(Consumer<Generation> listener) {
        this.listener = Objects.requireNonNull(listener, "listener");
        return this;
    }

    // ---------------------------------------------------------------- running

    private Activation head() {
        if (head != null) {
            return head;
        }
        if (task.kind() == Benchmark.Kind.REGRESSION) {
            return Activation.IDENTITY;
        }
        return Math.toIntExact(task.outputShape().size()) == 1 ? Activation.SIGMOID : Activation.SOFTMAX;
    }

    /**
     * Builds a model of a design, untrained, with this search's input, front end and head.
     *
     * @param genome the design
     * @param seed the seed for the starting weights
     * @return the model
     */
    public Model build(Genome genome, long seed) {
        return genome.build(task.inputShape(), frontEnd, Math.toIntExact(task.outputShape().size()), head(), seed);
    }

    /**
     * Applies a design's hyperparameters, and this search's trainer settings, to a trainer
     * that already has the task's defaults.
     *
     * @param genome the design
     * @return the settings
     */
    public Consumer<Trainer> settings(Genome genome) {
        return t -> {
            if (!Double.isNaN(genome.learningRate()) || genome.optimizer() != null) {
                float lr = (float) (Double.isNaN(genome.learningRate()) ? 0.01 : genome.learningRate());
                t.optimizer(switch (genome.optimizer() == null ? "adam" : genome.optimizer()) {
                    case "sgd" -> Optimizer.sgd(lr);
                    case "momentum" -> Optimizer.momentum(lr, 0.9f);
                    default -> Optimizer.adam(lr);
                });
            }
            if (device != null) {
                t.device(device);
            }
            trainerSettings.accept(t);
        };
    }

    private Fitness.Score train(Genome genome, Dataset train, Dataset valid, long seed) {
        Model model = build(genome, seed);
        Trainer t = Trainer.of(model).loss(task.loss()).metric(task.metric()).seed(seed);
        task.defaults().accept(t);
        if (epochs > 0) {
            t.epochs(epochs);
        }
        settings(genome).accept(t);
        long start = System.nanoTime();
        t.fit(train);
        double seconds = (System.nanoTime() - start) / 1e9;
        Evaluation e = t.evaluate(valid);
        boolean higher = task.kind() == Benchmark.Kind.CLASSIFICATION;
        return new Fitness.Score(e.get(task.metric()), higher, e.loss(),
                genome.effectiveParameters(Math.toIntExact(task.outputShape().size())), genome.connections(), seconds);
    }

    /** Higher fitness first; among equals, the smaller design. */
    private static final Comparator<Individual> RANK = Comparator.comparingDouble(Individual::fitness).reversed()
            .thenComparingLong(i -> i.score().parameters());

    private double fit(Fitness.Score s) {
        double f = fitness.of(s);
        return Double.isNaN(f) ? Double.NEGATIVE_INFINITY : f;
    }

    private Genome scaled(Genome g, SplittableRandom random) {
        if (neuronScale >= 1) {
            return g;
        }
        return g.with(d -> {
            for (Genome.Draft.RegionDraft r : d.regions) {
                int keep = Math.max(1, (int) Math.round(r.neurons.size() * neuronScale));
                while (r.neurons.size() > keep) {
                    d.removeNeuron(r, random.nextInt(r.neurons.size()));
                }
            }
        });
    }

    private Genome mutate(Genome g, SplittableRandom random) {
        Genome.Draft d = g.draft();
        for (Mutation m : mutations) {
            m.apply(d, random);
        }
        return d.freeze();
    }

    private Individual pick(List<Individual> ranked, SplittableRandom random) {
        Individual best = null;
        int bestRank = Integer.MAX_VALUE;
        for (int i = 0; i < tournament; i++) {
            int r = random.nextInt(ranked.size());
            if (r < bestRank) {
                bestRank = r;
                best = ranked.get(r);
            }
        }
        return best;
    }

    /**
     * Runs the search.
     *
     * @return the result
     */
    public Result run() {
        if (template == null) {
            throw new IllegalStateException("set a template design first");
        }
        if (!template.isViable()) {
            throw new IllegalStateException("the template cannot be built: " + template);
        }
        build(template, seed);   // fails early if the template does not fit the task
        if (mutations.isEmpty() && mutationNames.isEmpty()) {
            mutationNames.addAll(List.of("connections", "neurons", "activations"));
        }
        for (String n : mutationNames) {
            mutations.add(Mutations.byName(n, maxNeurons()));
        }
        mutationNames.clear();
        SplittableRandom random = new SplittableRandom(seed);
        Dataset train = task.train();
        Dataset valid = train;
        if (validation > 0 && train.size() >= 50) {
            Dataset.Split split = train.shuffle(seed).split(1 - validation);
            train = split.train();
            valid = split.test();
        }
        long trainSeed = seed * 31 + 7;
        long rescoreSeed = trainSeed + 1;

        Map<Genome, Individual> seen = new HashMap<>();
        Genome start = scaled(template, random);
        List<Genome> candidates = new ArrayList<>();
        candidates.add(start);
        int attempts = 0;
        while (candidates.size() < population && attempts++ < population * 50) {
            Genome g = start;
            int rounds = 1 + random.nextInt(3);
            for (int i = 0; i < rounds; i++) {
                g = mutate(g, random);
            }
            if (g.isViable() && !candidates.contains(g)) {
                candidates.add(g);
            }
        }

        List<Generation> history = new ArrayList<>();
        List<Individual> ranked = List.of();
        ExecutorService pool = threads > 1 ? Executors.newFixedThreadPool(threads) : null;
        try {
            for (int gen = 0; gen < generations; gen++) {
                long t0 = System.nanoTime();
                List<Genome> fresh = candidates.stream().filter(g -> !seen.containsKey(g)).distinct().toList();
                List<Fitness.Score> scores = evaluate(fresh, train, valid, trainSeed, pool);
                for (int i = 0; i < fresh.size(); i++) {
                    Fitness.Score s = scores.get(i);
                    seen.put(fresh.get(i), new Individual(fresh.get(i), fit(s), s, 1));
                }
                List<Individual> pop = new ArrayList<>();
                for (Genome g : candidates) {
                    pop.add(seen.get(g));
                }
                // second opinion for the best that have only been trained once, until the
                // top few have all had one: a lucky first score cannot stay on top
                pop.sort(RANK);
                int retrained = 0;
                while (true) {
                    List<Individual> top = pop.stream().map(i -> seen.get(i.genome())).distinct().limit(rescore)
                            .filter(i -> i.evaluations() == 1).toList();
                    if (top.isEmpty()) {
                        break;
                    }
                    List<Fitness.Score> again = evaluate(top.stream().map(Individual::genome).toList(), train,
                            valid, rescoreSeed, pool);
                    for (int i = 0; i < top.size(); i++) {
                        Individual old = top.get(i);
                        seen.put(old.genome(), new Individual(old.genome(), (old.fitness() + fit(again.get(i))) / 2,
                                old.score(), 2));
                    }
                    retrained += top.size();
                    pop.replaceAll(i -> seen.get(i.genome()));
                    pop.sort(RANK);
                }
                ranked = List.copyOf(pop);
                double mean = ranked.stream().mapToDouble(Individual::fitness)
                        .filter(Double::isFinite).average().orElse(Double.NaN);
                Generation record = new Generation(gen, ranked.getFirst(), mean, fresh.size() + retrained,
                        (System.nanoTime() - t0) / 1e9);
                history.add(record);
                listener.accept(record);
                if (gen == generations - 1) {
                    break;
                }
                candidates = breed(ranked, random);
            }
        } finally {
            if (pool != null) {
                pool.shutdownNow();
            }
        }
        return new Result(List.copyOf(history), ranked, template);
    }

    private List<Genome> breed(List<Individual> ranked, SplittableRandom random) {
        List<Genome> next = new ArrayList<>();
        for (Individual i : ranked) {
            if (next.size() >= elites) {
                break;
            }
            if (!next.contains(i.genome())) {
                next.add(i.genome());
            }
        }
        int attempts = 0;
        while (next.size() < population && attempts++ < population * 50) {
            Individual a = pick(ranked, random);
            Genome child;
            if (random.nextDouble() < crossoverRate) {
                Individual b = pick(ranked, random);
                child = a.fitness() >= b.fitness() ? crossover.cross(a.genome(), b.genome(), random)
                        : crossover.cross(b.genome(), a.genome(), random);
            } else {
                child = a.genome();
            }
            child = mutate(child, random);
            if (child.isViable() && !next.contains(child)) {
                next.add(child);
            }
        }
        return next;
    }

    private List<Fitness.Score> evaluate(List<Genome> genomes, Dataset train, Dataset valid, long seed,
                                         ExecutorService pool) {
        List<Fitness.Score> out = new ArrayList<>();
        if (pool == null) {
            for (Genome g : genomes) {
                out.add(safely(g, train, valid, seed));
            }
            return out;
        }
        List<Future<Fitness.Score>> futures = new ArrayList<>();
        for (Genome g : genomes) {
            futures.add(pool.submit(() -> safely(g, train, valid, seed)));
        }
        try {
            for (Future<Fitness.Score> f : futures) {
                out.add(f.get());
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("interrupted", e);
        } catch (ExecutionException e) {
            throw new IllegalStateException("training a candidate failed", e.getCause());
        }
        return out;
    }

    private Fitness.Score safely(Genome g, Dataset train, Dataset valid, long seed) {
        try {
            return train(g, train, valid, seed);
        } catch (RuntimeException e) {
            // a design that cannot be built or trained is simply unfit
            return new Fitness.Score(Double.NaN, task.kind() == Benchmark.Kind.CLASSIFICATION, Double.NaN,
                    g.effectiveParameters(Math.toIntExact(task.outputShape().size())), g.connections(), 0);
        }
    }

    // ---------------------------------------------------------------- comparing

    /**
     * Trains a design on the whole training data with the task's own settings (not the
     * search's budget) and tests it on the test data, for a fair comparison with a
     * hand-built model run through {@link Benchmark#run}.
     *
     * @param genome the design
     * @param name its name in the report
     * @return the report
     */
    public Benchmark.Report benchmark(Genome genome, String name) {
        return Benchmark.run(task, name, build(genome, 1), settings(genome));
    }

    /**
     * Trains and tests a hand-built model the same way as {@link #benchmark(Genome, String)}.
     *
     * @param model makes the model
     * @param name its name in the report
     * @return the report
     */
    public Benchmark.Report benchmark(Supplier<Model> model, String name) {
        Consumer<Trainer> s = device == null ? trainerSettings : trainerSettings.andThen(t -> t.device(device));
        return Benchmark.run(task, name, model.get(), s);
    }
}
