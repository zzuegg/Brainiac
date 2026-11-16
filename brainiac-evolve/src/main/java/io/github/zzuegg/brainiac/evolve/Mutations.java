package io.github.zzuegg.brainiac.evolve;

import io.github.zzuegg.brainiac.evolve.Genome.Draft;
import io.github.zzuegg.brainiac.evolve.Genome.Draft.ProjectionDraft;
import io.github.zzuegg.brainiac.evolve.Genome.Draft.RegionDraft;
import io.github.zzuegg.brainiac.evolve.Genome.Neuron;
import io.github.zzuegg.brainiac.nn.Activation;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;

/**
 * The built-in {@link Mutation}s, one per kind of gene. Every rate, bound and choice set is
 * a parameter; {@link #byName} gives each one with sensible defaults.
 */
public final class Mutations {

    /** Whether activations change one neuron at a time or a whole region at once. */
    public enum Scope {
        /** Every neuron may get its own activation. */
        NEURON,
        /** All neurons of a region share one activation. */
        REGION
    }

    /** The activations {@link #activations(double, Scope)} chooses from by default. */
    public static final List<Activation> DEFAULT_ACTIVATIONS = List.of(Activation.RELU, Activation.TANH,
            Activation.SIGMOID, Activation.GELU, Activation.SOFTSIGN, Activation.SILU, Activation.IDENTITY);

    /** The optimizers {@link #optimizer(double, String...)} chooses from by default. */
    public static final List<String> OPTIMIZERS = List.of("adam", "momentum", "sgd");

    private Mutations() {
    }

    /**
     * Flips connections on or off at random positions of every projection.
     *
     * @param fraction the expected fraction of each projection's possible connections to
     *     flip; at least one flip per projection happens with probability {@code min(1, 10 * fraction)}
     * @return the mutation, named "connections"
     */
    public static Mutation connections(double fraction) {
        return Mutation.named("connections", (g, random) -> {
            for (ProjectionDraft p : g.projections) {
                int[] from = g.ids(p.from());
                int[] to = g.ids(p.to());
                long possible = (long) from.length * to.length;
                if (possible == 0) {
                    continue;
                }
                double expected = possible * fraction;
                int flips = (int) expected + (random.nextDouble() < expected - (int) expected ? 1 : 0);
                if (flips == 0 && random.nextDouble() < Math.min(1, 10 * fraction)) {
                    flips = 1;
                }
                for (int f = 0; f < flips; f++) {
                    long k = Genome.key(from[random.nextInt(from.length)], to[random.nextInt(to.length)]);
                    if (!p.connections.remove(k)) {
                        p.connections.add(k);
                    }
                }
            }
        });
    }

    /**
     * Grows or thins whole projections: with probability {@code rate} per projection, adds
     * or removes a tenth of its possible connections.
     *
     * @param rate the probability per projection
     * @return the mutation, named "density"
     */
    public static Mutation density(double rate) {
        return Mutation.named("density", (g, random) -> {
            for (ProjectionDraft p : g.projections) {
                if (random.nextDouble() >= rate) {
                    continue;
                }
                int[] from = g.ids(p.from());
                int[] to = g.ids(p.to());
                int n = (int) Math.max(1, (long) from.length * to.length / 10);
                boolean grow = random.nextBoolean();
                if (grow) {
                    for (int i = 0; i < n; i++) {
                        p.connections.add(Genome.key(from[random.nextInt(from.length)], to[random.nextInt(to.length)]));
                    }
                } else {
                    List<Long> on = new ArrayList<>(p.connections);
                    for (int i = 0; i < n && !on.isEmpty(); i++) {
                        p.connections.remove(on.remove(random.nextInt(on.size())));
                    }
                }
            }
        });
    }

    /**
     * Adds a projection that does not exist yet (feed-forward, feedback or lateral) at a
     * random density, or removes one; the input always keeps at least one projection.
     *
     * @param rate the probability of adding one, and separately of removing one
     * @return the mutation, named "projections"
     */
    public static Mutation projections(double rate) {
        return Mutation.named("projections", (g, random) -> {
            if (random.nextDouble() < rate && !g.regions.isEmpty()) {
                List<String> sources = new ArrayList<>();
                sources.add(Genome.INPUT);
                g.regions.forEach(r -> sources.add(r.name()));
                String from = sources.get(random.nextInt(sources.size()));
                String to = g.regions.get(random.nextInt(g.regions.size())).name();
                if (g.projection(from, to) == null) {
                    g.connectRandomly(from, to, 0.05 + 0.25 * random.nextDouble(), random);
                }
            }
            if (random.nextDouble() < rate && g.projections.size() > 1) {
                ProjectionDraft p = g.projections.get(random.nextInt(g.projections.size()));
                long inputs = g.projections.stream().filter(q -> q.from().equals(Genome.INPUT)).count();
                if (!p.from().equals(Genome.INPUT) || inputs > 1) {
                    g.projections.remove(p);
                }
            }
        });
    }

    /**
     * Adds or removes neurons: with probability {@code rate} per region, adds or removes
     * up to a tenth of its neurons (at least one). A new neuron copies the activation of
     * a random neighbor and is wired at the density of each projection it joins.
     *
     * @param rate the probability per region
     * @param min the fewest neurons a region may have
     * @param max the most neurons a region may have
     * @return the mutation, named "neurons"
     */
    public static Mutation neurons(double rate, int min, int max) {
        if (min < 1 || max < min) {
            throw new IllegalArgumentException("need 1 <= min <= max, got " + min + " and " + max);
        }
        return Mutation.named("neurons", (g, random) -> {
            for (RegionDraft r : g.regions) {
                if (random.nextDouble() >= rate) {
                    continue;
                }
                int size = r.neurons.size();
                int n = 1 + random.nextInt(Math.max(1, size / 10));
                if (random.nextBoolean()) {
                    n = Math.min(n, max - size);
                    for (int i = 0; i < n; i++) {
                        addNeuron(g, r, random);
                    }
                } else {
                    n = Math.min(n, size - min);
                    for (int i = 0; i < n; i++) {
                        g.removeNeuron(r, random.nextInt(r.neurons.size()));
                    }
                }
            }
        });
    }

    private static void addNeuron(Draft g, RegionDraft r, java.util.SplittableRandom random) {
        Activation a = r.neurons.isEmpty() ? Activation.TANH
                : r.neurons.get(random.nextInt(r.neurons.size())).activation();
        int id = g.newId();
        int before = r.neurons.size();
        for (ProjectionDraft p : g.projections) {
            if (p.to().equals(r.name())) {
                int[] from = g.ids(p.from());
                double density = before == 0 ? 0.5 : (double) p.connections.size() / ((long) from.length * before);
                for (int s : from) {
                    if (random.nextDouble() < density) {
                        p.connections.add(Genome.key(s, id));
                    }
                }
            }
            if (p.from().equals(r.name()) && !p.to().equals(r.name())) {
                int[] to = g.ids(p.to());
                double density = before == 0 ? 0.5 : (double) p.connections.size() / ((long) before * to.length);
                for (int t : to) {
                    if (random.nextDouble() < density) {
                        p.connections.add(Genome.key(id, t));
                    }
                }
            }
        }
        r.neurons.add(new Neuron(id, a));
    }

    /**
     * Changes activations, chosen from {@link #DEFAULT_ACTIVATIONS}.
     *
     * @param rate the probability per neuron (or per region)
     * @param scope per neuron or per region
     * @return the mutation, named "activations"
     */
    public static Mutation activations(double rate, Scope scope) {
        return activations(rate, scope, DEFAULT_ACTIVATIONS.toArray(Activation[]::new));
    }

    /**
     * Changes activations.
     *
     * @param rate the probability per neuron (or per region)
     * @param scope per neuron or per region
     * @param choices the activations to choose from; any without learnable parameters
     * @return the mutation, named "activations", or "region-activations" for {@link Scope#REGION}
     */
    public static Mutation activations(double rate, Scope scope, Activation... choices) {
        if (choices.length == 0) {
            throw new IllegalArgumentException("give at least one activation to choose from");
        }
        Activation[] set = choices.clone();
        Objects.requireNonNull(scope, "scope");
        return Mutation.named(scope == Scope.REGION ? "region-activations" : "activations", (g, random) -> {
            for (RegionDraft r : g.regions) {
                if (scope == Scope.REGION) {
                    if (random.nextDouble() < rate) {
                        Activation a = set[random.nextInt(set.length)];
                        r.neurons.replaceAll(n -> n.withActivation(a));
                    }
                } else {
                    r.neurons.replaceAll(n -> random.nextDouble() < rate
                            ? n.withActivation(set[random.nextInt(set.length)]) : n);
                }
            }
        });
    }

    /**
     * Turns input-controlled activation on or off for single neurons: a switching neuron
     * gets control weights on its incoming connections and chooses among its region's
     * candidates by what its inputs are (see {@link io.github.zzuegg.brainiac.nn.Brain.Switching}).
     * A region's first switching neuron gives it {@link Genome#DEFAULT_SWITCHING}.
     *
     * @param rate the probability per neuron
     * @return the mutation, named "switching"
     */
    public static Mutation switching(double rate) {
        return Mutation.named("switching", (g, random) -> {
            for (RegionDraft r : g.regions) {
                r.neurons.replaceAll(n -> random.nextDouble() < rate ? n.withSwitching(!n.switches()) : n);
            }
        });
    }

    /**
     * Flips a region's switching between soft (a learned blend of the candidates) and hard
     * (only the most likely candidate counts).
     *
     * @param rate the probability per region with switching neurons
     * @return the mutation, named "switch-mode"
     */
    public static Mutation switchMode(double rate) {
        return Mutation.named("switch-mode", (g, random) -> {
            for (RegionDraft r : g.regions) {
                if (switches(r) && random.nextDouble() < rate) {
                    var s = settings(r);
                    r.switching = new io.github.zzuegg.brainiac.nn.Brain.Switching(s.candidates(), !s.hard());
                }
            }
        });
    }

    /**
     * Adds a candidate to a region's switching neurons, or removes one (keeping at least two).
     *
     * @param rate the probability per region with switching neurons
     * @param choices the activations that can be added, without learnable parameters
     * @return the mutation, named "switch-candidates"
     */
    public static Mutation switchCandidates(double rate, Activation... choices) {
        Activation[] set = choices.length == 0 ? DEFAULT_ACTIVATIONS.toArray(Activation[]::new) : choices.clone();
        return Mutation.named("switch-candidates", (g, random) -> {
            for (RegionDraft r : g.regions) {
                if (!switches(r) || random.nextDouble() >= rate) {
                    continue;
                }
                var s = settings(r);
                List<Activation> cs = new ArrayList<>(s.candidates());
                boolean add = cs.size() <= 2 || random.nextBoolean();
                if (add) {
                    Activation a = set[random.nextInt(set.length)];
                    if (!cs.contains(a)) {
                        cs.add(a);
                    }
                } else {
                    cs.remove(random.nextInt(cs.size()));
                }
                if (cs.size() >= 2) {
                    r.switching = new io.github.zzuegg.brainiac.nn.Brain.Switching(cs, s.hard());
                }
            }
        });
    }

    private static boolean switches(RegionDraft r) {
        return r.neurons.stream().anyMatch(Neuron::switches);
    }

    private static io.github.zzuegg.brainiac.nn.Brain.Switching settings(RegionDraft r) {
        return r.switching != null ? r.switching : Genome.DEFAULT_SWITCHING;
    }

    /**
     * Scales the learning rate by a random factor {@code e^(sigma * N(0, 1))}, kept in
     * {@code [min, max]}. A genome that still uses the task's optimizer starts from 0.01.
     *
     * @param rate the probability of a change
     * @param sigma the spread of the factor on a log scale
     * @param min the smallest learning rate
     * @param max the largest learning rate
     * @return the mutation, named "learning-rate"
     */
    public static Mutation learningRate(double rate, double sigma, double min, double max) {
        return Mutation.named("learning-rate", (g, random) -> {
            if (random.nextDouble() < rate) {
                double lr = Double.isNaN(g.learningRate) ? 0.01 : g.learningRate;
                g.learningRate = Math.clamp(lr * Math.exp(sigma * random.nextGaussian()), min, max);
            }
        });
    }

    /**
     * Picks another optimizer.
     *
     * @param rate the probability of a change
     * @param choices optimizer names: "adam", "momentum", "sgd"
     * @return the mutation, named "optimizer"
     */
    public static Mutation optimizer(double rate, String... choices) {
        for (String c : choices) {
            if (!OPTIMIZERS.contains(c)) {
                throw new IllegalArgumentException("unknown optimizer " + c + "; choose from " + OPTIMIZERS);
            }
        }
        String[] set = choices.length == 0 ? OPTIMIZERS.toArray(String[]::new) : choices.clone();
        return Mutation.named("optimizer", (g, random) -> {
            if (random.nextDouble() < rate) {
                g.optimizer = set[random.nextInt(set.length)];
            }
        });
    }

    /**
     * Adds or removes one tick (update step per input), so feedback and lateral
     * projections get time to act.
     *
     * @param rate the probability of a change
     * @param max the most ticks
     * @return the mutation, named "ticks"
     */
    public static Mutation ticks(double rate, int max) {
        return Mutation.named("ticks", (g, random) -> {
            if (random.nextDouble() < rate) {
                g.ticks = Math.clamp(g.ticks + (random.nextBoolean() ? 1 : -1), 1, max);
            }
        });
    }

    /**
     * Adds a new region (a new layer, in effect) fed by a random existing region or the
     * input and feeding a random region after it, or removes a region that is not the only
     * output.
     *
     * @param rate the probability of adding one, and separately of removing one
     * @param minNeurons the fewest neurons of a new region
     * @param maxNeurons the most neurons of a new region
     * @param maxRegions the most regions
     * @return the mutation, named "regions"
     */
    public static Mutation regions(double rate, int minNeurons, int maxNeurons, int maxRegions) {
        return Mutation.named("regions", (g, random) -> {
            if (random.nextDouble() < rate && g.regions.size() < maxRegions && !g.regions.isEmpty()) {
                int at = random.nextInt(g.regions.size());       // new region goes before this one
                String from = at == 0 ? Genome.INPUT : g.regions.get(random.nextInt(at)).name();
                String to = g.regions.get(at + random.nextInt(g.regions.size() - at)).name();
                String name = freshName(g);
                Activation a = g.regions.get(at).neurons.isEmpty() ? Activation.TANH
                        : g.regions.get(at).neurons.get(0).activation();
                g.addRegion(name, minNeurons + random.nextInt(maxNeurons - minNeurons + 1), a);
                g.regions.add(at, g.regions.removeLast());
                g.connectRandomly(from, name, 0.5, random);
                g.connectRandomly(name, to, 0.5, random);
            }
            if (random.nextDouble() < rate && g.regions.size() > 1) {
                RegionDraft r = g.regions.get(random.nextInt(g.regions.size()));
                if (!(g.outputs.size() == 1 && g.outputs.contains(r.name()))) {
                    g.removeRegion(r.name());
                }
            }
        });
    }

    private static String freshName(Draft g) {
        for (int i = 1; ; i++) {
            String n = "r" + i;
            if (g.region(n) == null) {
                return n;
            }
        }
    }

    // ---------------------------------------------------------------- by name

    private static final Map<String, Function<Integer, Mutation>> DEFAULTS = new LinkedHashMap<>();

    static {
        DEFAULTS.put("connections", max -> connections(0.02));
        DEFAULTS.put("density", max -> density(0.1));
        DEFAULTS.put("projections", max -> projections(0.1));
        DEFAULTS.put("neurons", max -> neurons(0.3, 1, max));
        DEFAULTS.put("activations", max -> activations(0.05, Scope.NEURON));
        DEFAULTS.put("region-activations", max -> activations(0.2, Scope.REGION));
        DEFAULTS.put("switching", max -> switching(0.05));
        DEFAULTS.put("switch-mode", max -> switchMode(0.1));
        DEFAULTS.put("switch-candidates", max -> switchCandidates(0.1));
        DEFAULTS.put("learning-rate", max -> learningRate(0.3, 0.5, 1e-4, 0.3));
        DEFAULTS.put("optimizer", max -> optimizer(0.1));
        DEFAULTS.put("ticks", max -> ticks(0.1, 4));
        DEFAULTS.put("regions", max -> regions(0.05, 2, Math.max(2, max / 4), 6));
    }

    /** @return the names {@link #byName} knows */
    public static List<String> names() {
        return List.copyOf(DEFAULTS.keySet());
    }

    /**
     * A built-in mutation with default settings: "connections", "density", "projections",
     * "neurons", "activations" (per neuron), "region-activations", "switching",
     * "switch-mode", "switch-candidates", "learning-rate",
     * "optimizer", "ticks" or "regions".
     *
     * @param name the name
     * @param maxNeurons the most neurons per region, for "neurons" and "regions"
     * @return the mutation
     */
    public static Mutation byName(String name, int maxNeurons) {
        Function<Integer, Mutation> f = DEFAULTS.get(name);
        if (f == null) {
            throw new IllegalArgumentException("unknown mutation " + name + "; choose from " + names());
        }
        return f.apply(maxNeurons);
    }

    /**
     * The built-in mutations with default settings.
     *
     * @param names comma-separated names, as for {@link #byName}
     * @param maxNeurons the most neurons per region
     * @return the mutations
     */
    public static List<Mutation> parse(String names, int maxNeurons) {
        return Arrays.stream(names.split(",")).map(String::trim).filter(s -> !s.isEmpty())
                .map(n -> byName(n, maxNeurons)).toList();
    }
}
