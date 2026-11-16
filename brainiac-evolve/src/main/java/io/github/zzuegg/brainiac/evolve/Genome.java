package io.github.zzuegg.brainiac.evolve;

import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Brain;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.TreeSet;
import java.util.function.Consumer;

/**
 * The design of one network, as the genetic algorithm sees it: named regions of neurons,
 * each neuron with its own id and activation, joined by projections that list exactly
 * which connections exist, plus a few training hyperparameters.
 *
 * <p>A genome describes the hidden part of a model. The input (the task's features, or
 * the output of a fixed front end such as a few convolutions) and the output layer are
 * fixed; {@link #build} wraps the genome's regions in a {@link Brain} between them.
 *
 * <p>Neuron ids are what crossover lines parents up by, and they are what connections
 * refer to, so a connection survives when other neurons are added or removed. The input
 * neurons have ids {@code 0 .. inputs - 1}.
 *
 * <p>Genomes are immutable. To change one, take a {@link #draft()}, edit it, and
 * {@link Draft#freeze() freeze} it again; that is what {@link Mutation}s do.
 */
public final class Genome {

    /** The name of the input in {@link Projection#from()}. */
    public static final String INPUT = Brain.INPUT;

    /**
     * A neuron.
     *
     * @param id its id, unique within the genome and kept by its copies in offspring
     * @param activation its activation
     */
    public record Neuron(int id, Activation activation) {
        /** Checks the fields. */
        public Neuron {
            Objects.requireNonNull(activation, "activation");
        }
    }

    /**
     * A region: a group of neurons updated together.
     *
     * @param name its name
     * @param neurons its neurons, in order
     */
    public record Region(String name, List<Neuron> neurons) {
        /** Copies the list. */
        public Region {
            neurons = List.copyOf(neurons);
        }
    }

    /**
     * The connections from one region (or the input) to another.
     *
     * @param from the source region, or {@link #INPUT}
     * @param to the target region
     * @param connections the connections, as sorted {@link #key keys}
     */
    public record Projection(String from, String to, long[] connections) {
        /** Copies the array. */
        public Projection {
            connections = connections.clone();
        }

        @Override
        public long[] connections() {
            return connections.clone();
        }

        /** @return the number of connections */
        public int size() {
            return connections.length;
        }

        @Override
        public boolean equals(Object o) {
            return o instanceof Projection p && from.equals(p.from) && to.equals(p.to)
                    && Arrays.equals(connections, p.connections);
        }

        @Override
        public int hashCode() {
            return Objects.hash(from, to, Arrays.hashCode(connections));
        }

        @Override
        public String toString() {
            return from + " -> " + to + " (" + connections.length + ")";
        }
    }

    /**
     * Packs a connection into one number.
     *
     * @param source the source neuron's id
     * @param target the target neuron's id
     * @return the key
     */
    public static long key(int source, int target) {
        return ((long) source << 32) | (target & 0xffffffffL);
    }

    /**
     * @param key a connection key
     * @return its source neuron's id
     */
    public static int source(long key) {
        return (int) (key >>> 32);
    }

    /**
     * @param key a connection key
     * @return its target neuron's id
     */
    public static int target(long key) {
        return (int) key;
    }

    private final int inputs;
    private final List<Region> regions;
    private final List<Projection> projections;
    private final List<String> outputs;
    private final int ticks;
    private final double learningRate;
    private final String optimizer;
    private final int nextId;

    private Genome(int inputs, List<Region> regions, List<Projection> projections, List<String> outputs, int ticks,
                   double learningRate, String optimizer, int nextId) {
        this.inputs = inputs;
        this.regions = List.copyOf(regions);
        this.projections = List.copyOf(projections);
        this.outputs = List.copyOf(outputs);
        this.ticks = ticks;
        this.learningRate = learningRate;
        this.optimizer = optimizer;
        this.nextId = nextId;
    }

    // ---------------------------------------------------------------- templates

    /**
     * A fully connected chain of hidden layers, like
     * {@code Model.sequential().input(inputs).dense(hidden[0], activation).dense(hidden[1], activation)...}:
     * a good template when the known working model is a plain multilayer perceptron.
     *
     * @param inputs the number of inputs
     * @param activation the hidden layers' activation
     * @param hidden the size of every hidden layer, at least one
     * @return the genome; its regions are named {@code h1, h2, ...}
     */
    public static Genome dense(int inputs, Activation activation, int... hidden) {
        if (hidden.length == 0) {
            throw new IllegalArgumentException("give at least one hidden layer");
        }
        Draft d = new Draft(inputs);
        String previous = INPUT;
        for (int l = 0; l < hidden.length; l++) {
            String name = "h" + (l + 1);
            d.addRegion(name, hidden[l], activation);
            d.connectFully(previous, name);
            previous = name;
        }
        d.outputs.add(previous);
        return d.freeze();
    }

    /**
     * A genome with the regions and projections of a {@link Brain} design, each projection
     * wired at random at its density between the regions' starting neurons.
     *
     * @param inputs the number of inputs the brain gets
     * @param spec the brain's design
     * @param seed the seed for the random wiring
     * @return the genome
     */
    public static Genome of(int inputs, Brain.Spec spec, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        Draft d = new Draft(inputs);
        for (Brain.RegionSpec r : spec.regions()) {
            d.addRegion(r.name(), Math.max(1, r.initial()), r.activation());
        }
        for (Brain.ProjectionSpec p : spec.projections()) {
            d.connectRandomly(p.from(), p.to(), p.density(), random);
        }
        d.outputs.addAll(spec.outputs());
        d.ticks = spec.ticks();
        return d.freeze();
    }

    // ---------------------------------------------------------------- reading

    /** @return the number of inputs */
    public int inputs() {
        return inputs;
    }

    /** @return the regions, in update order */
    public List<Region> regions() {
        return regions;
    }

    /** @return the projections */
    public List<Projection> projections() {
        return projections;
    }

    /** @return the regions whose neurons feed the output layer */
    public List<String> outputs() {
        return outputs;
    }

    /** @return update steps per input; more than 1 lets feedback and lateral projections act */
    public int ticks() {
        return ticks;
    }

    /** @return the learning rate, or NaN to use the task's optimizer unchanged */
    public double learningRate() {
        return learningRate;
    }

    /** @return the optimizer ("adam", "momentum" or "sgd"), or null for the task's */
    public String optimizer() {
        return optimizer;
    }

    /** @return the number of hidden neurons */
    public int neurons() {
        return regions.stream().mapToInt(r -> r.neurons().size()).sum();
    }

    /** @return the number of connections */
    public long connections() {
        return projections.stream().mapToLong(Projection::size).sum();
    }

    /**
     * The parameters a trained model of this design really uses: connections, biases and
     * the output layer.
     *
     * @param outputs the size of the output layer
     * @return the count
     */
    public long effectiveParameters(int outputs) {
        int fed = 0;
        for (Region r : regions) {
            if (this.outputs.contains(r.name())) {
                fed += r.neurons().size();
            }
        }
        return connections() + neurons() + (long) fed * outputs + outputs;
    }

    /**
     * @param name a region's name
     * @return the region
     */
    public Region region(String name) {
        return regions.stream().filter(r -> r.name().equals(name)).findFirst()
                .orElseThrow(() -> new IllegalArgumentException("no region named " + name));
    }

    /**
     * @return how often each activation is used, over all hidden neurons
     */
    public Map<String, Integer> activationCounts() {
        Map<String, Integer> out = new LinkedHashMap<>();
        for (Region r : regions) {
            for (Neuron n : r.neurons()) {
                out.merge(n.activation().toString(), 1, Integer::sum);
            }
        }
        return out;
    }

    /**
     * Checks that a model can be built: every region has a neuron, the input reaches a
     * region, and every output region exists.
     *
     * @return whether this genome can be built
     */
    public boolean isViable() {
        if (regions.isEmpty() || outputs.isEmpty()) {
            return false;
        }
        for (Region r : regions) {
            if (r.neurons().isEmpty()) {
                return false;
            }
        }
        return projections.stream().anyMatch(p -> p.from().equals(INPUT) && p.size() > 0)
                && outputs.stream().allMatch(o -> regions.stream().anyMatch(r -> r.name().equals(o)));
    }

    /** @return a mutable copy */
    public Draft draft() {
        return new Draft(this);
    }

    /**
     * Applies changes to a copy.
     *
     * @param edit the changes
     * @return the changed copy
     */
    public Genome with(Consumer<Draft> edit) {
        Draft d = draft();
        edit.accept(d);
        return d.freeze();
    }

    // ---------------------------------------------------------------- building

    /**
     * Builds the brain layer of this design. Its wiring is random; {@link #wire} sets the
     * exact connections once the model is built.
     *
     * @return the brain's design
     */
    public Brain.Spec brainSpec() {
        Brain.Builder b = Brain.builder();
        for (Region r : regions) {
            Activation[] perUnit = r.neurons().stream().map(Neuron::activation).toArray(Activation[]::new);
            b.region(r.name(), perUnit.length).activation(Activation.perUnit(perUnit));
        }
        for (Projection p : projections) {
            if (p.size() == 0) {
                continue;
            }
            long possible = (long) size(p.from()) * size(p.to());
            b.connect(p.from(), p.to(), Math.min(1, (double) p.size() / possible));
        }
        return b.output(outputs.toArray(String[]::new)).ticks(ticks).build();
    }

    private int size(String region) {
        return region.equals(INPUT) ? inputs : region(region).neurons().size();
    }

    private Map<Integer, Integer> positions(String region) {
        Map<Integer, Integer> pos = new HashMap<>();
        if (region.equals(INPUT)) {
            for (int i = 0; i < inputs; i++) {
                pos.put(i, i);
            }
        } else {
            List<Neuron> ns = region(region).neurons();
            for (int i = 0; i < ns.size(); i++) {
                pos.put(ns.get(i).id(), i);
            }
        }
        return pos;
    }

    /**
     * Sets a brain built from {@link #brainSpec()} to exactly this genome's connections.
     *
     * @param brain the brain
     * @param random the source of the new weights
     */
    public void wire(Brain brain, SplittableRandom random) {
        int j = 0;
        for (Projection p : projections) {
            if (p.size() == 0) {
                continue;
            }
            Map<Integer, Integer> from = positions(p.from());
            Map<Integer, Integer> to = positions(p.to());
            int cols = to.size();
            boolean[] mask = new boolean[from.size() * cols];
            for (long k : p.connections) {
                Integer a = from.get(source(k));
                Integer b = to.get(target(k));
                if (a != null && b != null) {
                    mask[a * cols + b] = true;
                }
            }
            brain.setConnections(j++, mask, random);
        }
    }

    /**
     * Builds a model of this design.
     *
     * @param inputShape the shape of one sample
     * @param frontEnd fixed layers between the input and the brain, such as convolutions;
     *     may be null. The brain gets their output flattened.
     * @param outputs the size of the output layer
     * @param head the output layer's activation
     * @param seed the seed for the starting weights
     * @return the model
     */
    public Model build(Shape inputShape, Consumer<Model.Builder> frontEnd, int outputs, Activation head, long seed) {
        Model.Builder b = Model.sequential().input(inputShape.toArray());
        if (frontEnd != null) {
            frontEnd.accept(b);
        }
        if (b.shape().rank() != 1) {
            b.flatten();
        }
        if (b.shape().dim(0) != inputs) {
            throw new IllegalArgumentException("the genome has " + inputs + " inputs, but the brain would get "
                    + b.shape().dim(0));
        }
        Model model = b.add(brainSpec()).dense(outputs, head).seed(seed).build();
        wire(brainOf(model), new SplittableRandom(seed ^ 0x5DEECE66DL));
        return model;
    }

    /**
     * @param model a model with a brain layer
     * @return its brain
     */
    public static Brain brainOf(Model model) {
        for (Layer l : model.layers()) {
            if (l instanceof Brain b) {
                return b;
            }
        }
        throw new IllegalArgumentException("the model has no brain layer");
    }

    // ---------------------------------------------------------------- identity

    @Override
    public boolean equals(Object o) {
        return o instanceof Genome g && inputs == g.inputs && ticks == g.ticks
                && Double.compare(learningRate, g.learningRate) == 0 && Objects.equals(optimizer, g.optimizer)
                && regions.equals(g.regions) && projections.equals(g.projections) && outputs.equals(g.outputs);
    }

    @Override
    public int hashCode() {
        return Objects.hash(inputs, regions, projections, outputs, ticks, learningRate, optimizer);
    }

    /**
     * A multi-line description: regions with their sizes and activations, projections
     * with their densities, and the hyperparameters.
     *
     * @return the description
     */
    public String describe() {
        StringBuilder sb = new StringBuilder();
        for (Region r : regions) {
            Map<String, Integer> acts = new LinkedHashMap<>();
            for (Neuron n : r.neurons()) {
                acts.merge(n.activation().toString(), 1, Integer::sum);
            }
            sb.append(String.format("  region %-12s %4d neurons  %s%s%n", r.name(), r.neurons().size(), acts,
                    outputs.contains(r.name()) ? "  -> output" : ""));
        }
        for (Projection p : projections) {
            long possible = (long) size(p.from()) * size(p.to());
            sb.append(String.format("  %-12s -> %-12s %6d of %6d connections (%.0f%%)%n", p.from(), p.to(), p.size(),
                    possible, possible == 0 ? 0 : 100.0 * p.size() / possible));
        }
        sb.append("  ticks ").append(ticks);
        if (!Double.isNaN(learningRate)) {
            sb.append(String.format(", learning rate %.4g", learningRate));
        }
        if (optimizer != null) {
            sb.append(", optimizer ").append(optimizer);
        }
        return sb.append('\n').toString();
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("Genome[");
        for (Region r : regions) {
            sb.append(r.name()).append('=').append(r.neurons().size()).append(' ');
        }
        return sb.append(connections()).append(" connections, ticks ").append(ticks).append(']').toString();
    }

    // ---------------------------------------------------------------- editing

    /**
     * A mutable copy of a genome. Removing a neuron also removes its connections; a
     * projection to or from a removed region is dropped when the draft is frozen.
     */
    public static final class Draft {

        /** A region being edited. */
        public static final class RegionDraft {
            private final String name;
            /** Its neurons, editable. */
            public final List<Neuron> neurons;

            RegionDraft(String name, List<Neuron> neurons) {
                this.name = name;
                this.neurons = new ArrayList<>(neurons);
            }

            /** @return its name */
            public String name() {
                return name;
            }
        }

        /** A projection being edited. */
        public static final class ProjectionDraft {
            private final String from;
            private final String to;
            /** Its connections as {@link Genome#key keys}, editable. */
            public final TreeSet<Long> connections;

            ProjectionDraft(String from, String to, TreeSet<Long> connections) {
                this.from = from;
                this.to = to;
                this.connections = connections;
            }

            /** @return the source region, or {@link Genome#INPUT} */
            public String from() {
                return from;
            }

            /** @return the target region */
            public String to() {
                return to;
            }
        }

        private final int inputs;
        /** The regions, in update order; editable. */
        public final List<RegionDraft> regions = new ArrayList<>();
        /** The projections; editable. */
        public final List<ProjectionDraft> projections = new ArrayList<>();
        /** The output regions; editable. */
        public final List<String> outputs = new ArrayList<>();
        /** Update steps per input. */
        public int ticks = 1;
        /** The learning rate, or NaN for the task's. */
        public double learningRate = Double.NaN;
        /** The optimizer ("adam", "momentum" or "sgd"), or null for the task's. */
        public String optimizer;
        private int nextId;

        /**
         * An empty draft.
         *
         * @param inputs the number of inputs
         */
        public Draft(int inputs) {
            if (inputs < 1) {
                throw new IllegalArgumentException("inputs must be at least 1, got " + inputs);
            }
            this.inputs = inputs;
            this.nextId = inputs;
        }

        Draft(Genome g) {
            this.inputs = g.inputs;
            for (Region r : g.regions) {
                regions.add(new RegionDraft(r.name(), r.neurons()));
            }
            for (Projection p : g.projections) {
                TreeSet<Long> c = new TreeSet<>();
                for (long k : p.connections) {
                    c.add(k);
                }
                projections.add(new ProjectionDraft(p.from(), p.to(), c));
            }
            outputs.addAll(g.outputs);
            ticks = g.ticks;
            learningRate = g.learningRate;
            optimizer = g.optimizer;
            nextId = g.nextId;
        }

        /** @return the number of inputs */
        public int inputs() {
            return inputs;
        }

        /** @return a new neuron id, never used before in this genome's line */
        public int newId() {
            return nextId++;
        }

        /**
         * Makes sure ids from now on are above {@code id}.
         *
         * @param id an id in use
         */
        public void reserve(int id) {
            nextId = Math.max(nextId, id + 1);
        }

        /**
         * @param name a region's name
         * @return the region, or null
         */
        public RegionDraft region(String name) {
            for (RegionDraft r : regions) {
                if (r.name.equals(name)) {
                    return r;
                }
            }
            return null;
        }

        /**
         * @param from the source region or {@link Genome#INPUT}
         * @param to the target region
         * @return the projection, or null
         */
        public ProjectionDraft projection(String from, String to) {
            for (ProjectionDraft p : projections) {
                if (p.from.equals(from) && p.to.equals(to)) {
                    return p;
                }
            }
            return null;
        }

        /**
         * The ids of a region's neurons, or of the inputs.
         *
         * @param region the region or {@link Genome#INPUT}
         * @return the ids
         */
        public int[] ids(String region) {
            if (region.equals(INPUT)) {
                int[] ids = new int[inputs];
                for (int i = 0; i < inputs; i++) {
                    ids[i] = i;
                }
                return ids;
            }
            return region(region).neurons.stream().mapToInt(Neuron::id).toArray();
        }

        /**
         * Adds a region of new neurons.
         *
         * @param name its name
         * @param neurons how many neurons
         * @param activation their activation
         * @return the region
         */
        public RegionDraft addRegion(String name, int neurons, Activation activation) {
            if (name.equals(INPUT) || region(name) != null) {
                throw new IllegalArgumentException("region name " + name + " is taken");
            }
            RegionDraft r = new RegionDraft(name, List.of());
            for (int i = 0; i < neurons; i++) {
                r.neurons.add(new Neuron(newId(), activation));
            }
            regions.add(r);
            return r;
        }

        /**
         * Gets a projection, adding an empty one if there is none.
         *
         * @param from the source region or {@link Genome#INPUT}
         * @param to the target region
         * @return the projection
         */
        public ProjectionDraft connect(String from, String to) {
            ProjectionDraft p = projection(from, to);
            if (p == null) {
                if (!from.equals(INPUT) && region(from) == null || region(to) == null) {
                    throw new IllegalArgumentException("unknown region in " + from + " -> " + to);
                }
                p = new ProjectionDraft(from, to, new TreeSet<>());
                projections.add(p);
            }
            return p;
        }

        /**
         * Connects every neuron of {@code from} to every neuron of {@code to}.
         *
         * @param from the source region or {@link Genome#INPUT}
         * @param to the target region
         */
        public void connectFully(String from, String to) {
            connectRandomly(from, to, 1, null);
        }

        /**
         * Adds each possible connection with probability {@code density}.
         *
         * @param from the source region or {@link Genome#INPUT}
         * @param to the target region
         * @param density the probability
         * @param random the random source; may be null when density is 1
         */
        public void connectRandomly(String from, String to, double density, SplittableRandom random) {
            ProjectionDraft p = connect(from, to);
            for (int a : ids(from)) {
                for (int b : ids(to)) {
                    if (density >= 1 || random.nextDouble() < density) {
                        p.connections.add(key(a, b));
                    }
                }
            }
        }

        /**
         * Removes a neuron and every connection to or from it.
         *
         * @param region the region
         * @param index the neuron's position in it
         */
        public void removeNeuron(RegionDraft region, int index) {
            int id = region.neurons.remove(index).id();
            for (ProjectionDraft p : projections) {
                if (p.from.equals(region.name)) {
                    p.connections.removeIf(k -> source(k) == id);
                }
                if (p.to.equals(region.name)) {
                    p.connections.removeIf(k -> target(k) == id);
                }
            }
        }

        /**
         * Removes a region with all its neurons and projections.
         *
         * @param name the region's name
         */
        public void removeRegion(String name) {
            regions.removeIf(r -> r.name.equals(name));
            projections.removeIf(p -> p.from.equals(name) || p.to.equals(name));
            outputs.remove(name);
        }

        /**
         * Makes the genome. Connections to neurons that no longer exist and empty
         * projections between missing regions are dropped.
         *
         * @return the genome
         */
        public Genome freeze() {
            List<Region> rs = new ArrayList<>();
            for (RegionDraft r : regions) {
                rs.add(new Region(r.name, r.neurons));
                for (Neuron n : r.neurons) {
                    reserve(n.id());
                }
            }
            List<Projection> ps = new ArrayList<>();
            for (ProjectionDraft p : projections) {
                if (!p.from.equals(INPUT) && region(p.from) == null || region(p.to) == null) {
                    continue;
                }
                java.util.Set<Integer> from = new java.util.HashSet<>();
                for (int id : ids(p.from)) {
                    from.add(id);
                }
                java.util.Set<Integer> to = new java.util.HashSet<>();
                for (int id : ids(p.to)) {
                    to.add(id);
                }
                long[] c = p.connections.stream().filter(k -> from.contains(source(k)) && to.contains(target(k)))
                        .mapToLong(Long::longValue).toArray();
                ps.add(new Projection(p.from, p.to, c));
            }
            List<String> outs = outputs.stream().filter(o -> region(o) != null).distinct().toList();
            return new Genome(inputs, rs, ps, outs, Math.max(1, ticks), learningRate, optimizer, nextId);
        }
    }
}
