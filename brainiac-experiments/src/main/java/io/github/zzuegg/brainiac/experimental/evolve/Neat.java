package io.github.zzuegg.brainiac.experimental.evolve;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.SplittableRandom;
import java.util.TreeMap;
import java.util.function.ToDoubleFunction;
import java.util.stream.IntStream;

/**
 * NeuroEvolution of Augmenting Topologies: evolves both the weights and the wiring of a
 * network, starting from the smallest one (inputs wired straight to outputs) and adding
 * nodes and connections only when they pay off.
 *
 * <ul>
 *   <li><b>Historical markings</b>: every new connection gets a global innovation number,
 *       so crossover can line up the genes of two different topologies without any
 *       graph matching.</li>
 *   <li><b>Speciation</b>: genomes are grouped by a compatibility distance (excess,
 *       disjoint and weight differences); fitness is shared inside a species, which
 *       protects new structure until its weights are tuned.</li>
 *   <li><b>Complexification</b>: add-node splits a connection (the old weight goes out,
 *       1 goes in), add-connection links two unconnected nodes without forming a
 *       cycle.</li>
 * </ul>
 *
 * <pre>{@code
 * Neat neat = Neat.of(2, 1).population(150).seed(1);
 * Neat.Genome best = neat.run(net -> xorFitness(net), 100, 15.5);
 * }</pre>
 *
 * <p>Stanley and Miikkulainen 2002, "Evolving Neural Networks through Augmenting
 * Topologies", <a href="https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf">ECJ 10(2)</a>.
 * Fitness is maximised and should be non-negative.
 */
public final class Neat {

    /** What a node does. */
    public enum NodeType {
        /** reads an input */
        INPUT,
        /** always 1 */
        BIAS,
        /** a hidden node added by mutation */
        HIDDEN,
        /** an output */
        OUTPUT
    }

    /**
     * A node gene.
     *
     * @param id the node's id, shared by every genome that has it
     * @param type its role
     * @param activation its activation (unused for inputs and the bias)
     */
    public record NodeGene(int id, NodeType type, NeatActivation activation) {
    }

    /**
     * A connection gene.
     *
     * @param innovation the historical marking
     * @param from the source node id
     * @param to the target node id
     * @param weight the weight
     * @param enabled whether it is expressed
     */
    public record ConnectionGene(int innovation, int from, int to, double weight, boolean enabled) {
        ConnectionGene withWeight(double w) {
            return new ConnectionGene(innovation, from, to, w, enabled);
        }

        ConnectionGene withEnabled(boolean e) {
            return new ConnectionGene(innovation, from, to, weight, e);
        }
    }

    /** A genome: node genes by id and connection genes by innovation number. */
    public static final class Genome {
        private final TreeMap<Integer, NodeGene> nodes;
        private final TreeMap<Integer, ConnectionGene> connections;
        private double fitness;

        Genome(TreeMap<Integer, NodeGene> nodes, TreeMap<Integer, ConnectionGene> connections) {
            this.nodes = nodes;
            this.connections = connections;
        }

        Genome copy() {
            Genome g = new Genome(new TreeMap<>(nodes), new TreeMap<>(connections));
            g.fitness = fitness;
            return g;
        }

        /** @return the node genes, by id */
        public List<NodeGene> nodes() {
            return List.copyOf(nodes.values());
        }

        /** @return the connection genes, by innovation number */
        public List<ConnectionGene> connections() {
            return List.copyOf(connections.values());
        }

        /** @return the fitness it was last given */
        public double fitness() {
            return fitness;
        }

        /** @return the number of hidden nodes */
        public int hiddenNodes() {
            return (int) nodes.values().stream().filter(n -> n.type() == NodeType.HIDDEN).count();
        }

        /** @return the number of enabled connections */
        public int enabledConnections() {
            return (int) connections.values().stream().filter(ConnectionGene::enabled).count();
        }

        /** @return the network this genome encodes */
        public Network network() {
            return new Network(this);
        }

        @Override
        public String toString() {
            return "Genome(" + hiddenNodes() + " hidden, " + enabledConnections() + " connections, fitness "
                    + fitness + ")";
        }
    }

    /** A feed-forward network decoded from a genome, evaluated in topological order. */
    public static final class Network {
        private final int inputs;
        private final int[] order;
        private final NeatActivation[] activation;
        private final int[][] sources;
        private final double[][] weights;
        private final int[] inputSlots;
        private final int biasSlot;
        private final int[] outputSlots;

        Network(Genome g) {
            Map<Integer, Integer> slot = new HashMap<>();
            List<NodeGene> all = new ArrayList<>(g.nodes.values());
            for (int i = 0; i < all.size(); i++) {
                slot.put(all.get(i).id(), i);
            }
            int n = all.size();
            List<List<Integer>> src = new ArrayList<>();
            List<List<Double>> w = new ArrayList<>();
            int[] indegree = new int[n];
            List<List<Integer>> out = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                src.add(new ArrayList<>());
                w.add(new ArrayList<>());
                out.add(new ArrayList<>());
            }
            for (ConnectionGene c : g.connections.values()) {
                if (!c.enabled()) {
                    continue;
                }
                int a = slot.get(c.from());
                int b = slot.get(c.to());
                src.get(b).add(a);
                w.get(b).add(c.weight());
                out.get(a).add(b);
                indegree[b]++;
            }
            ArrayDeque<Integer> ready = new ArrayDeque<>();
            for (int i = 0; i < n; i++) {
                if (indegree[i] == 0) {
                    ready.add(i);
                }
            }
            int[] ord = new int[n];
            int k = 0;
            while (!ready.isEmpty()) {
                int i = ready.poll();
                ord[k++] = i;
                for (int j : out.get(i)) {
                    if (--indegree[j] == 0) {
                        ready.add(j);
                    }
                }
            }
            if (k != n) {
                throw new IllegalStateException("genome has a cycle");
            }
            order = ord;
            activation = new NeatActivation[n];
            sources = new int[n][];
            weights = new double[n][];
            List<Integer> in = new ArrayList<>();
            List<Integer> outs = new ArrayList<>();
            int bias = -1;
            for (int i = 0; i < n; i++) {
                NodeGene ng = all.get(i);
                activation[i] = ng.activation();
                sources[i] = src.get(i).stream().mapToInt(Integer::intValue).toArray();
                weights[i] = w.get(i).stream().mapToDouble(Double::doubleValue).toArray();
                switch (ng.type()) {
                    case INPUT -> in.add(i);
                    case BIAS -> bias = i;
                    case OUTPUT -> outs.add(i);
                    default -> {
                    }
                }
            }
            inputs = in.size();
            inputSlots = in.stream().mapToInt(Integer::intValue).toArray();
            outputSlots = outs.stream().mapToInt(Integer::intValue).toArray();
            biasSlot = bias;
        }

        /**
         * Computes the outputs.
         *
         * @param in one value per input
         * @return one value per output
         */
        public double[] activate(double... in) {
            if (in.length != inputs) {
                throw new IllegalArgumentException("expected " + inputs + " inputs, got " + in.length);
            }
            double[] value = new double[order.length];
            boolean[] fixed = new boolean[order.length];
            for (int i = 0; i < inputs; i++) {
                value[inputSlots[i]] = in[i];
                fixed[inputSlots[i]] = true;
            }
            if (biasSlot >= 0) {
                value[biasSlot] = 1;
                fixed[biasSlot] = true;
            }
            for (int i : order) {
                if (fixed[i]) {
                    continue;
                }
                double s = 0;
                int[] src = sources[i];
                for (int j = 0; j < src.length; j++) {
                    s += weights[i][j] * value[src[j]];
                }
                value[i] = activation[i].applyAsDouble(s);
            }
            double[] out = new double[outputSlots.length];
            for (int i = 0; i < out.length; i++) {
                out[i] = value[outputSlots[i]];
            }
            return out;
        }
    }

    /**
     * One generation's summary.
     *
     * @param index the generation
     * @param bestFitness the best fitness
     * @param meanFitness the mean fitness
     * @param species the number of species
     * @param bestHidden hidden nodes of the best genome
     * @param bestConnections enabled connections of the best genome
     */
    public record Generation(int index, double bestFitness, double meanFitness, int species, int bestHidden,
            int bestConnections) {
    }

    private static final class Species {
        final int id;
        Genome representative;
        final List<Genome> members = new ArrayList<>();
        double best = Double.NEGATIVE_INFINITY;
        int lastImproved;

        Species(int id, Genome representative, int generation) {
            this.id = id;
            this.representative = representative;
            this.lastImproved = generation;
        }
    }

    private final int inputs;
    private final int outputs;
    private int populationSize = 150;
    private long seed = 1;
    private NeatActivation outputActivation = NeatActivation.SIGMOID;
    private NeatActivation[] hiddenActivations = {NeatActivation.SIGMOID};
    private double weightMutationRate = 0.8;
    private double perturbRate = 0.9;
    private double perturbPower = 0.5;
    private double addNodeRate = 0.03;
    private double addConnectionRate = 0.05;
    private double activationMutationRate = 0.1;
    private double c1 = 1;
    private double c2 = 1;
    private double c3 = 0.4;
    private double threshold = 3.0;
    private int targetSpecies = 0;
    private int stagnation = 15;
    private double survival = 0.2;
    private double crossoverRate = 0.75;
    private double interspeciesRate = 0.001;
    private double initialWeightRange = 1;
    private boolean parallel;

    private SplittableRandom random;
    private final Map<Long, Integer> innovations = new HashMap<>();
    private final Map<Integer, Integer> splitNodes = new HashMap<>();
    private int nextInnovation;
    private int nextNode;
    private List<Genome> population;
    private final List<Species> species = new ArrayList<>();
    private int nextSpecies;
    private int generation;
    private Genome best;
    private final List<Generation> history = new ArrayList<>();

    private Neat(int inputs, int outputs) {
        this.inputs = inputs;
        this.outputs = outputs;
    }

    /**
     * Starts setting up a search.
     *
     * @param inputs the number of network inputs (a bias node is added)
     * @param outputs the number of network outputs
     * @return the search, to configure
     */
    public static Neat of(int inputs, int outputs) {
        if (inputs < 1 || outputs < 1) {
            throw new IllegalArgumentException("need inputs and outputs");
        }
        return new Neat(inputs, outputs);
    }

    /**
     * Sets the population size.
     *
     * @param size the number of genomes
     * @return this
     */
    public Neat population(int size) {
        this.populationSize = size;
        return this;
    }

    /**
     * Sets the seed.
     *
     * @param s the seed
     * @return this
     */
    public Neat seed(long s) {
        this.seed = s;
        return this;
    }

    /**
     * Sets the output nodes' activation.
     *
     * @param a the activation
     * @return this
     */
    public Neat outputActivation(NeatActivation a) {
        this.outputActivation = Objects.requireNonNull(a);
        return this;
    }

    /**
     * Sets the activations a new hidden node may get; with more than one, mutation may also
     * change a hidden node's activation (as in a CPPN).
     *
     * @param list the choices
     * @return this
     */
    public Neat hiddenActivations(NeatActivation... list) {
        if (list.length == 0) {
            throw new IllegalArgumentException("need at least one activation");
        }
        this.hiddenActivations = list.clone();
        return this;
    }

    /**
     * Sets the structural mutation rates.
     *
     * @param addNode the chance a child gets a new node
     * @param addConnection the chance a child gets a new connection
     * @return this
     */
    public Neat structuralRates(double addNode, double addConnection) {
        this.addNodeRate = addNode;
        this.addConnectionRate = addConnection;
        return this;
    }

    /**
     * Sets the weight mutation settings.
     *
     * @param rate the chance a child's weights are mutated
     * @param power the standard deviation of a weight perturbation
     * @return this
     */
    public Neat weightMutation(double rate, double power) {
        this.weightMutationRate = rate;
        this.perturbPower = power;
        return this;
    }

    /**
     * Sets the compatibility distance coefficients and threshold.
     *
     * @param excess weight of excess genes
     * @param disjoint weight of disjoint genes
     * @param weightDiff weight of the mean weight difference of matching genes
     * @param thresh the distance below which two genomes share a species
     * @return this
     */
    public Neat compatibility(double excess, double disjoint, double weightDiff, double thresh) {
        this.c1 = excess;
        this.c2 = disjoint;
        this.c3 = weightDiff;
        this.threshold = thresh;
        return this;
    }

    /**
     * Adapts the compatibility threshold each generation to keep about this many species
     * (0, the default, keeps it fixed).
     *
     * @param count the wanted number of species
     * @return this
     */
    public Neat targetSpecies(int count) {
        this.targetSpecies = count;
        return this;
    }

    /**
     * Sets how many generations a species may go without improving before it is dropped.
     *
     * @param generations the patience
     * @return this
     */
    public Neat stagnation(int generations) {
        this.stagnation = generations;
        return this;
    }

    /**
     * Evaluates genomes in parallel; the fitness function must be thread-safe.
     *
     * @param on whether to evaluate in parallel
     * @return this
     */
    public Neat parallel(boolean on) {
        this.parallel = on;
        return this;
    }

    /** @return the best genome seen so far, or null before the first generation */
    public Genome best() {
        return best;
    }

    /** @return one entry per generation so far */
    public List<Generation> history() {
        return List.copyOf(history);
    }

    /** @return the current population */
    public List<Genome> population() {
        return population == null ? List.of() : List.copyOf(population);
    }

    /**
     * Evolves until the fitness target is reached or the generations run out.
     *
     * @param fitness scores a network (higher is better, non-negative)
     * @param generations the maximum number of generations
     * @param target stop as soon as a genome reaches this fitness
     * @return the best genome
     */
    public Genome run(ToDoubleFunction<Network> fitness, int generations, double target) {
        for (int g = 0; g < generations; g++) {
            step(fitness);
            if (best.fitness >= target) {
                break;
            }
        }
        return best;
    }

    /**
     * Runs one generation: evaluate, speciate, reproduce.
     *
     * @param fitness scores a network
     * @return the generation's summary
     */
    public Generation step(ToDoubleFunction<Network> fitness) {
        if (population == null) {
            initialise();
        }
        List<Genome> pop = population;
        IntStream range = IntStream.range(0, pop.size());
        (parallel ? range.parallel() : range).forEach(i -> pop.get(i).fitness = fitness.applyAsDouble(
                pop.get(i).network()));
        Genome champion = pop.stream().max(Comparator.comparingDouble(Genome::fitness)).orElseThrow();
        if (best == null || champion.fitness > best.fitness) {
            best = champion.copy();
        }
        speciate(pop);
        double mean = pop.stream().mapToDouble(Genome::fitness).average().orElse(0);
        Generation summary = new Generation(generation, champion.fitness, mean, species.size(),
                champion.hiddenNodes(), champion.enabledConnections());
        history.add(summary);
        population = reproduce();
        generation++;
        return summary;
    }

    private void initialise() {
        random = new SplittableRandom(seed);
        nextNode = 0;
        TreeMap<Integer, NodeGene> nodes = new TreeMap<>();
        for (int i = 0; i < inputs; i++) {
            nodes.put(nextNode, new NodeGene(nextNode++, NodeType.INPUT, NeatActivation.LINEAR));
        }
        nodes.put(nextNode, new NodeGene(nextNode++, NodeType.BIAS, NeatActivation.LINEAR));
        for (int o = 0; o < outputs; o++) {
            nodes.put(nextNode, new NodeGene(nextNode++, NodeType.OUTPUT, outputActivation));
        }
        population = new ArrayList<>();
        for (int p = 0; p < populationSize; p++) {
            TreeMap<Integer, ConnectionGene> conns = new TreeMap<>();
            for (int a = 0; a <= inputs; a++) {
                for (int o = 0; o < outputs; o++) {
                    int to = inputs + 1 + o;
                    int inn = innovation(a, to);
                    conns.put(inn, new ConnectionGene(inn, a, to,
                            random.nextDouble(-initialWeightRange, initialWeightRange), true));
                }
            }
            population.add(new Genome(new TreeMap<>(nodes), conns));
        }
    }

    private int innovation(int from, int to) {
        return innovations.computeIfAbsent(((long) from << 32) | (to & 0xffffffffL), k -> nextInnovation++);
    }

    /**
     * The compatibility distance of two genomes.
     *
     * @param a one genome
     * @param b another
     * @return {@code c1 E / N + c2 D / N + c3 W}
     */
    public double distance(Genome a, Genome b) {
        int maxA = a.connections.isEmpty() ? -1 : a.connections.lastKey();
        int maxB = b.connections.isEmpty() ? -1 : b.connections.lastKey();
        int excess = 0;
        int disjoint = 0;
        int matching = 0;
        double weightDiff = 0;
        for (ConnectionGene c : a.connections.values()) {
            ConnectionGene d = b.connections.get(c.innovation());
            if (d != null) {
                matching++;
                weightDiff += Math.abs(c.weight() - d.weight());
            } else if (c.innovation() > maxB) {
                excess++;
            } else {
                disjoint++;
            }
        }
        for (ConnectionGene c : b.connections.values()) {
            if (!a.connections.containsKey(c.innovation())) {
                if (c.innovation() > maxA) {
                    excess++;
                } else {
                    disjoint++;
                }
            }
        }
        int n = Math.max(a.connections.size(), b.connections.size());
        double norm = n < 20 ? 1 : n;
        return c1 * excess / norm + c2 * disjoint / norm + c3 * (matching == 0 ? 0 : weightDiff / matching);
    }

    private void speciate(List<Genome> pop) {
        for (Species s : species) {
            s.members.clear();
        }
        for (Genome g : pop) {
            Species home = null;
            for (Species s : species) {
                if (distance(g, s.representative) < threshold) {
                    home = s;
                    break;
                }
            }
            if (home == null) {
                home = new Species(nextSpecies++, g, generation);
                species.add(home);
            }
            home.members.add(g);
        }
        species.removeIf(s -> s.members.isEmpty());
        for (Species s : species) {
            double top = s.members.stream().mapToDouble(Genome::fitness).max().orElse(0);
            if (top > s.best) {
                s.best = top;
                s.lastImproved = generation;
            }
            s.representative = s.members.get(random.nextInt(s.members.size()));
        }
        if (targetSpecies > 0) {
            threshold = Math.max(0.3, threshold + (species.size() < targetSpecies ? -0.3
                    : species.size() > targetSpecies ? 0.3 : 0));
        }
    }

    private List<Genome> reproduce() {
        // drop stagnant species, but always keep the two best
        species.sort(Comparator.comparingDouble((Species s) -> -s.best));
        List<Species> alive = new ArrayList<>();
        for (int i = 0; i < species.size(); i++) {
            Species s = species.get(i);
            if (i < 2 || generation - s.lastImproved < stagnation) {
                alive.add(s);
            }
        }
        species.retainAll(alive);
        double min = Double.POSITIVE_INFINITY;
        for (Species s : species) {
            for (Genome g : s.members) {
                min = Math.min(min, g.fitness);
            }
        }
        double[] share = new double[species.size()];
        double total = 0;
        for (int i = 0; i < species.size(); i++) {
            Species s = species.get(i);
            double sum = 0;
            for (Genome g : s.members) {
                sum += g.fitness - min + 1e-6;
            }
            share[i] = sum / s.members.size();
            total += share[i];
        }
        int[] quota = new int[species.size()];
        int assigned = 0;
        for (int i = 0; i < species.size(); i++) {
            quota[i] = (int) Math.floor(share[i] / total * populationSize);
            assigned += quota[i];
        }
        for (int i = 0; assigned < populationSize; i = (i + 1) % species.size()) {
            quota[i]++;
            assigned++;
        }
        List<Genome> next = new ArrayList<>(populationSize);
        for (int i = 0; i < species.size(); i++) {
            Species s = species.get(i);
            s.members.sort(Comparator.comparingDouble((Genome g) -> -g.fitness));
            int left = quota[i];
            if (left > 0 && s.members.size() >= 5) {
                next.add(s.members.getFirst().copy());
                left--;
            }
            int parents = Math.max(1, (int) Math.ceil(s.members.size() * survival));
            for (int k = 0; k < left; k++) {
                Genome mother = s.members.get(random.nextInt(parents));
                Genome child;
                if (random.nextDouble() < crossoverRate) {
                    Genome father;
                    if (random.nextDouble() < interspeciesRate && species.size() > 1) {
                        Species other = species.get(random.nextInt(species.size()));
                        father = other.members.get(random.nextInt(other.members.size()));
                    } else {
                        father = s.members.get(random.nextInt(parents));
                    }
                    child = crossover(mother, father);
                } else {
                    child = mother.copy();
                }
                mutate(child);
                next.add(child);
            }
        }
        return next;
    }

    private Genome crossover(Genome a, Genome b) {
        if (b.fitness > a.fitness) {
            Genome t = a;
            a = b;
            b = t;
        }
        TreeMap<Integer, ConnectionGene> conns = new TreeMap<>();
        for (ConnectionGene c : a.connections.values()) {
            ConnectionGene d = b.connections.get(c.innovation());
            ConnectionGene pick = d != null && random.nextBoolean() ? d : c;
            if (d != null && (!c.enabled() || !d.enabled())) {
                pick = pick.withEnabled(random.nextDouble() >= 0.75);
            }
            conns.put(pick.innovation(), pick);
        }
        TreeMap<Integer, NodeGene> nodes = new TreeMap<>();
        for (NodeGene n : a.nodes.values()) {
            NodeGene other = b.nodes.get(n.id());
            nodes.put(n.id(), other != null && random.nextBoolean() ? other : n);
        }
        Genome child = new Genome(nodes, conns);
        if (hasCycle(child)) {
            return a.copy();
        }
        return child;
    }

    private void mutate(Genome g) {
        if (random.nextDouble() < weightMutationRate) {
            for (ConnectionGene c : new ArrayList<>(g.connections.values())) {
                double w = random.nextDouble() < perturbRate
                        ? c.weight() + random.nextGaussian() * perturbPower
                        : random.nextDouble(-2, 2);
                g.connections.put(c.innovation(), c.withWeight(Math.max(-8, Math.min(8, w))));
            }
        }
        if (hiddenActivations.length > 1 && random.nextDouble() < activationMutationRate) {
            List<NodeGene> hidden = g.nodes.values().stream().filter(n -> n.type() == NodeType.HIDDEN).toList();
            if (!hidden.isEmpty()) {
                NodeGene n = hidden.get(random.nextInt(hidden.size()));
                g.nodes.put(n.id(), new NodeGene(n.id(), NodeType.HIDDEN,
                        hiddenActivations[random.nextInt(hiddenActivations.length)]));
            }
        }
        if (random.nextDouble() < addNodeRate) {
            addNode(g);
        }
        if (random.nextDouble() < addConnectionRate) {
            addConnection(g);
        }
    }

    private void addNode(Genome g) {
        List<ConnectionGene> enabled = g.connections.values().stream().filter(ConnectionGene::enabled).toList();
        if (enabled.isEmpty()) {
            return;
        }
        ConnectionGene c = enabled.get(random.nextInt(enabled.size()));
        Integer id = splitNodes.get(c.innovation());
        if (id == null || g.nodes.containsKey(id)) {
            id = nextNode++;
            if (!splitNodes.containsKey(c.innovation())) {
                splitNodes.put(c.innovation(), id);
            }
        }
        g.connections.put(c.innovation(), c.withEnabled(false));
        g.nodes.put(id, new NodeGene(id, NodeType.HIDDEN, hiddenActivations[random.nextInt(hiddenActivations.length)]));
        int in = innovation(c.from(), id);
        int out = innovation(id, c.to());
        g.connections.put(in, new ConnectionGene(in, c.from(), id, 1, true));
        g.connections.put(out, new ConnectionGene(out, id, c.to(), c.weight(), true));
    }

    private void addConnection(Genome g) {
        List<NodeGene> nodes = new ArrayList<>(g.nodes.values());
        Set<Long> existing = new HashSet<>();
        for (ConnectionGene c : g.connections.values()) {
            existing.add(((long) c.from() << 32) | (c.to() & 0xffffffffL));
        }
        for (int attempt = 0; attempt < 30; attempt++) {
            NodeGene a = nodes.get(random.nextInt(nodes.size()));
            NodeGene b = nodes.get(random.nextInt(nodes.size()));
            if (a.type() == NodeType.OUTPUT || b.type() == NodeType.INPUT || b.type() == NodeType.BIAS
                    || a.id() == b.id() || existing.contains(((long) a.id() << 32) | (b.id() & 0xffffffffL))
                    || reaches(g, b.id(), a.id())) {
                continue;
            }
            int inn = innovation(a.id(), b.id());
            g.connections.put(inn, new ConnectionGene(inn, a.id(), b.id(), random.nextDouble(-1, 1), true));
            return;
        }
    }

    /** Whether {@code to} can be reached from {@code from} along connections (enabled or not). */
    private static boolean reaches(Genome g, int from, int to) {
        Map<Integer, List<Integer>> out = new HashMap<>();
        for (ConnectionGene c : g.connections.values()) {
            out.computeIfAbsent(c.from(), k -> new ArrayList<>()).add(c.to());
        }
        ArrayDeque<Integer> stack = new ArrayDeque<>(List.of(from));
        Set<Integer> seen = new HashSet<>();
        while (!stack.isEmpty()) {
            int n = stack.pop();
            if (n == to) {
                return true;
            }
            if (seen.add(n)) {
                stack.addAll(out.getOrDefault(n, List.of()));
            }
        }
        return false;
    }

    private static boolean hasCycle(Genome g) {
        for (ConnectionGene c : g.connections.values()) {
            if (reaches(g, c.to(), c.from())) {
                return true;
            }
        }
        return false;
    }

    /** @return the current compatibility threshold */
    public double threshold() {
        return threshold;
    }

    @Override
    public String toString() {
        return "Neat(" + inputs + " -> " + outputs + ", population " + populationSize + ", "
                + Arrays.toString(hiddenActivations) + ")";
    }
}
