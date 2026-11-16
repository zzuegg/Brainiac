package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * A network of named <b>regions</b> of neurons joined by <b>sparse projections</b> that can
 * grow and be pruned while it learns.
 *
 * <pre>{@code
 * Brain.Spec brain = Brain.builder()
 *         .region("sensory", 128).initialNeurons(16).activation(Activation.RELU)
 *         .region("association", 256).initialNeurons(16).activation(Activation.TANH)
 *         .connect("input", "sensory", 0.3)          // 30% of possible connections
 *         .connect("sensory", "association", 0.2)
 *         .connect("association", "sensory", 0.1)    // feedback, read from the previous tick
 *         .connect("association", "association", 0.1) // lateral (recurrent)
 *         .output("association")
 *         .ticks(3)                                   // let activity settle for 3 steps
 *         .build();
 * Model model = Model.sequential().input(64).add(brain).dense(10, Activation.SOFTMAX).build();
 * }</pre>
 *
 * <p>Every region has a fixed <em>capacity</em> of neurons, of which some are active; the
 * rest are dormant and output zero. Every projection is a weight matrix with a mask that
 * says which connections exist. So tensor shapes never change, and a growing brain trains
 * with any optimizer, on any device, and saves like any other layer.
 *
 * <p>Regions are updated in the order they were declared. A projection from an earlier
 * region (or the input) is feed-forward and sees this tick's activity; one from the same
 * or a later region is recurrent and sees the previous tick's. With one tick, recurrent
 * projections have no effect.
 *
 * <p>Structural changes, usually driven by a {@code Plasticity} listener during training:
 * <ul>
 *   <li>{@link #prune(double)} removes the weakest connections;</li>
 *   <li>{@link #grow(int, int, boolean, SplittableRandom)} adds connections, where the loss
 *       gradient is largest (as in RigL) or at random (as in SET);</li>
 *   <li>{@link #addNeurons(String, int, SplittableRandom)} wakes dormant neurons and wires
 *       them in at each projection's density (neurogenesis);</li>
 *   <li>{@link #removeDeadNeurons(double)} puts neurons that never fire back to sleep.</li>
 * </ul>
 */
public final class Brain implements Layer {

    /** The name of the brain's input in {@link Builder#connect}. */
    public static final String INPUT = "input";

    /**
     * A region's settings.
     *
     * @param name its name
     * @param capacity the most neurons it can have
     * @param initial how many are active at the start
     * @param activation the neurons' activation
     */
    public record RegionSpec(String name, int capacity, int initial, Activation activation) {
    }

    /**
     * A projection's settings.
     *
     * @param from the source region, or {@link #INPUT}
     * @param to the target region
     * @param density the fraction of possible connections (between active neurons) to make
     */
    public record ProjectionSpec(String from, String to, double density) {
    }

    /**
     * A region's current size.
     *
     * @param name the region
     * @param active active neurons
     * @param capacity the most neurons it can have
     */
    public record RegionStats(String name, int active, int capacity) {
    }

    /**
     * A projection's current wiring.
     *
     * @param from the source
     * @param to the target
     * @param connections existing connections
     * @param possible connections possible between the active neurons at both ends
     */
    public record ProjectionStats(String from, String to, long connections, long possible) {
        /** @return connections divided by possible ones */
        public double density() {
            return possible == 0 ? 0 : (double) connections / possible;
        }
    }

    private final int inputs;
    private final List<RegionSpec> regionSpecs;
    private final List<ProjectionSpec> projectionSpecs;
    private final List<String> outputs;
    private final int ticks;
    private final Var[] biases;
    private final Var[] neuronMasks;
    private final Var[] weights;
    private final Var[] masks;
    private final int[] source;   // region index, or -1 for the input
    private final int[] target;
    private Var[] probes;
    private double[][] gradientEma;
    private double[][] activity;
    private boolean tracking;

    private Brain(int inputs, List<RegionSpec> regions, List<ProjectionSpec> projections, List<String> outputs,
                  int ticks, SplittableRandom random) {
        this.inputs = inputs;
        this.regionSpecs = List.copyOf(regions);
        this.projectionSpecs = List.copyOf(projections);
        this.outputs = List.copyOf(outputs);
        this.ticks = ticks;
        int r = regions.size();
        int p = projections.size();
        biases = new Var[r];
        neuronMasks = new Var[r];
        weights = new Var[p];
        masks = new Var[p];
        source = new int[p];
        target = new int[p];
        activity = new double[r][];
        for (int i = 0; i < r; i++) {
            RegionSpec s = regions.get(i);
            biases[i] = Var.param(Tensor.zeros(Shape.of(1, s.capacity())));
            float[] m = new float[s.capacity()];
            Arrays.fill(m, 0, s.initial(), 1);
            neuronMasks[i] = Var.of(Tensor.of(Shape.of(1, s.capacity()), m));
            activity[i] = new double[s.capacity()];
        }
        for (int j = 0; j < p; j++) {
            ProjectionSpec s = projections.get(j);
            source[j] = s.from().equals(INPUT) ? -1 : regionIndex(s.from());
            target[j] = regionIndex(s.to());
            weights[j] = Var.param(Tensor.zeros(Shape.of(sourceSize(j), regions.get(target[j]).capacity())));
            masks[j] = Var.of(Tensor.zeros(Shape.of(sourceSize(j), regions.get(target[j]).capacity())));
        }
        if (random != null) {
            for (int j = 0; j < p; j++) {
                wire(j, random);
            }
        }
    }

    private int regionIndex(String name) {
        for (int i = 0; i < regionSpecs.size(); i++) {
            if (regionSpecs.get(i).name().equals(name)) {
                return i;
            }
        }
        throw new IllegalArgumentException("no region named " + name);
    }

    private int sourceSize(int projection) {
        return source[projection] < 0 ? inputs : regionSpecs.get(source[projection]).capacity();
    }

    private boolean[] active(int region) {
        if (region < 0) {
            boolean[] all = new boolean[inputs];
            Arrays.fill(all, true);
            return all;
        }
        float[] m = neuronMasks[region].value().to(DType.FLOAT32).toArray();
        boolean[] a = new boolean[m.length];
        for (int i = 0; i < m.length; i++) {
            a[i] = m[i] != 0;
        }
        return a;
    }

    private static int count(boolean[] a) {
        int n = 0;
        for (boolean b : a) {
            n += b ? 1 : 0;
        }
        return n;
    }

    private float initScale(int projection) {
        int activeSources = count(active(source[projection]));
        double fanIn = Math.max(1, activeSources * projectionSpecs.get(projection).density());
        // incoming projections of the same region share the variance budget
        long incoming = Arrays.stream(target).filter(t -> t == target[projection]).count();
        double gain = regionSpecs.get(target[projection]).activation() == Activation.RELU ? 2 : 1;
        return (float) Math.sqrt(gain / (fanIn * incoming));
    }

    /** Makes the projection's starting connections between active neurons at its density. */
    private void wire(int j, SplittableRandom random) {
        boolean[] from = active(source[j]);
        boolean[] to = active(target[j]);
        int cols = to.length;
        float[] w = weights[j].value().to(DType.FLOAT32).toArray();
        float[] m = masks[j].value().to(DType.FLOAT32).toArray();
        double density = projectionSpecs.get(j).density();
        float scale = initScale(j);
        for (int a = 0; a < from.length; a++) {
            for (int b = 0; b < cols; b++) {
                if (from[a] && to[b] && random.nextDouble() < density) {
                    m[a * cols + b] = 1;
                    w[a * cols + b] = (float) (random.nextGaussian() * scale);
                }
            }
        }
        set(weights[j], w);
        set(masks[j], m);
    }

    private static void set(Var v, float[] values) {
        Tensor old = v.value();
        v.assign(Tensor.of(old.shape(), values).to(old.dtype()).to(old.device()));
    }

    /** @return a builder */
    public static Builder builder() {
        return new Builder();
    }

    /** Collects regions and projections; build a {@link Spec} with {@link #build()}. */
    public static final class Builder {
        private final List<RegionSpec> regions = new ArrayList<>();
        private final List<ProjectionSpec> projections = new ArrayList<>();
        private final List<String> outputs = new ArrayList<>();
        private int ticks = 1;

        private Builder() {
        }

        /**
         * Adds a region whose neurons are all active, with {@code tanh}.
         *
         * @param name a unique name, not {@code input}
         * @param capacity the most neurons it can have
         * @return this builder
         */
        public Builder region(String name, int capacity) {
            Objects.requireNonNull(name, "name");
            if (name.equals(INPUT) || regions.stream().anyMatch(r -> r.name().equals(name))) {
                throw new IllegalArgumentException("region name " + name + " is taken");
            }
            if (capacity < 1) {
                throw new IllegalArgumentException("capacity must be at least 1");
            }
            regions.add(new RegionSpec(name, capacity, capacity, Activation.TANH));
            return this;
        }

        private RegionSpec last() {
            if (regions.isEmpty()) {
                throw new IllegalStateException("add a region first");
            }
            return regions.removeLast();
        }

        /**
         * @param n how many neurons of the last region start active; the rest can grow
         * @return this builder
         */
        public Builder initialNeurons(int n) {
            RegionSpec r = last();
            if (n < 0 || n > r.capacity()) {
                regions.add(r);
                throw new IllegalArgumentException("initial neurons must be in [0, " + r.capacity() + "]");
            }
            regions.add(new RegionSpec(r.name(), r.capacity(), n, r.activation()));
            return this;
        }

        /**
         * @param activation the last region's activation (parameter-free activations only)
         * @return this builder
         */
        public Builder activation(Activation activation) {
            RegionSpec r = last();
            regions.add(new RegionSpec(r.name(), r.capacity(), r.initial(), Objects.requireNonNull(activation)));
            return this;
        }

        /**
         * Connects two regions sparsely.
         *
         * @param from a region or {@link #INPUT}
         * @param to a region
         * @param density the fraction of possible connections to make, in {@code (0, 1]}
         * @return this builder
         */
        public Builder connect(String from, String to, double density) {
            if (!(density > 0 && density <= 1)) {
                throw new IllegalArgumentException("density must be in (0, 1], got " + density);
            }
            projections.add(new ProjectionSpec(from, to, density));
            return this;
        }

        /**
         * @param names the regions whose activity is the brain's output, side by side
         * @return this builder
         */
        public Builder output(String... names) {
            outputs.addAll(List.of(names));
            return this;
        }

        /**
         * @param n how many update steps each input gets (at least 1)
         * @return this builder
         */
        public Builder ticks(int n) {
            if (n < 1) {
                throw new IllegalArgumentException("ticks must be at least 1");
            }
            this.ticks = n;
            return this;
        }

        /** @return the layer spec */
        public Spec build() {
            if (regions.isEmpty()) {
                throw new IllegalStateException("a brain needs at least one region");
            }
            List<String> names = regions.stream().map(RegionSpec::name).toList();
            for (ProjectionSpec p : projections) {
                if (!p.from().equals(INPUT) && !names.contains(p.from()) || !names.contains(p.to())) {
                    throw new IllegalArgumentException("unknown region in " + p.from() + " -> " + p.to());
                }
            }
            if (projections.stream().noneMatch(p -> p.from().equals(INPUT))) {
                throw new IllegalStateException("connect the input to at least one region");
            }
            List<String> out = outputs.isEmpty() ? List.of(names.getLast()) : List.copyOf(outputs);
            for (String o : out) {
                if (!names.contains(o)) {
                    throw new IllegalArgumentException("unknown output region " + o);
                }
            }
            return new Spec(List.copyOf(regions), List.copyOf(projections), out, ticks);
        }
    }

    /**
     * Describes a brain; its input must be a flat vector.
     *
     * @param regions the regions, in update order
     * @param projections the projections
     * @param outputs the output regions
     * @param ticks update steps per input
     */
    public record Spec(List<RegionSpec> regions, List<ProjectionSpec> projections, List<String> outputs, int ticks)
            implements LayerSpec {

        @Override
        public Shape outputShape(Shape input) {
            if (input.rank() != 1) {
                throw new IllegalArgumentException("a brain takes flat input; flatten() first, got " + input);
            }
            int size = 0;
            for (String o : outputs) {
                size += regions.stream().filter(r -> r.name().equals(o)).findFirst().orElseThrow().capacity();
            }
            return Shape.of(size);
        }

        @Override
        public Brain create(Shape input, SplittableRandom random) {
            outputShape(input);
            return new Brain(input.dim(0), regions, projections, outputs, ticks, random);
        }
    }

    // ---------------------------------------------------------------- forward

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        int r = regionSpecs.size();
        Var[] state = new Var[r];
        for (int t = 0; t < ticks; t++) {
            Var[] next = new Var[r];
            for (int i = 0; i < r; i++) {
                Var pre = null;
                for (int j = 0; j < weights.length; j++) {
                    if (target[j] != i) {
                        continue;
                    }
                    Var src = source[j] < 0 ? input : source[j] < i ? next[source[j]] : state[source[j]];
                    if (src == null) {
                        continue;
                    }
                    Var w = weights[j].mul(masks[j].value());
                    if (tracking && training) {
                        w = w.add(probe(j));
                    }
                    Var term = src.matmul(w);
                    pre = pre == null ? term : pre.add(term);
                }
                Var b = biases[i];
                pre = pre == null ? b.mul(Tensor.full(Shape.of(input.shape().dim(0), 1), 1, b.value().dtype(),
                        b.value().device())) : pre.add(b);
                next[i] = regionSpecs.get(i).activation().forward(pre).mul(neuronMasks[i].value());
            }
            state = next;
        }
        if (tracking && training) {
            for (int i = 0; i < r; i++) {
                float[] mean = state[i].value().abs().mean(0, false).to(DType.FLOAT32).toArray();
                for (int k = 0; k < mean.length; k++) {
                    activity[i][k] = 0.9 * activity[i][k] + 0.1 * mean[k];
                }
            }
        }
        List<Var> outs = new ArrayList<>();
        for (String o : outputs) {
            outs.add(state[regionIndex(o)]);
        }
        return outs.size() == 1 ? outs.get(0) : Var.concat(1, outs.toArray(Var[]::new));
    }

    private Var probe(int j) {
        if (probes == null) {
            probes = new Var[weights.length];
            gradientEma = new double[weights.length][];
        }
        Tensor w = weights[j].value();
        if (probes[j] == null || !probes[j].value().device().equals(w.device())
                || probes[j].value().dtype() != w.dtype()) {
            probes[j] = Var.param(Tensor.zeros(w.shape(), w.dtype(), w.device()));
            if (gradientEma[j] == null) {
                gradientEma[j] = new double[Math.toIntExact(w.shape().size())];
            }
        }
        return probes[j];
    }

    @Override
    public Shape outputShape(Shape input) {
        int size = 0;
        for (String o : outputs) {
            size += regionSpecs.get(regionIndex(o)).capacity();
        }
        return Shape.of(size);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(Arrays.asList(weights));
        p.addAll(Arrays.asList(biases));
        return p;
    }

    @Override
    public List<Var> buffers() {
        List<Var> b = new ArrayList<>(Arrays.asList(masks));
        b.addAll(Arrays.asList(neuronMasks));
        return b;
    }

    // ---------------------------------------------------------------- explicit wiring

    /**
     * Which connections of a projection exist, row by row: entry {@code a * targetSize + b}
     * is the connection from source neuron {@code a} to target neuron {@code b}.
     *
     * @param projection the projection's index, in the order they were connected
     * @return one flag per possible connection
     */
    public boolean[] connectionMask(int projection) {
        float[] m = masks[projection].value().to(DType.FLOAT32).toArray();
        boolean[] out = new boolean[m.length];
        for (int k = 0; k < m.length; k++) {
            out[k] = m[k] != 0;
        }
        return out;
    }

    /**
     * Replaces a projection's connections with exactly the given ones, such as a wiring
     * found by an evolutionary search. New weights are random, scaled like the starting
     * wiring; connections that are off get weight 0.
     *
     * @param projection the projection's index, in the order they were connected
     * @param mask one flag per possible connection, row by row, as in {@link #connectionMask}
     * @param random the random source for the new weights
     */
    public void setConnections(int projection, boolean[] mask, SplittableRandom random) {
        int size = sourceSize(projection) * regionSpecs.get(target[projection]).capacity();
        if (mask.length != size) {
            throw new IllegalArgumentException("projection " + projection + " has " + size
                    + " possible connections, got a mask of " + mask.length);
        }
        float scale = initScale(projection);
        float[] w = new float[size];
        float[] m = new float[size];
        for (int k = 0; k < size; k++) {
            if (mask[k]) {
                m[k] = 1;
                w[k] = (float) (random.nextGaussian() * scale);
            }
        }
        set(weights[projection], w);
        set(masks[projection], m);
    }

    /**
     * A projection's weights, row by row as in {@link #connectionMask}.
     *
     * @param projection the projection's index
     * @return a copy of the weights
     */
    public float[] weightValues(int projection) {
        return weights[projection].value().to(DType.FLOAT32).toArray();
    }

    /**
     * Overwrites a projection's weights; entries whose connection is off are kept at 0.
     *
     * @param projection the projection's index
     * @param values the weights, row by row as in {@link #connectionMask}
     */
    public void setWeightValues(int projection, float[] values) {
        float[] m = masks[projection].value().to(DType.FLOAT32).toArray();
        if (values.length != m.length) {
            throw new IllegalArgumentException("projection " + projection + " has " + m.length
                    + " weights, got " + values.length);
        }
        float[] w = values.clone();
        for (int k = 0; k < w.length; k++) {
            if (m[k] == 0) {
                w[k] = 0;
            }
        }
        set(weights[projection], w);
    }

    /**
     * A region's biases, one per neuron of its capacity.
     *
     * @param region the region's name
     * @return a copy of the biases
     */
    public float[] biasValues(String region) {
        return biases[regionIndex(region)].value().to(DType.FLOAT32).toArray();
    }

    /**
     * Overwrites a region's biases.
     *
     * @param region the region's name
     * @param values one bias per neuron of its capacity
     */
    public void setBiasValues(String region, float[] values) {
        set(biases[regionIndex(region)], values);
    }

    // ---------------------------------------------------------------- plasticity

    /**
     * Turns on tracking of neuron activity and of the loss gradient for missing
     * connections, which {@link #grow} with {@code byGradient} and
     * {@link #removeDeadNeurons} use. Call {@link #collectGradients()} after every
     * backward pass while it is on.
     *
     * @param on whether to track
     */
    public void track(boolean on) {
        this.tracking = on;
    }

    /**
     * Folds the gradients of the last backward pass into a running average per possible
     * connection, then clears them.
     */
    public void collectGradients() {
        if (probes == null) {
            return;
        }
        for (int j = 0; j < probes.length; j++) {
            if (probes[j] == null || probes[j].grad() == null) {
                continue;
            }
            float[] g = probes[j].grad().to(DType.FLOAT32).toArray();
            double[] e = gradientEma[j];
            for (int k = 0; k < g.length; k++) {
                e[k] = 0.9 * e[k] + 0.1 * Math.abs(g[k]);
            }
            probes[j].zeroGrad();
        }
    }

    /**
     * Removes the weakest connections (smallest absolute weight) of every projection.
     *
     * @param fraction the fraction of each projection's connections to remove
     * @return how many were removed in each projection
     */
    public int[] prune(double fraction) {
        int[] removed = new int[weights.length];
        for (int j = 0; j < weights.length; j++) {
            float[] w = weights[j].value().to(DType.FLOAT32).toArray();
            float[] m = masks[j].value().to(DType.FLOAT32).toArray();
            List<Integer> on = new ArrayList<>();
            for (int k = 0; k < m.length; k++) {
                if (m[k] != 0) {
                    on.add(k);
                }
            }
            int n = (int) Math.floor(on.size() * fraction);
            if (n == 0) {
                continue;
            }
            on.sort((a, b) -> Float.compare(Math.abs(w[a]), Math.abs(w[b])));
            for (int k = 0; k < n; k++) {
                m[on.get(k)] = 0;
                w[on.get(k)] = 0;
            }
            set(weights[j], w);
            set(masks[j], m);
            removed[j] = n;
        }
        return removed;
    }

    /**
     * Adds connections between active neurons in one projection: where the tracked loss
     * gradient is largest, or at random. New connections start at 0 when chosen by
     * gradient (so they do not disturb the output until trained), else small and random.
     *
     * @param projection the projection's index, in the order they were connected
     * @param count how many to add
     * @param byGradient choose by gradient (needs {@link #track}) instead of at random
     * @param random the random source
     * @return how many were added (fewer if the projection is full)
     */
    public int grow(int projection, int count, boolean byGradient, SplittableRandom random) {
        int j = projection;
        boolean[] from = active(source[j]);
        boolean[] to = active(target[j]);
        int cols = to.length;
        float[] w = weights[j].value().to(DType.FLOAT32).toArray();
        float[] m = masks[j].value().to(DType.FLOAT32).toArray();
        List<Integer> free = new ArrayList<>();
        for (int a = 0; a < from.length; a++) {
            for (int b = 0; b < cols; b++) {
                if (from[a] && to[b] && m[a * cols + b] == 0) {
                    free.add(a * cols + b);
                }
            }
        }
        int n = Math.min(count, free.size());
        if (n == 0) {
            return 0;
        }
        double[] score = byGradient && gradientEma != null ? gradientEma[j] : null;
        if (score != null) {
            free.sort((a, b) -> Double.compare(score[b], score[a]));
        } else {
            for (int i = free.size() - 1; i > 0; i--) {
                int k = random.nextInt(i + 1);
                Integer tmp = free.get(i);
                free.set(i, free.get(k));
                free.set(k, tmp);
            }
        }
        float scale = initScale(j);
        for (int i = 0; i < n; i++) {
            int k = free.get(i);
            m[k] = 1;
            w[k] = score != null ? 0 : (float) (random.nextGaussian() * scale * 0.1);
        }
        set(weights[j], w);
        set(masks[j], m);
        return n;
    }

    /**
     * Wakes dormant neurons in a region and wires them to the active neurons around them,
     * at each projection's density.
     *
     * @param region the region's name
     * @param count how many neurons to add
     * @param random the random source
     * @return how many were added (fewer if the region is full)
     */
    public int addNeurons(String region, int count, SplittableRandom random) {
        int i = regionIndex(region);
        float[] mask = neuronMasks[i].value().to(DType.FLOAT32).toArray();
        List<Integer> woken = new ArrayList<>();
        for (int k = 0; k < mask.length && woken.size() < count; k++) {
            if (mask[k] == 0) {
                mask[k] = 1;
                woken.add(k);
                activity[i][k] = 1;     // give it time before it can count as dead
            }
        }
        if (woken.isEmpty()) {
            return 0;
        }
        set(neuronMasks[i], mask);
        for (int j = 0; j < weights.length; j++) {
            if (source[j] != i && target[j] != i) {
                continue;
            }
            boolean[] from = active(source[j]);
            boolean[] to = active(target[j]);
            int cols = to.length;
            float[] w = weights[j].value().to(DType.FLOAT32).toArray();
            float[] m = masks[j].value().to(DType.FLOAT32).toArray();
            double density = projectionSpecs.get(j).density();
            float scale = initScale(j);
            for (int k : woken) {
                if (target[j] == i) {
                    for (int a = 0; a < from.length; a++) {
                        if (from[a] && random.nextDouble() < density) {
                            m[a * cols + k] = 1;
                            w[a * cols + k] = (float) (random.nextGaussian() * scale);
                        }
                    }
                }
                if (source[j] == i) {
                    for (int b = 0; b < cols; b++) {
                        if (to[b] && random.nextDouble() < density) {
                            m[k * cols + b] = 1;
                            w[k * cols + b] = (float) (random.nextGaussian() * scale * 0.1);
                        }
                    }
                }
            }
            set(weights[j], w);
            set(masks[j], m);
        }
        return woken.size();
    }

    /**
     * Puts neurons whose average absolute activity during training stayed below a
     * threshold back to sleep and removes their connections. Needs {@link #track}.
     *
     * @param threshold the activity below which a neuron counts as dead
     * @return how many were removed per region
     */
    public Map<String, Integer> removeDeadNeurons(double threshold) {
        Map<String, Integer> removed = new LinkedHashMap<>();
        for (int i = 0; i < regionSpecs.size(); i++) {
            float[] mask = neuronMasks[i].value().to(DType.FLOAT32).toArray();
            List<Integer> dead = new ArrayList<>();
            int alive = 0;
            for (int k = 0; k < mask.length; k++) {
                if (mask[k] != 0) {
                    alive++;
                    if (activity[i][k] < threshold) {
                        dead.add(k);
                    }
                }
            }
            if (dead.size() >= alive) {
                dead = dead.subList(0, Math.max(0, alive - 1));   // keep at least one neuron
            }
            if (dead.isEmpty()) {
                continue;
            }
            for (int k : dead) {
                mask[k] = 0;
            }
            set(neuronMasks[i], mask);
            for (int j = 0; j < weights.length; j++) {
                if (source[j] != i && target[j] != i) {
                    continue;
                }
                int cols = regionSpecs.get(target[j]).capacity();
                int rows = sourceSize(j);
                float[] w = weights[j].value().to(DType.FLOAT32).toArray();
                float[] m = masks[j].value().to(DType.FLOAT32).toArray();
                for (int k : dead) {
                    if (target[j] == i) {
                        for (int a = 0; a < rows; a++) {
                            m[a * cols + k] = 0;
                            w[a * cols + k] = 0;
                        }
                    }
                    if (source[j] == i) {
                        for (int b = 0; b < cols; b++) {
                            m[k * cols + b] = 0;
                            w[k * cols + b] = 0;
                        }
                    }
                }
                set(weights[j], w);
                set(masks[j], m);
            }
            removed.put(regionSpecs.get(i).name(), dead.size());
        }
        return removed;
    }

    /**
     * How strongly the loss wants more connections into each region: the mean tracked
     * gradient over its missing incoming connections. Needs {@link #track}.
     *
     * @return the pressure per region, by name
     */
    public Map<String, Double> growthPressure() {
        Map<String, Double> p = new LinkedHashMap<>();
        for (int i = 0; i < regionSpecs.size(); i++) {
            double sum = 0;
            long n = 0;
            for (int j = 0; j < weights.length; j++) {
                if (target[j] != i || gradientEma == null || gradientEma[j] == null) {
                    continue;
                }
                for (double g : gradientEma[j]) {
                    sum += g;
                    n++;
                }
            }
            p.put(regionSpecs.get(i).name(), n == 0 ? 0 : sum / n);
        }
        return p;
    }

    // ---------------------------------------------------------------- inspection

    /** @return every region's current size, in update order */
    public List<RegionStats> regions() {
        List<RegionStats> s = new ArrayList<>();
        for (int i = 0; i < regionSpecs.size(); i++) {
            s.add(new RegionStats(regionSpecs.get(i).name(), count(active(i)), regionSpecs.get(i).capacity()));
        }
        return s;
    }

    /** @return every projection's current wiring, in the order they were connected */
    public List<ProjectionStats> projections() {
        List<ProjectionStats> s = new ArrayList<>();
        for (int j = 0; j < weights.length; j++) {
            float[] m = masks[j].value().to(DType.FLOAT32).toArray();
            long on = 0;
            for (float v : m) {
                on += v != 0 ? 1 : 0;
            }
            s.add(new ProjectionStats(projectionSpecs.get(j).from(), projectionSpecs.get(j).to(), on,
                    (long) count(active(source[j])) * count(active(target[j]))));
        }
        return s;
    }

    /** @return the existing connections in all projections */
    public long connections() {
        return projections().stream().mapToLong(ProjectionStats::connections).sum();
    }

    /** @return the active neurons in all regions */
    public int activeNeurons() {
        return regions().stream().mapToInt(RegionStats::active).sum();
    }

    /** @return the settings this brain was built from */
    public Spec spec() {
        return new Spec(regionSpecs, projectionSpecs, outputs, ticks);
    }

    /** @return a table of regions and projections */
    public String describe() {
        StringBuilder sb = new StringBuilder();
        for (RegionStats r : regions()) {
            sb.append(String.format("  region %-14s %5d / %d neurons%n", r.name(), r.active(), r.capacity()));
        }
        for (ProjectionStats p : projections()) {
            sb.append(String.format("  %-14s -> %-14s %7d connections (%.1f%% of possible)%n", p.from(), p.to(),
                    p.connections(), 100 * p.density()));
        }
        return sb.toString();
    }

    @Override
    public String toString() {
        return "Brain(" + regionSpecs.size() + " regions, " + activeNeurons() + " neurons, " + connections()
                + " connections" + (ticks > 1 ? ", " + ticks + " ticks" : "") + ")";
    }

    Device device() {
        return weights[0].value().device();
    }

    static Config write(Brain b) {
        Config.Builder c = Config.builder().put("inputs", b.inputs).put("ticks", b.ticks)
                .put("regions", b.regionSpecs.size()).put("projections", b.projectionSpecs.size())
                .put("outputs", String.join(",", b.outputs));
        for (int i = 0; i < b.regionSpecs.size(); i++) {
            RegionSpec r = b.regionSpecs.get(i);
            c.put("region" + i + ".name", r.name()).put("region" + i + ".capacity", r.capacity())
                    .put("region" + i + ".initial", r.initial()).put("region" + i + ".activation", r.activation());
        }
        for (int j = 0; j < b.projectionSpecs.size(); j++) {
            ProjectionSpec p = b.projectionSpecs.get(j);
            c.put("projection" + j + ".from", p.from()).put("projection" + j + ".to", p.to())
                    .put("projection" + j + ".density", (float) p.density());
        }
        return c.build();
    }

    static Brain read(Config c) {
        List<RegionSpec> regions = new ArrayList<>();
        for (int i = 0; i < c.getInt("regions"); i++) {
            regions.add(new RegionSpec(c.getString("region" + i + ".name"), c.getInt("region" + i + ".capacity"),
                    c.getInt("region" + i + ".initial"), c.getActivation("region" + i + ".activation")));
        }
        List<ProjectionSpec> projections = new ArrayList<>();
        for (int j = 0; j < c.getInt("projections"); j++) {
            projections.add(new ProjectionSpec(c.getString("projection" + j + ".from"),
                    c.getString("projection" + j + ".to"), c.getFloat("projection" + j + ".density")));
        }
        return new Brain(c.getInt("inputs"), regions, projections, List.of(c.getString("outputs").split(",")),
                c.getInt("ticks"), null);
    }
}
