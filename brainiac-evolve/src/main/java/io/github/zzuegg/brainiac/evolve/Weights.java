package io.github.zzuegg.brainiac.evolve;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Brain;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Trained weights of a design, stored by neuron id instead of by position, so a child can
 * start from what its parents learned: a connection, bias or output weight the child shares
 * with a parent (same neuron ids) gets the parent's trained value; everything new starts
 * random as usual.
 *
 * <p>Weights of a fixed front end are not carried over.
 */
public final class Weights {

    /** No weights: every child starts from scratch. */
    public static final Weights NONE = new Weights(Map.of(), Map.of(), Map.of(), null, Map.of(), Map.of());

    private final Map<Long, Float> connections;
    private final Map<Integer, Float> biases;
    private final Map<Long, Float> head;
    private final float[] headBias;
    /** Control weights of switching neurons, by connection. */
    private final Map<Long, Float> controls;
    /** Per switching neuron: "id/b" its control bias, "id/u/act" and "id/e/act" per candidate. */
    private final Map<String, Float> switching;

    Weights(Map<Long, Float> connections, Map<Integer, Float> biases, Map<Long, Float> head, float[] headBias,
            Map<Long, Float> controls, Map<String, Float> switching) {
        this.connections = connections;
        this.biases = biases;
        this.head = head;
        this.headBias = headBias;
        this.controls = controls;
        this.switching = switching;
    }

    /** The key of a switching neuron's control bias. */
    static String biasKey(int id) {
        return id + "/b";
    }

    /** The key of a switching neuron's scale ({@code u}) or offset ({@code e}) for a candidate. */
    static String candidateKey(int id, char which, io.github.zzuegg.brainiac.nn.Activation a) {
        return id + "/" + which + "/" + a;
    }

    /**
     * Reads a switching region's control weights and own parameters into the maps.
     *
     * @param region the region
     * @param from per incoming projection, the ids of its sources
     * @param control per incoming projection, its control weights, row by row with
     *     {@code width} entries per row
     * @param mask per incoming projection, its connections, laid out like {@code control}
     * @param own the control bias, scales and offsets, as {@link Brain#switchingValues}
     *     gives them, with {@code width} entries per neuron row
     * @param width the row length
     * @param controls where control weights go, by connection
     * @param switching where the own parameters go, by {@link #biasKey} and {@link #candidateKey}
     */
    static void readSwitching(Genome.Region region, List<int[]> from, List<float[]> control, List<boolean[]> mask,
                              float[][] own, int width, Map<Long, Float> controls, Map<String, Float> switching) {
        List<Genome.Neuron> ns = region.neurons();
        for (int p = 0; p < from.size(); p++) {
            int[] src = from.get(p);
            for (int a = 0; a < src.length; a++) {
                for (int b = 0; b < ns.size(); b++) {
                    int k = a * width + b;
                    if (mask.get(p)[k]) {
                        controls.put(Genome.key(src[a], ns.get(b).id()), control.get(p)[k]);
                    }
                }
            }
        }
        List<io.github.zzuegg.brainiac.nn.Activation> cs = region.switching().candidates();
        for (int b = 0; b < ns.size(); b++) {
            int id = ns.get(b).id();
            switching.put(biasKey(id), own[0][b]);
            for (int k = 0; k < cs.size(); k++) {
                switching.put(candidateKey(id, 'u', cs.get(k)), own[1][k * width + b]);
                switching.put(candidateKey(id, 'e', cs.get(k)), own[2][k * width + b]);
            }
        }
    }

    /** @return the connection weights by connection key; not a copy */
    Map<Long, Float> connectionValues() {
        return connections;
    }

    /** @return the number of stored connection weights */
    public int connections() {
        return connections.size();
    }

    /** @return whether nothing is stored */
    public boolean isEmpty() {
        return connections.isEmpty() && biases.isEmpty() && head.isEmpty() && controls.isEmpty();
    }

    /**
     * Combines two parents' weights: this one's where both have a value, the other's where
     * only it has one.
     *
     * @param other the other parent's weights
     * @return the combined weights
     */
    public Weights orElse(Weights other) {
        if (other.isEmpty()) {
            return this;
        }
        if (isEmpty()) {
            return other;
        }
        Map<Long, Float> c = new HashMap<>(other.connections);
        c.putAll(connections);
        Map<Integer, Float> b = new HashMap<>(other.biases);
        b.putAll(biases);
        Map<Long, Float> h = new HashMap<>(other.head);
        h.putAll(head);
        Map<Long, Float> ctl = new HashMap<>(other.controls);
        ctl.putAll(controls);
        Map<String, Float> sw = new HashMap<>(other.switching);
        sw.putAll(switching);
        return new Weights(c, b, h, headBias != null ? headBias : other.headBias, ctl, sw);
    }

    /** Combines a value both parents have; see {@link #crossover(Weights, Mix)}. */
    @FunctionalInterface
    public interface Mix {
        /**
         * @param mine this parent's value
         * @param theirs the other parent's value
         * @return the child's value
         */
        float apply(float mine, float theirs);
    }

    /**
     * Combines two parents' weights value by value: where both have one, {@code mix}
     * decides; where only one has it, the child takes it.
     *
     * @param other the other parent's weights
     * @param mix the combination of shared values
     * @return the child's weights
     */
    public Weights crossover(Weights other, Mix mix) {
        float[] hb = headBias;
        if (headBias == null) {
            hb = other.headBias;
        } else if (other.headBias != null && other.headBias.length == headBias.length) {
            hb = new float[headBias.length];
            for (int i = 0; i < hb.length; i++) {
                hb[i] = mix.apply(headBias[i], other.headBias[i]);
            }
        }
        return new Weights(merge(connections, other.connections, mix), merge(biases, other.biases, mix),
                merge(head, other.head, mix), hb, merge(controls, other.controls, mix),
                merge(switching, other.switching, mix));
    }

    /**
     * Uniform crossover: each shared value from either parent with equal chance.
     *
     * @param other the other parent's weights
     * @param random picks the parent per value
     * @return the child's weights
     */
    public Weights crossover(Weights other, java.util.random.RandomGenerator random) {
        return crossover(other, (a, b) -> random.nextBoolean() ? a : b);
    }

    /**
     * Blending: each shared value {@code (1 - t) * mine + t * theirs}.
     *
     * @param other the other parent's weights
     * @param t how far towards the other parent, usually in {@code [0, 1]}
     * @return the child's weights
     */
    public Weights blend(Weights other, float t) {
        return crossover(other, (a, b) -> (1 - t) * a + t * b);
    }

    private static <K> Map<K, Float> merge(Map<K, Float> mine, Map<K, Float> theirs, Mix mix) {
        Map<K, Float> out = new HashMap<>(theirs);
        for (Map.Entry<K, Float> e : mine.entrySet()) {
            Float t = theirs.get(e.getKey());
            out.put(e.getKey(), t == null ? e.getValue() : mix.apply(e.getValue(), t));
        }
        return out;
    }

    private static Dense head(Model model) {
        return (Dense) model.layers().getLast();
    }

    private static float[] values(Var v) {
        return v.value().to(DType.FLOAT32).toArray();
    }

    /**
     * Reads the trained weights of a model built from a genome.
     *
     * @param genome the design
     * @param model the trained model, as built by {@link Genome#build}
     * @return its weights by neuron id
     */
    public static Weights of(Genome genome, Model model) {
        Brain brain = Genome.brainOf(model);
        Map<Long, Float> c = new HashMap<>();
        int j = 0;
        for (Genome.Projection p : genome.projections()) {
            if (p.size() == 0) {
                continue;
            }
            int[] from = ids(genome, p.from());
            int[] to = ids(genome, p.to());
            boolean[] mask = brain.connectionMask(j);
            float[] w = brain.weightValues(j);
            for (int a = 0; a < from.length; a++) {
                for (int b = 0; b < to.length; b++) {
                    int k = a * to.length + b;
                    if (mask[k]) {
                        c.put(Genome.key(from[a], to[b]), w[k]);
                    }
                }
            }
            j++;
        }
        Map<Integer, Float> b = new HashMap<>();
        for (Genome.Region r : genome.regions()) {
            float[] v = brain.biasValues(r.name());
            for (int i = 0; i < r.neurons().size(); i++) {
                b.put(r.neurons().get(i).id(), v[i]);
            }
        }
        Map<Long, Float> ctl = new HashMap<>();
        Map<String, Float> sw = new HashMap<>();
        for (Genome.Region r : genome.regions()) {
            if (r.switching() == null) {
                continue;
            }
            List<int[]> from = new ArrayList<>();
            List<float[]> control = new ArrayList<>();
            List<boolean[]> mask = new ArrayList<>();
            int k = 0;
            for (Genome.Projection p : genome.projections()) {
                if (p.size() == 0) {
                    continue;
                }
                if (p.to().equals(r.name())) {
                    from.add(ids(genome, p.from()));
                    control.add(brain.controlValues(k));
                    mask.add(brain.connectionMask(k));
                }
                k++;
            }
            readSwitching(r, from, control, mask, brain.switchingValues(r.name()), r.neurons().size(), ctl, sw);
        }
        Dense dense = head(model);
        return fromHead(genome, c, b, values(dense.parameters().get(0)), values(dense.parameters().get(1)), ctl, sw);
    }

    static Weights fromHead(Genome genome, Map<Long, Float> c, Map<Integer, Float> b, float[] hw, float[] hb,
                            Map<Long, Float> controls, Map<String, Float> switching) {
        int outputs = hb.length;
        Map<Long, Float> h = new HashMap<>();
        int row = 0;
        for (String o : genome.outputs()) {
            for (Genome.Neuron n : genome.region(o).neurons()) {
                for (int k = 0; k < outputs; k++) {
                    h.put(Genome.key(n.id(), k), hw[row * outputs + k]);
                }
                row++;
            }
        }
        return new Weights(c, b, h, hb, controls, switching);
    }

    /**
     * Overwrites the weights of a freshly built model wherever this holds a value for the
     * same connection, neuron or output; the rest keep their starting values.
     *
     * @param genome the model's design
     * @param model the model, as built by {@link Genome#build}
     * @return how many values were copied
     */
    public int applyTo(Genome genome, Model model) {
        if (isEmpty()) {
            return 0;
        }
        int copied = 0;
        Brain brain = Genome.brainOf(model);
        int j = 0;
        for (Genome.Projection p : genome.projections()) {
            if (p.size() == 0) {
                continue;
            }
            int[] from = ids(genome, p.from());
            int[] to = ids(genome, p.to());
            boolean[] mask = brain.connectionMask(j);
            float[] w = brain.weightValues(j);
            for (int a = 0; a < from.length; a++) {
                for (int b = 0; b < to.length; b++) {
                    int k = a * to.length + b;
                    Float v = mask[k] ? connections.get(Genome.key(from[a], to[b])) : null;
                    if (v != null) {
                        w[k] = v;
                        copied++;
                    }
                }
            }
            brain.setWeightValues(j, w);
            if (genome.region(p.to()).switching() != null) {
                float[] cv = brain.controlValues(j);
                for (int a = 0; a < from.length; a++) {
                    for (int b = 0; b < to.length; b++) {
                        int k = a * to.length + b;
                        Float v = mask[k] ? controls.get(Genome.key(from[a], to[b])) : null;
                        if (v != null) {
                            cv[k] = v;
                            copied++;
                        }
                    }
                }
                brain.setControlValues(j, cv);
            }
            j++;
        }
        for (Genome.Region r : genome.regions()) {
            if (r.switching() == null) {
                continue;
            }
            float[][] own = brain.switchingValues(r.name());
            int width = r.neurons().size();
            List<io.github.zzuegg.brainiac.nn.Activation> cs = r.switching().candidates();
            for (int b = 0; b < width; b++) {
                int id = r.neurons().get(b).id();
                copied += copy(switching.get(biasKey(id)), own[0], b);
                for (int k = 0; k < cs.size(); k++) {
                    copied += copy(switching.get(candidateKey(id, 'u', cs.get(k))), own[1], k * width + b);
                    copied += copy(switching.get(candidateKey(id, 'e', cs.get(k))), own[2], k * width + b);
                }
            }
            brain.setSwitchingValues(r.name(), own);
        }
        for (Genome.Region r : genome.regions()) {
            float[] v = brain.biasValues(r.name());
            for (int i = 0; i < r.neurons().size(); i++) {
                Float x = biases.get(r.neurons().get(i).id());
                if (x != null) {
                    v[i] = x;
                    copied++;
                }
            }
            brain.setBiasValues(r.name(), v);
        }
        Dense dense = head(model);
        Var weight = dense.parameters().get(0);
        Var bias = dense.parameters().get(1);
        float[] hw = values(weight);
        int outputs = Math.toIntExact(bias.value().shape().size());
        int row = 0;
        for (String o : genome.outputs()) {
            for (Genome.Neuron n : genome.region(o).neurons()) {
                for (int k = 0; k < outputs; k++) {
                    Float x = head.get(Genome.key(n.id(), k));
                    if (x != null) {
                        hw[row * outputs + k] = x;
                        copied++;
                    }
                }
                row++;
            }
        }
        set(weight, hw);
        if (headBias != null && headBias.length == outputs) {
            set(bias, headBias);
        }
        return copied;
    }

    private static int copy(Float value, float[] into, int at) {
        if (value == null) {
            return 0;
        }
        into[at] = value;
        return 1;
    }

    private static void set(Var v, float[] values) {
        Tensor old = v.value();
        v.assign(Tensor.of(old.shape(), values).to(old.dtype()).to(old.device()));
    }

    static int[] ids(Genome genome, String region) {
        if (region.equals(Genome.INPUT)) {
            int[] ids = new int[genome.inputs()];
            for (int i = 0; i < ids.length; i++) {
                ids[i] = i;
            }
            return ids;
        }
        return genome.region(region).neurons().stream().mapToInt(Genome.Neuron::id).toArray();
    }
}
