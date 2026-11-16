package io.github.zzuegg.brainiac.evolve;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Brain;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.HashMap;
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
    public static final Weights NONE = new Weights(Map.of(), Map.of(), Map.of(), null);

    private final Map<Long, Float> connections;
    private final Map<Integer, Float> biases;
    private final Map<Long, Float> head;
    private final float[] headBias;

    Weights(Map<Long, Float> connections, Map<Integer, Float> biases, Map<Long, Float> head, float[] headBias) {
        this.connections = connections;
        this.biases = biases;
        this.head = head;
        this.headBias = headBias;
    }

    /** @return the number of stored connection weights */
    public int connections() {
        return connections.size();
    }

    /** @return whether nothing is stored */
    public boolean isEmpty() {
        return connections.isEmpty() && biases.isEmpty() && head.isEmpty();
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
        return new Weights(c, b, h, headBias != null ? headBias : other.headBias);
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
        Dense dense = head(model);
        return fromHead(genome, c, b, values(dense.parameters().get(0)), values(dense.parameters().get(1)));
    }

    static Weights fromHead(Genome genome, Map<Long, Float> c, Map<Integer, Float> b, float[] hw, float[] hb) {
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
        return new Weights(c, b, h, hb);
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
            j++;
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
