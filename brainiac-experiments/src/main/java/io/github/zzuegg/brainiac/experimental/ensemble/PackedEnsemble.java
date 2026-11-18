package io.github.zzuegg.brainiac.experimental.ensemble;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Packed ensembles (Laurent et al., ICLR 2023,
 * <a href="https://arxiv.org/abs/2210.09184">arXiv:2210.09184</a>): {@code M} members that
 * share nothing, run as one dense model through grouped layers. Every member is an MLP of width
 * {@code α · hidden / M}; the input is repeated once per member, the hidden layers are grouped
 * linear maps with {@code M · γ} groups (each member split into {@code γ} further groups), and
 * the head has {@code M} groups. Output {@code [n, M, classes]} from {@link #members}, the mean
 * of the members' logits from {@link #forward} (as {@code TabM} does).
 *
 * <p>{@link Grouped} is the grouped linear layer on its own, a drop-in {@link Layer} with a
 * {@link LayerSpec}.
 */
public final class PackedEnsemble implements Layer {

    /**
     * A grouped linear layer: input {@code [..., groups · in]} split into {@code groups} slices,
     * each mapped by its own {@code in -> out} matrix, output {@code [..., groups · out]}.
     */
    public static final class Grouped implements Layer {

        private final int groups;
        private final int in;
        private final int out;
        private final Var weight;
        private final Var bias;

        /**
         * Creates the layer with He-uniform weights.
         *
         * @param groups the number of groups
         * @param in input width per group
         * @param out output width per group
         * @param random the initializer
         */
        public Grouped(int groups, int in, int out, SplittableRandom random) {
            this.groups = groups;
            this.in = in;
            this.out = out;
            float[] w = new float[groups * in * out];
            double bound = Math.sqrt(6.0 / in);
            for (int i = 0; i < w.length; i++) {
                w[i] = (float) ((2 * random.nextDouble() - 1) * bound);
            }
            this.weight = Var.param(Tensor.of(Shape.of(groups, in, out), w));
            this.bias = Var.param(Tensor.zeros(Shape.of(groups * out)));
        }

        /**
         * @param groups the number of groups
         * @param out output width per group
         * @return a spec for inputs whose last dimension is a multiple of {@code groups}
         */
        public static LayerSpec spec(int groups, int out) {
            return new LayerSpec() {
                @Override
                public Shape outputShape(Shape input) {
                    int[] d = input.toArray();
                    d[d.length - 1] = groups * out;
                    return Shape.of(d);
                }

                @Override
                public Layer create(Shape input, SplittableRandom random) {
                    return new Grouped(groups, input.dim(-1) / groups, out, random);
                }
            };
        }

        @Override
        public Var forward(Var input) {
            int n = input.shape().dim(0);
            Var x = input.reshape(n, groups, in).permute(1, 0, 2);
            return x.matmul(weight).permute(1, 0, 2).reshape(n, groups * out).add(bias);
        }

        @Override
        public List<Var> parameters() {
            return List.of(weight, bias);
        }

        @Override
        public int outputSize(int inputSize) {
            return groups * out;
        }

        @Override
        public String toString() {
            return "Grouped(" + groups + " x " + in + "->" + out + ")";
        }
    }

    private final int members;
    private final int inputs;
    private final int classes;
    private final List<Grouped> layers = new ArrayList<>();

    /**
     * Creates the ensemble.
     *
     * @param inputs input features
     * @param hidden the hidden width of the equivalent single model, per layer
     * @param classes the number of classes
     * @param members the members {@code M}
     * @param alpha the width factor {@code α}
     * @param gamma the subgroups per member {@code γ} in hidden layers
     * @param seed the seed
     */
    public PackedEnsemble(int inputs, int[] hidden, int classes, int members, double alpha, int gamma, long seed) {
        this.members = members;
        this.inputs = inputs;
        this.classes = classes;
        SplittableRandom r = new SplittableRandom(seed);
        int prev = inputs;
        for (int l = 0; l < hidden.length; l++) {
            int width = Math.max(gamma, (int) Math.round(alpha * hidden[l] / members / gamma) * gamma);
            int groups = l == 0 ? members : members * gamma;
            int inPer = l == 0 ? inputs : prev / gamma;
            layers.add(new Grouped(groups, inPer, l == 0 ? width : width / gamma, r));
            prev = width;
        }
        layers.add(new Grouped(members, prev, classes, r));
    }

    /** @return the ensemble size */
    public int size() {
        return members;
    }

    /**
     * Every member's logits.
     *
     * @param input {@code [n, inputs]}
     * @param training the training flag
     * @return {@code [n, M, classes]}
     */
    public Var members(Var input, boolean training) {
        int n = input.shape().dim(0);
        Var h = input.reshape(n, 1, inputs).expand(Shape.of(n, members, inputs)).reshape(n, members * inputs);
        for (int l = 0; l < layers.size(); l++) {
            h = layers.get(l).forward(h, training);
            if (l < layers.size() - 1) {
                h = h.relu();
            }
        }
        return h.reshape(n, members, classes);
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        return members(input, training).mean(1, false);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        for (Grouped g : layers) {
            p.addAll(g.parameters());
        }
        return p;
    }

    @Override
    public int outputSize(int inputSize) {
        return classes;
    }

    /** @return the multiply-adds of one input */
    public long multiplyAdds() {
        long s = 0;
        for (Grouped g : layers) {
            s += (long) g.groups * g.in * g.out;
        }
        return s;
    }

    /**
     * Each member's probabilities.
     *
     * @return one audit member per packed member
     */
    public List<EnsembleAudit.Member> auditMembers() {
        List<EnsembleAudit.Member> out = new ArrayList<>();
        for (int i = 0; i < members; i++) {
            int m = i;
            out.add(x -> {
                Tensor z = Var.noGrad(() -> members(Var.of(x), false).value());
                return EnsembleAudit.softmaxRows(z.slice(1, m, m + 1).reshape(z.shape().dim(0), classes));
            });
        }
        return out;
    }

    @Override
    public String toString() {
        return "PackedEnsemble(M=" + members + ", " + layers + ")";
    }
}
