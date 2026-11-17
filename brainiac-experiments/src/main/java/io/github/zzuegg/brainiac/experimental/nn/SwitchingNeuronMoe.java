package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.nn.SwitchingDense;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * A neuron-level mixture of experts built from switching neurons (experiment X20). Each
 * routed expert is a small group of {@link SwitchingDense} neurons, which choose their own
 * activation per input, followed by a linear read-out; a router sends every token to its
 * {@code topK} groups, as in {@link FineGrainedMoe}. Two levels of input-dependent
 * computation stack: the router picks which neuron groups run, and inside a group each
 * neuron picks how it responds. Groups can stay tiny (a handful of neurons), so this scales
 * a single switching layer out to many cheaply routed specialists.
 *
 * <p>The router, load balancing and dispatch are those of {@link FineGrainedMoe}, available
 * through {@link #moe()}.
 */
public final class SwitchingNeuronMoe implements Layer {

    /** One group: switching neurons, then a linear read-out. */
    private static final class Group implements Layer {
        private final SwitchingDense neurons;
        private final Dense readout;

        Group(int in, int units, int out, Activation[] candidates, RandomGenerator random) {
            neurons = new SwitchingDense(in, units, candidates, false, random);
            readout = new Dense(units, out, Activation.IDENTITY, Init.XAVIER, random);
        }

        @Override
        public Var forward(Var input, boolean training) {
            return readout.forward(neurons.forward(input, training), training);
        }

        @Override
        public Var forward(Var input) {
            return forward(input, false);
        }

        @Override
        public List<Var> parameters() {
            List<Var> p = new ArrayList<>(neurons.parameters());
            p.addAll(readout.parameters());
            return p;
        }

        @Override
        public List<Var> buffers() {
            return neurons.buffers();
        }
    }

    private final FineGrainedMoe moe;

    /**
     * Creates the layer.
     *
     * @param features the input width
     * @param outputs the output width
     * @param groups the number of neuron groups (routed experts)
     * @param neuronsPerGroup the switching neurons in each group
     * @param topK groups per token
     * @param candidates the activations each neuron chooses between, at least two
     * @param random the random source
     */
    public SwitchingNeuronMoe(int features, int outputs, int groups, int neuronsPerGroup, int topK,
            Activation[] candidates, RandomGenerator random) {
        List<Layer> experts = new ArrayList<>();
        for (int i = 0; i < groups; i++) {
            experts.add(new Group(features, neuronsPerGroup, outputs, candidates, random));
        }
        this.moe = new FineGrainedMoe(features, outputs, topK, experts, List.of(), random);
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}.
     *
     * @param outputs the output width
     * @param groups the number of neuron groups
     * @param neuronsPerGroup the switching neurons in each group
     * @param topK groups per token
     * @param candidates the activations each neuron chooses between
     * @return the spec
     */
    public static LayerSpec of(int outputs, int groups, int neuronsPerGroup, int topK, Activation... candidates) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                int[] d = input.toArray();
                d[d.length - 1] = outputs;
                return Shape.of(d);
            }

            @Override
            public Layer create(Shape input, java.util.SplittableRandom random) {
                return new SwitchingNeuronMoe(input.dim(-1), outputs, groups, neuronsPerGroup, topK, candidates, random);
            }
        };
    }

    /** @return the underlying router and dispatcher, for balance losses and loads */
    public FineGrainedMoe moe() {
        return moe;
    }

    @Override
    public Var forward(Var input) {
        return moe.forward(input);
    }

    @Override
    public List<Var> parameters() {
        return moe.parameters();
    }

    @Override
    public int outputSize(int inputSize) {
        return moe.outputSize(inputSize);
    }

    @Override
    public String toString() {
        return "SwitchingNeuronMoe(" + moe + ")";
    }
}
