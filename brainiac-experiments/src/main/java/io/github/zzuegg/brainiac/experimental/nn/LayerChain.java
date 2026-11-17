package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Several layers applied one after the other, as a single {@link Layer}: a stack of blocks
 * or a block with extra layers around it can then go anywhere one layer spec is expected
 * (for example into a shoot-out harness that takes one mixer). Parameters, buffers,
 * auxiliary losses and state resets are passed through.
 */
public final class LayerChain implements Layer {

    private final List<Layer> layers;

    /**
     * Creates the chain.
     *
     * @param layers the layers, in order
     */
    public LayerChain(List<? extends Layer> layers) {
        this.layers = List.copyOf(layers);
    }

    /**
     * Describes a chain; each spec is created for the output shape of the one before.
     *
     * @param specs the layer specs, in order
     * @return the spec
     */
    public static LayerSpec of(LayerSpec... specs) {
        List<LayerSpec> list = List.of(specs);
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                Shape s = input;
                for (LayerSpec spec : list) {
                    s = spec.outputShape(s);
                }
                return s;
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                List<Layer> made = new ArrayList<>();
                Shape s = input;
                for (LayerSpec spec : list) {
                    made.add(spec.create(s, random));
                    s = spec.outputShape(s);
                }
                return new LayerChain(made);
            }
        };
    }

    /**
     * Describes {@code count} copies of one spec, each with its own weights.
     *
     * @param spec the layer spec
     * @param count how many
     * @return the spec
     */
    public static LayerSpec repeat(LayerSpec spec, int count) {
        LayerSpec[] specs = new LayerSpec[count];
        java.util.Arrays.fill(specs, spec);
        return of(specs);
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        Var x = input;
        for (Layer l : layers) {
            x = l.forward(x, training);
        }
        return x;
    }

    /** @return the layers, in order */
    public List<Layer> layers() {
        return layers;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        layers.forEach(l -> p.addAll(l.parameters()));
        return p;
    }

    @Override
    public List<Var> buffers() {
        List<Var> p = new ArrayList<>();
        layers.forEach(l -> p.addAll(l.buffers()));
        return p;
    }

    @Override
    public Var auxiliaryLoss() {
        Var sum = null;
        for (Layer l : layers) {
            Var a = l.auxiliaryLoss();
            if (a != null) {
                sum = sum == null ? a : sum.add(a);
            }
        }
        return sum;
    }

    @Override
    public void resetState() {
        layers.forEach(Layer::resetState);
    }

    @Override
    public Shape outputShape(Shape inputShape) {
        Shape s = inputShape;
        for (Layer l : layers) {
            s = l.outputShape(s);
        }
        return s;
    }

    @Override
    public String toString() {
        return "LayerChain" + layers;
    }
}
