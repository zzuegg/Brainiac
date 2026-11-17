package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.MultiInputLayer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * A graph convolution (Kipf and Welling, 2016, <a href="https://arxiv.org/abs/1609.02907">arXiv:1609.02907</a>):
 * {@code H' = act(D^-1/2 (A + I) D^-1/2 H W + b)}. Every node averages its own and its
 * neighbours' transformed features, weighted by inverse square-root degrees; stacking
 * {@code k} layers mixes information from {@code k} hops away. The workhorse of
 * semi-supervised node classification.
 *
 * <p>Inputs: node features {@code [n, in]} and the adjacency {@code [n, n]} (a constant).
 */
public final class GcnLayer implements MultiInputLayer {

    private final int in;
    private final int out;
    private final Var weight;
    private final Var bias;
    private final Activation activation;

    /**
     * Creates the layer with Glorot-uniform weights.
     *
     * @param in input features per node
     * @param out output features per node
     * @param activation the activation (bound to {@code out})
     * @param random the initialization source
     */
    public GcnLayer(int in, int out, Activation activation, SplittableRandom random) {
        this.in = in;
        this.out = out;
        float bound = (float) Math.sqrt(6.0 / (in + out));
        this.weight = Var.param(Tensor.uniform(Shape.of(in, out), -bound, bound, random));
        this.bias = Var.param(Tensor.zeros(Shape.of(1, out)));
        this.activation = Objects.requireNonNull(activation).bind(out);
    }

    /**
     * @param out output features per node
     * @param activation the activation
     * @return a spec for {@code Node.merge(spec, adjacency)}
     */
    public static MultiInputLayer.Spec spec(int out, Activation activation) {
        return new MultiInputLayer.Spec() {
            @Override
            public Shape outputShape(List<Shape> inputs) {
                return Shape.of(inputs.get(0).dim(0), out);
            }

            @Override
            public MultiInputLayer create(List<Shape> inputs, SplittableRandom random) {
                return new GcnLayer(inputs.get(0).dim(-1), out, activation, random);
            }
        };
    }

    @Override
    public Var forward(List<Var> inputs, boolean training) {
        if (inputs.size() != 2) {
            throw new IllegalArgumentException("GcnLayer takes node features and an adjacency matrix");
        }
        Tensor propagate = GraphOps.gcnNormalize(inputs.get(1).value());
        Var h = Var.of(propagate).matmul(inputs.get(0).matmul(weight)).add(bias);
        return activation.forward(h, training);
    }

    @Override
    public Shape outputShape(List<Shape> inputs) {
        return Shape.of(inputs.get(0).dim(0), out);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(weight, bias));
        p.addAll(activation.parameters());
        return p;
    }

    @Override
    public int outputSize(int inputSize) {
        return out;
    }

    @Override
    public String toString() {
        return "GcnLayer(" + in + " -> " + out + ")";
    }
}
