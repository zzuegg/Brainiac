package io.github.zzuegg.brainiac.experimental.graph;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.GcnLayer;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A deep GCN stack for node classification with optional residual skips and a jumping-knowledge
 * readout (Xu et al., ICML 2018, <a href="https://arxiv.org/abs/1806.03536">arXiv:1806.03536</a>).
 * An input projection maps the features to {@code hidden}; {@code depth} ReLU {@link GcnLayer}s
 * follow, each optionally with its input-to-output skip ({@link GcnLayer#withSkip}); the
 * readout combines the layers' outputs by {@link Readout#CONCAT} or element-wise
 * {@link Readout#MAX} (jumping knowledge), or takes only the {@link Readout#LAST} one (a plain
 * stack), and a linear head maps it to class logits. Combining every depth lets each node use
 * the neighbourhood size that suits it, so depth does not wash out the representation.
 */
public final class JumpingKnowledgeGcn {

    /** How the layers' outputs are combined. */
    public enum Readout {
        /** only the last layer: a plain stack */
        LAST,
        /** concatenation of every layer's output */
        CONCAT,
        /** element-wise maximum over the layers */
        MAX
    }

    private final Var inW;
    private final Var inB;
    private final List<GcnLayer> layers = new ArrayList<>();
    private final Var outW;
    private final Var outB;
    private final Readout readout;

    /**
     * Creates the network.
     *
     * @param in input features per node
     * @param hidden hidden features per node
     * @param classes output classes
     * @param depth graph convolutions
     * @param readout the layer combination
     * @param skip whether each convolution adds its input
     * @param seed the seed
     */
    public JumpingKnowledgeGcn(int in, int hidden, int classes, int depth, Readout readout, boolean skip,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        this.readout = readout;
        float b0 = (float) Math.sqrt(6.0 / (in + hidden));
        inW = Var.param(Tensor.uniform(Shape.of(in, hidden), -b0, b0, r));
        inB = Var.param(Tensor.zeros(Shape.of(1, hidden)));
        for (int l = 0; l < depth; l++) {
            layers.add(new GcnLayer(hidden, hidden, Activation.RELU, r).withSkip(skip));
        }
        int width = readout == Readout.CONCAT ? hidden * depth : hidden;
        float b1 = (float) Math.sqrt(6.0 / (width + classes));
        outW = Var.param(Tensor.uniform(Shape.of(width, classes), -b1, b1, r));
        outB = Var.param(Tensor.zeros(Shape.of(1, classes)));
    }

    /**
     * Class logits for every node.
     *
     * @param x node features {@code [n, in]}
     * @param adjacency the adjacency {@code [n, n]}
     * @param training whether training
     * @return logits {@code [n, classes]}
     */
    public Var forward(Var x, Var adjacency, boolean training) {
        Var h = x.matmul(inW).add(inB).relu();
        List<Var> outs = new ArrayList<>();
        for (GcnLayer layer : layers) {
            h = layer.forward(List.of(h, adjacency), training);
            outs.add(h);
        }
        Var z;
        if (readout == Readout.LAST || outs.size() == 1) {
            z = h;
        } else if (readout == Readout.CONCAT) {
            z = Var.concat(1, outs.toArray(new Var[0]));
        } else {
            z = outs.get(0);
            for (int l = 1; l < outs.size(); l++) {
                z = z.maximum(outs.get(l));
            }
        }
        return z.matmul(outW).add(outB);
    }

    /**
     * The trainable parameters.
     *
     * @return the parameters
     */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(inW, inB));
        for (GcnLayer l : layers) {
            p.addAll(l.parameters());
        }
        p.addAll(List.of(outW, outB));
        return p;
    }
}
