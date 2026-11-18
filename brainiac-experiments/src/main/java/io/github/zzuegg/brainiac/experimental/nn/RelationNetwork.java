package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.MultiInputLayer;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * A relation network: {@code RN(O, q) = f(sum_{i,j} g(o_i, o_j, q))}. A shared MLP {@code g}
 * scores every ordered pair of objects (optionally conditioned on a question {@code q}), the
 * pair outputs are summed, and an MLP {@code f} reads the answer. The architecture builds in
 * the inductive bias that the answer depends on relations between objects, which is what
 * plain CNN+MLP pipelines miss on relational questions (Sort-of-CLEVR). Objects can be CNN
 * feature-map cells with their coordinates.
 *
 * <p>As a {@link MultiInputLayer} it takes objects {@code [batch, n, d]} and optionally a
 * question {@code [batch, q]}, and returns {@code [batch, outputs]}.
 *
 * <p>Paper: Santoro et al., "A simple neural network module for relational reasoning"
 * (NeurIPS 2017), <a href="https://arxiv.org/abs/1706.01427">arXiv:1706.01427</a>.
 */
public final class RelationNetwork implements MultiInputLayer {

    private final int features;
    private final int question;
    private final int outputs;
    private final List<Var> g = new ArrayList<>();
    private final List<Var> f = new ArrayList<>();

    /**
     * Creates the network.
     *
     * @param features the features per object
     * @param question the question features, 0 for none
     * @param width the hidden width of {@code g} and {@code f}
     * @param outputs the output features
     * @param random the source of initial weights
     */
    public RelationNetwork(int features, int question, int width, int outputs, RandomGenerator random) {
        this.features = features;
        this.question = question;
        this.outputs = outputs;
        g.add(ClassicOps.matrix(random, 2 * features + question, width));
        g.add(ClassicOps.zeros(width));
        g.add(ClassicOps.matrix(random, width, width));
        g.add(ClassicOps.zeros(width));
        f.add(ClassicOps.matrix(random, width, width));
        f.add(ClassicOps.zeros(width));
        f.add(ClassicOps.matrix(random, width, outputs));
        f.add(ClassicOps.zeros(outputs));
    }

    /**
     * Describes the network for graph builders: inputs {@code (objects)} or {@code (objects, question)}.
     *
     * @param width the hidden width
     * @param outputs the output features
     * @return the spec
     */
    public static MultiInputLayer.Spec spec(int width, int outputs) {
        return new MultiInputLayer.Spec() {
            @Override
            public Shape outputShape(List<Shape> inputs) {
                return Shape.of(outputs);
            }

            @Override
            public MultiInputLayer create(List<Shape> inputs, SplittableRandom random) {
                return new RelationNetwork(inputs.get(0).dim(1), inputs.size() > 1 ? inputs.get(1).dim(0) : 0, width,
                        outputs, random);
            }
        };
    }

    @Override
    public Var forward(List<Var> inputs, boolean training) {
        Var objects = inputs.get(0);
        if (objects.shape().rank() != 3 || objects.shape().dim(2) != features) {
            throw new IllegalArgumentException(this + " expects objects [batch, n, " + features + "], got "
                    + objects.shape());
        }
        int b = objects.shape().dim(0);
        int n = objects.shape().dim(1);
        // g's first layer is linear in (o_i, o_j, q): project each object once and broadcast the
        // sum over pairs, O(n d w + n^2 w) instead of O(n^2 d w).
        Var w0 = g.get(0);
        int width = w0.shape().dim(1);
        Shape grid = Shape.of(b, n, n, width);
        Var left = objects.matmul(w0.slice(0, 0, features)).reshape(b, n, 1, width).expand(grid);
        Var right = objects.matmul(w0.slice(0, features, 2 * features)).reshape(b, 1, n, width).expand(grid);
        Var pre = left.add(right);
        if (question > 0) {
            Var q = inputs.get(1).matmul(w0.slice(0, 2 * features, 2 * features + question));
            pre = pre.add(q.reshape(b, 1, 1, width).expand(grid));
        }
        Var h = pre.reshape(b * n * n, width).add(g.get(1)).relu().matmul(g.get(2)).add(g.get(3)).relu();
        Var pooled = h.reshape(b, n * n, width).sum(1, false);
        return pooled.matmul(f.get(0)).add(f.get(1)).relu().matmul(f.get(2)).add(f.get(3));
    }

    @Override
    public Shape outputShape(List<Shape> inputs) {
        return Shape.of(outputs);
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(outputs);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(g);
        p.addAll(f);
        return p;
    }

    @Override
    public String toString() {
        return "RelationNetwork(" + features + (question > 0 ? " + q" + question : "") + " -> " + outputs + ")";
    }
}
