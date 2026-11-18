package io.github.zzuegg.brainiac.experimental.graph;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.MultiInputLayer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * An encode-process-decode message-passing processor for neural algorithmic reasoning
 * (Veličković et al., 2020, <a href="https://arxiv.org/abs/1910.10593">arXiv:1910.10593</a>;
 * CLRS, <a href="https://arxiv.org/abs/2205.15659">arXiv:2205.15659</a>). One shared
 * processor step runs once per algorithm step:
 * <pre>
 * c_i = [enc(x_i) || h_i]                     (inputs re-injected every step)
 * m_ij = relu(c_i A + c_j B + w_ij a + b)      over j in N(i) + {i}
 * m_i = max_j m_ij   (or sum_j for the ablation)
 * g_i = sigmoid([c_i || m_i] W_g),  h_i' = g_i * relu([c_i || m_i] W_u) + (1 - g_i) * h_i
 * </pre>
 * Max aggregation aligns with the "best of the neighbours" structure of BFS, Bellman-Ford
 * and Prim (a node relaxes to its best neighbour), which is what lets a processor trained
 * on 8-16 node graphs run on graphs several times larger; the gate lets nodes keep a
 * finished state. After each step two decoders read the state: per-node class logits
 * ({@link #nodeLogits}) and predecessor-pointer logits over each node's neighbours and
 * itself ({@link #pointerLogits}). Training on every step's state (the <em>hints</em>), not
 * just the final output, is the supervision that the CLRS benchmark uses.
 *
 * <p>Inputs to {@link #forward}: node inputs {@code [n, in]} and edge weights {@code [n, n]}
 * (0 = no edge); it runs {@code n} steps and returns the final state {@code [n, width]}.
 */
public final class AlgorithmicProcessor implements MultiInputLayer {

    /** The neighbourhood aggregation. */
    public enum Aggregation {
        /** element-wise maximum, aligned with relaxation */
        MAX,
        /** sum */
        SUM
    }

    private final int width;
    private final Aggregation aggregation;
    private final Var enc;
    private final Var encBias;
    private final Var a;
    private final Var b;
    private final Var edge;
    private final Var msgBias;
    private final Var wg;
    private final Var bg;
    private final Var wu;
    private final Var bu;
    private final Var node;
    private final Var nodeBias;
    private final Var p1;
    private final Var p2;
    private final Var p3;
    private final Var pv;

    /**
     * Creates the processor.
     *
     * @param in node input features
     * @param width the latent width
     * @param classes node output classes
     * @param aggregation the aggregation
     * @param random the initialization source
     */
    public AlgorithmicProcessor(int in, int width, int classes, Aggregation aggregation, SplittableRandom random) {
        this.width = width;
        this.aggregation = Objects.requireNonNull(aggregation);
        enc = GraphMath.weight(in, width, random);
        encBias = GraphMath.bias(width);
        a = GraphMath.weight(2 * width, width, random);
        b = GraphMath.weight(2 * width, width, random);
        edge = GraphMath.weight(1, width, random);
        msgBias = GraphMath.bias(width);
        wg = GraphMath.weight(3 * width, width, random);
        bg = GraphMath.bias(width);
        wu = GraphMath.weight(3 * width, width, random);
        bu = GraphMath.bias(width);
        node = GraphMath.weight(width, classes, random);
        nodeBias = GraphMath.bias(classes);
        p1 = GraphMath.weight(width, width, random);
        p2 = GraphMath.weight(width, width, random);
        p3 = GraphMath.weight(1, width, random);
        pv = GraphMath.weight(width, 1, random);
    }

    /** @return the latent width */
    public int width() {
        return width;
    }

    private static float[] neighbourhood(Tensor weights) {
        int n = weights.shape().dim(0);
        float[] w = weights.toArray();
        float[] mask = new float[n * n];
        for (int i = 0; i < n * n; i++) {
            mask[i] = w[i] != 0 || i / n == i % n ? 1 : 0;
        }
        return mask;
    }

    /**
     * One processor step.
     *
     * @param x node inputs {@code [n, in]}
     * @param h the state {@code [n, width]}
     * @param weights edge weights {@code [n, n]}
     * @return the next state
     */
    public Var step(Var x, Var h, Tensor weights) {
        int n = h.shape().dim(0);
        Var c = Var.concat(1, GraphMath.linear(x, enc, encBias), h);
        Var w = Var.of(weights.reshape(n, n, 1));
        Var m = c.matmul(a).reshape(n, 1, width).add(c.matmul(b).reshape(1, n, width)).add(w.matmul(edge))
                .add(msgBias).relu();
        float[] mask = neighbourhood(weights);
        Var agg;
        if (aggregation == Aggregation.MAX) {
            float[] pen = new float[n * n];
            for (int i = 0; i < n * n; i++) {
                pen[i] = mask[i] != 0 ? 0 : -1e9f;
            }
            agg = m.add(Tensor.of(Shape.of(n, n, 1), pen)).max(1, false);
        } else {
            agg = m.mul(Tensor.of(Shape.of(n, n, 1), mask)).sum(1, false);
        }
        Var cm = Var.concat(1, c, agg);
        Var gate = GraphMath.linear(cm, wg, bg).sigmoid();
        return gate.mul(GraphMath.linear(cm, wu, bu).relu()).add(gate.neg().add(1f).mul(h));
    }

    /**
     * Runs the processor.
     *
     * @param x node inputs {@code [n, in]}
     * @param weights edge weights {@code [n, n]}
     * @param steps processor steps
     * @return the state after every step
     */
    public List<Var> run(Var x, Tensor weights, int steps) {
        int n = x.shape().dim(0);
        Var h = Var.of(Tensor.zeros(Shape.of(n, width)));
        List<Var> states = new ArrayList<>();
        for (int s = 0; s < steps; s++) {
            h = step(x, h, weights);
            states.add(h);
        }
        return states;
    }

    /**
     * @param h a state {@code [n, width]}
     * @return per-node class logits {@code [n, classes]}
     */
    public Var nodeLogits(Var h) {
        return GraphMath.linear(h, node, nodeBias);
    }

    /**
     * Predecessor-pointer logits, restricted to each node's neighbours and itself.
     *
     * @param h a state {@code [n, width]}
     * @param weights edge weights {@code [n, n]}
     * @return {@code [n, n]}: row {@code i} scores the node {@code i} points to
     */
    public Var pointerLogits(Var h, Tensor weights) {
        int n = h.shape().dim(0);
        Var w = Var.of(weights.reshape(n, n, 1));
        Var s = h.matmul(p1).reshape(n, 1, width).add(h.matmul(p2).reshape(1, n, width)).add(w.matmul(p3)).relu()
                .matmul(pv).reshape(n, n);
        float[] mask = neighbourhood(weights);
        float[] pen = new float[n * n];
        for (int i = 0; i < n * n; i++) {
            pen[i] = mask[i] != 0 ? 0 : -1e9f;
        }
        return s.add(Tensor.of(Shape.of(n, n), pen));
    }

    @Override
    public Var forward(List<Var> inputs, boolean training) {
        List<Var> states = run(inputs.get(0), inputs.get(1).value(), inputs.get(0).shape().dim(0));
        return states.getLast();
    }

    @Override
    public Shape outputShape(List<Shape> inputs) {
        return Shape.of(inputs.get(0).dim(0), width);
    }

    @Override
    public List<Var> parameters() {
        return List.of(enc, encBias, a, b, edge, msgBias, wg, bg, wu, bu, node, nodeBias, p1, p2, p3, pv);
    }

    @Override
    public int outputSize(int inputSize) {
        return width;
    }

    @Override
    public String toString() {
        return "AlgorithmicProcessor(" + width + ", " + aggregation + ")";
    }
}
