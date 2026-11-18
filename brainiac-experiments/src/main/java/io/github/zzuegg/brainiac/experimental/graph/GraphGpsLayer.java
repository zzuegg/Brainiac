package io.github.zzuegg.brainiac.experimental.graph;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.GinLayer;
import io.github.zzuegg.brainiac.experimental.nn.GraphOps;
import io.github.zzuegg.brainiac.nn.MultiInputLayer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * One GraphGPS layer (Rampášek et al., 2022, <a href="https://arxiv.org/abs/2205.12454">arXiv:2205.12454</a>):
 * a local message-passing update (a {@link GinLayer}) and a global multi-head
 * self-attention update over all nodes of the same graph run side by side on the same
 * input, their residual outputs are summed and normalised, and a two-layer MLP with a
 * residual finishes the layer:
 * <pre>
 * h_local = LN(x + GIN(x, A)),  h_global = LN(x + MHA(x)),  h = h_local + h_global
 * out = LN(h + MLP(h))
 * </pre>
 * Attention lets every node read every other node in one step, so long-range
 * interactions do not have to squeeze through edges; the structure comes from the local
 * path and from positional encodings ({@link GraphPositionalEncodings}) concatenated to the
 * input features. With {@code attention = false} the layer is the "GIN + PE" ablation.
 *
 * <p>Inputs: node features {@code [n, width]}, the adjacency {@code [n, n]} and, for a
 * block-diagonal batch, an optional {@code [n, n]} attention mask from
 * {@link #sameGraphMask} so attention stays inside each graph.
 */
public final class GraphGpsLayer implements MultiInputLayer {

    private final int width;
    private final int heads;
    private final boolean attention;
    private final GinLayer local;
    private final Var wq;
    private final Var wk;
    private final Var wv;
    private final Var wo;
    private final Var w1;
    private final Var b1;
    private final Var w2;
    private final Var b2;

    /**
     * Creates the layer.
     *
     * @param width node features in and out
     * @param heads attention heads; must divide {@code width}
     * @param attention whether the global attention path is used
     * @param random the initialization source
     */
    public GraphGpsLayer(int width, int heads, boolean attention, SplittableRandom random) {
        if (width % heads != 0) {
            throw new IllegalArgumentException("heads must divide width");
        }
        this.width = width;
        this.heads = heads;
        this.attention = attention;
        this.local = new GinLayer(width, width, width, random);
        this.wq = GraphMath.weight(width, width, random);
        this.wk = GraphMath.weight(width, width, random);
        this.wv = GraphMath.weight(width, width, random);
        this.wo = GraphMath.weight(width, width, random);
        this.w1 = GraphMath.weight(width, 2 * width, random);
        this.b1 = GraphMath.bias(2 * width);
        this.w2 = GraphMath.weight(2 * width, width, random);
        this.b2 = GraphMath.bias(width);
    }

    /**
     * @param heads attention heads
     * @param attention whether the global attention path is used
     * @return a spec for {@code Node.merge(spec, adjacency[, mask])}
     */
    public static MultiInputLayer.Spec spec(int heads, boolean attention) {
        return new MultiInputLayer.Spec() {
            @Override
            public Shape outputShape(List<Shape> inputs) {
                return inputs.get(0);
            }

            @Override
            public MultiInputLayer create(List<Shape> inputs, SplittableRandom random) {
                return new GraphGpsLayer(inputs.get(0).dim(-1), heads, attention, random);
            }
        };
    }

    /**
     * The attention mask of a block-diagonal batch: 1 where two nodes share a graph.
     *
     * @param membership {@code [graphs, n]}
     * @return {@code [n, n]}
     */
    public static Tensor sameGraphMask(Tensor membership) {
        return membership.transpose().matmul(membership);
    }

    /**
     * Multi-head self-attention over the nodes.
     *
     * @param x {@code [n, width]}
     * @param mask {@code [n, n]} (1 = may attend), or {@code null} for all pairs
     * @return {@code [n, width]}
     */
    public Var attend(Var x, Tensor mask) {
        int n = x.shape().dim(0);
        int d = width / heads;
        Var q = x.matmul(wq).reshape(n, heads, d).permute(1, 0, 2);
        Var k = x.matmul(wk).reshape(n, heads, d).permute(1, 2, 0);
        Var v = x.matmul(wv).reshape(n, heads, d).permute(1, 0, 2);
        Var scores = q.matmul(k).mul((float) (1 / Math.sqrt(d)));
        if (mask != null) {
            scores = scores.add(mask.sub(1f).mul(1e9f).reshape(1, n, n));
        }
        Var att = GraphOps.softmax(scores);
        return att.matmul(v).permute(1, 0, 2).reshape(n, width).matmul(wo);
    }

    @Override
    public Var forward(List<Var> inputs, boolean training) {
        Var x = inputs.get(0);
        Var h = GraphMath.layerNorm(x.add(local.forward(List.of(x, inputs.get(1)), training)));
        if (attention) {
            Tensor mask = inputs.size() > 2 ? inputs.get(2).value() : null;
            h = h.add(GraphMath.layerNorm(x.add(attend(x, mask))));
        }
        return GraphMath.layerNorm(h.add(GraphMath.linear(GraphMath.linear(h, w1, b1).relu(), w2, b2)));
    }

    @Override
    public Shape outputShape(List<Shape> inputs) {
        return inputs.get(0);
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(local.parameters());
        if (attention) {
            p.addAll(List.of(wq, wk, wv, wo));
        }
        p.addAll(List.of(w1, b1, w2, b2));
        return p;
    }

    @Override
    public int outputSize(int inputSize) {
        return width;
    }

    @Override
    public String toString() {
        return "GraphGpsLayer(" + width + ", heads=" + heads + (attention ? "" : ", no attention") + ")";
    }
}
