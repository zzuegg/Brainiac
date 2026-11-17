package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * An E(n)-equivariant graph layer (EGNN; Satorras, Hoogeboom and Welling,
 * <a href="https://arxiv.org/abs/2102.09844">2102.09844</a>) on a fully connected point set.
 * Messages see the features of both ends and only their squared distance, so they are
 * invariant; coordinates move along the difference vectors, so they rotate, reflect and
 * translate with the input:
 *
 * <pre>
 * m_ij = phi_e(h_i, h_j, |x_i - x_j|^2)
 * x_i' = x_i + 1/(n-1) sum_(j != i) (x_i - x_j) phi_x(m_ij)
 * h_i' = h_i + phi_h(h_i, sum_(j != i) m_ij)
 * </pre>
 *
 * <p>To fit the single-input {@link Layer} contract the input packs both:
 * {@code [batch, nodes, coordinates + features]}, coordinates first; the output has the same
 * layout and width, so layers stack.
 */
public final class EquivariantGraphLayer implements Layer {
    private final int coordinates;
    private final int features;
    private final int hidden;
    private final Var edge1;
    private final Var edgeBias1;
    private final Var edge2;
    private final Var edgeBias2;
    private final Var coord1;
    private final Var coordBias1;
    private final Var coord2;
    private final Var node1;
    private final Var nodeBias1;
    private final Var node2;
    private final Var nodeBias2;

    private EquivariantGraphLayer(int coordinates, int features, int hidden, SplittableRandom random) {
        this.coordinates = coordinates;
        this.features = features;
        this.hidden = hidden;
        int in = 2 * features + 1;
        this.edge1 = weight(in, hidden, random);
        this.edgeBias1 = Var.param(Tensor.zeros(Shape.of(hidden)));
        this.edge2 = weight(hidden, hidden, random);
        this.edgeBias2 = Var.param(Tensor.zeros(Shape.of(hidden)));
        this.coord1 = weight(hidden, hidden, random);
        this.coordBias1 = Var.param(Tensor.zeros(Shape.of(hidden)));
        // small last layer: coordinates start close to the identity map
        this.coord2 = Var.param(Init.XAVIER.create(Shape.of(hidden, 1), hidden, 1, random).mul(0.01f));
        this.node1 = weight(features + hidden, hidden, random);
        this.nodeBias1 = Var.param(Tensor.zeros(Shape.of(hidden)));
        this.node2 = weight(hidden, features, random);
        this.nodeBias2 = Var.param(Tensor.zeros(Shape.of(features)));
    }

    private static Var weight(int in, int out, SplittableRandom random) {
        return Var.param(Init.XAVIER.create(Shape.of(in, out), in, out, random));
    }

    /**
     * Creates a layer.
     *
     * @param coordinates the space dimension n
     * @param features the node feature width
     * @param hidden the message width
     * @param seed the seed
     * @return the layer
     */
    public static EquivariantGraphLayer of(int coordinates, int features, int hidden, long seed) {
        return new EquivariantGraphLayer(coordinates, features, hidden, new SplittableRandom(seed));
    }

    /**
     * Describes the layer for model builders; input {@code [nodes, coordinates + features]}.
     *
     * @param coordinates the space dimension
     * @param hidden the message width
     * @return the spec
     */
    public static LayerSpec spec(int coordinates, int hidden) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return input;
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new EquivariantGraphLayer(coordinates, input.dim(-1) - coordinates, hidden, random);
            }
        };
    }

    private static Var silu(Var x) {
        return x.mul(x.sigmoid());
    }

    @Override
    public Var forward(Var input) {
        Shape s = input.shape();
        if (s.rank() != 3 || s.dim(2) != coordinates + features) {
            throw new IllegalArgumentException("EquivariantGraphLayer expects [batch, nodes, " + (coordinates + features)
                    + "], got " + s);
        }
        int b = s.dim(0);
        int n = s.dim(1);
        Var x = input.slice(2, 0, coordinates);
        Var h = input.slice(2, coordinates, coordinates + features);
        Var diff = x.reshape(b, n, 1, coordinates).expand(Shape.of(b, n, n, coordinates))
                .sub(x.reshape(b, 1, n, coordinates).expand(Shape.of(b, n, n, coordinates)));
        Var d2 = diff.square().sum(3, true);
        Var hi = h.reshape(b, n, 1, features).expand(Shape.of(b, n, n, features));
        Var hj = h.reshape(b, 1, n, features).expand(Shape.of(b, n, n, features));
        Var m = silu(silu(Var.concat(3, hi, hj, d2).matmul(edge1).add(edgeBias1)).matmul(edge2).add(edgeBias2));
        float[] off = new float[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                off[i * n + j] = i == j ? 0 : 1;
            }
        }
        Tensor mask = Tensor.of(Shape.of(1, n, n, 1), off, input.value().device());
        m = m.mul(mask);
        Var weights = silu(m.matmul(coord1).add(coordBias1)).matmul(coord2);       // [b, n, n, 1]
        Var x2 = x.add(diff.mul(weights).sum(2, false).div(Math.max(1, n - 1)));
        Var aggregate = m.sum(2, false);                                             // [b, n, hidden]
        Var h2 = h.add(silu(Var.concat(2, h, aggregate).matmul(node1).add(nodeBias1)).matmul(node2).add(nodeBias2));
        return Var.concat(2, x2, h2);
    }

    @Override
    public List<Var> parameters() {
        return List.of(edge1, edgeBias1, edge2, edgeBias2, coord1, coordBias1, coord2, node1, nodeBias1, node2,
                nodeBias2);
    }

    @Override
    public int outputSize(int inputSize) {
        return inputSize;
    }

    /** @return the message width */
    public int hidden() {
        return hidden;
    }
}
