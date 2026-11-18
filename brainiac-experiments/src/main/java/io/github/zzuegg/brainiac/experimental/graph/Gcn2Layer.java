package io.github.zzuegg.brainiac.experimental.graph;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.GraphOps;
import io.github.zzuegg.brainiac.nn.MultiInputLayer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * A GCNII layer (Chen et al., 2020, <a href="https://arxiv.org/abs/2007.02133">arXiv:2007.02133</a>),
 * which lets graph convolutions be stacked 32 or 64 deep without over-smoothing:
 * <pre>
 * H' = relu( ((1 - alpha) S H + alpha H0) ((1 - beta_l) I + beta_l W) )
 * beta_l = log(lambda / l + 1)
 * </pre>
 * The <em>initial residual</em> mixes a fraction {@code alpha} of the first layer's
 * representation {@code H0} into every layer, so no depth can wash it out; the <em>identity
 * mapping</em> keeps the weight close to the identity, more so the deeper the layer
 * ({@code beta_l} shrinks with {@code l}), so a deep stack starts as (and stays near) a
 * personalised-PageRank propagation.
 *
 * <p>Inputs: node features {@code [n, width]}, the adjacency {@code [n, n]} and the initial
 * representation {@code H0} {@code [n, width]}.
 */
public final class Gcn2Layer implements MultiInputLayer {

    private final int width;
    private final float alpha;
    private final float beta;
    private final Var weight;

    /**
     * Creates the layer.
     *
     * @param width node features in and out
     * @param alpha the initial-residual weight (0.1 in the paper)
     * @param lambda the identity-mapping strength (0.5 in the paper)
     * @param layer this layer's 1-based depth {@code l}
     * @param random the initialization source
     */
    public Gcn2Layer(int width, float alpha, float lambda, int layer, SplittableRandom random) {
        this.width = width;
        this.alpha = alpha;
        this.beta = (float) Math.log(lambda / layer + 1);
        this.weight = GraphMath.weight(width, width, random);
    }

    /**
     * @param alpha the initial-residual weight
     * @param lambda the identity-mapping strength
     * @param layer the 1-based depth
     * @return a spec for {@code Node.merge(spec, adjacency, h0)}
     */
    public static MultiInputLayer.Spec spec(float alpha, float lambda, int layer) {
        return new MultiInputLayer.Spec() {
            @Override
            public Shape outputShape(List<Shape> inputs) {
                return inputs.get(0);
            }

            @Override
            public MultiInputLayer create(List<Shape> inputs, SplittableRandom random) {
                return new Gcn2Layer(inputs.get(0).dim(-1), alpha, lambda, layer, random);
            }
        };
    }

    /** @return {@code beta_l} */
    public float beta() {
        return beta;
    }

    @Override
    public Var forward(List<Var> inputs, boolean training) {
        Tensor s = GraphOps.gcnNormalize(inputs.get(1).value());
        Var support = Var.of(s).matmul(inputs.get(0)).mul(1 - alpha).add(inputs.get(2).mul(alpha));
        return support.mul(1 - beta).add(support.matmul(weight).mul(beta)).relu();
    }

    @Override
    public Shape outputShape(List<Shape> inputs) {
        return inputs.get(0);
    }

    @Override
    public List<Var> parameters() {
        return List.of(weight);
    }

    @Override
    public int outputSize(int inputSize) {
        return width;
    }

    @Override
    public String toString() {
        return "Gcn2Layer(" + width + ", alpha=" + alpha + ", beta=" + beta + ")";
    }
}
