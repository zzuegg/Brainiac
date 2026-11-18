package io.github.zzuegg.brainiac.experimental.graph;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.GraphOps;
import io.github.zzuegg.brainiac.nn.MultiInputLayer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * Principal neighbourhood aggregation (Corso et al., 2020, <a href="https://arxiv.org/abs/2004.05718">arXiv:2004.05718</a>).
 * A single aggregator loses information about the multiset of neighbours (mean forgets
 * how many, max forgets how many share the top value, sum confuses many small with few
 * large). PNA uses four aggregators (mean, max, min, standard deviation) of transformed
 * neighbours {@code h_j W_pre}, each under three degree scalers
 * <pre>
 * identity: 1,   amplification: log(d + 1) / delta,   attenuation: delta / log(d + 1)
 * </pre>
 * ({@code delta} is the average {@code log(d + 1)} over the training graphs), and mixes
 * the node's own features with the 12 resulting vectors by a linear layer:
 * {@code h' = relu([h || 12 aggregates] W + b)}.
 *
 * <p>Inputs: node features {@code [n, in]} and the adjacency {@code [n, n]}.
 */
public final class PnaLayer implements MultiInputLayer {

    private final int out;
    private final float delta;
    private final Var pre;
    private final Var preBias;
    private final Var post;
    private final Var postBias;

    /**
     * Creates the layer.
     *
     * @param in input features per node
     * @param out output features per node
     * @param delta the average {@code log(degree + 1)} of the training graphs ({@link #delta})
     * @param random the initialization source
     */
    public PnaLayer(int in, int out, float delta, SplittableRandom random) {
        this.out = out;
        this.delta = delta;
        this.pre = GraphMath.weight(in, in, random);
        this.preBias = GraphMath.bias(in);
        this.post = GraphMath.weight(13 * in, out, random);
        this.postBias = GraphMath.bias(out);
    }

    /**
     * @param out output features per node
     * @param delta the average {@code log(degree + 1)}
     * @return a spec for {@code Node.merge(spec, adjacency)}
     */
    public static MultiInputLayer.Spec spec(int out, float delta) {
        return new MultiInputLayer.Spec() {
            @Override
            public Shape outputShape(List<Shape> inputs) {
                return Shape.of(inputs.get(0).dim(0), out);
            }

            @Override
            public MultiInputLayer create(List<Shape> inputs, SplittableRandom random) {
                return new PnaLayer(inputs.get(0).dim(-1), out, delta, random);
            }
        };
    }

    /**
     * The degree normaliser {@code delta}: the mean of {@code log(d + 1)} over all nodes.
     *
     * @param adjacency a (batched) training adjacency
     * @return {@code delta}
     */
    public static float delta(Tensor adjacency) {
        int n = adjacency.shape().dim(0);
        float[] a = GraphMath.binary(adjacency);
        double s = 0;
        for (int i = 0; i < n; i++) {
            int d = 0;
            for (int j = 0; j < n; j++) {
                d += a[i * n + j] != 0 ? 1 : 0;
            }
            s += Math.log(d + 1);
        }
        return (float) (s / n);
    }

    @Override
    public Var forward(List<Var> inputs, boolean training) {
        Var x = inputs.get(0);
        Tensor adjacency = inputs.get(1).value();
        int n = x.shape().dim(0);
        float[] a = GraphMath.binary(adjacency);
        float[] amp = new float[n];
        float[] att = new float[n];
        for (int i = 0; i < n; i++) {
            int d = 0;
            for (int j = 0; j < n; j++) {
                d += a[i * n + j] != 0 ? 1 : 0;
            }
            double l = Math.log(d + 1);
            amp[i] = (float) (l / delta);
            att[i] = d == 0 ? 0 : (float) (delta / l);
        }
        Tensor binary = Tensor.of(Shape.of(n, n), a);
        Var m = GraphMath.linear(x, pre, preBias);
        Var mean = Var.of(GraphOps.meanNormalize(binary)).matmul(m);
        Var meanSq = Var.of(GraphOps.meanNormalize(binary)).matmul(m.square());
        Var std = meanSq.sub(mean.square()).relu().add(1e-5f).sqrt();
        Var max = GraphMath.neighbourMax(m, binary);
        Var min = GraphMath.neighbourMax(m.neg(), binary).neg();
        Var aggregates = Var.concat(1, mean, max, min, std);
        Var scaled = Var.concat(1, aggregates, aggregates.mul(Tensor.of(Shape.of(n, 1), amp)),
                aggregates.mul(Tensor.of(Shape.of(n, 1), att)));
        return GraphMath.linear(Var.concat(1, x, scaled), post, postBias).relu();
    }

    @Override
    public Shape outputShape(List<Shape> inputs) {
        return Shape.of(inputs.get(0).dim(0), out);
    }

    @Override
    public List<Var> parameters() {
        return List.of(pre, preBias, post, postBias);
    }

    @Override
    public int outputSize(int inputSize) {
        return out;
    }

    @Override
    public String toString() {
        return "PnaLayer(-> " + out + ", delta=" + delta + ")";
    }
}
