package io.github.zzuegg.brainiac.experimental.vision;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * EdgeConv, the layer of the dynamic graph CNN (DGCNN, Wang et al., 2019). For every point it
 * finds the {@code k} nearest other points in the current feature space (so the graph is
 * rebuilt at every layer, from the features rather than only the coordinates), applies a
 * shared two-layer MLP to each edge feature {@code (x_i, x_j - x_i)}, and max-pools over the
 * neighbours. The {@code x_j - x_i} term sees local geometry that a per-point MLP such as
 * {@link PointNet} cannot. The neighbour search is not differentiated.
 *
 * <p>Input {@code [batch, points, in]}, output {@code [batch, points, out]}.
 *
 * <p>Paper: <a href="https://arxiv.org/abs/1801.07829">arXiv:1801.07829</a>.
 */
public final class EdgeConv implements Layer {

    private final int inputs;
    private final int outputs;
    private final int k;
    private final Dense first;
    private final Dense second;

    /**
     * Creates the layer.
     *
     * @param inputs the input feature width
     * @param outputs the output width
     * @param k the neighbours per point
     * @param random the source of initial weights
     */
    public EdgeConv(int inputs, int outputs, int k, RandomGenerator random) {
        this.inputs = inputs;
        this.outputs = outputs;
        this.k = k;
        this.first = new Dense(2 * inputs, outputs, Activation.RELU, Init.HE, random);
        this.second = new Dense(outputs, outputs, Activation.RELU, Init.HE, random);
    }

    /**
     * The {@code k} nearest other points of every point, by Euclidean distance.
     *
     * @param features {@code [batch, points, c]}
     * @param k the neighbours per point
     * @return flat row indices into {@code [batch * points]}, {@code k} per point
     */
    public static float[] neighbours(Tensor features, int k) {
        int b = features.shape().dim(0);
        int n = features.shape().dim(1);
        int c = features.shape().dim(2);
        float[] f = features.toArray();
        float[] out = new float[b * n * k];
        double[] dist = new double[n];
        Integer[] order = new Integer[n];
        for (int s = 0; s < b; s++) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < n; j++) {
                    double d = 0;
                    for (int q = 0; q < c; q++) {
                        double e = f[(s * n + i) * c + q] - f[(s * n + j) * c + q];
                        d += e * e;
                    }
                    dist[j] = j == i ? Double.MAX_VALUE : d;
                    order[j] = j;
                }
                java.util.Arrays.sort(order, (x, y) -> Double.compare(dist[x], dist[y]));
                for (int m = 0; m < k; m++) {
                    out[(s * n + i) * k + m] = s * n + order[m];
                }
            }
        }
        return out;
    }

    @Override
    public Var forward(Var input) {
        int b = input.shape().dim(0);
        int n = input.shape().dim(1);
        Var flat = input.reshape(b * n, inputs);
        Tensor idx = Tensor.of(Shape.of(b * n * k), neighbours(input.value(), k));
        Var xj = flat.gather(0, idx);
        Var xi = flat.repeatInterleave(k, 0);
        Var edge = second.forward(first.forward(Var.concat(1, xi, xj.sub(xi))));
        return edge.reshape(b * n, k, outputs).max(1, false).reshape(b, n, outputs);
    }

    @Override
    public int outputSize(int inputSize) {
        return outputs;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(first.parameters());
        p.addAll(second.parameters());
        return p;
    }

    @Override
    public String toString() {
        return "EdgeConv(" + inputs + " -> " + outputs + ", k " + k + ")";
    }
}
