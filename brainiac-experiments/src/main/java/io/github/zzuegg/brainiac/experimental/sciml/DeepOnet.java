package io.github.zzuegg.brainiac.experimental.sciml;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.nn.MultiInputLayer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * DeepONet (Lu, Jin, Pang, Zhang and Karniadakis, 2021): an operator network
 * {@code G(u)(y) = sum_k b_k(u) t_k(y) + b0}. A branch net encodes the input function sampled
 * at fixed sensors, a trunk net encodes a query coordinate, and their dot product is the
 * output, so the learned operator can be queried at any point, not only on a training grid.
 *
 * <p>As a {@link MultiInputLayer} it takes {@code [u, y]}: sensor values {@code [b, m]} and
 * query points, either shared {@code [q, d]} or per example {@code [b, q, d]}, and returns
 * {@code [b, q]}.
 *
 * <p>Paper: <a href="https://arxiv.org/abs/1910.03193">arXiv:1910.03193</a>.
 */
public final class DeepOnet implements MultiInputLayer {

    private final int sensors;
    private final int coordinates;
    private final int basis;
    private final SimpleMlp branch;
    private final SimpleMlp trunk;
    private final Var bias;

    private DeepOnet(int sensors, int coordinates, int width, int depth, int basis, long seed) {
        this.sensors = sensors;
        this.coordinates = coordinates;
        this.basis = basis;
        this.branch = SimpleMlp.of(seed, Var::relu, sizes(sensors, width, depth, basis));
        this.trunk = SimpleMlp.tanh(seed + 1, sizes(coordinates, width, depth, basis));
        this.bias = Var.param(Tensor.zeros(Shape.of(1)));
    }

    private static int[] sizes(int in, int width, int depth, int out) {
        int[] s = new int[depth + 2];
        s[0] = in;
        for (int i = 1; i <= depth; i++) {
            s[i] = width;
        }
        s[depth + 1] = out;
        return s;
    }

    /**
     * Creates the network.
     *
     * @param sensors the input-function sensors {@code m}
     * @param coordinates the query dimension {@code d}
     * @param width hidden width of both nets
     * @param depth hidden layers of both nets
     * @param basis the latent basis size {@code p}
     * @param seed the seed
     * @return the network
     */
    public static DeepOnet of(int sensors, int coordinates, int width, int depth, int basis, long seed) {
        return new DeepOnet(sensors, coordinates, width, depth, basis, seed);
    }

    /**
     * Evaluates the operator.
     *
     * @param u sensor values {@code [b, m]}
     * @param y query points {@code [q, d]} or {@code [b, q, d]}
     * @return {@code [b, q]}
     */
    public Var forward(Var u, Var y) {
        if (u.shape().dim(1) != sensors) {
            throw new IllegalArgumentException("DeepOnet expects " + sensors + " sensors, got " + u.shape());
        }
        Var b = branch.forward(u);   // [b, p]
        if (y.shape().rank() == 2) {
            return b.matmul(trunk.forward(y).transpose()).add(bias);
        }
        int batch = y.shape().dim(0);
        int q = y.shape().dim(1);
        Var t = trunk.forward(y.reshape(batch * q, coordinates)).reshape(batch, q, basis);
        return t.matmul(b.reshape(batch, basis, 1)).reshape(batch, q).add(bias);
    }

    @Override
    public Var forward(List<Var> inputs, boolean training) {
        return forward(inputs.get(0), inputs.get(1));
    }

    @Override
    public Shape outputShape(List<Shape> inputs) {
        Shape y = inputs.get(1);
        return Shape.of(inputs.get(0).dim(0), y.dim(y.rank() - 2));
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(branch.parameters());
        p.addAll(trunk.parameters());
        p.add(bias);
        return p;
    }

    @Override
    public String toString() {
        return "DeepOnet(sensors=" + sensors + ", basis=" + basis + ")";
    }
}
