package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * A self-referential weight matrix (Irie, Schlag, Csordas and Schmidhuber 2022,
 * <a href="https://arxiv.org/abs/2202.05780">arXiv:2202.05780</a>): one matrix that, at
 * every step, produces its output together with a key, a query and a learning rate for
 * modifying itself, and then applies the modification:
 * <pre>
 * [y, k, q, beta] = W_(t-1) x_t
 * v = W_(t-1) phi(q),  v_old = W_(t-1) phi(k)            (phi = softmax)
 * W_t = W_(t-1) + sigmoid(beta) (v - v_old) phi(k)^T      (delta rule on the whole matrix)
 * </pre>
 * The matrix rewrites the rows that produce outputs and the rows that produce its own keys,
 * queries and learning rates, so it learns how to learn within a sequence. Only the initial
 * matrix {@code W_0} is trained, by backpropagation through the sequence. Per head; heads
 * split the input. Input {@code [batch, steps, features]}, output
 * {@code [batch, steps, outputs]}; every sequence starts from {@code W_0}.
 */
public final class SelfReferentialWeightMatrix implements Layer {

    private final int features;
    private final int outputs;
    private final int heads;
    private final int in;
    private final int out;
    private final Var w0;
    private Tensor lastWeights;

    /**
     * Creates the layer.
     *
     * @param features the input features (divisible by {@code heads})
     * @param outputs the output features (divisible by {@code heads})
     * @param heads the heads
     * @param random the random source for {@code W_0}
     */
    public SelfReferentialWeightMatrix(int features, int outputs, int heads, SplittableRandom random) {
        if (features % heads != 0 || outputs % heads != 0) {
            throw new IllegalArgumentException("features and outputs must be divisible by heads");
        }
        this.features = features;
        this.outputs = outputs;
        this.heads = heads;
        this.in = features / heads;
        this.out = outputs / heads;
        int rows = out + 2 * in + 1;
        w0 = Var.param(Tensor.normal(Shape.of(1, heads, rows, in), 0, (float) (1 / Math.sqrt(in)), random));
    }

    /**
     * Describes the layer for model builders, on inputs {@code [steps, features]}.
     *
     * @param outputs the output features
     * @param heads the heads
     * @return the spec
     */
    public static LayerSpec spec(int outputs, int heads) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(input.dim(0), outputs);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new SelfReferentialWeightMatrix(input.dim(-1), outputs, heads, random);
            }
        };
    }

    /** @return the matrices at the end of the last forward pass, {@code [batch, heads, rows, in]} */
    public Tensor lastWeights() {
        return lastWeights;
    }

    @Override
    public Var forward(Var input) {
        int n = input.shape().dim(0);
        int t = input.shape().dim(1);
        if (input.shape().rank() != 3 || input.shape().dim(2) != features) {
            throw new IllegalArgumentException(this + " expects [batch, steps, " + features + "], got " + input.shape());
        }
        int rows = out + 2 * in + 1;
        Var x = input.reshape(n, t, heads, in).permute(0, 2, 1, 3);   // [n, h, t, in]
        Var w = w0.expand(Shape.of(n, heads, rows, in));
        Var[] ys = new Var[t];
        for (int s = 0; s < t; s++) {
            Var xs = x.slice(2, s, s + 1);                       // [n, h, 1, in]
            Var all = xs.matmul(w.transpose());                  // [n, h, 1, rows]
            ys[s] = all.slice(-1, 0, out);
            Var k = Activation.SOFTMAX.forward(all.slice(-1, out, out + in));
            Var q = Activation.SOFTMAX.forward(all.slice(-1, out + in, out + 2 * in));
            Var beta = all.slice(-1, out + 2 * in, rows).sigmoid();
            Var v = q.matmul(w.transpose());                     // [n, h, 1, rows]
            Var vOld = k.matmul(w.transpose());
            w = w.add(v.sub(vOld).mul(beta).transpose().matmul(k));
        }
        lastWeights = w.value();
        return Var.concat(2, ys).permute(0, 2, 1, 3).reshape(n, t, outputs);
    }

    @Override
    public List<Var> parameters() {
        return List.of(w0);
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(input.dim(0), outputs);
    }

    @Override
    public String toString() {
        return "SelfReferentialWeightMatrix(" + features + " -> " + outputs + ", " + heads + " heads)";
    }
}
