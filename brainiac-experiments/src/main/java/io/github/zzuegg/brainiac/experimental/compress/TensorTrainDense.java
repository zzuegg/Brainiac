package io.github.zzuegg.brainiac.experimental.compress;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * A dense layer stored as a tensor train (Novikov et al. 2015, "Tensorizing neural networks",
 * <a href="https://arxiv.org/abs/1509.06569">arXiv:1509.06569</a>): the weight
 * {@code W [in, out]} with {@code in = i_1 ... i_d} and {@code out = j_1 ... j_d} is the chain of
 * cores {@code G_k [r_{k-1}, i_k, j_k, r_k]}, so it stores {@code sum r_{k-1} i_k j_k r_k} values
 * instead of {@code in * out}. The forward pass contracts the input with one core at a time and
 * never rebuilds {@code W}; gradients reach the cores through ordinary {@link Var} operations, so
 * the layer fine-tunes like any other. A drop-in {@link Layer} for a core {@link Dense}
 * ({@link #from}).
 */
public final class TensorTrainDense implements Layer {

    private final int[] inFactors;
    private final int[] outFactors;
    private final List<Var> cores = new ArrayList<>();
    private final Var bias;
    private final Activation activation;

    /**
     * @param cores the cores {@code [r_{k-1}, i_k, j_k, r_k]}, {@code r_0 = r_d = 1}
     * @param bias the bias {@code [out]}, or null
     * @param activation the activation
     */
    public TensorTrainDense(List<Tensor> cores, Tensor bias, Activation activation) {
        int d = cores.size();
        this.inFactors = new int[d];
        this.outFactors = new int[d];
        int out = 1;
        for (int k = 0; k < d; k++) {
            Shape s = cores.get(k).shape();
            inFactors[k] = s.dim(1);
            outFactors[k] = s.dim(2);
            out *= outFactors[k];
            this.cores.add(Var.param(cores.get(k)));
        }
        this.bias = Var.param(bias != null ? bias : Tensor.zeros(Shape.of(out)));
        this.activation = activation.bind(out);
    }

    /**
     * Compresses a trained dense layer by TT-SVD.
     *
     * @param dense the layer
     * @param inFactors the factorisation of its input size
     * @param outFactors the factorisation of its output size
     * @param maxRank the largest TT rank
     * @return the layer
     */
    public static TensorTrainDense from(Dense dense, int[] inFactors, int[] outFactors, int maxRank) {
        return new TensorTrainDense(TensorDecomposition.tensorTrain(dense.weight().value(), inFactors, outFactors,
                maxRank), dense.bias() == null ? null : dense.bias().value(), dense.activation());
    }

    /**
     * Contracts a batch with the train.
     *
     * @param x {@code [batch, in]}
     * @return {@code x W}, {@code [batch, out]}
     */
    public Var multiply(Var x) {
        int b = x.shape().dim(0);
        int d = inFactors.length;
        int done = 1;          // product of the j already produced
        int rank = 1;          // r_{k-1}
        int restIn = 1;
        for (int f : inFactors) {
            restIn *= f;
        }
        Var t = x.reshape(b, restIn);   // [b * done, rank * i_k * restAfter]
        for (int k = 0; k < d; k++) {
            int ik = inFactors[k];
            int jk = outFactors[k];
            restIn /= ik;
            Var core = cores.get(k);
            int rk = core.shape().dim(3);
            // [b*done, rank*ik, rest] -> [b*done*rest, rank*ik]
            Var m = t.reshape(b * done, rank * ik, restIn).permute(0, 2, 1).reshape(b * done * restIn, rank * ik);
            Var y = m.matmul(core.reshape(rank * ik, jk * rk));   // [b*done*rest, jk*rk]
            // -> [b*done, jk, rk, rest]
            t = y.reshape(b * done, restIn, jk, rk).permute(0, 2, 3, 1).reshape(b * done * jk, rk * restIn);
            done *= jk;
            rank = rk;
        }
        return t.reshape(b, done);
    }

    @Override
    public Var forward(Var input) {
        return activation.forward(multiply(input).add(bias));
    }

    /** @return the number of stored weights (cores only) */
    public long weightCount() {
        long n = 0;
        for (Var c : cores) {
            n += c.shape().size();
        }
        return n;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(cores);
        p.add(bias);
        return p;
    }

    @Override
    public int outputSize(int inputSize) {
        int out = 1;
        for (int f : outFactors) {
            out *= f;
        }
        return out;
    }

    @Override
    public String toString() {
        return "TensorTrainDense(" + cores.size() + " cores, " + weightCount() + " weights)";
    }
}
