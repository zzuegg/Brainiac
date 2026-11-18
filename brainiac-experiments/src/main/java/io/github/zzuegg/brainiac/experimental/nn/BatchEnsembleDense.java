package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * A BatchEnsemble dense layer (Wen, Tran and Ba, "BatchEnsemble: An Alternative Approach to
 * Efficient Ensemble and Lifelong Learning", 2020,
 * <a href="https://arxiv.org/abs/2002.06715">arXiv:2002.06715</a>): {@code k} ensemble members
 * share one weight matrix {@code W} and each owns rank-one adapters, so member {@code i} computes
 * {@code ((x * r_i) W) * s_i + b_i}. The ensemble costs one matrix of parameters plus
 * {@code k (in + 2 out)} and runs all members in one batched product. The building block of TabM.
 *
 * <p>Input {@code [batch, k, in]} (one slice per member), output {@code [batch, k, out]}.
 */
public final class BatchEnsembleDense implements Layer {

    private final int members;
    private final Var weight;
    private final Var r;
    private final Var s;
    private final Var bias;

    /**
     * Creates the layer. The adapters start at {@code +-1} with random signs, as in TabM, so the
     * members differ from the first step.
     *
     * @param in input width
     * @param out output width
     * @param members the ensemble size {@code k}
     * @param random the initialization source
     */
    public BatchEnsembleDense(int in, int out, int members, SplittableRandom random) {
        this.members = members;
        this.weight = ClassicOps.matrix(random, in, out);
        this.r = Var.param(signs(Shape.of(members, in), random));
        this.s = Var.param(signs(Shape.of(members, out), random));
        this.bias = TabularOps.zeros(members, out);
    }

    /**
     * @param out output width
     * @param members the ensemble size
     * @return a spec for {@code [batch, k, in]} inputs
     */
    public static LayerSpec spec(int out, int members) {
        return ClassicOps.spec(sh -> ClassicOps.withLast(sh, out),
                (sh, rnd) -> new BatchEnsembleDense(sh.dim(-1), out, members, rnd));
    }

    private static Tensor signs(Shape shape, SplittableRandom random) {
        float[] v = new float[Math.toIntExact(shape.size())];
        for (int i = 0; i < v.length; i++) {
            v[i] = random.nextBoolean() ? 1f : -1f;
        }
        return Tensor.of(shape, v);
    }

    /** @return the ensemble size */
    public int members() {
        return members;
    }

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        if (input.shape().rank() != 3 || input.shape().dim(1) != members) {
            throw new IllegalArgumentException(this + " expects [batch, " + members + ", in], got " + input.shape());
        }
        return input.mul(r).matmul(weight).mul(s).add(bias);
    }

    @Override
    public List<Var> parameters() {
        return List.of(weight, r, s, bias);
    }

    @Override
    public String toString() {
        return "BatchEnsembleDense(" + weight.shape() + ", k=" + members + ")";
    }
}
