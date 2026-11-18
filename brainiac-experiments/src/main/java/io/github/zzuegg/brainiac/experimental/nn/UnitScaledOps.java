package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Unit scaling (Blake et al. 2023, <a href="https://arxiv.org/abs/2303.11257">arXiv:2303.11257</a>;
 * u-muP, Blake et al. 2024, <a href="https://arxiv.org/abs/2407.17465">arXiv:2407.17465</a>):
 * operations whose forward <em>and</em> backward passes keep unit variance at initialization,
 * with weights initialized from {@code N(0, 1)}. A matmul divides its output by
 * {@code sqrt(fan_in)}, but its input gradient by {@code sqrt(fan_out)} and its weight gradient by
 * {@code sqrt(batch)}; separate scales per direction are what lets every tensor (activations,
 * weights, gradients) sit near 1, so FP8 and bf16 need no loss scaling or per-tensor scales.
 * The backward scales change only the size of each gradient tensor, not its direction, which
 * Adam-type optimizers ignore; {@code experimental.train.UnitMup} supplies the matching
 * learning rates.
 */
public final class UnitScaledOps {

    /** The standard deviation of {@code tanh(z)} for {@code z ~ N(0, 1)}. */
    public static final float TANH_STD = 0.6278f;

    private UnitScaledOps() {
    }

    /**
     * A unit-scaled matmul {@code x W / sqrt(fan_in)}.
     *
     * @param x {@code [batch, in]}
     * @param w {@code [in, out]}, initialized {@code N(0, 1)}
     * @return {@code [batch, out]}
     */
    public static Var matmul(Var x, Var w) {
        int in = w.shape().dim(0);
        int out = w.shape().dim(1);
        int batch = x.shape().dim(0);
        float fwd = (float) (1 / Math.sqrt(in));
        float bx = (float) (1 / Math.sqrt(out));
        float bw = (float) (1 / Math.sqrt(batch));
        return Var.custom(t -> t[0].matmul(t[1]).mul(fwd), (t, o, g) -> new Tensor[] {
                g.matmul(t[1].transpose()).mul(bx),
                t[0].transpose().matmul(g).mul(bw)
        }, x, w);
    }

    /**
     * The muP readout: {@code x W / fan_in}, whose output shrinks with width as muP requires, with
     * unit-scaled gradients.
     *
     * @param x {@code [batch, in]}
     * @param w {@code [in, out]}, initialized {@code N(0, 1)}
     * @return {@code [batch, out]}
     */
    public static Var readout(Var x, Var w) {
        int in = w.shape().dim(0);
        int batch = x.shape().dim(0);
        float fwd = 1f / in;
        float bx = (float) (1 / Math.sqrt(w.shape().dim(1)));
        float bw = (float) (1 / Math.sqrt(batch));
        return Var.custom(t -> t[0].matmul(t[1]).mul(fwd), (t, o, g) -> new Tensor[] {
                g.matmul(t[1].transpose()).mul(bx),
                t[0].transpose().matmul(g).mul(bw)
        }, x, w);
    }

    /**
     * A unit-scaled tanh, {@code tanh(x) / 0.6278}, unit variance for unit-normal inputs.
     *
     * @param x the input
     * @return the scaled activation
     */
    public static Var tanh(Var x) {
        return x.tanh().mul(1 / TANH_STD);
    }

    /**
     * A unit-scaled residual connection, {@code sqrt(1 - tau) skip + sqrt(tau) branch}.
     *
     * @param skip the skip path
     * @param branch the residual branch
     * @param tau the branch's share of the variance, in {@code (0, 1)}
     * @return the sum, unit variance when both inputs are
     */
    public static Var residualAdd(Var skip, Var branch, float tau) {
        return skip.mul((float) Math.sqrt(1 - tau)).add(branch.mul((float) Math.sqrt(tau)));
    }

    /**
     * Softmax cross-entropy whose value is the batch mean but whose gradient to the logits is
     * per example ({@code p - y}, not {@code (p - y) / batch}), so it stays unit-scaled however
     * large the batch.
     *
     * @param logits {@code [batch, classes]}
     * @param labels {@code [batch]} class ids
     * @return the mean loss
     */
    public static Var crossEntropy(Var logits, Tensor labels) {
        int batch = logits.shape().dim(0);
        int classes = logits.shape().dim(1);
        float[] y = labels.toArray();
        float[] onehot = new float[batch * classes];
        for (int i = 0; i < batch; i++) {
            onehot[i * classes + (int) y[i]] = 1;
        }
        Tensor target = Tensor.of(Shape.of(batch, classes), onehot);
        return Var.custom(t -> {
            Tensor lse = t[0].logsumexp(1, true);
            return lse.sub(t[0].mul(target).sum(1, true)).mean();
        }, (t, o, g) -> {
            Tensor p = t[0].sub(t[0].logsumexp(1, true)).exp();
            return new Tensor[] {p.sub(target).mul(g)};
        }, logits);
    }
}
