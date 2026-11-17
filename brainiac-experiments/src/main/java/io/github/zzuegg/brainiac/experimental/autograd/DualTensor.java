package io.github.zzuegg.brainiac.experimental.autograd;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * A dual number of tensors for <em>forward-mode</em> automatic differentiation: a value
 * {@code primal} and its directional derivative {@code tangent}, carried together through
 * every operation by the chain rule. One forward pass gives the Jacobian-vector product
 * {@code J v} of a whole computation for a chosen direction {@code v}, with no stored graph
 * and no backward pass, which is what forward gradients (Baydin et al. 2022,
 * <a href="https://arxiv.org/abs/2202.08587">arXiv:2202.08587</a>) build on.
 *
 * <pre>{@code
 * DualTensor w = DualTensor.of(weights, direction);   // seed the direction
 * DualTensor x = DualTensor.constant(inputs);
 * DualTensor loss = x.matmul(w).tanh().square().mean();
 * float directional = loss.tangent().item();          // d loss / d w . direction
 * }</pre>
 *
 * @param primal the value
 * @param tangent the derivative along the seeded direction, same shape
 */
public record DualTensor(Tensor primal, Tensor tangent) {

    /** Checks the shapes. */
    public DualTensor {
        if (!primal.shape().equals(tangent.shape())) {
            throw new IllegalArgumentException("tangent shape " + tangent.shape() + " differs from " + primal.shape());
        }
    }

    /**
     * @param primal the value
     * @param tangent the direction
     * @return the dual number
     */
    public static DualTensor of(Tensor primal, Tensor tangent) {
        return new DualTensor(primal, tangent);
    }

    /**
     * @param value a value that does not depend on the direction
     * @return the value with a zero tangent
     */
    public static DualTensor constant(Tensor value) {
        return new DualTensor(value, Tensor.zeros(value.shape(), value.dtype(), value.device()));
    }

    /** @return the shape */
    public Shape shape() {
        return primal.shape();
    }

    /**
     * @param o another dual
     * @return the sum
     */
    public DualTensor add(DualTensor o) {
        return new DualTensor(primal.add(o.primal), sumLike(tangent.add(o.tangent), primal.add(o.primal)));
    }

    /**
     * @param c a constant
     * @return the sum
     */
    public DualTensor add(float c) {
        return new DualTensor(primal.add(c), tangent);
    }

    /**
     * @param o another dual
     * @return the difference
     */
    public DualTensor sub(DualTensor o) {
        return add(o.neg());
    }

    /** @return the negation */
    public DualTensor neg() {
        return new DualTensor(primal.neg(), tangent.neg());
    }

    /**
     * Product rule: {@code (a b)' = a' b + a b'}.
     *
     * @param o another dual
     * @return the elementwise product
     */
    public DualTensor mul(DualTensor o) {
        Tensor p = primal.mul(o.primal);
        return new DualTensor(p, sumLike(tangent.mul(o.primal).add(primal.mul(o.tangent)), p));
    }

    /**
     * @param c a constant
     * @return the scaled dual
     */
    public DualTensor mul(float c) {
        return new DualTensor(primal.mul(c), tangent.mul(c));
    }

    /**
     * @param c a constant tensor, broadcast
     * @return the elementwise product
     */
    public DualTensor mul(Tensor c) {
        Tensor p = primal.mul(c);
        return new DualTensor(p, sumLike(tangent.mul(c), p));
    }

    /**
     * {@code (A B)' = A' B + A B'}.
     *
     * @param o the right operand
     * @return the matrix product
     */
    public DualTensor matmul(DualTensor o) {
        return new DualTensor(primal.matmul(o.primal), tangent.matmul(o.primal).add(primal.matmul(o.tangent)));
    }

    /** @return {@code tanh}, with {@code tanh' = 1 - tanh^2} */
    public DualTensor tanh() {
        Tensor y = primal.tanh();
        return new DualTensor(y, tangent.mul(y.square().neg().add(1)));
    }

    /** @return {@code max(0, x)} */
    public DualTensor relu() {
        return new DualTensor(primal.relu(), tangent.mul(primal.step()));
    }

    /** @return the logistic sigmoid */
    public DualTensor sigmoid() {
        Tensor y = primal.sigmoid();
        return new DualTensor(y, tangent.mul(y.mul(y.neg().add(1))));
    }

    /** @return {@code exp} */
    public DualTensor exp() {
        Tensor y = primal.exp();
        return new DualTensor(y, tangent.mul(y));
    }

    /** @return the natural logarithm */
    public DualTensor log() {
        return new DualTensor(primal.log(), tangent.div(primal));
    }

    /** @return the elementwise square */
    public DualTensor square() {
        return new DualTensor(primal.square(), tangent.mul(primal).mul(2));
    }

    /** @return the sum of all values, a scalar */
    public DualTensor sum() {
        return new DualTensor(primal.sum(), tangent.sum());
    }

    /** @return the mean of all values */
    public DualTensor mean() {
        return new DualTensor(primal.mean(), tangent.mean());
    }

    /**
     * @param axis the axis
     * @param keepDims whether to keep it with size 1
     * @return the sum along the axis
     */
    public DualTensor sum(int axis, boolean keepDims) {
        return new DualTensor(primal.sum(axis, keepDims), tangent.sum(axis, keepDims));
    }

    /**
     * Log-softmax over the last axis of {@code [rows, classes]}:
     * {@code (log softmax(x))' = x' - sum(softmax(x) x')}.
     *
     * @return the log-probabilities
     */
    public DualTensor logSoftmax() {
        Tensor shifted = primal.sub(primal.max(1, true));
        Tensor lse = shifted.exp().sum(1, true).log();
        Tensor logp = shifted.sub(lse);
        Tensor p = logp.exp();
        return new DualTensor(logp, tangent.sub(p.mul(tangent).sum(1, true)));
    }

    /**
     * Mean cross-entropy of logits against class ids.
     *
     * @param ids {@code [rows]} class ids
     * @return a scalar
     */
    public DualTensor crossEntropy(Tensor ids) {
        int rows = shape().dim(0);
        int classes = shape().dim(1);
        Tensor target = ids.reshape(rows, 1).equalTo(Tensor.arange(classes).to(ids.device()));
        return logSoftmax().mul(target).sum().mul(-1f / rows);
    }

    /** Reduces a broadcast tangent to the result's shape (it already has it unless broadcast). */
    private static Tensor sumLike(Tensor t, Tensor like) {
        return t.shape().equals(like.shape()) ? t : t.expand(like.shape());
    }
}
