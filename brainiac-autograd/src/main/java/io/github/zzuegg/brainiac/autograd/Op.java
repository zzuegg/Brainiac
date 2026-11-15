package io.github.zzuegg.brainiac.autograd;

import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * A differentiable operation with a hand-written gradient, for when composing existing
 * {@link Var} operations is not enough: a function autograd cannot express, a faster
 * fused computation, or a deliberately different gradient (such as a straight-through
 * estimator). Apply it with {@link Var#apply(Op, Var...)} and test its gradient with
 * {@link GradCheck#check(Op, Tensor...)}.
 *
 * <pre>{@code
 * // cube with its own backward pass
 * Op cube = new Op() {
 *     public Tensor forward(Tensor... in) { return in[0].mul(in[0]).mul(in[0]); }
 *     public Tensor[] backward(Tensor grad, Tensor out, Tensor... in) {
 *         return new Tensor[] {grad.mul(in[0].square()).mul(3)};
 *     }
 * };
 * Var y = Var.apply(cube, x);
 * }</pre>
 */
public interface Op {

    /**
     * Computes the result.
     *
     * @param inputs the input values, in the order passed to {@link Var#apply}
     * @return the result
     */
    Tensor forward(Tensor... inputs);

    /**
     * Computes the gradient of every input from the gradient of the result.
     *
     * @param grad the gradient of the result, same shape as {@code output}
     * @param output what {@link #forward} returned
     * @param inputs the input values
     * @return one gradient per input, each with that input's shape; {@code null} for an
     *     input that gets no gradient
     */
    Tensor[] backward(Tensor grad, Tensor output, Tensor... inputs);

    /**
     * Names the operation in error messages; the class name unless overridden.
     *
     * @return the name
     */
    default String name() {
        return getClass().getSimpleName().isEmpty() ? "custom op" : getClass().getSimpleName();
    }
}
