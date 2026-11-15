package io.github.zzuegg.brainiac.autograd;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Deque;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * A tensor value that remembers how it was computed, so gradients can flow back to it.
 *
 * <p>Create trainable values with {@link #param(Tensor)} and fixed inputs with
 * {@link #of(Tensor)}. Every operation on a {@code Var} returns a new {@code Var} and,
 * when any input needs a gradient, records itself. Calling {@link #backward()} on a
 * scalar result walks those records in reverse and stores {@code d(result)/d(param)} in
 * each parameter's {@link #grad()}.
 *
 * <pre>{@code
 * Var w = Var.param(Tensor.uniform(Shape.of(2, 8), -1, 1, rng));
 * Var x = Var.of(inputs);
 * Var loss = x.matmul(w).tanh().sub(targets).square().mean();
 * loss.backward();
 * Tensor dw = w.grad();
 * }</pre>
 *
 * <p>Gradients are computed with ordinary tensor operations, so they run on whatever
 * device the values live on. Gradients accumulate across {@code backward()} calls until
 * {@link #zeroGrad()} is called.
 */
public final class Var {

    /** Computes the gradients of a node's inputs from the gradient of its output. */
    @FunctionalInterface
    private interface Backward {
        Tensor[] apply(Tensor grad);
    }

    private static final Var[] NO_INPUTS = new Var[0];

    private Tensor value;
    private final boolean requiresGrad;
    private final Var[] inputs;
    private final Backward backward;
    private Tensor grad;

    private Var(Tensor value, boolean requiresGrad, Var[] inputs, Backward backward) {
        this.value = Objects.requireNonNull(value, "value");
        this.requiresGrad = requiresGrad;
        this.inputs = inputs;
        this.backward = backward;
    }

    /**
     * Wraps a fixed value, such as input data or a target. No gradient is computed for it.
     *
     * @param value the value
     * @return a constant {@code Var}
     */
    public static Var of(Tensor value) {
        return new Var(value, false, NO_INPUTS, null);
    }

    /**
     * Wraps a trainable value, such as a weight matrix. After {@link #backward()} its
     * gradient is available from {@link #grad()}.
     *
     * @param value the initial value
     * @return a parameter {@code Var}
     */
    public static Var param(Tensor value) {
        return new Var(value, true, NO_INPUTS, null);
    }

    /**
     * Returns the value.
     *
     * @return the value
     */
    public Tensor value() {
        return value;
    }

    /**
     * Returns the shape of the value.
     *
     * @return the shape
     */
    public Shape shape() {
        return value.shape();
    }

    /**
     * Returns whether gradients flow to or through this value.
     *
     * @return {@code true} for parameters and values computed from them
     */
    public boolean requiresGrad() {
        return requiresGrad;
    }

    /**
     * Returns the accumulated gradient of a parameter.
     *
     * @return the gradient, same shape as the value, or {@code null} before the first
     *     {@link #backward()} that reaches this parameter
     */
    public Tensor grad() {
        return grad;
    }

    /**
     * Replaces the value of a leaf (a parameter or a constant), for example after an
     * optimizer step or to update a layer's running statistics. Values computed earlier
     * keep the value they were computed from.
     *
     * @param newValue the new value; must have the same shape
     * @throws IllegalStateException if this value was computed from other values
     */
    public void assign(Tensor newValue) {
        if (backward != null) {
            throw new IllegalStateException("only leaf values (parameters and constants) can be assigned");
        }
        if (!newValue.shape().equals(value.shape())) {
            throw new IllegalArgumentException("new value shape " + newValue.shape() + " does not match " + value.shape());
        }
        value = newValue;
    }

    /** Clears the accumulated gradient. */
    public void zeroGrad() {
        grad = null;
    }

    /**
     * Returns the same value cut off from the graph: no gradient flows back through it.
     *
     * @return a constant {@code Var} with this value
     */
    public Var detach() {
        return of(value);
    }

    // ---------------------------------------------------------------- backward pass

    /**
     * Computes the gradient of this scalar with respect to every parameter it depends on
     * and adds it to their {@link #grad()}.
     *
     * @throws IllegalStateException if this value has more than one element or does not
     *     depend on any parameter
     */
    public void backward() {
        if (value.size() != 1) {
            throw new IllegalStateException(
                    "backward() needs a scalar such as a loss, got shape " + shape() + "; use backward(Tensor)");
        }
        backward(Tensor.full(shape(), 1f, value.device()));
    }

    /**
     * Back-propagates a given output gradient: computes {@code seed . d(this)/d(param)}
     * for every parameter this value depends on and adds it to their {@link #grad()}.
     *
     * @param seed the gradient of the final result with respect to this value; same shape
     * @throws IllegalStateException if this value does not depend on any parameter
     */
    public void backward(Tensor seed) {
        if (!requiresGrad) {
            throw new IllegalStateException("this value does not depend on any parameter");
        }
        if (!seed.shape().equals(shape())) {
            throw new IllegalArgumentException("seed shape " + seed.shape() + " does not match " + shape());
        }
        Map<Var, Tensor> grads = new IdentityHashMap<>();
        grads.put(this, seed);
        for (Var node : topologicalOrder()) {
            Tensor g = grads.remove(node);
            if (g == null) {
                continue;
            }
            if (node.backward == null) {
                node.grad = node.grad == null ? g : node.grad.add(g);
                continue;
            }
            Tensor[] inputGrads = node.backward.apply(g);
            for (int i = 0; i < node.inputs.length; i++) {
                Var in = node.inputs[i];
                if (in.requiresGrad) {
                    grads.merge(in, inputGrads[i], Tensor::add);
                }
            }
        }
    }

    /** Returns this node and its ancestors that need gradients, outputs before inputs. */
    private List<Var> topologicalOrder() {
        List<Var> postOrder = new ArrayList<>();
        Map<Var, Boolean> visited = new IdentityHashMap<>();
        Deque<Var> stack = new ArrayDeque<>();
        Deque<Integer> nextChild = new ArrayDeque<>();
        stack.push(this);
        nextChild.push(0);
        visited.put(this, Boolean.TRUE);
        // Iterative depth-first search, so deep graphs cannot overflow the call stack.
        while (!stack.isEmpty()) {
            Var node = stack.peek();
            int i = nextChild.pop();
            if (i < node.inputs.length) {
                nextChild.push(i + 1);
                Var child = node.inputs[i];
                if (child.requiresGrad && visited.put(child, Boolean.TRUE) == null) {
                    stack.push(child);
                    nextChild.push(0);
                }
            } else {
                stack.pop();
                postOrder.add(node);
            }
        }
        Collections.reverse(postOrder);
        return postOrder;
    }

    private static Var op(Tensor out, Var[] inputs, Backward backward) {
        for (Var in : inputs) {
            if (in.requiresGrad) {
                return new Var(out, true, inputs, backward);
            }
        }
        return of(out);
    }

    private Var unary(Tensor out, Backward backward) {
        return op(out, new Var[] {this}, backward);
    }

    private Var binary(Var other, Tensor out, Backward backward) {
        return op(out, new Var[] {this, other}, backward);
    }

    // ---------------------------------------------------------------- arithmetic

    /**
     * Adds element-wise with broadcasting.
     *
     * @param other the value to add
     * @return {@code this + other}
     */
    public Var add(Var other) {
        Shape a = shape();
        Shape b = other.shape();
        return binary(other, value.add(other.value), g -> new Tensor[] {g.sumTo(a), g.sumTo(b)});
    }

    /**
     * Adds a fixed tensor element-wise with broadcasting.
     *
     * @param other the tensor to add
     * @return {@code this + other}
     */
    public Var add(Tensor other) {
        return add(of(other));
    }

    /**
     * Adds a constant.
     *
     * @param c the constant
     * @return {@code this + c}
     */
    public Var add(float c) {
        return unary(value.add(c), g -> new Tensor[] {g});
    }

    /**
     * Subtracts element-wise with broadcasting.
     *
     * @param other the value to subtract
     * @return {@code this - other}
     */
    public Var sub(Var other) {
        Shape a = shape();
        Shape b = other.shape();
        return binary(other, value.sub(other.value), g -> new Tensor[] {g.sumTo(a), g.neg().sumTo(b)});
    }

    /**
     * Subtracts a fixed tensor element-wise with broadcasting.
     *
     * @param other the tensor to subtract
     * @return {@code this - other}
     */
    public Var sub(Tensor other) {
        return sub(of(other));
    }

    /**
     * Subtracts a constant.
     *
     * @param c the constant
     * @return {@code this - c}
     */
    public Var sub(float c) {
        return unary(value.sub(c), g -> new Tensor[] {g});
    }

    /**
     * Multiplies element-wise with broadcasting.
     *
     * @param other the value to multiply by
     * @return {@code this * other}
     */
    public Var mul(Var other) {
        Tensor a = value;
        Tensor b = other.value;
        return binary(other, a.mul(b),
                g -> new Tensor[] {g.mul(b).sumTo(a.shape()), g.mul(a).sumTo(b.shape())});
    }

    /**
     * Multiplies element-wise by a fixed tensor with broadcasting.
     *
     * @param other the tensor to multiply by
     * @return {@code this * other}
     */
    public Var mul(Tensor other) {
        return mul(of(other));
    }

    /**
     * Multiplies by a constant.
     *
     * @param c the constant
     * @return {@code this * c}
     */
    public Var mul(float c) {
        return unary(value.mul(c), g -> new Tensor[] {g.mul(c)});
    }

    /**
     * Divides element-wise with broadcasting.
     *
     * @param other the divisor
     * @return {@code this / other}
     */
    public Var div(Var other) {
        Tensor a = value;
        Tensor b = other.value;
        return binary(other, a.div(b), g -> new Tensor[] {
            g.div(b).sumTo(a.shape()),
            g.mul(a).div(b.square()).neg().sumTo(b.shape())
        });
    }

    /**
     * Divides element-wise by a fixed tensor with broadcasting.
     *
     * @param other the divisor
     * @return {@code this / other}
     */
    public Var div(Tensor other) {
        return div(of(other));
    }

    /**
     * Divides by a constant.
     *
     * @param c the divisor
     * @return {@code this / c}
     */
    public Var div(float c) {
        return unary(value.div(c), g -> new Tensor[] {g.div(c)});
    }

    /**
     * Raises every value to a constant power.
     *
     * @param p the exponent
     * @return {@code this ^ p}
     */
    public Var pow(float p) {
        Tensor x = value;
        return unary(x.pow(p), g -> new Tensor[] {g.mul(x.pow(p - 1)).mul(p)});
    }

    /** @return {@code -this} */
    public Var neg() {
        return unary(value.neg(), g -> new Tensor[] {g.neg()});
    }

    // ---------------------------------------------------------------- functions

    /** @return each value squared */
    public Var square() {
        Tensor x = value;
        return unary(x.square(), g -> new Tensor[] {g.mul(x).mul(2)});
    }

    /** @return the square root of each value */
    public Var sqrt() {
        Tensor out = value.sqrt();
        return unary(out, g -> new Tensor[] {g.div(out.mul(2))});
    }

    /** @return {@code e} raised to each value */
    public Var exp() {
        Tensor out = value.exp();
        return unary(out, g -> new Tensor[] {g.mul(out)});
    }

    /** @return the natural logarithm of each value */
    public Var log() {
        Tensor x = value;
        return unary(x.log(), g -> new Tensor[] {g.div(x)});
    }

    /** @return the absolute values; the gradient at 0 is 0 */
    public Var abs() {
        Tensor x = value;
        return unary(x.abs(), g -> new Tensor[] {g.mul(x.sign())});
    }

    /** @return the hyperbolic tangent of each value */
    public Var tanh() {
        Tensor out = value.tanh();
        return unary(out, g -> new Tensor[] {g.mul(out.square().neg().add(1))});
    }

    /** @return the logistic sigmoid of each value */
    public Var sigmoid() {
        Tensor out = value.sigmoid();
        return unary(out, g -> new Tensor[] {g.mul(out).mul(out.neg().add(1))});
    }

    /** @return {@code max(0, x)} for each value; the gradient at 0 is 0 */
    public Var relu() {
        Tensor x = value;
        return unary(x.relu(), g -> new Tensor[] {g.mul(x.step())});
    }

    // ---------------------------------------------------------------- linear algebra and shape

    /**
     * Multiplies matrices: {@code [m, k] x [k, n] -> [m, n]}.
     *
     * @param other the right-hand matrix
     * @return the matrix product
     */
    public Var matmul(Var other) {
        Tensor a = value;
        Tensor b = other.value;
        return binary(other, a.matmul(b),
                g -> new Tensor[] {g.matmul(b.transpose()), a.transpose().matmul(g)});
    }

    /**
     * Multiplies by a fixed matrix.
     *
     * @param other the right-hand matrix
     * @return the matrix product
     */
    public Var matmul(Tensor other) {
        return matmul(of(other));
    }

    /** @return the value with its last two dimensions swapped */
    public Var transpose() {
        return unary(value.transpose(), g -> new Tensor[] {g.transpose()});
    }

    /**
     * Returns the same values in a new shape.
     *
     * @param dims the new dimensions
     * @return the reshaped value
     */
    public Var reshape(int... dims) {
        Shape original = shape();
        return unary(value.reshape(dims), g -> new Tensor[] {g.reshape(original)});
    }

    // ---------------------------------------------------------------- reductions

    /** @return the sum of all values, as a scalar */
    public Var sum() {
        Shape s = shape();
        return unary(value.sum(), g -> new Tensor[] {g.expand(s)});
    }

    /**
     * Sums along one axis.
     *
     * @param axis the axis; negative values count from the end
     * @param keepDims keep the axis with size 1 instead of removing it
     * @return the sums
     */
    public Var sum(int axis, boolean keepDims) {
        Shape s = shape();
        Tensor keptSum = value.sum(axis, true);
        Shape kept = keptSum.shape();
        return unary(squeeze(keptSum, axis, keepDims), g -> new Tensor[] {g.reshape(kept).expand(s)});
    }

    /** @return the mean of all values, as a scalar */
    public Var mean() {
        Shape s = shape();
        float n = s.size();
        return unary(value.mean(), g -> new Tensor[] {g.div(n).expand(s)});
    }

    /**
     * Averages along one axis.
     *
     * @param axis the axis; negative values count from the end
     * @param keepDims keep the axis with size 1 instead of removing it
     * @return the means
     */
    public Var mean(int axis, boolean keepDims) {
        Shape s = shape();
        Tensor keptMean = value.mean(axis, true);
        Shape kept = keptMean.shape();
        float n = s.dim(axis);
        return unary(squeeze(keptMean, axis, keepDims), g -> new Tensor[] {g.reshape(kept).div(n).expand(s)});
    }

    /**
     * Takes the maximum along one axis. The gradient flows to the maximum; if several
     * values tie, each of them receives it.
     *
     * @param axis the axis; negative values count from the end
     * @param keepDims keep the axis with size 1 instead of removing it
     * @return the maxima
     */
    public Var max(int axis, boolean keepDims) {
        Tensor x = value;
        Tensor keptMax = x.max(axis, true);
        Shape kept = keptMax.shape();
        return unary(squeeze(keptMax, axis, keepDims), g -> new Tensor[] {x.equalTo(keptMax).mul(g.reshape(kept))});
    }

    /** Removes a size-1 axis that a keepDims reduction left, unless it should stay. */
    private static Tensor squeeze(Tensor kept, int axis, boolean keepDims) {
        if (keepDims) {
            return kept;
        }
        int[] dims = kept.shape().toArray();
        int a = axis < 0 ? dims.length + axis : axis;
        int[] out = new int[dims.length - 1];
        System.arraycopy(dims, 0, out, 0, a);
        System.arraycopy(dims, a + 1, out, a, dims.length - a - 1);
        return kept.reshape(out);
    }

    @Override
    public String toString() {
        return (requiresGrad ? "Var(grad) " : "Var ") + value;
    }
}
