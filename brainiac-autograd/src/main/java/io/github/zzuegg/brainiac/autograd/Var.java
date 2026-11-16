package io.github.zzuegg.brainiac.autograd;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.tensor.spi.ActivationOp;

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
    /** Gradients computed by {@link #backward(Tensor)} from this value, for {@link #release()}. */
    private List<Tensor> backwardValues;
    /** The parameters' gradients {@link #backward(Tensor)} computed: never released here. */
    private List<Tensor> backwardGrads;

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
     * @param newValue the new value; must have the same shape, and may have another
     *     number format
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

    /**
     * Replaces the accumulated gradient, for gradient clipping, accumulation or custom
     * update rules.
     *
     * @param newGrad the gradient, same shape as the value, or {@code null} to clear it
     */
    public void setGrad(Tensor newGrad) {
        if (newGrad != null && !newGrad.shape().equals(value.shape())) {
            throw new IllegalArgumentException("gradient shape " + newGrad.shape() + " does not match " + value.shape());
        }
        grad = newGrad;
    }

    /** Clears the accumulated gradient. */
    public void zeroGrad() {
        grad = null;
    }

    /**
     * Converts to another number format. Gradients flow back converted to this value's
     * format, so a float16 computation can train float32 parameters.
     *
     * @param type the new type
     * @return the converted value; {@code this} if it already has that type
     */
    public Var to(DType type) {
        if (type == value.dtype()) {
            return this;
        }
        DType original = value.dtype();
        return unary(value.to(type), g -> new Tensor[] {g.to(original)});
    }

    /** @return the number format of the value */
    public DType dtype() {
        return value.dtype();
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
        backward(Tensor.full(shape(), 1f, value.dtype(), value.device()));
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
        List<Tensor> created = Tensor.created(() -> {
            Map<Var, Tensor> grads = new IdentityHashMap<>();
            grads.put(this, seed);
            for (Var node : topologicalOrder()) {
                Tensor g = grads.remove(node);
                if (g == null) {
                    continue;
                }
                // a gradient always has its value's number format
                g = g.to(node.value.dtype());
                if (node.backward == null) {
                    node.grad = node.grad == null ? g : node.grad.add(g).to(node.value.dtype());
                    continue;
                }
                Tensor[] inputGrads = node.backward.apply(g);
                for (int i = 0; i < node.inputs.length; i++) {
                    Var in = node.inputs[i];
                    if (in.requiresGrad) {
                        grads.merge(in, inputGrads[i].to(in.value.dtype()), Tensor::add);
                    }
                }
            }
        });
        if (backwardValues == null) {
            backwardValues = new ArrayList<>(created.size());
            backwardGrads = new ArrayList<>();
        }
        backwardValues.addAll(created);   // gradients and what they were computed with
        for (Var node : topologicalOrder()) {
            if (node.backward == null && node.grad != null) {
                backwardGrads.add(node.grad);   // an optimizer may keep them, such as for momentum
            }
        }
    }

    /**
     * Frees the device memory of every value computed on the way to this one and of the
     * values {@link #backward()} computed from it on the way to the parameters' gradients,
     * rather than waiting for the garbage collector. Parameters, constants, their
     * gradients, this value itself and {@code keep} stay. Use it once a training step is done: the intermediate values must not be used
     * afterwards, and anything not yet computed from them (see {@link Tensor#evaluate})
     * must be computed first.
     *
     * @param keep further values that stay in use, such as metrics computed from an
     *     intermediate value
     */
    public void release(Tensor... keep) {
        List<Tensor> dead = new ArrayList<>(backwardValues == null ? List.of() : backwardValues);
        List<Tensor> kept = new ArrayList<>(List.of(keep));
        kept.add(value);
        if (backwardGrads != null) {
            kept.addAll(backwardGrads);
        }
        Map<Var, Boolean> visited = new IdentityHashMap<>();
        Deque<Var> stack = new ArrayDeque<>();
        stack.push(this);
        visited.put(this, Boolean.TRUE);
        while (!stack.isEmpty()) {
            Var node = stack.pop();
            if (node.inputs.length == 0) {
                kept.add(node.value);
                if (node.grad != null) {
                    kept.add(node.grad);
                }
            } else if (node != this) {
                dead.add(node.value);
            }
            for (Var in : node.inputs) {
                if (visited.put(in, Boolean.TRUE) == null) {
                    stack.push(in);
                }
            }
        }
        backwardValues = null;
        backwardGrads = null;
        // Gradients and kept values may still be computed from the intermediate ones.
        // (Values of parameters and constants never are: only values that need no
        // gradient are computed into constants.)
        List<Tensor> computed = new ArrayList<>(List.of(keep));
        computed.add(value);
        for (Var node : visited.keySet()) {
            if (node.backward == null && node.grad != null) {
                computed.add(node.grad);
            }
        }
        Tensor.evaluate(computed);
        Tensor.release(dead, kept);
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
        // a constant, but computed in this step: its inputs say so to release()
        return new Var(out, false, inputs, null);
    }

    private Var unary(Tensor out, Backward backward) {
        return op(out, new Var[] {this}, backward);
    }

    private Var binary(Var other, Tensor out, Backward backward) {
        return op(out, new Var[] {this, other}, backward);
    }

    /**
     * Applies a custom operation with a hand-written gradient.
     *
     * @param op the operation
     * @param inputs its inputs
     * @return the result, recorded so gradients flow through {@link Op#backward}
     */
    public static Var apply(Op op, Var... inputs) {
        Objects.requireNonNull(op, "op");
        Tensor[] values = new Tensor[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            values[i] = Objects.requireNonNull(inputs[i], "input " + i).value;
        }
        Tensor out = Objects.requireNonNull(op.forward(values.clone()), () -> op.name() + " returned null");
        return op(out, inputs.clone(), g -> {
            if (!g.shape().equals(out.shape())) {
                throw new IllegalStateException(op.name() + ": gradient shape " + g.shape()
                        + " does not match output " + out.shape());
            }
            Tensor[] grads = op.backward(g, out, values.clone());
            if (grads == null || grads.length != values.length) {
                throw new IllegalStateException(op.name() + ".backward must return " + values.length
                        + " gradients, got " + (grads == null ? "null" : grads.length));
            }
            for (int i = 0; i < grads.length; i++) {
                if (grads[i] == null && !inputs[i].requiresGrad) {
                    continue;   // never read
                }
                if (grads[i] == null) {
                    grads[i] = Tensor.zeros(values[i].shape(), values[i].dtype(), values[i].device());
                } else if (!grads[i].shape().equals(values[i].shape())) {
                    throw new IllegalStateException(op.name() + ": gradient " + i + " has shape " + grads[i].shape()
                            + ", input " + i + " has shape " + values[i].shape());
                }
            }
            return grads;
        });
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
        return activation(ActivationOp.TANH, 0);
    }

    /** @return the logistic sigmoid of each value */
    public Var sigmoid() {
        return activation(ActivationOp.SIGMOID, 0);
    }

    /** @return {@code max(0, x)} for each value; the gradient at 0 is 0 */
    public Var relu() {
        return activation(ActivationOp.RELU, 0);
    }

    /**
     * Applies an activation function with a fused kernel: one pass forward and one pass
     * backward, instead of a chain of simpler operations.
     *
     * @param op the activation
     * @param param its parameter (such as the leaky ReLU slope or the ELU alpha); ignored
     *     by activations without one
     * @return {@code f(this)}
     */
    public Var activation(ActivationOp op, float param) {
        Tensor x = value;
        return unary(x.activation(op, param), g -> new Tensor[] {x.activationBackward(op, param, g)});
    }

    /** @return the Gauss error function of each value */
    public Var erf() {
        Tensor x = value;
        // d/dx erf(x) = 2 / sqrt(pi) * e^(-x^2)
        return unary(x.erf(), g -> new Tensor[] {g.mul(x.square().neg().exp().mul((float) (2 / Math.sqrt(Math.PI))))});
    }

    /** @return {@code log(1 + e^x)} for each value, computed without overflow */
    public Var softplus() {
        return activation(ActivationOp.SOFTPLUS, 0);
    }

    /**
     * Returns the element-wise maximum with broadcasting. Where both are equal, the
     * gradient goes to {@code other}.
     *
     * @param other the other value
     * @return {@code max(this, other)}
     */
    public Var maximum(Var other) {
        Tensor a = value;
        Tensor b = other.value;
        Shape sa = shape();
        Shape sb = other.shape();
        return binary(other, a.maximum(b), g -> {
            Tensor pick = a.greater(b);
            return new Tensor[] {g.mask(pick).sumTo(sa), g.mask(pick.neg().add(1)).sumTo(sb)};
        });
    }

    /**
     * Returns the element-wise minimum with broadcasting. Where both are equal, the
     * gradient goes to {@code other}.
     *
     * @param other the other value
     * @return {@code min(this, other)}
     */
    public Var minimum(Var other) {
        Tensor a = value;
        Tensor b = other.value;
        Shape sa = shape();
        Shape sb = other.shape();
        return binary(other, a.minimum(b), g -> {
            Tensor pick = b.greater(a);
            return new Tensor[] {g.mask(pick).sumTo(sa), g.mask(pick.neg().add(1)).sumTo(sb)};
        });
    }

    /**
     * @param c the lower bound
     * @return {@code max(this, c)}; the gradient is 0 where the bound applies
     */
    public Var maximum(float c) {
        Tensor x = value;
        Tensor bound = Tensor.full(Shape.of(), c, x.device());
        return unary(x.maximum(bound), g -> new Tensor[] {g.mask(x.greater(bound))});
    }

    /**
     * @param c the upper bound
     * @return {@code min(this, c)}; the gradient is 0 where the bound applies
     */
    public Var minimum(float c) {
        Tensor x = value;
        Tensor bound = Tensor.full(Shape.of(), c, x.device());
        return unary(x.minimum(bound), g -> new Tensor[] {g.mask(bound.greater(x))});
    }

    /**
     * Limits every value to {@code [low, high]}.
     *
     * @param low the lower bound
     * @param high the upper bound
     * @return the clamped values; the gradient is 0 where a bound applies
     */
    public Var clamp(float low, float high) {
        if (!(low <= high)) {
            throw new IllegalArgumentException("clamp needs low <= high, got " + low + " and " + high);
        }
        return maximum(low).minimum(high);
    }

    /**
     * Picks values from {@code ifTrue} where {@code condition} is non-zero and from
     * {@code ifFalse} elsewhere, like {@code condition ? ifTrue : ifFalse} per element.
     * Values that are not picked never leak into the result, even if infinite or NaN,
     * and get a gradient of exactly 0. The condition itself gets no gradient.
     *
     * <p>As in other autograd libraries, a branch that overflows can still produce NaN
     * gradients upstream ({@code 0 * infinity}), so keep both branches finite:
     *
     * <pre>{@code
     * // ELU: exp of min(x, 0) so the unused branch cannot overflow
     * Var elu = Var.where(x.greater(0), x, x.minimum(0).exp().sub(1));
     * }</pre>
     *
     * @param condition the condition; broadcasts
     * @param ifTrue values where the condition holds
     * @param ifFalse values elsewhere
     * @return the picked values
     */
    public static Var where(Tensor condition, Var ifTrue, Var ifFalse) {
        Shape sa = ifTrue.shape();
        Shape sb = ifFalse.shape();
        Tensor otherwise = condition.equalTo(Tensor.full(Shape.of(), 0f, condition.device()));
        return op(Tensor.where(condition, ifTrue.value, ifFalse.value), new Var[] {ifTrue, ifFalse},
                g -> new Tensor[] {g.mask(condition).sumTo(sa), g.mask(otherwise).sumTo(sb)});
    }

    /**
     * @param other the value to compare with
     * @return a constant {@code 1} where {@code this > other}, else {@code 0}; no gradient
     */
    public Tensor greater(float other) {
        return value.greater(Tensor.full(Shape.of(), other, value.device()));
    }

    // ---------------------------------------------------------------- linear algebra and shape

    /**
     * Multiplies matrices: {@code [m, k] x [k, n] -> [m, n]}, or batches of them with
     * leading dimensions broadcast as in {@link Tensor#matmul(Tensor)}.
     *
     * @param other the right-hand matrix
     * @return the matrix product
     */
    public Var matmul(Var other) {
        Tensor a = value;
        Tensor b = other.value;
        if (a.rank() == 2 && b.rank() == 2) {
            return binary(other, a.matmul(b),
                    g -> new Tensor[] {g.matmul(b.transpose()), a.transpose().matmul(g)});
        }
        return binary(other, a.matmul(b), g -> new Tensor[] {
            g.matmul(b.transpose()).sumTo(a.shape()),
            a.rank() > 2 && b.rank() == 2
                    ? a.reshape(Math.toIntExact(a.size() / a.shape().dim(-1)), a.shape().dim(-1)).transpose()
                            .matmul(g.reshape(Math.toIntExact(g.size() / g.shape().dim(-1)), g.shape().dim(-1)))
                    : a.transpose().matmul(g).sumTo(b.shape())
        });
    }

    /**
     * Reorders dimensions.
     *
     * @param axes for each output dimension, the input dimension it takes
     * @return the permuted value
     */
    public Var permute(int... axes) {
        int[] inverse = new int[axes.length];
        for (int i = 0; i < axes.length; i++) {
            inverse[axes[i] < 0 ? axes[i] + axes.length : axes[i]] = i;
        }
        return unary(value.permute(axes), g -> new Tensor[] {g.permute(inverse)});
    }

    /**
     * Repeats size-1 dimensions (and adds leading ones) to reach a larger shape, as
     * broadcasting would.
     *
     * @param target the shape to expand to
     * @return the expanded value
     */
    public Var expand(Shape target) {
        Shape original = shape();
        return unary(value.expand(target), g -> new Tensor[] {g.sumTo(original)});
    }

    /**
     * Picks entries along an axis by index; see {@link Tensor#gather(int, Tensor)}. The
     * gradient adds up for repeated indices, as an embedding table needs.
     *
     * @param axis the axis to index
     * @param indices whole-number indices; not differentiated
     * @return the picked entries
     */
    public Var gather(int axis, Tensor indices) {
        int a = axis < 0 ? shape().rank() + axis : axis;
        int length = shape().dim(a);
        return unary(value.gather(a, indices), g -> new Tensor[] {g.scatterAdd(a, indices, length)});
    }

    /**
     * Unrolls sliding windows into rows; see {@link Tensor#im2col}.
     *
     * @param geometry the windows
     * @param padValue the value read from padding
     * @return the unrolled windows
     */
    public Var im2col(io.github.zzuegg.brainiac.tensor.ConvGeometry geometry, float padValue) {
        return unary(value.im2col(geometry, padValue), g -> new Tensor[] {g.col2im(geometry)});
    }

    /**
     * Adds unrolled windows back into images; see {@link Tensor#col2im}.
     *
     * @param geometry the windows
     * @return the images
     */
    public Var col2im(io.github.zzuegg.brainiac.tensor.ConvGeometry geometry) {
        return unary(value.col2im(geometry), g -> new Tensor[] {g.im2col(geometry, 0f)});
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

    /**
     * Returns the part of this value from {@code start} to {@code end} (exclusive) along
     * one axis.
     *
     * @param axis the axis; negative counts from the end
     * @param start the first index to keep
     * @param end one past the last index to keep
     * @return the slice
     */
    public Var slice(int axis, int start, int end) {
        Tensor out = value.slice(axis, start, end);
        int a = axis < 0 ? shape().rank() + axis : axis;
        Shape original = shape();
        return unary(out, g -> {
            List<Tensor> parts = new ArrayList<>(3);
            if (start > 0) {
                int[] before = original.toArray();
                before[a] = start;
                parts.add(Tensor.zeros(Shape.of(before), g.device()));
            }
            parts.add(g);
            if (end < original.dim(a)) {
                int[] after = original.toArray();
                after[a] = original.dim(a) - end;
                parts.add(Tensor.zeros(Shape.of(after), g.device()));
            }
            return new Tensor[] {parts.size() == 1 ? g : Tensor.concat(a, parts.toArray(Tensor[]::new))};
        });
    }

    /**
     * Joins values along one axis; see {@link Tensor#concat(int, Tensor...)}.
     *
     * @param axis the axis to join along; negative counts from the end
     * @param parts the values, at least one
     * @return the joined value
     */
    public static Var concat(int axis, Var... parts) {
        Tensor[] values = new Tensor[parts.length];
        for (int i = 0; i < parts.length; i++) {
            values[i] = parts[i].value;
        }
        Tensor out = Tensor.concat(axis, values);
        int a = axis < 0 ? out.rank() + axis : axis;
        return op(out, parts.clone(), g -> {
            Tensor[] grads = new Tensor[parts.length];
            int start = 0;
            for (int i = 0; i < parts.length; i++) {
                int length = values[i].shape().dim(a);
                grads[i] = g.slice(a, start, start + length);
                start += length;
            }
            return grads;
        });
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
