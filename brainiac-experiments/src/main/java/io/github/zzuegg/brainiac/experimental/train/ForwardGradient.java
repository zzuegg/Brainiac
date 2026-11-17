package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Forward gradients (Baydin, Pearlmutter, Syme, Wood and Torr 2022,
 * <a href="https://arxiv.org/abs/2202.08587">arXiv:2202.08587</a>): an unbiased estimate of
 * the gradient from one directional derivative. Draw a random direction {@code v ~ N(0, I)}
 * over all parameters, compute the directional derivative {@code d = grad f . v} (a
 * Jacobian-vector product) and use {@code g = d v}. Since {@code E[v v^T] = I},
 * {@code E[g] = grad f}.
 *
 * <p>The Jacobian-vector product comes from core nested autograd
 * ({@link Var#gradients(Var, Var, List, boolean)}): the vector-Jacobian product with a
 * symbolic seed {@code u} is linear in {@code u}, and differentiating its dot product with
 * {@code v} with respect to {@code u} gives {@code J v}. This works for outputs of any shape;
 * the estimator only needs the scalar case.
 *
 * <p>{@link #estimate} writes the estimate into the parameters' gradients, so any
 * {@code Optimizer} can step on it. Averaging several directions lowers the variance.
 */
public final class ForwardGradient {

    /**
     * The value of a function and its derivative along a direction.
     *
     * @param value the function's value
     * @param tangent {@code J v}, shaped like {@code value}
     */
    public record Jvp(Tensor value, Tensor tangent) {
    }

    private final SplittableRandom random;
    private final int directions;

    /**
     * @param seed the seed of the random directions
     * @param directions directions averaged per estimate, at least 1
     */
    public ForwardGradient(long seed, int directions) {
        if (directions < 1) {
            throw new IllegalArgumentException("need at least one direction");
        }
        this.random = new SplittableRandom(seed);
        this.directions = directions;
    }

    /**
     * The derivative of {@code f} at the parameters' values along {@code tangents}: one
     * Jacobian-vector product.
     *
     * @param params the parameters
     * @param tangents one direction tensor per parameter
     * @param f the function of the parameters
     * @return the value and the directional derivative
     */
    public static Jvp jvp(List<Var> params, List<Tensor> tangents, Function<List<Var>, Var> f) {
        Var out = f.apply(params);
        Var seed = Var.param(Tensor.zeros(out.shape(), out.dtype(), out.value().device()));
        List<Var> vjp = Var.gradients(out, seed, params, true);
        Var dot = null;
        for (int i = 0; i < params.size(); i++) {
            Var term = vjp.get(i).mul(Var.of(tangents.get(i))).sum();
            dot = dot == null ? term : dot.add(term);
        }
        Tensor tangent = dot == null ? Tensor.zeros(out.shape(), out.dtype(), out.value().device())
                : Var.gradients(dot, List.of(seed), false).get(0).value();
        return new Jvp(out.value(), tangent);
    }

    /**
     * Estimates the gradient and stores it in every parameter's {@code grad}.
     *
     * @param params the parameters
     * @param f the scalar loss as a function of the parameters
     * @return the loss at the current parameters
     */
    public float estimate(List<Var> params, Function<List<Var>, Var> f) {
        Tensor[] acc = new Tensor[params.size()];
        float loss = 0;
        for (int k = 0; k < directions; k++) {
            List<Tensor> v = new ArrayList<>(params.size());
            for (Var p : params) {
                v.add(Tensor.normal(p.shape(), 0, 1, random));
            }
            Jvp out = jvp(params, v, f);
            loss = out.value().item();
            float d = out.tangent().item() / directions;
            for (int i = 0; i < acc.length; i++) {
                Tensor g = v.get(i).mul(d);
                acc[i] = acc[i] == null ? g : acc[i].add(g);
            }
        }
        for (int i = 0; i < acc.length; i++) {
            params.get(i).setGrad(acc[i]);
        }
        return loss;
    }
}
