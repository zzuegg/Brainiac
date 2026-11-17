package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.autograd.DualTensor;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Forward gradients (Baydin, Pearlmutter, Syme, Wood and Torr 2022,
 * <a href="https://arxiv.org/abs/2202.08587">arXiv:2202.08587</a>): an unbiased estimate of
 * the gradient from forward passes only. Draw a random direction {@code v ~ N(0, I)} over
 * all parameters, compute the directional derivative {@code d = grad f . v} with one
 * forward-mode pass ({@link DualTensor}), and use {@code g = d v}. Since
 * {@code E[v v^T] = I}, {@code E[g] = grad f}.
 *
 * <p>{@link #estimate} writes the estimate into the parameters' gradients, so any
 * {@code Optimizer} can step on it. Averaging several directions lowers the variance.
 */
public final class ForwardGradient {

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
     * The directional derivative of {@code f} at the parameters' values along {@code
     * tangents}: one Jacobian-vector product.
     *
     * @param params the parameters
     * @param tangents one direction tensor per parameter
     * @param f the function of the parameters' duals, returning a scalar
     * @return the value and the directional derivative
     */
    public static DualTensor jvp(List<Var> params, List<Tensor> tangents,
            Function<List<DualTensor>, DualTensor> f) {
        List<DualTensor> duals = new ArrayList<>(params.size());
        for (int i = 0; i < params.size(); i++) {
            duals.add(DualTensor.of(params.get(i).value(), tangents.get(i)));
        }
        return f.apply(duals);
    }

    /**
     * Estimates the gradient and stores it in every parameter's {@code grad}.
     *
     * @param params the parameters
     * @param f the loss as a function of the parameters' duals
     * @return the loss at the current parameters
     */
    public float estimate(List<Var> params, Function<List<DualTensor>, DualTensor> f) {
        Tensor[] acc = new Tensor[params.size()];
        float loss = 0;
        for (int k = 0; k < directions; k++) {
            List<Tensor> v = new ArrayList<>(params.size());
            for (Var p : params) {
                v.add(Tensor.normal(p.shape(), 0, 1, random));
            }
            DualTensor out = jvp(params, v, f);
            loss = out.primal().item();
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
