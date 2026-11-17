package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.Map;

/**
 * Rewrites a parameter's gradient before the optimizer sees it. Wrap any optimizer with
 * {@link #wrap(Optimizer, GradientTransform...)}:
 *
 * <ul>
 *   <li>{@link #adaptiveClipping} (AGC, Brock et al. 2021,
 *       <a href="https://arxiv.org/abs/2102.06171">arXiv:2102.06171</a>): clips each
 *       output unit's gradient so that {@code |g| <= clip * max(|w|, eps)} for that unit,
 *       the trick that lets normaliser-free ResNets train with large batches.</li>
 *   <li>{@link #centralization} (Yong et al. 2020,
 *       <a href="https://arxiv.org/abs/2004.01461">arXiv:2004.01461</a>): subtracts the mean
 *       over the input axis from weight gradients, constraining weights to a hyperplane;
 *       smoother loss and faster training.</li>
 * </ul>
 */
@FunctionalInterface
public interface GradientTransform {

    /**
     * @param parameter the parameter
     * @param grad its gradient
     * @return the new gradient, same shape
     */
    Tensor apply(Var parameter, Tensor grad);

    /**
     * Adaptive gradient clipping, unit-wise: for {@code [in, out]} weights each output
     * column (one unit's incoming weights) is clipped; vectors are clipped as a whole.
     *
     * @param clip the largest allowed ratio of gradient norm to weight norm, for example 0.01
     * @param eps the smallest weight norm used, {@code 1e-3}
     * @return the transform
     */
    static GradientTransform adaptiveClipping(float clip, float eps) {
        return (p, g) -> {
            boolean matrix = g.rank() == 2;
            Tensor wn = matrix ? p.value().square().sum(0, true).sqrt() : p.value().square().sum().sqrt();
            Tensor gn = matrix ? g.square().sum(0, true).sqrt() : g.square().sum().sqrt();
            Tensor maxNorm = wn.maximum(Tensor.full(Shape.of(), eps)).mul(clip);
            Tensor factor = maxNorm.div(gn.maximum(Tensor.full(Shape.of(), 1e-6f)))
                    .minimum(Tensor.full(Shape.of(), 1f));
            return g.mul(factor);
        };
    }

    /**
     * Gradient centralization for parameters of rank 2 or more (others pass through):
     * subtracts the mean over every axis but the last (the units).
     *
     * @return the transform
     */
    static GradientTransform centralization() {
        return (p, g) -> {
            if (g.rank() < 2) {
                return g;
            }
            int units = g.shape().dim(-1);
            Tensor flat = g.reshape(Math.toIntExact(g.size() / units), units);
            return flat.sub(flat.mean(0, true)).reshape(g.shape());
        };
    }

    /**
     * An optimizer that transforms every gradient, in order, before {@code inner} steps.
     *
     * @param inner the optimizer
     * @param transforms the transforms
     * @return the wrapped optimizer
     */
    static Optimizer wrap(Optimizer inner, GradientTransform... transforms) {
        List<GradientTransform> list = List.of(transforms);
        return new Optimizer() {
            @Override
            public void step(List<Var> parameters) {
                for (Var p : parameters) {
                    if (p.grad() != null) {
                        Tensor g = p.grad();
                        for (GradientTransform t : list) {
                            g = t.apply(p, g);
                        }
                        p.setGrad(g);
                    }
                }
                inner.step(parameters);
            }

            @Override
            public void forget(Var parameter) {
                inner.forget(parameter);
            }

            @Override
            public float learningRate() {
                return inner.learningRate();
            }

            @Override
            public void learningRate(float learningRate) {
                inner.learningRate(learningRate);
            }

            @Override
            public Map<String, Tensor> state(List<Var> parameters) {
                return inner.state(parameters);
            }

            @Override
            public void loadState(List<Var> parameters, Map<String, Tensor> state) {
                inner.loadState(parameters, state);
            }

            @Override
            public String toString() {
                return inner + " with " + list.size() + " gradient transform(s)";
            }
        };
    }
}
