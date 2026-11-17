package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Supplier;

/**
 * MeZO, the memory-efficient zeroth-order optimizer (Malladi et al. 2023,
 * <a href="https://arxiv.org/abs/2305.17333">arXiv:2305.17333</a>): SPSA gradient estimates
 * from two forward passes, with the random direction <em>regenerated from a seed</em>
 * instead of stored, so training needs no more memory than inference. Per step:
 *
 * <pre>
 *   draw seed;  w += eps z;  l+ = loss;  w -= 2 eps z;  l- = loss;  w += eps z
 *   g = (l+ - l-) / (2 eps);  w -= lr (g z + weightDecay w)     (z regenerated each time)
 * </pre>
 *
 * <p>No backward pass is needed, so the loss may be non-differentiable (accuracy, a
 * reward). Workers that share the seeds can reproduce each other's updates from the two
 * scalars alone. As an {@link Optimizer}, {@link #step(List)} is plain SGD on existing
 * gradients, and {@link #estimate} writes the SPSA estimate into the gradients for use
 * with any other optimizer.
 */
public final class MeZo implements Optimizer {

    private float learningRate;
    private final float epsilon;
    private final float weightDecay;
    private final SplittableRandom seeds;

    /**
     * @param learningRate the step size, for example {@code 1e-3f}
     * @param epsilon the perturbation size, for example {@code 1e-3f}
     * @param weightDecay the weight decay
     * @param seed the seed of the seed sequence
     */
    public MeZo(float learningRate, float epsilon, float weightDecay, long seed) {
        learningRate(learningRate);
        this.epsilon = epsilon;
        this.weightDecay = weightDecay;
        this.seeds = new SplittableRandom(seed);
    }

    /**
     * The direction for one parameter, regenerated from the step's seed.
     *
     * @param p the parameter
     * @param index its index in the parameter list
     * @param seed the step's seed
     * @return a standard normal tensor of the parameter's shape
     */
    public static Tensor direction(Var p, int index, long seed) {
        return Tensor.normal(p.shape(), 0, 1, new SplittableRandom(seed * 0x9E3779B97F4A7C15L + index));
    }

    private void shift(List<Var> params, long seed, float amount) {
        for (int i = 0; i < params.size(); i++) {
            Var p = params.get(i);
            p.assign(p.value().add(direction(p, i, seed).mul(amount)));
        }
    }

    /**
     * Two forward passes and the projected gradient {@code (l+ - l-) / (2 eps)} along the
     * seed's direction; the weights are left unchanged.
     *
     * @param params the parameters
     * @param loss evaluates the loss at the current weights
     * @param seed the direction's seed
     * @return the projected gradient
     */
    public float projectedGradient(List<Var> params, Supplier<Float> loss, long seed) {
        shift(params, seed, epsilon);
        float plus = loss.get();
        shift(params, seed, -2 * epsilon);
        float minus = loss.get();
        shift(params, seed, epsilon);
        return (plus - minus) / (2 * epsilon);
    }

    /**
     * One in-place MeZO step.
     *
     * @param params the parameters
     * @param loss evaluates the loss on the current batch at the current weights
     * @return the projected gradient
     */
    public float step(List<Var> params, Supplier<Float> loss) {
        long seed = seeds.nextLong();
        float g = projectedGradient(params, loss, seed);
        for (int i = 0; i < params.size(); i++) {
            Var p = params.get(i);
            Tensor update = direction(p, i, seed).mul(g);
            if (weightDecay > 0) {
                update = update.add(p.value().mul(weightDecay));
            }
            p.assign(p.value().sub(update.mul(learningRate)));
        }
        return g;
    }

    /**
     * Writes the SPSA gradient estimate {@code g z} into every parameter's gradient.
     *
     * @param params the parameters
     * @param loss evaluates the loss
     * @return the projected gradient
     */
    public float estimate(List<Var> params, Supplier<Float> loss) {
        long seed = seeds.nextLong();
        float g = projectedGradient(params, loss, seed);
        for (int i = 0; i < params.size(); i++) {
            params.get(i).setGrad(direction(params.get(i), i, seed).mul(g));
        }
        return g;
    }

    @Override
    public void step(List<Var> parameters) {
        for (Var p : parameters) {
            if (p.grad() != null) {
                p.assign(p.value().sub(p.grad().mul(learningRate)));
                p.zeroGrad();
            }
        }
    }

    @Override
    public float learningRate() {
        return learningRate;
    }

    @Override
    public void learningRate(float learningRate) {
        if (!(learningRate > 0)) {
            throw new IllegalArgumentException("learningRate must be positive");
        }
        this.learningRate = learningRate;
    }

    @Override
    public String toString() {
        return "MeZO(lr=" + learningRate + ", eps=" + epsilon + ")";
    }
}
