package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

/**
 * Sharpness-Aware Minimization (Foret et al. 2021, <a href="https://arxiv.org/abs/2010.01412">arXiv:2010.01412</a>)
 * and its scale-invariant variant ASAM (Kwon et al. 2021,
 * <a href="https://arxiv.org/abs/2102.11600">arXiv:2102.11600</a>), wrapping any optimizer.
 * Each step first climbs to the worst point in a small neighbourhood,
 * {@code e = rho g / |g|} (ASAM: {@code e = rho T^2 g / |T g|} with {@code T = |w| + eta}),
 * takes the gradient there, then steps from the original weights with it. This seeks flat
 * minima, which generalise better.
 *
 * <p>Two gradient evaluations per step: either call {@link #step(List, Supplier)} with a
 * function that computes the loss and runs {@code backward()}, or call
 * {@link #perturb(List)} after the first backward pass and {@link #step(List)} after the
 * second. A plain {@link #step(List)} without a perturbation is the base optimizer's step.
 */
public final class SharpnessAware implements Optimizer {

    private final Optimizer base;
    private final float rho;
    private final boolean adaptive;
    private final float eta;
    private Map<Var, Tensor> perturbation;

    private SharpnessAware(Optimizer base, float rho, boolean adaptive, float eta) {
        this.base = base;
        this.rho = rho;
        this.adaptive = adaptive;
        this.eta = eta;
    }

    /**
     * @param base the optimizer that takes the actual step
     * @param rho the neighbourhood radius, for example 0.05
     * @return SAM
     */
    public static SharpnessAware sam(Optimizer base, float rho) {
        return new SharpnessAware(base, rho, false, 0);
    }

    /**
     * @param base the optimizer that takes the actual step
     * @param rho the radius, larger than SAM's, for example 0.5
     * @param eta keeps zero weights perturbable, for example 0.01
     * @return ASAM
     */
    public static SharpnessAware asam(Optimizer base, float rho, float eta) {
        return new SharpnessAware(base, rho, true, eta);
    }

    /**
     * Moves the weights to the sharpest nearby point using their current gradients, and
     * clears the gradients.
     *
     * @param parameters the parameters, with gradients
     */
    public void perturb(List<Var> parameters) {
        perturbation = new IdentityHashMap<>();
        double norm = 0;
        for (Var p : parameters) {
            if (p.grad() != null) {
                Tensor tg = adaptive ? p.grad().mul(p.value().abs().add(eta)) : p.grad();
                norm += tg.square().sum().item();
            }
        }
        float scale = (float) (rho / (Math.sqrt(norm) + 1e-12));
        for (Var p : parameters) {
            if (p.grad() == null) {
                continue;
            }
            Tensor e = p.grad().mul(scale);
            if (adaptive) {
                e = e.mul(p.value().abs().add(eta).square());
            }
            perturbation.put(p, e);
            p.assign(p.value().add(e));
            p.zeroGrad();
        }
    }

    @Override
    public void step(List<Var> parameters) {
        if (perturbation != null) {
            for (Var p : parameters) {
                Tensor e = perturbation.get(p);
                if (e != null) {
                    p.assign(p.value().sub(e));
                }
            }
            perturbation = null;
        }
        base.step(parameters);
    }

    /**
     * One full SAM step.
     *
     * @param parameters the parameters
     * @param lossAndBackward computes the loss on the current batch, calls {@code backward()}
     *     on it and returns it
     * @return the loss at the original weights
     */
    public float step(List<Var> parameters, Supplier<Var> lossAndBackward) {
        float loss = lossAndBackward.get().value().item();
        perturb(parameters);
        lossAndBackward.get();
        step(parameters);
        return loss;
    }

    @Override
    public float learningRate() {
        return base.learningRate();
    }

    @Override
    public void learningRate(float learningRate) {
        base.learningRate(learningRate);
    }

    @Override
    public Map<String, Tensor> state(List<Var> parameters) {
        return base.state(parameters);
    }

    @Override
    public void loadState(List<Var> parameters, Map<String, Tensor> state) {
        base.loadState(parameters, state);
    }

    @Override
    public String toString() {
        return (adaptive ? "ASAM(" : "SAM(") + base + ", rho=" + rho + ")";
    }
}
