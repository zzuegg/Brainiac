package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Supplier;

/**
 * AdaHessian (Yao et al. 2021, <a href="https://arxiv.org/abs/2006.00719">arXiv:2006.00719</a>):
 * Adam with its second moment built from a Hutchinson estimate of the Hessian diagonal,
 * {@code D = z * (H z)} with a Rademacher {@code z} and one exact Hessian-vector product by
 * double backpropagation. {@code D} is averaged over the spatial axes of convolution kernels
 * (rank 3 or more), its square goes into the running average {@code v}, and the step is
 * {@code m_hat / (sqrt(v_hat)^k + eps)} with the Hessian power {@code k} in {@code (0, 1]}.
 *
 * <p>Call {@link #updateHessian} with a loss closure before each {@link #step} (or every few
 * steps; the last estimate is reused), as for {@link Sophia#updateHessianHutchinson}. An HVP
 * costs about two gradients.
 */
public final class AdaHessian extends SlotOptimizer {

    private final float beta1;
    private final float beta2;
    private final float epsilon;
    private final float power;
    private final float weightDecay;
    private final Map<Var, Tensor> diagonal = new IdentityHashMap<>();
    private final SplittableRandom random;

    /**
     * @param learningRate the step size, for example 0.15 in the paper (it is larger than Adam's)
     * @param beta1 the momentum decay, 0.9
     * @param beta2 the second-moment decay, 0.999
     * @param epsilon added to the denominator, 1e-4
     * @param power the Hessian power {@code k}, 1 for the full Newton-like scaling
     * @param weightDecay the decoupled weight decay
     * @param seed the seed of the Rademacher vectors
     */
    public AdaHessian(float learningRate, float beta1, float beta2, float epsilon, float power, float weightDecay,
            long seed) {
        super(learningRate, "m", "v");
        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
        this.power = power;
        this.weightDecay = weightDecay;
        this.random = new SplittableRandom(seed);
    }

    /**
     * @param learningRate the step size
     * @param seed the seed
     * @return AdaHessian with the paper's defaults ({@code 0.9, 0.999, 1e-4, k = 1})
     */
    public static AdaHessian of(float learningRate, long seed) {
        return new AdaHessian(learningRate, 0.9f, 0.999f, 1e-4f, 1f, 0, seed);
    }

    /**
     * Estimates the Hessian diagonal at the current parameters with one Hutchinson probe.
     * Leaves the parameters' gradients untouched.
     *
     * @param parameters the parameters
     * @param loss computes the loss on the current batch
     */
    public void updateHessian(List<Var> parameters, Supplier<Var> loss) {
        List<Var> grads = Var.gradients(loss.get(), parameters, true);
        List<Tensor> z = new ArrayList<>(parameters.size());
        Var gz = null;
        for (int i = 0; i < parameters.size(); i++) {
            Tensor zi = CurvatureProducts.rademacher(parameters.get(i).shape(), random);
            z.add(zi);
            Var term = grads.get(i).mul(Var.of(zi)).sum();
            gz = gz == null ? term : gz.add(term);
        }
        if (gz == null) {
            return;
        }
        List<Var> hz = Var.gradients(gz, parameters, false);
        for (int i = 0; i < parameters.size(); i++) {
            Tensor d = z.get(i).mul(hz.get(i).value());
            if (d.rank() >= 3) {
                int spatial = 1;
                for (int a = 2; a < d.rank(); a++) {
                    spatial *= d.shape().dim(a);
                }
                int lead = Math.toIntExact(d.size() / spatial);
                d = d.reshape(lead, spatial).abs().mean(1, true).expand(
                        io.github.zzuegg.brainiac.tensor.Shape.of(lead, spatial)).reshape(d.shape());
            }
            diagonal.put(parameters.get(i), d);
        }
    }

    @Override
    protected Tensor[] update(Var p, Tensor g, Tensor[] s, int t) {
        Tensor d = diagonal.getOrDefault(p, g);
        Tensor m = s[0] == null ? g.mul(1 - beta1) : s[0].mul(beta1).add(g.mul(1 - beta1));
        Tensor v = s[1] == null ? d.square().mul(1 - beta2) : s[1].mul(beta2).add(d.square().mul(1 - beta2));
        Tensor mHat = m.div((float) (1 - Math.pow(beta1, t)));
        Tensor vHat = v.div((float) (1 - Math.pow(beta2, t)));
        Tensor denom = power == 1 ? vHat.sqrt() : vHat.sqrt().pow(power);
        Tensor w = weightDecay > 0 ? p.value().mul(1 - learningRate() * weightDecay) : p.value();
        p.assign(w.sub(mHat.div(denom.add(epsilon)).mul(learningRate())));
        return new Tensor[] {m, v};
    }

    @Override
    public void forget(Var parameter) {
        super.forget(parameter);
        diagonal.remove(parameter);
    }

    @Override
    public String toString() {
        return "AdaHessian(lr=" + learningRate() + ", k=" + power + ")";
    }
}
