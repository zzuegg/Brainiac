package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Wraps any optimizer and adds annealed Gaussian noise to every gradient before the step:
 * {@code g + N(0, eta / (1 + t)^gamma)}. The noise helps deep or badly initialized networks
 * escape saddle points and poor early minima, and fades so it does not hurt the end of
 * training. A drop-in {@link Optimizer}.
 *
 * <p>Paper: Neelakantan et al., "Adding Gradient Noise Improves Learning for Very Deep
 * Networks" (2015), <a href="https://arxiv.org/abs/1511.06807">arXiv:1511.06807</a>.
 */
public final class GradientNoise implements Optimizer {

    private final Optimizer inner;
    private final float eta;
    private final float gamma;
    private final SplittableRandom random;
    private int step;

    /**
     * Creates the wrapper.
     *
     * @param inner the optimizer that takes the step
     * @param eta the initial noise variance; the paper uses 0.01, 0.3 or 1
     * @param gamma the decay exponent; the paper uses 0.55
     * @param seed the noise seed
     */
    public GradientNoise(Optimizer inner, float eta, float gamma, long seed) {
        this.inner = inner;
        this.eta = eta;
        this.gamma = gamma;
        this.random = new SplittableRandom(seed);
    }

    /** @return the noise standard deviation the next step will use */
    public float currentStd() {
        return (float) Math.sqrt(eta / Math.pow(1 + step, gamma));
    }

    @Override
    public void step(List<Var> parameters) {
        float std = currentStd();
        for (Var p : parameters) {
            Tensor g = p.grad();
            if (g != null && std > 0) {
                p.setGrad(g.add(Tensor.normal(g.shape(), 0, std, random).to(g.device())));
            }
        }
        step++;
        inner.step(parameters);
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
        return "GradientNoise(" + inner + ", eta=" + eta + ", gamma=" + gamma + ")";
    }
}
