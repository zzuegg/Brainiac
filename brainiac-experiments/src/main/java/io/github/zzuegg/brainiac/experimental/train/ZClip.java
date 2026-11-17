package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.Map;

/**
 * ZClip (Kumar et al. 2025, "ZClip: Adaptive Spike Mitigation for LLM Pre-Training",
 * <a href="https://arxiv.org/abs/2504.02507">arXiv:2504.02507</a>): gradient clipping
 * whose threshold follows the run. It keeps an exponential moving average {@code mu} and
 * variance {@code sigma^2} of the global gradient norm and computes each step's z-score
 * {@code z = (||g|| - mu) / sigma}. Above {@code zThreshold} the gradient is a spike and
 * is rescaled to the norm {@code mu + (zThreshold^2 / z) sigma} (reciprocal clipping: the
 * bigger the spike, the harder the clip), and the statistics are updated with the clipped
 * norm so a spike cannot inflate the threshold. A fixed clipping norm is either too loose
 * early, when norms are large, or too tight late, when they have shrunk; ZClip adapts.
 *
 * <p>The first {@code warmup} steps only collect statistics (their mean and variance seed
 * the averages). Wrap any optimizer: {@code ZClip.wrap(Optimizer.adamW(...))}.
 */
public final class ZClip implements Optimizer {

    private final Optimizer inner;
    private final float alpha;
    private final float zThreshold;
    private final int warmup;
    private double mean;
    private double variance;
    private double sum;
    private double sumSq;
    private int seen;
    private int clipped;
    private double lastNorm;

    /**
     * @param inner the optimizer
     * @param alpha the decay of the norm statistics, 0.97
     * @param zThreshold the z-score above which a step is clipped, 2.5
     * @param warmup steps that only collect statistics, 25
     */
    public ZClip(Optimizer inner, float alpha, float zThreshold, int warmup) {
        if (!(alpha > 0) || alpha >= 1 || !(zThreshold > 0) || warmup < 2) {
            throw new IllegalArgumentException("bad ZClip settings");
        }
        this.inner = inner;
        this.alpha = alpha;
        this.zThreshold = zThreshold;
        this.warmup = warmup;
    }

    /**
     * ZClip with the paper's defaults ({@code alpha = 0.97}, {@code z = 2.5}, 25 warm-up steps).
     *
     * @param inner the optimizer
     * @return the clipped optimizer
     */
    public static ZClip wrap(Optimizer inner) {
        return new ZClip(inner, 0.97f, 2.5f, 25);
    }

    /**
     * The global norm of the parameters' gradients.
     *
     * @param parameters the parameters
     * @return {@code sqrt(sum ||g||^2)}
     */
    public static double globalNorm(List<Var> parameters) {
        Tensor total = Tensor.full(Shape.of(), 0f);
        for (Var p : parameters) {
            if (p.grad() != null) {
                total = total.add(p.grad().square().sum());
            }
        }
        return Math.sqrt(total.item());
    }

    @Override
    public void step(List<Var> parameters) {
        double norm = globalNorm(parameters);
        lastNorm = norm;
        if (!Double.isFinite(norm)) {   // nothing sensible to clip to: drop the step
            for (Var p : parameters) {
                p.zeroGrad();
            }
            clipped++;
            return;
        }
        seen++;
        double used = norm;
        if (seen <= warmup) {
            sum += norm;
            sumSq += norm * norm;
            if (seen == warmup) {
                mean = sum / warmup;
                variance = Math.max(sumSq / warmup - mean * mean, 0);
            }
        } else {
            double std = Math.sqrt(variance) + 1e-12;
            double z = (norm - mean) / std;
            if (z > zThreshold) {
                used = mean + zThreshold * zThreshold / z * std;
                float factor = (float) (used / norm);
                for (Var p : parameters) {
                    if (p.grad() != null) {
                        p.setGrad(p.grad().mul(factor));
                    }
                }
                clipped++;
            }
            double delta = used - mean;
            mean = alpha * mean + (1 - alpha) * used;
            variance = alpha * variance + (1 - alpha) * delta * delta;
        }
        inner.step(parameters);
    }

    /** @return the steps clipped so far (non-finite steps included) */
    public int clipped() {
        return clipped;
    }

    /** @return the current mean of the gradient norm */
    public double meanNorm() {
        return mean;
    }

    /** @return the current standard deviation of the gradient norm */
    public double stdNorm() {
        return Math.sqrt(variance);
    }

    /** @return the raw gradient norm of the last step */
    public double lastNorm() {
        return lastNorm;
    }

    @Override
    public void resetState(Var parameter, Tensor keep) {
        inner.resetState(parameter, keep);
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
        return "ZClip(" + inner + ", z=" + zThreshold + ")";
    }
}
