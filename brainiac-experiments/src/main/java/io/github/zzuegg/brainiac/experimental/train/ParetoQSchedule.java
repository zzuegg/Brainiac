package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.ParetoQDense;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.Objects;

/**
 * ParetoQ's training budget split: train in full precision for most of the run and switch the
 * {@link ParetoQDense} layers to quantization-aware training for the last part (the paper found
 * about 90% full precision and 10% QAT best at every bit width, and better than either QAT from
 * scratch or post-training quantization of the finished model).
 *
 * <p>It is an {@link Optimizer} wrapper, so it drops into any training loop: it counts
 * {@link #step} calls and, after {@code switchAt} of them, initializes the layers' scales from
 * the trained weights, turns quantization on and clears the inner optimizer's state for the
 * scales (their moments from the float phase are meaningless).
 *
 * <p>Liu et al., "ParetoQ: Scaling Laws in Extremely Low-bit LLM Quantization" (2025),
 * <a href="https://arxiv.org/abs/2502.02631">arXiv:2502.02631</a>.
 */
public final class ParetoQSchedule implements Optimizer {

    private final Optimizer inner;
    private final List<ParetoQDense> layers;
    private final long switchAt;
    private long steps;

    /**
     * @param inner the optimizer that does the updates
     * @param layers the layers to switch
     * @param totalSteps the run's length
     * @param qatShare the share of steps trained quantized (0.1 in the paper; 1 is QAT from scratch)
     */
    public ParetoQSchedule(Optimizer inner, List<ParetoQDense> layers, long totalSteps, double qatShare) {
        if (qatShare < 0 || qatShare > 1) {
            throw new IllegalArgumentException("qatShare must be in [0, 1]");
        }
        this.inner = Objects.requireNonNull(inner, "inner");
        this.layers = List.copyOf(layers);
        this.switchAt = Math.round(totalSteps * (1 - qatShare));
        if (switchAt == 0) {
            switchOn();
        }
    }

    private void switchOn() {
        for (ParetoQDense l : layers) {
            if (!l.quantized()) {
                l.quantize(true);
                List<Var> p = l.parameters();
                inner.forget(p.get(p.size() - 1));
            }
        }
    }

    /** @return whether the layers are quantized now */
    public boolean quantizing() {
        return steps >= switchAt;
    }

    @Override
    public void step(List<Var> parameters) {
        inner.step(parameters);
        steps++;
        if (steps == switchAt) {
            switchOn();
        }
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
    public String toString() {
        return "ParetoQSchedule(QAT from step " + switchAt + ", " + inner + ")";
    }
}
