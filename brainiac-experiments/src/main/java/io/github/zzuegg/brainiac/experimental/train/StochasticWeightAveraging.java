package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.EpochResult;
import io.github.zzuegg.brainiac.train.TrainingListener;

import java.util.ArrayList;
import java.util.List;

/**
 * Stochastic Weight Averaging (Izmailov et al. 2018, <a href="https://arxiv.org/abs/1803.05407">arXiv:1803.05407</a>):
 * the <em>uniform</em> average of the weights visited late in training, every
 * {@code every} steps from step {@code start}. The average lies in a flatter, wider
 * region than any single iterate and generalises better (the core {@code Ema} keeps an
 * exponential average instead). As a {@link TrainingListener} it samples after optimizer
 * steps; {@link #swap()} exchanges the averaged and current weights.
 */
public final class StochasticWeightAveraging implements TrainingListener {

    private final List<Var> parameters;
    private final long start;
    private final int every;
    private List<Tensor> average;
    private int count;

    /**
     * @param parameters the parameters to average
     * @param start the first step that is averaged
     * @param every steps between samples
     */
    public StochasticWeightAveraging(List<Var> parameters, long start, int every) {
        this.parameters = List.copyOf(parameters);
        this.start = start;
        this.every = every;
    }

    /** @return how many weight samples are in the average */
    public int samples() {
        return count;
    }

    /** Adds the current weights to the average. */
    public void sample() {
        if (average == null) {
            average = new ArrayList<>();
            for (Var p : parameters) {
                average.add(p.value());
            }
        } else {
            for (int i = 0; i < parameters.size(); i++) {
                Tensor a = average.get(i);
                average.set(i, a.add(parameters.get(i).value().sub(a).div(count + 1)));
            }
        }
        count++;
        Tensor.evaluate(average);
    }

    @Override
    public void onBatch(int epoch, long step, float loss) {
        if (step >= start && (step - start) % every == 0) {
            sample();
        }
    }

    @Override
    public void onEpoch(EpochResult result) {
    }

    /** Exchanges the averaged weights and the current ones (call again to switch back). */
    public void swap() {
        if (average == null) {
            throw new IllegalStateException("no weights averaged yet");
        }
        for (int i = 0; i < parameters.size(); i++) {
            Tensor current = parameters.get(i).value();
            parameters.get(i).assign(average.get(i));
            average.set(i, current);
        }
    }
}
