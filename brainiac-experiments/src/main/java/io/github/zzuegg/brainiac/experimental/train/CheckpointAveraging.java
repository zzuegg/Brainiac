package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.EpochResult;
import io.github.zzuegg.brainiac.train.TrainingListener;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;

/**
 * LAWA, latest weight averaging (Kaddour 2022, <a href="https://arxiv.org/abs/2209.14981">arXiv:2209.14981</a>;
 * Sanyal et al. 2023, <a href="https://arxiv.org/abs/2306.03241">arXiv:2306.03241</a>): the
 * uniform average of the last {@code k} checkpoints, taken every {@code every} steps
 * <em>during</em> training. Averaging a sliding window of recent, already-good checkpoints
 * cancels the noise of a high learning rate the way a decayed rate would, so the averaged
 * model is ahead of the raw one early in training. Unlike the core {@code Ema} it forgets
 * old weights completely after {@code k} checkpoints, and unlike
 * {@link StochasticWeightAveraging} it does not average everything since a start step.
 *
 * <p>As a {@link TrainingListener} it takes checkpoints after optimizer steps;
 * {@link #swap()} exchanges the averaged and current weights for evaluation.
 */
public final class CheckpointAveraging implements TrainingListener {

    private final List<Var> parameters;
    private final int window;
    private final int every;
    private final ArrayDeque<List<Tensor>> checkpoints = new ArrayDeque<>();
    private List<Tensor> swapped;

    /**
     * @param parameters the parameters to average
     * @param window checkpoints kept, {@code k}
     * @param every steps between checkpoints
     */
    public CheckpointAveraging(List<Var> parameters, int window, int every) {
        if (window < 1 || every < 1) {
            throw new IllegalArgumentException("window and every must be positive");
        }
        this.parameters = List.copyOf(parameters);
        this.window = window;
        this.every = every;
    }

    /** Takes a checkpoint of the current weights, dropping the oldest beyond the window. */
    public void checkpoint() {
        List<Tensor> snap = new ArrayList<>(parameters.size());
        for (Var p : parameters) {
            snap.add(p.value());
        }
        checkpoints.addLast(snap);
        if (checkpoints.size() > window) {
            checkpoints.removeFirst();
        }
    }

    /** @return the checkpoints currently in the window */
    public int size() {
        return checkpoints.size();
    }

    /**
     * The average of the checkpoints in the window.
     *
     * @return one tensor per parameter
     */
    public List<Tensor> average() {
        if (checkpoints.isEmpty()) {
            throw new IllegalStateException("no checkpoint taken yet");
        }
        List<Tensor> sum = null;
        for (List<Tensor> c : checkpoints) {
            if (sum == null) {
                sum = new ArrayList<>(c);
            } else {
                for (int i = 0; i < sum.size(); i++) {
                    sum.set(i, sum.get(i).add(c.get(i)));
                }
            }
        }
        List<Tensor> out = new ArrayList<>(sum.size());
        for (Tensor s : sum) {
            out.add(s.div(checkpoints.size()));
        }
        Tensor.evaluate(out);
        return out;
    }

    /**
     * Puts the averaged weights in the parameters, or the training weights back if they
     * are averaged now.
     */
    public void swap() {
        if (swapped == null) {
            List<Tensor> avg = average();
            swapped = new ArrayList<>();
            for (int i = 0; i < parameters.size(); i++) {
                swapped.add(parameters.get(i).value());
                parameters.get(i).assign(avg.get(i));
            }
        } else {
            for (int i = 0; i < parameters.size(); i++) {
                parameters.get(i).assign(swapped.get(i));
            }
            swapped = null;
        }
    }

    @Override
    public void onBatch(int epoch, long step, float loss) {
        if (step % every == 0) {
            checkpoint();
        }
    }

    @Override
    public void onEpoch(EpochResult result) {
    }
}
