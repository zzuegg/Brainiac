package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.experimental.nn.ScheduledDropout;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.train.EpochResult;
import io.github.zzuegg.brainiac.train.TrainingListener;

import java.util.ArrayList;
import java.util.List;
import java.util.function.LongToDoubleFunction;

/**
 * A {@link TrainingListener} that sets the rate of every {@link ScheduledDropout} layer of a
 * model after each optimizer step: early dropout, late dropout (Liu et al. 2023,
 * <a href="https://arxiv.org/abs/2303.01500">arXiv:2303.01500</a>) or any schedule.
 *
 * <pre>{@code
 * Trainer.of(model).listener(DropoutSchedule.early(model, 0.1f, 500)).fit(data);
 * }</pre>
 */
public final class DropoutSchedule implements TrainingListener {

    private final List<ScheduledDropout> layers;
    private final LongToDoubleFunction rateAtStep;

    /**
     * @param layers the layers to control
     * @param rateAtStep the rate after a given optimizer step (from 1)
     */
    public DropoutSchedule(List<ScheduledDropout> layers, LongToDoubleFunction rateAtStep) {
        this.layers = List.copyOf(layers);
        this.rateAtStep = rateAtStep;
        apply(0);
    }

    /**
     * @param model a model
     * @return its {@link ScheduledDropout} layers
     */
    public static List<ScheduledDropout> layersOf(Model model) {
        List<ScheduledDropout> out = new ArrayList<>();
        for (Layer l : model.layers()) {
            if (l instanceof ScheduledDropout d) {
                out.add(d);
            }
        }
        return out;
    }

    /**
     * Early dropout: {@code rate} for the first {@code steps} steps, then none.
     *
     * @param model the model
     * @param rate the early rate
     * @param steps how long it lasts
     * @return the schedule
     */
    public static DropoutSchedule early(Model model, float rate, long steps) {
        return new DropoutSchedule(layersOf(model), s -> s < steps ? rate : 0);
    }

    /**
     * Late dropout: none for the first {@code steps} steps, then {@code rate}.
     *
     * @param model the model
     * @param rate the late rate
     * @param steps when it starts
     * @return the schedule
     */
    public static DropoutSchedule late(Model model, float rate, long steps) {
        return new DropoutSchedule(layersOf(model), s -> s < steps ? 0 : rate);
    }

    /**
     * Sets the rates for a step.
     *
     * @param step the optimizer steps taken
     */
    public void apply(long step) {
        float r = (float) rateAtStep.applyAsDouble(step);
        for (ScheduledDropout d : layers) {
            d.rate(r);
        }
    }

    @Override
    public void onBatch(int epoch, long step, float loss) {
        apply(step);
    }

    @Override
    public void onEpoch(EpochResult result) {
    }
}
