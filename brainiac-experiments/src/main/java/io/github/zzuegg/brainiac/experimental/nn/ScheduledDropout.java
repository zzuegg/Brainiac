package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

/**
 * Inverted dropout whose rate can change during training, for dropout schedules such as
 * early and late dropout (Liu et al. 2023, <a href="https://arxiv.org/abs/2303.01500">arXiv:2303.01500</a>):
 * dropout only in the first epochs helps underfitting models fit better, dropout only at
 * the end regularises overfitting ones. A schedule (for example the experimental
 * {@code DropoutSchedule} training listener) sets {@link #rate(float)}; otherwise it
 * behaves like the core {@code Dropout}.
 */
public final class ScheduledDropout implements Layer {

    private final SplittableRandom random;
    private volatile float rate;

    /**
     * @param rate the starting rate, in {@code [0, 1)}
     * @param random the random source
     */
    public ScheduledDropout(float rate, SplittableRandom random) {
        rate(rate);
        this.random = random;
    }

    /**
     * @param rate the starting rate
     * @return a spec for model builders
     */
    public static LayerSpec spec(float rate) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return input;
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new ScheduledDropout(rate, random.split());
            }
        };
    }

    /** @return the current rate */
    public float rate() {
        return rate;
    }

    /**
     * @param rate the new rate, in {@code [0, 1)}
     */
    public void rate(float rate) {
        if (!(rate >= 0 && rate < 1)) {
            throw new IllegalArgumentException("dropout rate must be in [0, 1), got " + rate);
        }
        this.rate = rate;
    }

    @Override
    public Var forward(Var input) {
        return input;
    }

    @Override
    public Var forward(Var input, boolean training) {
        float r = rate;
        if (!training || r == 0) {
            return input;
        }
        Tensor keep = Tensor.random(input.shape(), random.nextLong(), input.value().device())
                .greater(Tensor.full(Shape.of(), r, input.value().device()));
        return input.mul(keep.div(1 - r));
    }

    @Override
    public String toString() {
        return "ScheduledDropout(" + rate + ")";
    }
}
