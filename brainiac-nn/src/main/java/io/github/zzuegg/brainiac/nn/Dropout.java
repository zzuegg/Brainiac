package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.random.RandomGenerator;

/**
 * Randomly zeroes a share of values while training ("inverted dropout"): kept values are
 * divided by {@code 1 - rate}, so predictions need no rescaling and pass through
 * unchanged. Helps a network not to rely on single units, which reduces overfitting.
 */
public final class Dropout implements Layer {

    private final float rate;
    private final RandomGenerator random;

    /**
     * Creates a dropout layer.
     *
     * @param rate the share of values to drop while training, in [0, 1)
     * @param random the random source for the masks
     */
    public Dropout(float rate, RandomGenerator random) {
        if (!(rate >= 0 && rate < 1)) {
            throw new IllegalArgumentException("dropout rate must be in [0, 1), got " + rate);
        }
        this.rate = rate;
        this.random = random;
    }

    @Override
    public Var forward(Var input) {
        return input;
    }

    @Override
    public Var forward(Var input, boolean training) {
        if (!training || rate == 0) {
            return input;
        }
        Tensor keep = Tensor.uniform(input.shape(), 0, 1, random).to(input.value().device())
                .greater(Tensor.full(Shape.of(), rate, input.value().device()));
        return input.mul(keep.div(1 - rate));
    }

    /** @return the share of values dropped while training */
    public float rate() {
        return rate;
    }

    @Override
    public String toString() {
        return "Dropout(" + rate + ")";
    }
}
