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
    private final boolean channelwise;

    /**
     * Creates a dropout layer.
     *
     * @param rate the share of values to drop while training, in [0, 1)
     * @param random the random source for the masks
     */
    public Dropout(float rate, RandomGenerator random) {
        this(rate, random, false);
    }

    /**
     * Creates a dropout layer that can drop whole channels.
     *
     * @param rate the share of values (or channels) to drop while training, in [0, 1)
     * @param random the random source for the masks
     * @param channelwise {@code true} to drop whole channels of {@code [batch, channels, ...]}
     *     inputs (spatial dropout), which suits convolutional layers whose neighbouring
     *     values are correlated
     */
    public Dropout(float rate, RandomGenerator random, boolean channelwise) {
        check(rate);
        this.rate = rate;
        this.random = random;
        this.channelwise = channelwise;
    }

    static void check(float rate) {
        if (!(rate >= 0 && rate < 1)) {
            throw new IllegalArgumentException("dropout rate must be in [0, 1), got " + rate);
        }
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
        Shape maskShape = input.shape();
        if (channelwise && maskShape.rank() > 2) {
            int[] dims = new int[maskShape.rank()];
            java.util.Arrays.fill(dims, 1);
            dims[0] = maskShape.dim(0);
            dims[1] = maskShape.dim(1);
            maskShape = Shape.of(dims);
        }
        Tensor keep = Tensor.random(maskShape, random.nextLong(), input.value().device())
                .greater(Tensor.full(Shape.of(), rate, input.value().device()));
        return input.mul(keep.div(1 - rate));
    }

    /** @return the share of values dropped while training */
    public float rate() {
        return rate;
    }

    /** @return whether whole channels are dropped */
    public boolean channelwise() {
        return channelwise;
    }

    @Override
    public String toString() {
        return (channelwise ? "SpatialDropout(" : "Dropout(") + rate + ")";
    }
}
