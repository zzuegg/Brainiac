package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional.PoolKind;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.Arrays;
import java.util.Objects;

/**
 * Pools each channel to a fixed output size whatever the input size, so one model can
 * take images of different sizes.
 */
public final class AdaptivePool implements Layer {

    private final PoolKind kind;
    private final int[] size;

    /**
     * Creates an adaptive pooling layer.
     *
     * @param kind {@code AVG}, {@code MAX}, {@code MIN} or {@code SUM}
     * @param size the output size per spatial dimension
     */
    public AdaptivePool(PoolKind kind, int... size) {
        this.kind = Objects.requireNonNull(kind, "kind");
        if (kind == PoolKind.LP) {
            throw new IllegalArgumentException("adaptive LP pooling is not supported");
        }
        if (size.length == 0) {
            throw new IllegalArgumentException("give at least one output size");
        }
        for (int s : size) {
            if (s < 1) {
                throw new IllegalArgumentException("output sizes must be at least 1, got " + Arrays.toString(size));
            }
        }
        this.size = size.clone();
    }

    /**
     * Adaptive average pooling.
     *
     * @param size the output size per spatial dimension
     * @return the layer
     */
    public static AdaptivePool avg(int... size) {
        return new AdaptivePool(PoolKind.AVG, size);
    }

    /**
     * Adaptive max pooling.
     *
     * @param size the output size per spatial dimension
     * @return the layer
     */
    public static AdaptivePool max(int... size) {
        return new AdaptivePool(PoolKind.MAX, size);
    }

    @Override
    public Var forward(Var input) {
        return Functional.adaptivePool(input, kind, size);
    }

    @Override
    public Shape outputShape(Shape input) {
        if (input.rank() != size.length + 1) {
            throw new IllegalArgumentException(this + " expects [channels, " + size.length
                    + " spatial dimensions], got " + input);
        }
        int[] out = new int[size.length + 1];
        out[0] = input.dim(0);
        System.arraycopy(size, 0, out, 1, size.length);
        return Shape.of(out);
    }

    @Override
    public String toString() {
        return "Adaptive" + kind.name().charAt(0) + kind.name().substring(1).toLowerCase(java.util.Locale.ROOT)
                + "Pool(" + Arrays.toString(size) + ")";
    }

    static Config write(AdaptivePool p) {
        return Config.builder().put("kind", p.kind.name()).put("size", p.size).build();
    }

    static AdaptivePool read(Config c) {
        return new AdaptivePool(PoolKind.valueOf(c.getString("kind")), c.getInts("size"));
    }
}
