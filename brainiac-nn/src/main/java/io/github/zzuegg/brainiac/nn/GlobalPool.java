package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional.PoolKind;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.Arrays;
import java.util.Objects;

/**
 * Pools each channel over all its positions: {@code [channels, spatial...]} becomes
 * {@code [channels]} (or {@code [channels, 1, ...]} with {@code keepDims}). The usual end
 * of a convolutional network before the classifier.
 */
public final class GlobalPool implements Layer {

    private final PoolKind kind;
    private final boolean keepDims;

    /**
     * Creates a global pooling layer.
     *
     * @param kind {@code AVG}, {@code MAX}, {@code MIN} or {@code SUM}
     * @param keepDims whether to keep the spatial dimensions as size 1
     */
    public GlobalPool(PoolKind kind, boolean keepDims) {
        this.kind = Objects.requireNonNull(kind, "kind");
        if (kind == PoolKind.LP) {
            throw new IllegalArgumentException("global LP pooling is not supported");
        }
        this.keepDims = keepDims;
    }

    /** @return global average pooling */
    public static GlobalPool avg() {
        return new GlobalPool(PoolKind.AVG, false);
    }

    /** @return global max pooling */
    public static GlobalPool max() {
        return new GlobalPool(PoolKind.MAX, false);
    }

    @Override
    public Var forward(Var input) {
        if (input.shape().rank() < 3) {
            throw new IllegalArgumentException("global pooling expects [batch, channels, spatial...], got "
                    + input.shape());
        }
        return Functional.globalPool(input, kind, keepDims);
    }

    @Override
    public Shape outputShape(Shape input) {
        if (input.rank() < 2) {
            throw new IllegalArgumentException("global pooling expects [channels, spatial...], got " + input);
        }
        if (!keepDims) {
            return Shape.of(input.dim(0));
        }
        int[] dims = new int[input.rank()];
        Arrays.fill(dims, 1);
        dims[0] = input.dim(0);
        return Shape.of(dims);
    }

    @Override
    public String toString() {
        return "Global" + kind.name().charAt(0) + kind.name().substring(1).toLowerCase(java.util.Locale.ROOT) + "Pool";
    }

    static Config write(GlobalPool p) {
        return Config.builder().put("kind", p.kind.name()).put("keepDims", p.keepDims).build();
    }

    static GlobalPool read(Config c) {
        return new GlobalPool(PoolKind.valueOf(c.getString("kind")), c.getBoolean("keepDims"));
    }
}
