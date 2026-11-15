package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.Arrays;

/**
 * Changes the shape of each sample without moving values. {@link #flatten()} turns
 * images into feature rows for dense layers; a dimension of {@code -1} is worked out from
 * the others.
 */
public final class Reshape implements Layer {

    private final int[] dims;

    /**
     * Creates a reshape layer.
     *
     * @param dims the new shape of one sample; at most one entry may be {@code -1}
     */
    public Reshape(int... dims) {
        if (dims.length == 0) {
            throw new IllegalArgumentException("give at least one dimension");
        }
        int unknown = 0;
        for (int d : dims) {
            if (d == -1) {
                unknown++;
            } else if (d < 1) {
                throw new IllegalArgumentException("dimensions must be at least 1 or -1, got " + Arrays.toString(dims));
            }
        }
        if (unknown > 1) {
            throw new IllegalArgumentException("at most one dimension may be -1");
        }
        this.dims = dims.clone();
    }

    /** @return a layer that flattens each sample into one row */
    public static Reshape flatten() {
        return new Reshape(-1);
    }

    @Override
    public Var forward(Var input) {
        int[] out = new int[dims.length + 1];
        out[0] = input.shape().dim(0);
        long sample = input.shape().size() / out[0];
        System.arraycopy(resolve(sample), 0, out, 1, dims.length);
        return input.reshape(out);
    }

    private int[] resolve(long size) {
        int[] out = dims.clone();
        long known = 1;
        int unknown = -1;
        for (int i = 0; i < out.length; i++) {
            if (out[i] == -1) {
                unknown = i;
            } else {
                known *= out[i];
            }
        }
        if (unknown >= 0) {
            if (size % known != 0) {
                throw new IllegalArgumentException("cannot reshape " + size + " values to " + Arrays.toString(dims));
            }
            out[unknown] = Math.toIntExact(size / known);
        } else if (known != size) {
            throw new IllegalArgumentException("cannot reshape " + size + " values to " + Arrays.toString(dims));
        }
        return out;
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(resolve(input.size()));
    }

    @Override
    public String toString() {
        return dims.length == 1 && dims[0] == -1 ? "Flatten" : "Reshape" + Arrays.toString(dims);
    }

    static Config write(Reshape r) {
        return Config.builder().put("dims", r.dims).build();
    }

    static Reshape read(Config c) {
        return new Reshape(c.getInts("dims"));
    }
}
