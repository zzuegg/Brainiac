package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional.Interpolation;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.Arrays;
import java.util.Objects;

/**
 * Upsamples or downsamples the spatial dimensions of images by a scale factor or to a
 * fixed size, with nearest, (bi/tri)linear or (bi)cubic interpolation.
 *
 * <pre>{@code
 * Resize.scale(2)                                      // nearest, twice the size
 * Resize.scale(Interpolation.LINEAR, 2).alignCorners(true)
 * Resize.to(Interpolation.CUBIC, 64, 64)
 * }</pre>
 */
public final class Resize implements Layer {

    private final Interpolation interpolation;
    private final float[] scale;
    private final int[] size;
    private final boolean alignCorners;

    private Resize(Interpolation interpolation, float[] scale, int[] size, boolean alignCorners) {
        this.interpolation = Objects.requireNonNull(interpolation, "interpolation");
        this.scale = scale;
        this.size = size;
        this.alignCorners = alignCorners;
    }

    /**
     * Nearest-neighbour upsampling by a factor.
     *
     * @param factors one factor for all dimensions, or one per dimension
     * @return the layer
     */
    public static Resize scale(float... factors) {
        return scale(Interpolation.NEAREST, factors);
    }

    /**
     * Resizes by a factor.
     *
     * @param interpolation how new cells are computed
     * @param factors one factor for all dimensions, or one per dimension
     * @return the layer
     */
    public static Resize scale(Interpolation interpolation, float... factors) {
        if (factors.length == 0) {
            throw new IllegalArgumentException("give at least one scale factor");
        }
        for (float f : factors) {
            if (!(f > 0)) {
                throw new IllegalArgumentException("scale factors must be positive, got " + Arrays.toString(factors));
            }
        }
        return new Resize(interpolation, factors.clone(), null, false);
    }

    /**
     * Resizes to a fixed size.
     *
     * @param interpolation how new cells are computed
     * @param size the output size per spatial dimension
     * @return the layer
     */
    public static Resize to(Interpolation interpolation, int... size) {
        for (int s : size) {
            if (s < 1) {
                throw new IllegalArgumentException("sizes must be at least 1, got " + Arrays.toString(size));
            }
        }
        return new Resize(interpolation, null, size.clone(), false);
    }

    /**
     * Maps corner cells onto each other (linear and cubic only).
     *
     * @param align whether to align corners
     * @return a copy with the setting
     */
    public Resize alignCorners(boolean align) {
        return new Resize(interpolation, scale, size, align);
    }

    private int[] target(Shape sample) {
        int r = sample.rank() - 1;
        int[] out = new int[r];
        for (int i = 0; i < r; i++) {
            if (size != null) {
                if (size.length != r) {
                    throw new IllegalArgumentException(this + " has " + size.length + " sizes for " + sample);
                }
                out[i] = size[i];
            } else {
                if (scale.length != 1 && scale.length != r) {
                    throw new IllegalArgumentException(this + " has " + scale.length + " factors for " + sample);
                }
                float f = scale.length == 1 ? scale[0] : scale[i];
                out[i] = Math.max(1, (int) Math.floor(sample.dim(i + 1) * f));
            }
        }
        return out;
    }

    @Override
    public Var forward(Var input) {
        int[] dims = input.shape().toArray();
        return Functional.resize(input, target(Shape.of(Arrays.copyOfRange(dims, 1, dims.length))), interpolation,
                alignCorners);
    }

    @Override
    public Shape outputShape(Shape input) {
        int[] t = target(input);
        int[] out = new int[t.length + 1];
        out[0] = input.dim(0);
        System.arraycopy(t, 0, out, 1, t.length);
        return Shape.of(out);
    }

    @Override
    public String toString() {
        return "Resize(" + interpolation.name().toLowerCase(java.util.Locale.ROOT) + ", "
                + (size != null ? Arrays.toString(size) : "x" + Arrays.toString(scale)) + ")";
    }

    static Config write(Resize r) {
        Config.Builder b = Config.builder().put("interpolation", r.interpolation.name())
                .put("alignCorners", r.alignCorners);
        if (r.size != null) {
            b.put("size", r.size);
        } else {
            b.put("scale", r.scale);
        }
        return b.build();
    }

    static Resize read(Config c) {
        Interpolation i = Interpolation.valueOf(c.getString("interpolation"));
        Resize r = c.has("size") ? to(i, c.getInts("size")) : scale(i, c.getFloats("scale"));
        return r.alignCorners(c.getBoolean("alignCorners"));
    }
}
