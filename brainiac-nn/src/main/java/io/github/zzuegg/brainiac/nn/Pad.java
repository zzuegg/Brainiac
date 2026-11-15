package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.Arrays;
import java.util.Objects;

/**
 * Pads (or with negative amounts, crops) the spatial dimensions of images:
 * {@code [channels, spatial...]}. See {@link PadMode} for how new cells are filled.
 */
public final class Pad implements Layer {

    private final int[] before;
    private final int[] after;
    private final PadMode mode;
    private final float value;

    /**
     * Creates a padding layer.
     *
     * @param before cells added before each spatial dimension (negative crops)
     * @param after cells added after each spatial dimension (negative crops)
     * @param mode how new cells are filled
     * @param value the fill value for {@link PadMode#CONSTANT}
     */
    public Pad(int[] before, int[] after, PadMode mode, float value) {
        if (before.length == 0 || before.length != after.length) {
            throw new IllegalArgumentException("give one amount per spatial dimension on each side");
        }
        this.before = before.clone();
        this.after = after.clone();
        this.mode = Objects.requireNonNull(mode, "mode");
        this.value = value;
    }

    /**
     * Pads every side of every spatial dimension by the same amount.
     *
     * @param spatialRank the number of spatial dimensions
     * @param amount the cells per side
     * @param mode how new cells are filled
     * @return the layer
     */
    public static Pad of(int spatialRank, int amount, PadMode mode) {
        int[] a = new int[spatialRank];
        Arrays.fill(a, amount);
        return new Pad(a, a, mode, 0);
    }

    /**
     * Crops every side of every spatial dimension by the same amount.
     *
     * @param spatialRank the number of spatial dimensions
     * @param amount the cells removed per side
     * @return the layer
     */
    public static Pad crop(int spatialRank, int amount) {
        return of(spatialRank, -amount, PadMode.CONSTANT);
    }

    @Override
    public Var forward(Var input) {
        return Functional.pad(input, before, after, mode, value);
    }

    @Override
    public Shape outputShape(Shape input) {
        if (input.rank() != before.length + 1) {
            throw new IllegalArgumentException(this + " expects [channels, " + before.length
                    + " spatial dimensions], got " + input);
        }
        int[] out = input.toArray();
        for (int i = 0; i < before.length; i++) {
            out[i + 1] += before[i] + after[i];
        }
        return Shape.of(out);
    }

    @Override
    public String toString() {
        return "Pad(" + Arrays.toString(before) + "/" + Arrays.toString(after) + ", "
                + mode.name().toLowerCase(java.util.Locale.ROOT) + ")";
    }

    static Config write(Pad p) {
        return Config.builder().put("before", p.before).put("after", p.after).put("mode", p.mode.name())
                .put("value", p.value).build();
    }

    static Pad read(Config c) {
        return new Pad(c.getInts("before"), c.getInts("after"), PadMode.valueOf(c.getString("mode")),
                c.getFloat("value"));
    }
}
