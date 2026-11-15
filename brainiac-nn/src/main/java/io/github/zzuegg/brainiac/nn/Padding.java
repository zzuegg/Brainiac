package io.github.zzuegg.brainiac.nn;

import java.util.Arrays;

/**
 * How much padding a convolution or pooling window gets around the image, per spatial
 * dimension.
 *
 * <pre>{@code
 * Padding.VALID            // none: the output shrinks
 * Padding.SAME             // output size = ceil(input / stride), split evenly (extra after)
 * Padding.FULL             // every partial overlap: kernel - 1 on both sides
 * Padding.of(1)            // 1 on both sides of every dimension
 * Padding.of(1, 2)         // 1 on both sides of the first dimension, 2 of the second
 * Padding.explicit(new int[] {0, 1}, new int[] {1, 1})  // before and after, per dimension
 * }</pre>
 */
public final class Padding {

    private enum Kind { VALID, SAME, FULL, EXPLICIT }

    /** No padding. */
    public static final Padding VALID = new Padding(Kind.VALID, null, null);

    /** Enough padding that the output size is {@code ceil(input / stride)}. */
    public static final Padding SAME = new Padding(Kind.SAME, null, null);

    /** Padding for every partial overlap: {@code dilation * (kernel - 1)} on each side. */
    public static final Padding FULL = new Padding(Kind.FULL, null, null);

    private final Kind kind;
    private final int[] before;
    private final int[] after;

    private Padding(Kind kind, int[] before, int[] after) {
        this.kind = kind;
        this.before = before;
        this.after = after;
    }

    /**
     * Pads both sides by the same amount.
     *
     * @param amounts one amount for every dimension, or one per dimension
     * @return the padding
     */
    public static Padding of(int... amounts) {
        return explicit(amounts, amounts);
    }

    /**
     * Pads each side by its own amount.
     *
     * @param before the padding before the image, one value for all dimensions or one per dimension
     * @param after the padding after the image, likewise
     * @return the padding
     */
    public static Padding explicit(int[] before, int[] after) {
        if (before.length == 0 || after.length == 0) {
            throw new IllegalArgumentException("padding needs at least one amount");
        }
        for (int v : before) {
            check(v);
        }
        for (int v : after) {
            check(v);
        }
        return new Padding(Kind.EXPLICIT, before.clone(), after.clone());
    }

    private static void check(int v) {
        if (v < 0) {
            throw new IllegalArgumentException("padding must not be negative, got " + v);
        }
    }

    /**
     * Works out the padding for one input.
     *
     * @param image the image size per dimension
     * @param kernel the window size per dimension
     * @param stride the stride per dimension
     * @param dilation the dilation per dimension
     * @return {@code {before, after}}, one entry per dimension each
     */
    int[][] resolve(int[] image, int[] kernel, int[] stride, int[] dilation) {
        int r = image.length;
        int[] b = new int[r];
        int[] a = new int[r];
        for (int i = 0; i < r; i++) {
            int span = dilation[i] * (kernel[i] - 1) + 1;
            switch (kind) {
                case VALID -> { }
                case FULL -> {
                    b[i] = span - 1;
                    a[i] = span - 1;
                }
                case SAME -> {
                    int out = Math.ceilDiv(image[i], stride[i]);
                    int total = Math.max((out - 1) * stride[i] + span - image[i], 0);
                    b[i] = total / 2;
                    a[i] = total - b[i];
                }
                case EXPLICIT -> {
                    b[i] = pick(before, i, r);
                    a[i] = pick(after, i, r);
                }
            }
        }
        return new int[][] {b, a};
    }

    /**
     * Works out the padding of a transposed convolution, which removes cells from the
     * output instead of adding them to the input. {@link #SAME} gives
     * {@code output = input * stride}.
     */
    int[][] resolveTransposed(int[] kernel, int[] stride, int[] dilation) {
        int r = kernel.length;
        int[] b = new int[r];
        int[] a = new int[r];
        for (int i = 0; i < r; i++) {
            int span = dilation[i] * (kernel[i] - 1) + 1;
            switch (kind) {
                case VALID -> { }
                case FULL -> {
                    b[i] = span - 1;
                    a[i] = span - 1;
                }
                case SAME -> {
                    // output (in - 1) * s + span - total = in * s  =>  total = span - s
                    int total = span - stride[i];
                    b[i] = Math.max(total, 0) / 2;
                    a[i] = total - b[i];
                }
                case EXPLICIT -> {
                    b[i] = pick(before, i, r);
                    a[i] = pick(after, i, r);
                }
            }
        }
        return new int[][] {b, a};
    }

    private static int pick(int[] values, int i, int rank) {
        if (values.length == 1) {
            return values[0];
        }
        if (values.length != rank) {
            throw new IllegalArgumentException("padding has " + values.length + " amounts for " + rank + " dimensions");
        }
        return values[i];
    }

    @Override
    public String toString() {
        return kind == Kind.EXPLICIT
                ? (Arrays.equals(before, after) ? "padding " + Arrays.toString(before)
                        : "padding " + Arrays.toString(before) + "/" + Arrays.toString(after))
                : kind.name().toLowerCase(java.util.Locale.ROOT);
    }
}
