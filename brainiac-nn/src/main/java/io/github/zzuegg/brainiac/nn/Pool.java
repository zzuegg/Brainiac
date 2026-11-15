package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional.PoolKind;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.Arrays;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * Pooling over sliding windows of each channel, for any number of spatial dimensions.
 *
 * <pre>{@code
 * Pool.max(2, 2)                                  // 2x2 max pooling, stride 2
 * Pool.avg(3, 3).stride(1).padding(Padding.SAME)  // smoothing that keeps the size
 * Pool.max(3, 3).stride(2).padding(1).ceilMode(true)
 * Pool.avg(2, 2).countPadding(false)              // padding cells do not count in the mean
 * Pool.lp(2, 3, 3)                                // power mean (L2 pooling)
 * Pool.min(2, 2); Pool.sum(2, 2);
 * }</pre>
 *
 * <p>The stride defaults to the window size. Max pooling sends the whole gradient to the
 * first largest value of each window.
 */
public final class Pool implements Layer {

    private final PoolKind kind;
    private final int[] kernel;
    private final int[] stride;
    private final int[] padBefore;
    private final int[] padAfter;
    private final int[] dilation;
    private final boolean ceil;
    private final boolean countPadding;
    private final float power;

    Pool(PoolKind kind, int[] kernel, int[] stride, int[] padBefore, int[] padAfter, int[] dilation, boolean ceil,
            boolean countPadding, float power) {
        this.kind = kind;
        this.kernel = kernel.clone();
        this.stride = stride.clone();
        this.padBefore = padBefore.clone();
        this.padAfter = padAfter.clone();
        this.dilation = dilation.clone();
        this.ceil = ceil;
        this.countPadding = countPadding;
        this.power = power;
    }

    /**
     * Max pooling.
     *
     * @param kernel the window size per spatial dimension
     * @return the spec
     */
    public static Spec max(int... kernel) {
        return new Spec(PoolKind.MAX, kernel, 0);
    }

    /**
     * Average pooling.
     *
     * @param kernel the window size per spatial dimension
     * @return the spec
     */
    public static Spec avg(int... kernel) {
        return new Spec(PoolKind.AVG, kernel, 0);
    }

    /**
     * Min pooling.
     *
     * @param kernel the window size per spatial dimension
     * @return the spec
     */
    public static Spec min(int... kernel) {
        return new Spec(PoolKind.MIN, kernel, 0);
    }

    /**
     * Sum pooling.
     *
     * @param kernel the window size per spatial dimension
     * @return the spec
     */
    public static Spec sum(int... kernel) {
        return new Spec(PoolKind.SUM, kernel, 0);
    }

    /**
     * Power-mean pooling, {@code (sum |x|^p)^(1/p)}: {@code p = 1} sums magnitudes, large
     * {@code p} approaches max pooling.
     *
     * @param p the power, positive
     * @param kernel the window size per spatial dimension
     * @return the spec
     */
    public static Spec lp(float p, int... kernel) {
        if (!(p > 0)) {
            throw new IllegalArgumentException("power must be positive, got " + p);
        }
        return new Spec(PoolKind.LP, kernel, p);
    }

    /** An immutable description of a pooling layer; every setter returns a new spec. */
    public static final class Spec implements LayerSpec {

        private final PoolKind kind;
        private final int[] kernel;
        private final float power;
        private int[] stride;
        private int[] dilation = {1};
        private Padding padding = Padding.VALID;
        private boolean ceil;
        private boolean countPadding = true;

        private Spec(PoolKind kind, int[] kernel, float power) {
            if (kernel.length == 0) {
                throw new IllegalArgumentException("give at least one window size");
            }
            for (int k : kernel) {
                if (k < 1) {
                    throw new IllegalArgumentException("window sizes must be at least 1, got " + Arrays.toString(kernel));
                }
            }
            this.kind = kind;
            this.kernel = kernel.clone();
            this.power = power;
            this.stride = kernel.clone();
        }

        private Spec copy() {
            Spec s = new Spec(kind, kernel, power);
            s.stride = stride;
            s.dilation = dilation;
            s.padding = padding;
            s.ceil = ceil;
            s.countPadding = countPadding;
            return s;
        }

        /**
         * Sets the step between windows; default is the window size.
         *
         * @param stride one value for all dimensions, or one per dimension
         * @return the new spec
         */
        public Spec stride(int... stride) {
            Spec s = copy();
            s.stride = expand(stride, kernel.length, "stride");
            return s;
        }

        /**
         * Sets the spacing between window cells.
         *
         * @param dilation one value for all dimensions, or one per dimension
         * @return the new spec
         */
        public Spec dilation(int... dilation) {
            Spec s = copy();
            s.dilation = expand(dilation, kernel.length, "dilation");
            return s;
        }

        /**
         * Sets the padding.
         *
         * @param padding the padding
         * @return the new spec
         */
        public Spec padding(Padding padding) {
            Spec s = copy();
            s.padding = Objects.requireNonNull(padding, "padding");
            return s;
        }

        /**
         * Pads both sides of every dimension equally.
         *
         * @param amounts one value for all dimensions, or one per dimension
         * @return the new spec
         */
        public Spec padding(int... amounts) {
            return padding(Padding.of(amounts));
        }

        /**
         * Lets a last, partial window produce an output.
         *
         * @param ceil whether to round the output size up
         * @return the new spec
         */
        public Spec ceilMode(boolean ceil) {
            Spec s = copy();
            s.ceil = ceil;
            return s;
        }

        /**
         * For average pooling: whether padding cells count in the divisor.
         *
         * @param count {@code true} (default) to divide by the window size
         * @return the new spec
         */
        public Spec countPadding(boolean count) {
            Spec s = copy();
            s.countPadding = count;
            return s;
        }

        private static int[] expand(int[] values, int rank, String what) {
            for (int v : values) {
                if (v < 1) {
                    throw new IllegalArgumentException(what + " must be at least 1, got " + Arrays.toString(values));
                }
            }
            if (values.length == rank) {
                return values.clone();
            }
            if (values.length == 1) {
                int[] out = new int[rank];
                Arrays.fill(out, values[0]);
                return out;
            }
            throw new IllegalArgumentException(what + " has " + values.length + " values for " + rank + " dimensions");
        }

        private Pool layer(Shape input) {
            int r = kernel.length;
            if (input.rank() != r + 1) {
                throw new IllegalArgumentException("a " + r + "-d pooling expects [channels, " + r
                        + " spatial dimensions], got " + input);
            }
            int[] d = expand(dilation, r, "dilation");
            int[] image = Arrays.copyOfRange(input.toArray(), 1, r + 1);
            int[][] pads = padding.resolve(image, kernel, stride, d);
            return new Pool(kind, kernel, stride, pads[0], pads[1], d, ceil, countPadding, power);
        }

        @Override
        public Shape outputShape(Shape input) {
            return layer(input).outputShape(input);
        }

        @Override
        public Layer create(Shape input, SplittableRandom random) {
            return layer(input);
        }
    }

    @Override
    public Var forward(Var input) {
        return Functional.pool(input, kind, kernel, stride, padBefore, padAfter, dilation, ceil, countPadding, power);
    }

    @Override
    public Shape outputShape(Shape input) {
        int[] image = Arrays.copyOfRange(input.toArray(), 1, input.rank());
        int[] pos = io.github.zzuegg.brainiac.tensor.ConvGeometry.sliding(1, 1, image, kernel, stride, padBefore,
                padAfter, dilation, ceil).positions();
        int[] out = new int[input.rank()];
        out[0] = input.dim(0);
        System.arraycopy(pos, 0, out, 1, pos.length);
        return Shape.of(out);
    }

    @Override
    public String toString() {
        return kind.name().charAt(0) + kind.name().substring(1).toLowerCase(java.util.Locale.ROOT) + "Pool"
                + kernel.length + "d(" + Arrays.toString(kernel) + (Arrays.equals(kernel, stride) ? ""
                        : ", stride " + Arrays.toString(stride)) + ")";
    }

    static Config write(Pool p) {
        return Config.builder()
                .put("kind", p.kind.name())
                .put("kernel", p.kernel)
                .put("stride", p.stride)
                .put("padBefore", p.padBefore)
                .put("padAfter", p.padAfter)
                .put("dilation", p.dilation)
                .put("ceil", p.ceil)
                .put("countPadding", p.countPadding)
                .put("power", p.power)
                .build();
    }

    static Pool read(Config c) {
        return new Pool(PoolKind.valueOf(c.getString("kind")), c.getInts("kernel"), c.getInts("stride"),
                c.getInts("padBefore"), c.getInts("padAfter"), c.getInts("dilation"), c.getBoolean("ceil"),
                c.getBoolean("countPadding"), c.getFloat("power"));
    }
}
