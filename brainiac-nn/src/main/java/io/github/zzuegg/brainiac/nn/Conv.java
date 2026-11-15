package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * A convolution layer over images with any number of spatial dimensions: 1-d signals
 * {@code [channels, length]}, 2-d images {@code [channels, height, width]}, 3-d volumes
 * and beyond. The number of kernel sizes picks the dimensionality.
 *
 * <p>Every option is on the spec returned by {@link #of(int, int...)}:
 *
 * <pre>{@code
 * Conv.of(32, 3, 3)                                  // 32 filters of 3x3, stride 1, no padding
 *     .stride(2)                                     // or stride(2, 1) per dimension
 *     .padding(Padding.SAME)                         // VALID, SAME, FULL, of(1), explicit(before, after)
 *     .padMode(PadMode.REFLECT)                      // CONSTANT (zeros), REFLECT, SYMMETRIC, REPLICATE, CIRCULAR
 *     .dilation(2)                                   // atrous / dilated
 *     .groups(4)                                     // grouped convolution
 *     .bias(false)
 *     .activation(Activation.RELU)
 *     .init(Init.HE);
 *
 * Conv.depthwise(3, 3).multiplier(2);                // one (or more) filters per channel
 * Conv.transposed(16, 4, 4).stride(2).padding(Padding.of(1));   // upsampling "deconvolution"
 * }</pre>
 *
 * <p>Weights are {@code [filters, channels / groups, kernel...]}, or
 * {@code [channels, filters / groups, kernel...]} for transposed convolutions, as in
 * PyTorch. Learnable activations get one parameter set per filter.
 */
public final class Conv implements Layer {

    private final boolean transposed;
    private final int channels;
    private final int filters;
    private final int[] stride;
    private final int[] dilation;
    private final int[] padBefore;
    private final int[] padAfter;
    private final int[] outputPadding;
    private final PadMode padMode;
    private final float padValue;
    private final int groups;
    private final Activation activation;
    private final Var weight;
    private final Var bias;

    /**
     * Creates a convolution from explicit settings and weights; most code uses
     * {@link #of(int, int...)} instead.
     *
     * @param settings the resolved settings
     * @param activation the activation
     * @param weight the filters
     * @param bias one value per filter, or {@code null} for none
     */
    public Conv(Settings settings, Activation activation, Tensor weight, Tensor bias) {
        this.transposed = settings.transposed;
        this.stride = settings.stride.clone();
        this.dilation = settings.dilation.clone();
        this.padBefore = settings.padBefore.clone();
        this.padAfter = settings.padAfter.clone();
        this.outputPadding = settings.outputPadding.clone();
        this.padMode = Objects.requireNonNull(settings.padMode, "padMode");
        this.padValue = settings.padValue;
        this.groups = settings.groups;
        Shape ws = weight.shape();
        int r = stride.length;
        if (ws.rank() != r + 2 || dilation.length != r || padBefore.length != r || padAfter.length != r
                || outputPadding.length != r) {
            throw new IllegalArgumentException("settings for " + r + " spatial dimensions do not fit weight " + ws);
        }
        this.channels = transposed ? ws.dim(0) : ws.dim(1) * groups;
        this.filters = transposed ? ws.dim(1) * groups : ws.dim(0);
        if (channels % groups != 0 || filters % groups != 0) {
            throw new IllegalArgumentException(groups + " groups do not divide " + channels + " channels and "
                    + filters + " filters");
        }
        if (bias != null && !bias.shape().equals(Shape.of(filters))) {
            throw new IllegalArgumentException("bias must be [" + filters + "], got " + bias.shape());
        }
        this.activation = Objects.requireNonNull(activation, "activation").bind(filters);
        this.weight = Var.param(weight);
        this.bias = bias == null ? null : Var.param(bias);
    }

    /**
     * The resolved geometry of a convolution, as stored in model files.
     *
     * @param transposed whether this is a transposed convolution
     * @param stride the step per spatial dimension
     * @param dilation the kernel cell spacing per spatial dimension
     * @param padBefore the padding before each spatial dimension
     * @param padAfter the padding after each spatial dimension
     * @param outputPadding extra output cells per dimension (transposed only)
     * @param padMode how padding cells are filled
     * @param padValue the value for {@link PadMode#CONSTANT}
     * @param groups the number of channel groups
     */
    public record Settings(boolean transposed, int[] stride, int[] dilation, int[] padBefore, int[] padAfter,
            int[] outputPadding, PadMode padMode, float padValue, int groups) {
    }

    // ---------------------------------------------------------------- specs

    /**
     * Describes a convolution with {@code filters} output channels; the number of
     * kernel sizes sets the spatial dimensions.
     *
     * @param filters the number of output channels
     * @param kernel the kernel size per spatial dimension
     * @return the spec
     */
    public static Spec of(int filters, int... kernel) {
        if (filters < 1) {
            throw new IllegalArgumentException("filters must be at least 1, got " + filters);
        }
        return new Spec(false, filters, 0, checkKernel(kernel));
    }

    /**
     * Describes a depthwise convolution: one filter (or {@link Spec#multiplier} filters)
     * per input channel, each seeing only its own channel.
     *
     * @param kernel the kernel size per spatial dimension
     * @return the spec
     */
    public static Spec depthwise(int... kernel) {
        return new Spec(false, -1, 1, checkKernel(kernel)).groups(-1);
    }

    /**
     * Describes a transposed convolution, which upsamples: with {@code stride(2)} and
     * {@code padding(Padding.SAME)} the output is twice the input size.
     *
     * @param filters the number of output channels
     * @param kernel the kernel size per spatial dimension
     * @return the spec
     */
    public static Spec transposed(int filters, int... kernel) {
        if (filters < 1) {
            throw new IllegalArgumentException("filters must be at least 1, got " + filters);
        }
        return new Spec(true, filters, 0, checkKernel(kernel));
    }

    private static int[] checkKernel(int[] kernel) {
        if (kernel.length == 0) {
            throw new IllegalArgumentException("give at least one kernel size");
        }
        for (int k : kernel) {
            if (k < 1) {
                throw new IllegalArgumentException("kernel sizes must be at least 1, got " + Arrays.toString(kernel));
            }
        }
        return kernel.clone();
    }

    /** An immutable description of a convolution; every setter returns a new spec. */
    public static final class Spec implements LayerSpec {

        private final boolean transposed;
        private final int filters;
        private final int multiplier;
        private final int[] kernel;
        private int[] stride = {1};
        private int[] dilation = {1};
        private int[] outputPadding = {0};
        private Padding padding = Padding.VALID;
        private PadMode padMode = PadMode.CONSTANT;
        private float padValue;
        private int groups = 1;
        private boolean bias = true;
        private Activation activation = Activation.IDENTITY;
        private Init init;

        private Spec(boolean transposed, int filters, int multiplier, int[] kernel) {
            this.transposed = transposed;
            this.filters = filters;
            this.multiplier = multiplier;
            this.kernel = kernel;
        }

        private Spec copy(int filters, int multiplier) {
            Spec s = new Spec(transposed, filters, multiplier, kernel);
            s.stride = stride;
            s.dilation = dilation;
            s.outputPadding = outputPadding;
            s.padding = padding;
            s.padMode = padMode;
            s.padValue = padValue;
            s.groups = groups;
            s.bias = bias;
            s.activation = activation;
            s.init = init;
            return s;
        }

        private Spec copy() {
            return copy(filters, multiplier);
        }

        /**
         * Sets the step between window positions.
         *
         * @param stride one value for all dimensions, or one per dimension
         * @return the new spec
         */
        public Spec stride(int... stride) {
            Spec s = copy();
            s.stride = positive(stride, "stride");
            return s;
        }

        /**
         * Sets the spacing between kernel cells (atrous convolution).
         *
         * @param dilation one value for all dimensions, or one per dimension
         * @return the new spec
         */
        public Spec dilation(int... dilation) {
            Spec s = copy();
            s.dilation = positive(dilation, "dilation");
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
         * Pads both sides of every dimension by the same number of cells.
         *
         * @param amounts one value for all dimensions, or one per dimension
         * @return the new spec
         */
        public Spec padding(int... amounts) {
            return padding(Padding.of(amounts));
        }

        /**
         * Sets how padding cells are filled; not available for transposed convolutions.
         *
         * @param mode the mode
         * @return the new spec
         */
        public Spec padMode(PadMode mode) {
            Spec s = copy();
            s.padMode = Objects.requireNonNull(mode, "mode");
            return s;
        }

        /**
         * Pads with a constant other than zero.
         *
         * @param value the padding value
         * @return the new spec
         */
        public Spec padValue(float value) {
            Spec s = copy();
            s.padMode = PadMode.CONSTANT;
            s.padValue = value;
            return s;
        }

        /**
         * Adds cells after each output dimension of a transposed convolution, to pick
         * among the output sizes a stride allows.
         *
         * @param amounts one value for all dimensions, or one per dimension
         * @return the new spec
         */
        public Spec outputPadding(int... amounts) {
            if (!transposed) {
                throw new IllegalStateException("outputPadding is only for transposed convolutions");
            }
            Spec s = copy();
            for (int a : amounts) {
                if (a < 0) {
                    throw new IllegalArgumentException("output padding must not be negative");
                }
            }
            s.outputPadding = amounts.clone();
            return s;
        }

        /**
         * Splits channels and filters into groups; each filter sees only its group's
         * channels. {@code -1} means one group per input channel (depthwise).
         *
         * @param groups the number of groups
         * @return the new spec
         */
        public Spec groups(int groups) {
            if (groups < 1 && groups != -1) {
                throw new IllegalArgumentException("groups must be at least 1, got " + groups);
            }
            Spec s = copy();
            s.groups = groups;
            return s;
        }

        /**
         * Sets the filters per input channel of a depthwise convolution.
         *
         * @param multiplier the depth multiplier
         * @return the new spec
         */
        public Spec multiplier(int multiplier) {
            if (filters != -1) {
                throw new IllegalStateException("multiplier is only for depthwise convolutions");
            }
            if (multiplier < 1) {
                throw new IllegalArgumentException("multiplier must be at least 1, got " + multiplier);
            }
            return copy(-1, multiplier);
        }

        /**
         * Turns the bias on or off; off is common before batch normalization.
         *
         * @param bias whether to add a learned bias per filter
         * @return the new spec
         */
        public Spec bias(boolean bias) {
            Spec s = copy();
            s.bias = bias;
            return s;
        }

        /**
         * Sets the activation applied to the output.
         *
         * @param activation the activation
         * @return the new spec
         */
        public Spec activation(Activation activation) {
            Spec s = copy();
            s.activation = Objects.requireNonNull(activation, "activation");
            return s;
        }

        /**
         * Sets the weight initializer; default is the activation's preferred one.
         *
         * @param init the initializer
         * @return the new spec
         */
        public Spec init(Init init) {
            Spec s = copy();
            s.init = Objects.requireNonNull(init, "init");
            return s;
        }

        private static int[] positive(int[] values, String what) {
            if (values.length == 0) {
                throw new IllegalArgumentException(what + " needs at least one value");
            }
            for (int v : values) {
                if (v < 1) {
                    throw new IllegalArgumentException(what + " must be at least 1, got " + Arrays.toString(values));
                }
            }
            return values.clone();
        }

        private static int[] expand(int[] values, int rank, String what) {
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

        private int rank() {
            return kernel.length;
        }

        private void checkInput(Shape input) {
            if (input.rank() != rank() + 1) {
                throw new IllegalArgumentException("a " + rank() + "-d convolution expects [channels, "
                        + rank() + " spatial dimensions], got " + input);
            }
        }

        private int resolvedGroups(int channels) {
            return groups == -1 ? channels : groups;
        }

        private int resolvedFilters(int channels) {
            return filters == -1 ? channels * multiplier : filters;
        }

        /**
         * Resolves the settings for an input shape.
         *
         * @param input the shape of one input sample, {@code [channels, spatial...]}
         * @return the settings
         */
        public Settings settings(Shape input) {
            checkInput(input);
            int r = rank();
            int[] s = expand(stride, r, "stride");
            int[] d = expand(dilation, r, "dilation");
            int[] image = Arrays.copyOfRange(input.toArray(), 1, r + 1);
            int[][] pads = transposed ? padding.resolveTransposed(kernel, s, d) : padding.resolve(image, kernel, s, d);
            int[] op = expand(outputPadding, r, "outputPadding");
            if (transposed) {
                // SAME may need one more output cell than padding can give
                for (int i = 0; i < r; i++) {
                    if (pads[1][i] < 0) {
                        op[i] -= pads[1][i];
                        pads[1][i] = 0;
                    }
                }
                if (padMode != PadMode.CONSTANT || padValue != 0) {
                    throw new IllegalArgumentException("transposed convolutions only support zero padding");
                }
            }
            return new Settings(transposed, s, d, pads[0], pads[1], op, padMode, padValue,
                    resolvedGroups(input.dim(0)));
        }

        @Override
        public Shape outputShape(Shape input) {
            Settings st = settings(input);
            int c = input.dim(0);
            int f = resolvedFilters(c);
            int g = st.groups();
            if (c % g != 0 || f % g != 0) {
                throw new IllegalArgumentException(g + " groups must divide " + c + " channels and " + f + " filters");
            }
            int r = rank();
            int[] out = new int[r + 1];
            out[0] = f;
            for (int i = 0; i < r; i++) {
                int in = input.dim(i + 1);
                int span = st.dilation()[i] * (kernel[i] - 1) + 1;
                if (transposed) {
                    out[i + 1] = (in - 1) * st.stride()[i] - st.padBefore()[i] - st.padAfter()[i] + span
                            + st.outputPadding()[i];
                } else {
                    int room = in + st.padBefore()[i] + st.padAfter()[i] - span;
                    if (room < 0) {
                        throw new IllegalArgumentException("kernel " + Arrays.toString(kernel) + " does not fit input "
                                + input + " with " + padding);
                    }
                    out[i + 1] = room / st.stride()[i] + 1;
                }
                if (out[i + 1] < 1) {
                    throw new IllegalArgumentException("convolution output would be empty for input " + input);
                }
            }
            return Shape.of(out);
        }

        @Override
        public Layer create(Shape input, SplittableRandom random) {
            outputShape(input);
            Settings st = settings(input);
            int c = input.dim(0);
            int f = resolvedFilters(c);
            int g = st.groups();
            int k = 1;
            for (int x : kernel) {
                k *= x;
            }
            int[] dims = new int[rank() + 2];
            dims[0] = transposed ? c : f;
            dims[1] = transposed ? f / g : c / g;
            System.arraycopy(kernel, 0, dims, 2, rank());
            Init weightInit = init != null ? init : activation.preferredInit();
            Tensor w = weightInit.create(Shape.of(dims), c / g * k, f / g * k, random);
            return new Conv(st, activation, w, bias ? Tensor.zeros(Shape.of(f)) : null);
        }

        @Override
        public String toString() {
            return (transposed ? "ConvTranspose" : filters == -1 ? "DepthwiseConv" : "Conv") + rank() + "d("
                    + (filters == -1 ? "x" + multiplier : filters) + ", kernel " + Arrays.toString(kernel) + ")";
        }
    }

    // ---------------------------------------------------------------- layer

    @Override
    public Var forward(Var input) {
        return forward(input, false);
    }

    @Override
    public Var forward(Var input, boolean training) {
        Shape s = input.shape();
        if (s.rank() != stride.length + 2 || s.dim(1) != channels) {
            throw new IllegalArgumentException(this + " expects [batch, " + channels + ", " + stride.length
                    + " spatial dimensions], got " + s);
        }
        Var y;
        if (transposed) {
            y = Functional.convTranspose(input, weight, bias, stride, padBefore, padAfter, outputPadding, dilation,
                    groups);
        } else {
            Var x = input;
            int[] before = padBefore;
            int[] after = padAfter;
            if (padMode != PadMode.CONSTANT || padValue != 0) {
                x = Functional.pad(x, padBefore, padAfter, padMode, padValue);
                before = new int[stride.length];
                after = new int[stride.length];
            }
            y = Functional.conv(x, weight, bias, stride, before, after, dilation, groups);
        }
        return applyActivation(activation, y);
    }

    /** Applies an activation per channel: learnable ones see channels as their last dimension. */
    static Var applyActivation(Activation activation, Var y) {
        if (activation == Activation.IDENTITY) {
            return y;
        }
        if (activation.parameters().isEmpty()) {
            return activation.forward(y);
        }
        return Functional.moveAxisBack(activation.forward(Functional.moveAxisLast(y, 1)), 1);
    }

    @Override
    public Shape outputShape(Shape input) {
        int r = stride.length;
        int[] out = new int[r + 1];
        out[0] = filters;
        for (int i = 0; i < r; i++) {
            int in = input.dim(i + 1);
            int k = weight.shape().dim(i + 2);
            int span = dilation[i] * (k - 1) + 1;
            out[i + 1] = transposed
                    ? (in - 1) * stride[i] - padBefore[i] - padAfter[i] + span + outputPadding[i]
                    : (in + padBefore[i] + padAfter[i] - span) / stride[i] + 1;
        }
        return Shape.of(out);
    }

    @Override
    public List<Var> parameters() {
        List<Var> params = new ArrayList<>();
        params.add(weight);
        if (bias != null) {
            params.add(bias);
        }
        params.addAll(activation.parameters());
        return params;
    }

    @Override
    public List<Var> buffers() {
        return activation.buffers();
    }

    /** @return the filters */
    public Var weight() {
        return weight;
    }

    /** @return the bias, or {@code null} */
    public Var bias() {
        return bias;
    }

    /** @return the activation */
    public Activation activation() {
        return activation;
    }

    /** @return the resolved settings */
    public Settings settings() {
        return new Settings(transposed, stride, dilation, padBefore, padAfter, outputPadding, padMode, padValue,
                groups);
    }

    @Override
    public String toString() {
        int[] kernel = Arrays.copyOfRange(weight.shape().toArray(), 2, stride.length + 2);
        StringBuilder sb = new StringBuilder(transposed ? "ConvTranspose" : "Conv").append(stride.length)
                .append("d(").append(channels).append(" -> ").append(filters).append(", kernel ")
                .append(Arrays.toString(kernel));
        if (Arrays.stream(stride).anyMatch(x -> x != 1)) {
            sb.append(", stride ").append(Arrays.toString(stride));
        }
        if (Arrays.stream(dilation).anyMatch(x -> x != 1)) {
            sb.append(", dilation ").append(Arrays.toString(dilation));
        }
        if (groups != 1) {
            sb.append(", groups ").append(groups);
        }
        if (activation != Activation.IDENTITY) {
            sb.append(", ").append(activation);
        }
        return sb.append(')').toString();
    }

    static Config write(Conv c) {
        Config.Builder b = Config.builder()
                .put("transposed", c.transposed)
                .put("weightShape", c.weight.shape().toArray())
                .put("bias", c.bias != null)
                .put("stride", c.stride)
                .put("dilation", c.dilation)
                .put("padBefore", c.padBefore)
                .put("padAfter", c.padAfter)
                .put("outputPadding", c.outputPadding)
                .put("padMode", c.padMode.name())
                .put("padValue", c.padValue)
                .put("groups", c.groups)
                .put("activation", c.activation);
        return b.build();
    }

    static Conv read(Config c) {
        int[] ws = c.getInts("weightShape");
        Settings st = new Settings(c.getBoolean("transposed"), c.getInts("stride"), c.getInts("dilation"),
                c.getInts("padBefore"), c.getInts("padAfter"), c.getInts("outputPadding"),
                PadMode.valueOf(c.getString("padMode")), c.getFloat("padValue"), c.getInt("groups"));
        int filters = c.getBoolean("transposed") ? ws[1] * st.groups() : ws[0];
        return new Conv(st, c.getActivation("activation"), Tensor.zeros(Shape.of(ws)),
                c.getBoolean("bias") ? Tensor.zeros(Shape.of(filters)) : null);
    }
}
