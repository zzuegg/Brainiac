package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.tensor.Shape;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Function;

/**
 * A point in a network graph: the model input, or the output of a layer, a sum or a
 * concatenation. Nodes describe the wiring only; {@link Model#graph(Node, Node)} turns
 * them into a model with fresh weights.
 *
 * <p>Use nodes when the network is not a straight line, for example with skip
 * connections or parallel branches:
 *
 * <pre>{@code
 * Node input = Node.input(4);
 * Node hidden = input.dense(32, Activation.RELU);
 * Node block = hidden.dense(32, Activation.RELU).dense(32).add(hidden).activation(Activation.RELU);
 * Node output = block.concat(hidden).dense(3, Activation.SOFTMAX);
 * Model model = Model.graph(input, output).seed(42).build();
 * }</pre>
 *
 * <p>Nodes are immutable, so one node can feed several others.
 */
public final class Node {

    enum Kind { INPUT, LAYER, ADD, CONCAT, MULTIPLY, MERGE }

    private static final AtomicLong IDS = new AtomicLong();

    private final long id;
    private final Kind kind;
    private final List<Node> inputs;
    private final Function<SplittableRandom, Layer> factory;
    private final Shape shape;

    private Node(Kind kind, List<Node> inputs, Function<SplittableRandom, Layer> factory, Shape shape) {
        this.id = IDS.incrementAndGet();
        this.kind = kind;
        this.inputs = List.copyOf(inputs);
        this.factory = factory;
        this.shape = shape;
    }

    /**
     * Creates the input of a network. One number is a row of features; more describe
     * images ({@code input(3, 32, 32)}, channels first) or sequences
     * ({@code input(steps, features)}).
     *
     * @param dims the shape of one input sample
     * @return the input node
     */
    public static Node input(int... dims) {
        if (dims.length == 0) {
            throw new IllegalArgumentException("input needs at least one dimension");
        }
        for (int d : dims) {
            if (d < 1) {
                throw new IllegalArgumentException("input needs at least 1 feature, got " + d);
            }
        }
        return new Node(Kind.INPUT, List.of(), null, Shape.of(dims));
    }

    /**
     * Adds a layer described by a spec; every built-in layer and option is available
     * this way.
     *
     * @param spec the layer spec
     * @return the layer's output
     */
    public Node add(LayerSpec spec) {
        Objects.requireNonNull(spec, "spec");
        Shape in = shape;
        return new Node(Kind.LAYER, List.of(this), random -> spec.create(in, random), spec.outputShape(in));
    }

    /**
     * Adds a fully connected layer without activation.
     *
     * @param units the number of outputs
     * @return the layer's output
     */
    public Node dense(int units) {
        return dense(units, Activation.IDENTITY);
    }

    /**
     * Adds a fully connected layer, initialized to suit the activation.
     *
     * @param units the number of outputs
     * @param activation the activation
     * @return the layer's output
     */
    public Node dense(int units, Activation activation) {
        return dense(units, activation, activation.preferredInit());
    }

    /**
     * Adds a fully connected layer. On inputs with several dimensions it works on the
     * last one, like a time-distributed layer.
     *
     * @param units the number of outputs
     * @param activation the activation
     * @param init how to initialize the weights
     * @return the layer's output
     */
    public Node dense(int units, Activation activation, Init init) {
        Objects.requireNonNull(activation, "activation");
        Objects.requireNonNull(init, "init");
        return add(Dense.spec(units, activation, init));
    }

    /**
     * Adds dropout; see {@link Dropout}.
     *
     * @param rate the share of values dropped while training, in [0, 1)
     * @return the layer's output
     */
    public Node dropout(float rate) {
        Dropout.check(rate);
        return new Node(Kind.LAYER, List.of(this), random -> new Dropout(rate, random.split()), shape);
    }

    /**
     * Adds batch normalization over the first sample dimension (features, or channels of
     * images); see {@link BatchNorm}.
     *
     * @return the layer's output
     */
    public Node batchNorm() {
        int in = shape.dim(0);
        return new Node(Kind.LAYER, List.of(this), random -> new BatchNorm(in), shape);
    }

    /**
     * Applies an activation function. Learnable activations get one parameter set per
     * value of the last dimension.
     *
     * @param activation the activation
     * @return the activated values
     */
    public Node activation(Activation activation) {
        Objects.requireNonNull(activation, "activation");
        int in = shape.dim(-1);
        return new Node(Kind.LAYER, List.of(this), random -> activation.bind(in), shape);
    }

    /**
     * Adds a layer instance. Using the same instance at several places shares its
     * weights between them.
     *
     * @param layer the layer
     * @return the layer's output
     */
    public Node layer(Layer layer) {
        Objects.requireNonNull(layer, "layer");
        return new Node(Kind.LAYER, List.of(this), random -> layer, layer.outputShape(shape));
    }

    /**
     * Adds a 1-d convolution over {@code [channels, length]}, without padding.
     * Use {@code add(Conv.of(...))} for every other option.
     *
     * @param filters the number of output channels
     * @param kernel the kernel length
     * @param activation the activation
     * @return the layer's output
     */
    public Node conv1d(int filters, int kernel, Activation activation) {
        return add(Conv.of(filters, kernel).activation(activation));
    }

    /**
     * Adds a 2-d convolution over {@code [channels, height, width]} with a square
     * kernel, without padding. Use {@code add(Conv.of(...))} for stride, padding,
     * dilation, groups and more.
     *
     * @param filters the number of output channels
     * @param kernel the kernel size
     * @param activation the activation
     * @return the layer's output
     */
    public Node conv2d(int filters, int kernel, Activation activation) {
        return add(Conv.of(filters, kernel, kernel).activation(activation));
    }

    /**
     * Adds a 3-d convolution over {@code [channels, depth, height, width]} with a cubic
     * kernel, without padding.
     *
     * @param filters the number of output channels
     * @param kernel the kernel size
     * @param activation the activation
     * @return the layer's output
     */
    public Node conv3d(int filters, int kernel, Activation activation) {
        return add(Conv.of(filters, kernel, kernel, kernel).activation(activation));
    }

    /**
     * Adds a depthwise separable convolution: a depthwise convolution (one filter per
     * channel) followed by a 1x1 convolution that mixes channels; far cheaper than a
     * full convolution. Uses SAME padding.
     *
     * @param filters the number of output channels
     * @param activation the activation after the pointwise step
     * @param kernel the kernel size per spatial dimension
     * @return the layer's output
     */
    public Node separableConv(int filters, Activation activation, int... kernel) {
        int[] ones = new int[kernel.length];
        java.util.Arrays.fill(ones, 1);
        return add(Conv.depthwise(kernel).padding(Padding.SAME).bias(false))
                .add(Conv.of(filters, ones).activation(activation));
    }

    /**
     * Adds max pooling with the stride equal to the window; the number of sizes sets
     * the spatial dimensions. Use {@code add(Pool.max(...))} for every option.
     *
     * @param kernel the window size per spatial dimension
     * @return the layer's output
     */
    public Node maxPool(int... kernel) {
        return add(Pool.max(kernel));
    }

    /**
     * Adds average pooling with the stride equal to the window.
     *
     * @param kernel the window size per spatial dimension
     * @return the layer's output
     */
    public Node avgPool(int... kernel) {
        return add(Pool.avg(kernel));
    }

    /**
     * Averages each channel over all positions: {@code [channels, ...] -> [channels]}.
     *
     * @return the layer's output
     */
    public Node globalAvgPool() {
        return layer(GlobalPool.avg());
    }

    /**
     * Takes the maximum of each channel over all positions.
     *
     * @return the layer's output
     */
    public Node globalMaxPool() {
        return layer(GlobalPool.max());
    }

    /**
     * Flattens each sample into one row of features, for example before dense layers.
     *
     * @return the layer's output
     */
    public Node flatten() {
        return layer(Reshape.flatten());
    }

    /**
     * Changes the shape of each sample; one dimension may be {@code -1}.
     *
     * @param dims the new shape of one sample
     * @return the layer's output
     */
    public Node reshape(int... dims) {
        return layer(new Reshape(dims));
    }

    /**
     * Adds layer normalization over the last dimension; see {@link LayerNorm}.
     *
     * @return the layer's output
     */
    public Node layerNorm() {
        return layer(LayerNorm.of(shape.dim(-1)));
    }

    /**
     * Adds group normalization over the channels (the first sample dimension); see
     * {@link GroupNorm}.
     *
     * @param groups the number of groups
     * @return the layer's output
     */
    public Node groupNorm(int groups) {
        return layer(new GroupNorm(groups, shape.dim(0), 1e-5f, true));
    }

    /**
     * Upsamples the spatial dimensions by repeating cells. Use {@code layer(Resize...)}
     * for linear or cubic interpolation.
     *
     * @param factor the scale factor
     * @return the layer's output
     */
    public Node upsample(int factor) {
        return layer(Resize.scale(factor));
    }

    /**
     * Adds spatial dropout: drops whole channels while training.
     *
     * @param rate the share of channels to drop, in [0, 1)
     * @return the layer's output
     */
    public Node spatialDropout(float rate) {
        return add(new LayerSpec() {
            @Override
            public io.github.zzuegg.brainiac.tensor.Shape outputShape(io.github.zzuegg.brainiac.tensor.Shape in) {
                Dropout.check(rate);
                return in;
            }

            @Override
            public Layer create(io.github.zzuegg.brainiac.tensor.Shape in, java.util.SplittableRandom random) {
                return new Dropout(rate, random.split(), true);
            }
        });
    }

    /**
     * Adds an LSTM over {@code [steps, features]}. Use {@code add(Recurrent.lstm(...))}
     * for bidirectional layers and every other option.
     *
     * @param units the hidden size
     * @param returnSequences {@code true} for every step {@code [steps, units]}, {@code false} for the last
     * @return the layer's output
     */
    public Node lstm(int units, boolean returnSequences) {
        return add(Recurrent.lstm(units).returnSequences(returnSequences));
    }

    /**
     * Adds a GRU over {@code [steps, features]}.
     *
     * @param units the hidden size
     * @param returnSequences {@code true} for every step, {@code false} for the last
     * @return the layer's output
     */
    public Node gru(int units, boolean returnSequences) {
        return add(Recurrent.gru(units).returnSequences(returnSequences));
    }

    /**
     * Adds a simple tanh RNN over {@code [steps, features]}.
     *
     * @param units the hidden size
     * @param returnSequences {@code true} for every step, {@code false} for the last
     * @return the layer's output
     */
    public Node rnn(int units, boolean returnSequences) {
        return add(Recurrent.rnn(units).returnSequences(returnSequences));
    }

    /**
     * Adds multi-head self-attention over {@code [steps, features]}. Use
     * {@code add(MultiHeadAttention.of(...))} for every option.
     *
     * @param heads the number of heads; must divide the features
     * @param causal whether each step sees only itself and earlier steps
     * @return the layer's output
     */
    public Node attention(int heads, boolean causal) {
        return add(MultiHeadAttention.of(heads).causal(causal));
    }

    /**
     * Adds a pre-norm transformer encoder block (or a GPT-style block if causal). Use
     * {@code add(TransformerBlock.of(...))} for every option.
     *
     * @param heads the number of heads; must divide the features
     * @param causal whether self-attention is causal
     * @return the layer's output
     */
    public Node transformer(int heads, boolean causal) {
        return add(TransformerBlock.of(heads).causal(causal));
    }

    /**
     * Adds fixed sinusoidal position encodings to {@code [steps, features]}.
     *
     * @return the layer's output
     */
    public Node positionalEncoding() {
        return layer(PositionalEncoding.sinusoidal(shape.dim(-1)));
    }

    /**
     * Reduces {@code [steps, features]} to {@code [features]}.
     *
     * @param kind mean, max, sum, first or last step
     * @return the layer's output
     */
    public Node sequencePool(SequencePool.Kind kind) {
        return layer(new SequencePool(kind));
    }

    /**
     * Adds this node and another element-wise, as in a residual (skip) connection.
     *
     * @param other a node of the same shape
     * @return the sum
     */
    public Node add(Node other) {
        Objects.requireNonNull(other, "other");
        if (!other.shape.equals(shape)) {
            throw new IllegalArgumentException("cannot add shapes " + shape + " and " + other.shape);
        }
        return new Node(Kind.ADD, List.of(this, other), null, shape);
    }

    /**
     * Multiplies this node and another element-wise, as in gating.
     *
     * @param other a node of the same shape
     * @return the product
     */
    public Node multiply(Node other) {
        Objects.requireNonNull(other, "other");
        if (!other.shape.equals(shape)) {
            throw new IllegalArgumentException("cannot multiply shapes " + shape + " and " + other.shape);
        }
        return new Node(Kind.MULTIPLY, List.of(this, other), null, shape);
    }

    /**
     * Joins this node and others along the first sample dimension: features side by
     * side, or the channels of images.
     *
     * @param others the nodes to append, at least one
     * @return the joined values
     */
    public Node concat(Node... others) {
        if (others.length == 0) {
            throw new IllegalArgumentException("concat needs at least one other node");
        }
        List<Node> parts = new ArrayList<>();
        parts.add(this);
        int[] dims = shape.toArray();
        for (Node n : others) {
            parts.add(Objects.requireNonNull(n, "node"));
            int[] d = n.shape.toArray();
            boolean ok = d.length == dims.length;
            for (int i = 1; ok && i < d.length; i++) {
                ok = d[i] == dims[i];
            }
            if (!ok) {
                throw new IllegalArgumentException("cannot concat shapes " + shape + " and " + n.shape
                        + ": all but the first dimension must match");
            }
            dims[0] += d[0];
        }
        return new Node(Kind.CONCAT, parts, null, Shape.of(dims));
    }

    /**
     * Feeds this node and others into a layer that takes several inputs, such as
     * cross-attention.
     *
     * @param spec the layer spec
     * @param others the other inputs, after this one
     * @return the layer's output
     */
    public Node merge(MultiInputLayer.Spec spec, Node... others) {
        Objects.requireNonNull(spec, "spec");
        List<Node> parts = new ArrayList<>();
        parts.add(this);
        for (Node n : others) {
            parts.add(Objects.requireNonNull(n, "node"));
        }
        List<Shape> shapes = parts.stream().map(Node::shape).toList();
        return new Node(Kind.MERGE, parts, random -> spec.create(shapes, random), spec.outputShape(shapes));
    }

    /** @return the number of values per sample this node produces */
    public int width() {
        return Math.toIntExact(shape.size());
    }

    /** @return the shape of one sample this node produces */
    public Shape shape() {
        return shape;
    }

    long id() {
        return id;
    }

    Kind kind() {
        return kind;
    }

    List<Node> inputs() {
        return inputs;
    }

    Layer createLayer(SplittableRandom random) {
        return factory.apply(random);
    }

    @Override
    public String toString() {
        return switch (kind) {
            case INPUT -> "Input" + shape;
            case ADD -> "Add" + shape;
            case MULTIPLY -> "Multiply" + shape;
            case CONCAT -> "Concat" + shape;
            case LAYER, MERGE -> "Node" + shape;
        };
    }
}
