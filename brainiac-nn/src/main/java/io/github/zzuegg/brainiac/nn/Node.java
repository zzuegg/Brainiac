package io.github.zzuegg.brainiac.nn;

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

    enum Kind { INPUT, LAYER, ADD, CONCAT }

    private static final AtomicLong IDS = new AtomicLong();

    private final long id;
    private final Kind kind;
    private final List<Node> inputs;
    private final Function<SplittableRandom, Layer> factory;
    private final int width;

    private Node(Kind kind, List<Node> inputs, Function<SplittableRandom, Layer> factory, int width) {
        this.id = IDS.incrementAndGet();
        this.kind = kind;
        this.inputs = List.copyOf(inputs);
        this.factory = factory;
        this.width = width;
    }

    /**
     * Creates the input of a network.
     *
     * @param features the number of input features
     * @return the input node
     */
    public static Node input(int features) {
        if (features < 1) {
            throw new IllegalArgumentException("input needs at least 1 feature, got " + features);
        }
        return new Node(Kind.INPUT, List.of(), null, features);
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
     * Adds a fully connected layer.
     *
     * @param units the number of outputs
     * @param activation the activation
     * @param init how to initialize the weights
     * @return the layer's output
     */
    public Node dense(int units, Activation activation, Init init) {
        Objects.requireNonNull(activation, "activation");
        Objects.requireNonNull(init, "init");
        if (units < 1) {
            throw new IllegalArgumentException("units must be at least 1, got " + units);
        }
        int in = width;
        return new Node(Kind.LAYER, List.of(this), random -> new Dense(in, units, activation, init, random), units);
    }

    /**
     * Adds dropout; see {@link Dropout}.
     *
     * @param rate the share of values dropped while training, in [0, 1)
     * @return the layer's output
     */
    public Node dropout(float rate) {
        if (!(rate >= 0 && rate < 1)) {
            throw new IllegalArgumentException("dropout rate must be in [0, 1), got " + rate);
        }
        return new Node(Kind.LAYER, List.of(this), random -> new Dropout(rate, random.split()), width);
    }

    /**
     * Adds batch normalization; see {@link BatchNorm}.
     *
     * @return the layer's output
     */
    public Node batchNorm() {
        int in = width;
        return new Node(Kind.LAYER, List.of(this), random -> new BatchNorm(in), width);
    }

    /**
     * Applies an activation function.
     *
     * @param activation the activation
     * @return the activated values
     */
    public Node activation(Activation activation) {
        Objects.requireNonNull(activation, "activation");
        int in = width;
        return new Node(Kind.LAYER, List.of(this), random -> activation.bind(in), width);
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
        return new Node(Kind.LAYER, List.of(this), random -> layer, layer.outputSize(width));
    }

    /**
     * Adds this node and another element-wise, as in a residual (skip) connection.
     *
     * @param other a node of the same width
     * @return the sum
     */
    public Node add(Node other) {
        Objects.requireNonNull(other, "other");
        if (other.width != width) {
            throw new IllegalArgumentException("cannot add widths " + width + " and " + other.width);
        }
        return new Node(Kind.ADD, List.of(this, other), null, width);
    }

    /**
     * Joins the features of this node and others side by side.
     *
     * @param others the nodes to append, at least one
     * @return the joined features, as wide as all parts together
     */
    public Node concat(Node... others) {
        if (others.length == 0) {
            throw new IllegalArgumentException("concat needs at least one other node");
        }
        List<Node> parts = new ArrayList<>();
        parts.add(this);
        int total = width;
        for (Node n : others) {
            parts.add(Objects.requireNonNull(n, "node"));
            total += n.width;
        }
        return new Node(Kind.CONCAT, parts, null, total);
    }

    /** @return the number of features this node produces */
    public int width() {
        return width;
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
            case INPUT -> "Input(" + width + ")";
            case ADD -> "Add(" + width + ")";
            case CONCAT -> "Concat(" + width + ")";
            case LAYER -> "Node(" + width + ")";
        };
    }
}
