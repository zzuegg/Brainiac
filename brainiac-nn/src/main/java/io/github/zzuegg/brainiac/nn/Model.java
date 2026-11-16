package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * A neural network that turns batches of feature rows into outputs. Build a straight
 * stack of layers with {@link #sequential()}, or wire any graph of layers with
 * {@link #graph(Node, Node)}.
 *
 * <pre>{@code
 * Model model = Model.sequential()
 *         .input(2)
 *         .dense(8, Activation.TANH)
 *         .dense(1, Activation.SIGMOID)
 *         .seed(42)
 *         .build();
 *
 * float[] out = model.predict(1, 0);
 * }</pre>
 */
public final class Model {

    private final Shape inputShape;
    private final Shape outputShape;
    private final List<Step> steps;
    private final List<Layer> layers;
    private final boolean chain;

    /**
     * One computation: a layer, a sum or a concatenation of earlier values. Value 0 is the
     * model input, step {@code i} produces value {@code i + 1}.
     */
    record Step(Node.Kind kind, int[] inputs, Layer layer) {
    }

    private Model(Shape inputShape, Shape outputShape, List<Step> steps) {
        this.inputShape = inputShape;
        this.outputShape = outputShape;
        this.steps = List.copyOf(steps);
        Set<Layer> seen = Collections.newSetFromMap(new IdentityHashMap<>());
        List<Layer> distinct = new ArrayList<>();
        boolean isChain = true;
        for (int i = 0; i < steps.size(); i++) {
            Step step = steps.get(i);
            isChain &= step.kind == Node.Kind.LAYER && step.inputs[0] == i;
            if (step.layer != null && seen.add(step.layer)) {
                distinct.add(step.layer);
            }
        }
        this.layers = List.copyOf(distinct);
        this.chain = isChain;
    }

    static Model fromSteps(Shape inputShape, Shape outputShape, List<Step> steps) {
        return new Model(inputShape, outputShape, steps);
    }

    List<Step> steps() {
        return steps;
    }

    /**
     * Saves the model, its structure and all weights and buffers, to a file.
     * {@link #load(Path)} restores it. Custom layers and activations load too once they
     * are registered with {@link Layer#register(String, Class, LayerCodec)} or
     * {@link Activation#register(Activation)}; unregistered ones are still saved, and the
     * weights can be restored into a model built in code with {@link #loadWeights(Path)}.
     *
     * @param path the file to write; replaced if it exists
     * @throws IOException if the file cannot be written
     */
    public void save(Path path) throws IOException {
        try (OutputStream out = new BufferedOutputStream(Files.newOutputStream(path))) {
            ModelFile.write(this, out);
        }
    }

    /**
     * An independent copy of this model, structure, weights and buffers, on a device: such
     * as a CPU copy of a GPU model for work too small to keep the GPU busy. Custom layers
     * and activations must be registered, as for {@link #save(Path)}.
     *
     * @param device where the copy's weights go
     * @return the copy
     * @throws IllegalStateException if the model has unregistered layers or activations
     */
    public Model copy(Device device) {
        try {
            java.io.ByteArrayOutputStream bytes = new java.io.ByteArrayOutputStream();
            ModelFile.write(this, bytes);
            return ModelFile.read(new java.io.ByteArrayInputStream(bytes.toByteArray()), device);
        } catch (IOException e) {
            throw new IllegalStateException("cannot copy " + this + ": " + e.getMessage(), e);
        }
    }

    /**
     * Loads a model saved with {@link #save(Path)} onto the default device.
     *
     * @param path the file
     * @return the model
     * @throws IOException if the file cannot be read, is not a model file, or contains
     *     layers or activations that are not registered
     */
    public static Model load(Path path) throws IOException {
        return load(path, Device.getDefault());
    }

    /**
     * Loads a model saved with {@link #save(Path)}.
     *
     * @param path the file
     * @param device the device to put the weights on
     * @return the model
     * @throws IOException if the file cannot be read, is not a model file, or contains
     *     layers or activations that are not registered
     */
    public static Model load(Path path, Device device) throws IOException {
        try (InputStream in = new BufferedInputStream(Files.newInputStream(path))) {
            return ModelFile.read(in, device);
        }
    }

    /**
     * Replaces this model's weights and buffers with those saved in a file. The file must
     * come from a model with the same structure; this also works for models with custom
     * layers.
     *
     * @param path the file
     * @return this model
     * @throws IOException if the file cannot be read or its tensors do not match
     */
    public Model loadWeights(Path path) throws IOException {
        try (InputStream in = new BufferedInputStream(Files.newInputStream(path))) {
            ModelFile.readWeights(this, in);
        }
        return this;
    }

    /**
     * Starts building a model whose layers run one after another.
     *
     * @return a new builder
     */
    public static Builder sequential() {
        return new Builder();
    }

    /**
     * Starts building a model from a graph of {@link Node}s, for networks with skip
     * connections, shared layers or parallel branches.
     *
     * @param input the node created with {@link Node#input(int...)}
     * @param output the node whose values the model returns
     * @return a new builder
     */
    public static GraphBuilder graph(Node input, Node output) {
        return new GraphBuilder(input, output);
    }

    /**
     * Runs the network on a batch in inference mode, recording the computation so
     * gradients can flow back.
     *
     * @param input a batch, shape {@code [batch, inputSize]}
     * @return the output, shape {@code [batch, outputSize]}
     */
    public Var forward(Var input) {
        return forward(input, false);
    }

    /**
     * Runs the network on a batch, recording the computation so gradients can flow back.
     *
     * @param input a batch, shape {@code [batch, inputShape...]}; flat rows
     *     {@code [batch, inputSize]} are reshaped to the input shape
     * @param training {@code true} while training (enables dropout and similar layers)
     * @return the output, shape {@code [batch, outputShape...]}
     */
    public Var forward(Var input, boolean training) {
        Var[] values = new Var[steps.size() + 1];
        values[0] = shaped(input).to(dtype());
        for (int i = 0; i < steps.size(); i++) {
            Step step = steps.get(i);
            int[] in = step.inputs;
            values[i + 1] = switch (step.kind) {
                case LAYER -> step.layer.forward(values[in[0]], training);
                case ADD -> values[in[0]].add(values[in[1]]);
                case CONCAT -> {
                    Var[] parts = new Var[in.length];
                    for (int k = 0; k < in.length; k++) {
                        parts[k] = values[in[k]];
                    }
                    yield Var.concat(1, parts);
                }
                case MULTIPLY -> values[in[0]].mul(values[in[1]]);
                case MERGE -> {
                    Var[] parts = new Var[in.length];
                    for (int k = 0; k < in.length; k++) {
                        parts[k] = values[in[k]];
                    }
                    yield ((MultiInputLayer) step.layer).forward(List.of(parts), training);
                }
                case INPUT -> throw new IllegalStateException("input inside the graph");
            };
        }
        return values[steps.size()];
    }

    private Var shaped(Var input) {
        Shape s = input.shape();
        if (inputShape.rank() > 1 && s.rank() == 2 && s.dim(1) == inputSize()) {
            int[] dims = new int[inputShape.rank() + 1];
            dims[0] = s.dim(0);
            System.arraycopy(inputShape.toArray(), 0, dims, 1, inputShape.rank());
            return input.reshape(dims);
        }
        return input;
    }

    /**
     * Runs the network on a batch.
     *
     * @param batch shape {@code [batch, inputShape...]} or flat {@code [batch, inputSize]}
     * @return the outputs, shape {@code [batch, outputSize]}
     */
    public Tensor predict(Tensor batch) {
        return forward(Var.of(batch.to(device()))).value();
    }

    /**
     * Runs the network on one sample.
     *
     * @param features the input features
     * @return the outputs
     */
    public float[] predict(float... features) {
        if (features.length != inputSize()) {
            throw new IllegalArgumentException("model expects " + inputSize() + " features, got " + features.length);
        }
        return predict(Tensor.of(Shape.of(1, inputSize()), features, device())).toArray();
    }

    /**
     * Runs the network on several samples.
     *
     * @param samples one row of features per sample
     * @return one row of outputs per sample
     */
    public float[][] predict(float[][] samples) {
        float[] flat = predict(Tensor.of(samples)).toArray();
        int outputSize = outputSize();
        float[][] out = new float[samples.length][outputSize];
        for (int i = 0; i < samples.length; i++) {
            System.arraycopy(flat, i * outputSize, out[i], 0, outputSize);
        }
        return out;
    }

    /**
     * Returns every trainable parameter, layer by layer, each once even when layers
     * share it.
     *
     * @return the parameters
     */
    public List<Var> parameters() {
        List<Var> params = new ArrayList<>();
        for (Layer layer : layers) {
            params.addAll(layer.parameters());
        }
        return distinct(params);
    }

    /**
     * Returns the non-trainable state of all layers, such as batch-norm running
     * statistics, in layer order.
     *
     * @return the buffers
     */
    public List<Var> buffers() {
        List<Var> buffers = new ArrayList<>();
        for (Layer layer : layers) {
            buffers.addAll(layer.buffers());
        }
        return distinct(buffers);
    }

    /** Drops repeats, which appear when layers share a parameter (e.g. one bound activation). */
    private static List<Var> distinct(List<Var> vars) {
        Set<Var> seen = Collections.newSetFromMap(new IdentityHashMap<>());
        List<Var> out = new ArrayList<>(vars.size());
        for (Var v : vars) {
            if (seen.add(v)) {
                out.add(v);
            }
        }
        return out;
    }

    /**
     * Returns the total number of trainable values.
     *
     * @return the parameter count
     */
    public long parameterCount() {
        long n = 0;
        for (Var p : parameters()) {
            n += p.value().size();
        }
        return n;
    }

    /**
     * Moves all parameters to a device. Inputs passed to {@link #predict} are moved there
     * automatically.
     *
     * @param device the target device
     * @return this model
     */
    public Model to(Device device) {
        for (Var p : parameters()) {
            p.assign(p.value().to(device));
        }
        for (Var b : buffers()) {
            b.assign(b.value().to(device));
        }
        return this;
    }

    /**
     * Converts all parameters and buffers to a number format: {@code FLOAT64} for precise
     * numerics, {@code FLOAT16} or {@code BFLOAT16} for half the memory. Inputs are
     * converted to it automatically. For training in half precision, prefer keeping the
     * model in float32 and using {@code Trainer.mixedPrecision}, which keeps exact float32
     * weights and scales the loss so small gradients survive.
     *
     * @param type the number format
     * @return this model
     */
    public Model to(DType type) {
        for (Var p : parameters()) {
            p.assign(p.value().to(type));
        }
        for (Var b : buffers()) {
            b.assign(b.value().to(type));
        }
        return this;
    }

    /**
     * Returns the number format of the parameters.
     *
     * @return the type; {@code FLOAT32} for a model without parameters
     */
    public DType dtype() {
        List<Var> params = parameters();
        return params.isEmpty() ? DType.FLOAT32 : params.get(0).value().dtype();
    }

    /**
     * Returns the device the parameters live on.
     *
     * @return the device; the default device for a model without parameters
     */
    public Device device() {
        List<Var> params = parameters();
        return params.isEmpty() ? Device.getDefault() : params.get(0).value().device();
    }

    /** @return the number of input values per sample */
    public int inputSize() {
        return Math.toIntExact(inputShape.size());
    }

    /** @return the number of output values per sample */
    public int outputSize() {
        return Math.toIntExact(outputShape.size());
    }

    /** @return the shape of one input sample, such as {@code [1, 28, 28]} for grey images */
    public Shape inputShape() {
        return inputShape;
    }

    /** @return the shape of one output sample */
    public Shape outputShape() {
        return outputShape;
    }

    /** Whether every step is a layer applied to the previous step's output. */
    boolean isChain() {
        for (int i = 0; i < steps.size(); i++) {
            Step step = steps.get(i);
            if (step.kind != Node.Kind.LAYER || step.inputs.length != 1 || step.inputs[0] != i) {
                return false;
            }
        }
        return true;
    }

    /** @return the layers, in order */
    public List<Layer> layers() {
        return layers;
    }

    /**
     * Returns a readable table of the layers and parameter counts.
     *
     * @return the summary
     */
    public String summary() {
        StringBuilder sb = new StringBuilder();
        sb.append("Input(").append(String.join(", ", java.util.Arrays.stream(inputShape.toArray())
                .mapToObj(Integer::toString).toList())).append(")\n");
        for (int i = 0; i < steps.size(); i++) {
            Step step = steps.get(i);
            if (!chain) {
                sb.append('#').append(i + 1).append(' ');
            }
            sb.append(switch (step.kind) {
                case LAYER -> step.layer.toString();
                case ADD -> "Add";
                case CONCAT -> "Concat";
                case MULTIPLY -> "Multiply";
                case MERGE -> step.layer.toString();
                case INPUT -> "Input";
            });
            if (!chain) {
                sb.append(" <- ");
                for (int k = 0; k < step.inputs.length; k++) {
                    sb.append(k > 0 ? ", " : "").append(step.inputs[k] == 0 ? "input" : "#" + step.inputs[k]);
                }
            }
            long n = 0;
            if (step.layer != null) {
                for (Var p : step.layer.parameters()) {
                    n += p.value().size();
                }
            }
            sb.append(n > 0 ? "  params: " + n : "").append('\n');
        }
        sb.append("Total params: ").append(parameterCount());
        return sb.toString();
    }

    @Override
    public String toString() {
        return "Model" + layers;
    }

    /** Builds a {@link Model} layer by layer. */
    public static final class Builder {

        private Shape inputShape;
        private Shape shape;
        private Long seed;
        private Device device;
        private final List<Function<SplittableRandom, Layer>> pending = new ArrayList<>();

        private Builder() {
        }

        /**
         * Sets the shape of one input sample. Must come first. One number is a row of
         * features; more describe images ({@code input(3, 32, 32)}: channels first) or
         * sequences ({@code input(steps, features)}).
         *
         * @param dims the input dimensions
         * @return this builder
         */
        public Builder input(int... dims) {
            if (inputShape != null) {
                throw new IllegalStateException("input(...) was already set");
            }
            if (dims.length == 0) {
                throw new IllegalArgumentException("input needs at least one dimension");
            }
            for (int d : dims) {
                if (d < 1) {
                    throw new IllegalArgumentException("input needs at least 1 feature, got " + d);
                }
            }
            inputShape = Shape.of(dims);
            shape = inputShape;
            return this;
        }

        /**
         * Adds a layer described by a spec, which knows its output shape for the current
         * input shape. All built-in layers are available this way, with every option.
         *
         * @param spec the layer spec, such as {@code Conv.of(32, 3, 3).stride(2)}
         * @return this builder
         */
        public Builder add(LayerSpec spec) {
            Objects.requireNonNull(spec, "spec");
            Shape in = current();
            Shape out = spec.outputShape(in);
            pending.add(random -> spec.create(in, random));
            shape = out;
            return this;
        }

        /** @return the shape of one sample after the layers added so far */
        public Shape shape() {
            return current();
        }

        /**
         * Adds a fully connected layer without activation.
         *
         * @param units the number of outputs
         * @return this builder
         */
        public Builder dense(int units) {
            return dense(units, Activation.IDENTITY);
        }

        /**
         * Adds a fully connected layer. Weights use the activation's
         * {@linkplain Activation#preferredInit() preferred init}: He for the ReLU family,
         * Xavier otherwise.
         *
         * @param units the number of outputs
         * @param activation the activation
         * @return this builder
         */
        public Builder dense(int units, Activation activation) {
            return dense(units, activation, activation.preferredInit());
        }

        /**
         * Adds a fully connected layer with a chosen initializer.
         *
         * @param units the number of outputs
         * @param activation the activation
         * @param init the weight initializer
         * @return this builder
         */
        public Builder dense(int units, Activation activation, Init init) {
            Objects.requireNonNull(activation, "activation");
            Objects.requireNonNull(init, "init");
            return add(Dense.spec(units, activation, init));
        }

        /**
         * Adds dropout: while training, each value is zeroed with probability
         * {@code rate} and the rest are scaled up to keep the expected sum. Reduces
         * overfitting; has no effect on predictions.
         *
         * @param rate the share of values to drop, in [0, 1)
         * @return this builder
         */
        public Builder dropout(float rate) {
            current();
            Dropout.check(rate);
            pending.add(random -> new Dropout(rate, random.split()));
            return this;
        }

        /**
         * Adds batch normalization over the current features: each feature is normalized
         * to zero mean and unit variance over the batch while training (and with running
         * averages for predictions), then scaled and shifted by learned values. Usually
         * placed between a linear {@code dense(...)} layer and its activation.
         *
         * @return this builder
         */
        public Builder batchNorm() {
            int in = current().dim(0);
            pending.add(random -> new BatchNorm(in));
            return this;
        }

        /**
         * Applies an activation function on its own, for example after
         * {@link #batchNorm()}.
         *
         * @param activation the activation
         * @return this builder
         */
        public Builder activation(Activation activation) {
            Objects.requireNonNull(activation, "activation");
            int in = current().dim(-1);
            pending.add(random -> activation.bind(in));
            return this;
        }

        /**
         * Adds any layer, such as an extra {@link Activation} or a custom one.
         *
         * @param layer the layer
         * @return this builder
         */
        public Builder layer(Layer layer) {
            Objects.requireNonNull(layer, "layer");
            Shape in = current();
            pending.add(random -> layer);
            shape = layer.outputShape(in);
            return this;
        }

        /**
         * Adds a 1-d convolution over {@code [channels, length]}, without padding.
         * Use {@code add(Conv.of(...))} for every other option.
         *
         * @param filters the number of output channels
         * @param kernel the kernel length
         * @param activation the activation
         * @return this builder
         */
        public Builder conv1d(int filters, int kernel, Activation activation) {
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
         * @return this builder
         */
        public Builder conv2d(int filters, int kernel, Activation activation) {
            return add(Conv.of(filters, kernel, kernel).activation(activation));
        }

        /**
         * Adds a 3-d convolution over {@code [channels, depth, height, width]} with a cubic
         * kernel, without padding.
         *
         * @param filters the number of output channels
         * @param kernel the kernel size
         * @param activation the activation
         * @return this builder
         */
        public Builder conv3d(int filters, int kernel, Activation activation) {
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
         * @return this builder
         */
        public Builder separableConv(int filters, Activation activation, int... kernel) {
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
         * @return this builder
         */
        public Builder maxPool(int... kernel) {
            return add(Pool.max(kernel));
        }

        /**
         * Adds average pooling with the stride equal to the window.
         *
         * @param kernel the window size per spatial dimension
         * @return this builder
         */
        public Builder avgPool(int... kernel) {
            return add(Pool.avg(kernel));
        }

        /**
         * Averages each channel over all positions: {@code [channels, ...] -> [channels]}.
         *
         * @return this builder
         */
        public Builder globalAvgPool() {
            return layer(GlobalPool.avg());
        }

        /**
         * Takes the maximum of each channel over all positions.
         *
         * @return this builder
         */
        public Builder globalMaxPool() {
            return layer(GlobalPool.max());
        }

        /**
         * Flattens each sample into one row of features, for example before dense layers.
         *
         * @return this builder
         */
        public Builder flatten() {
            return layer(Reshape.flatten());
        }

        /**
         * Changes the shape of each sample; one dimension may be {@code -1}.
         *
         * @param dims the new shape of one sample
         * @return this builder
         */
        public Builder reshape(int... dims) {
            return layer(new Reshape(dims));
        }

        /**
         * Adds layer normalization over the last dimension; see {@link LayerNorm}.
         *
         * @return this builder
         */
        public Builder layerNorm() {
            return layer(LayerNorm.of(current().dim(-1)));
        }

        /**
         * Adds group normalization over the channels (the first sample dimension); see
         * {@link GroupNorm}.
         *
         * @param groups the number of groups
         * @return this builder
         */
        public Builder groupNorm(int groups) {
            return layer(new GroupNorm(groups, current().dim(0), 1e-5f, true));
        }

        /**
         * Upsamples the spatial dimensions by repeating cells. Use {@code layer(Resize...)}
         * for linear or cubic interpolation.
         *
         * @param factor the scale factor
         * @return this builder
         */
        public Builder upsample(int factor) {
            return layer(Resize.scale(factor));
        }

        /**
         * Adds spatial dropout: drops whole channels while training.
         *
         * @param rate the share of channels to drop, in [0, 1)
         * @return this builder
         */
        public Builder spatialDropout(float rate) {
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
         * @return this builder
         */
        public Builder lstm(int units, boolean returnSequences) {
            return add(Recurrent.lstm(units).returnSequences(returnSequences));
        }

        /**
         * Adds a GRU over {@code [steps, features]}.
         *
         * @param units the hidden size
         * @param returnSequences {@code true} for every step, {@code false} for the last
         * @return this builder
         */
        public Builder gru(int units, boolean returnSequences) {
            return add(Recurrent.gru(units).returnSequences(returnSequences));
        }

        /**
         * Adds a simple tanh RNN over {@code [steps, features]}.
         *
         * @param units the hidden size
         * @param returnSequences {@code true} for every step, {@code false} for the last
         * @return this builder
         */
        public Builder rnn(int units, boolean returnSequences) {
            return add(Recurrent.rnn(units).returnSequences(returnSequences));
        }

        /**
         * Adds multi-head self-attention over {@code [steps, features]}. Use
         * {@code add(MultiHeadAttention.of(...))} for every option.
         *
         * @param heads the number of heads; must divide the features
         * @param causal whether each step sees only itself and earlier steps
         * @return this builder
         */
        public Builder attention(int heads, boolean causal) {
            return add(MultiHeadAttention.of(heads).causal(causal));
        }

        /**
         * Adds a pre-norm transformer encoder block (or a GPT-style block if causal). Use
         * {@code add(TransformerBlock.of(...))} for every option.
         *
         * @param heads the number of heads; must divide the features
         * @param causal whether self-attention is causal
         * @return this builder
         */
        public Builder transformer(int heads, boolean causal) {
            return add(TransformerBlock.of(heads).causal(causal));
        }

        /**
         * Adds fixed sinusoidal position encodings to {@code [steps, features]}.
         *
         * @return this builder
         */
        public Builder positionalEncoding() {
            return layer(PositionalEncoding.sinusoidal(current().dim(-1)));
        }

        /**
         * Reduces {@code [steps, features]} to {@code [features]}.
         *
         * @param kind mean, max, sum, first or last step
         * @return this builder
         */
        public Builder sequencePool(SequencePool.Kind kind) {
            return layer(new SequencePool(kind));
        }

        /**
         * Looks up a learned {@code dimension}-vector for every token id; see {@link Embedding}
         * for padding, pretrained and frozen vectors.
         *
         * @param vocabulary the number of token ids
         * @param dimension the size of each vector
         * @return this builder
         */
        public Builder embedding(int vocabulary, int dimension) {
            return add(Embedding.of(vocabulary, dimension));
        }

        /**
         * Fixes the random seed for weight initialization, so builds are reproducible.
         * Without it every build gets different starting weights.
         *
         * @param seed the seed
         * @return this builder
         */
        public Builder seed(long seed) {
            this.seed = seed;
            return this;
        }

        /**
         * Places the parameters on a device. Default: the default device.
         *
         * @param device the device
         * @return this builder
         */
        public Builder device(Device device) {
            this.device = Objects.requireNonNull(device, "device");
            return this;
        }

        /**
         * Creates the model.
         *
         * @return the model
         */
        public Model build() {
            if (pending.isEmpty()) {
                throw new IllegalStateException("a model needs at least one layer");
            }
            SplittableRandom random = seed == null ? new SplittableRandom() : new SplittableRandom(seed);
            List<Step> steps = new ArrayList<>();
            for (var factory : pending) {
                steps.add(new Step(Node.Kind.LAYER, new int[] {steps.size()}, factory.apply(random)));
            }
            Model model = new Model(inputShape, shape, steps);
            if (device != null) {
                model.to(device);
            }
            return model;
        }

        private Shape current() {
            if (inputShape == null) {
                throw new IllegalStateException("call input(features) before adding layers");
            }
            return shape;
        }
    }

    /** Builds a {@link Model} from a graph of {@link Node}s; see {@link Model#graph(Node, Node)}. */
    public static final class GraphBuilder {

        private final Node input;
        private final Node output;
        private Long seed;
        private Device device;

        private GraphBuilder(Node input, Node output) {
            this.input = Objects.requireNonNull(input, "input");
            this.output = Objects.requireNonNull(output, "output");
            if (input.kind() != Node.Kind.INPUT) {
                throw new IllegalArgumentException("the first argument must be created with Node.input(features)");
            }
        }

        /**
         * Sets the seed for weight initialization and dropout, for reproducible runs.
         *
         * @param seed the seed
         * @return this builder
         */
        public GraphBuilder seed(long seed) {
            this.seed = seed;
            return this;
        }

        /**
         * Sets the device the weights live on; defaults to {@link Device#getDefault()}.
         *
         * @param device the device
         * @return this builder
         */
        public GraphBuilder device(Device device) {
            this.device = Objects.requireNonNull(device, "device");
            return this;
        }

        /**
         * Creates the model with fresh weights. Every call gives a new, independent model,
         * except for layer instances passed to {@link Node#layer(Layer)}, which are shared.
         *
         * @return the model
         */
        public Model build() {
            if (output == input) {
                throw new IllegalStateException("a model needs at least one layer");
            }
            // collect the nodes the output depends on, then order them by creation, which
            // is always a valid evaluation order since nodes only refer to older nodes
            List<Node> nodes = new ArrayList<>();
            Set<Node> seen = Collections.newSetFromMap(new IdentityHashMap<>());
            List<Node> stack = new ArrayList<>(List.of(output));
            while (!stack.isEmpty()) {
                Node n = stack.removeLast();
                if (!seen.add(n)) {
                    continue;
                }
                if (n.kind() == Node.Kind.INPUT && n != input) {
                    throw new IllegalArgumentException("the output depends on " + n + ", which is not the model input");
                }
                nodes.add(n);
                stack.addAll(n.inputs());
            }
            if (!seen.contains(input)) {
                throw new IllegalArgumentException("the output does not depend on the input");
            }
            nodes.sort(Comparator.comparingLong(Node::id));
            SplittableRandom random = seed == null ? new SplittableRandom() : new SplittableRandom(seed);
            Map<Node, Integer> index = new HashMap<>();
            index.put(input, 0);
            List<Step> steps = new ArrayList<>();
            for (Node n : nodes) {
                if (n == input) {
                    continue;
                }
                int[] in = n.inputs().stream().mapToInt(index::get).toArray();
                Layer layer = n.kind() == Node.Kind.LAYER || n.kind() == Node.Kind.MERGE ? n.createLayer(random) : null;
                steps.add(new Step(n.kind(), in, layer));
                index.put(n, steps.size());
            }
            Model model = new Model(input.shape(), output.shape(), steps);
            if (device != null) {
                model.to(device);
            }
            return model;
        }
    }
}
