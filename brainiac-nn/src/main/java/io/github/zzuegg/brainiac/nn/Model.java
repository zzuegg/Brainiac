package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * A neural network: layers applied one after another to batches of feature rows.
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

    private final int inputSize;
    private final int outputSize;
    private final List<Layer> layers;

    private Model(int inputSize, int outputSize, List<Layer> layers) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;
        this.layers = List.copyOf(layers);
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
     * @param input a batch, shape {@code [batch, inputSize]}
     * @param training {@code true} while training (enables dropout and similar layers)
     * @return the output, shape {@code [batch, outputSize]}
     */
    public Var forward(Var input, boolean training) {
        Var x = input;
        for (Layer layer : layers) {
            x = layer.forward(x, training);
        }
        return x;
    }

    /**
     * Runs the network on a batch.
     *
     * @param batch shape {@code [batch, inputSize]}
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
        if (features.length != inputSize) {
            throw new IllegalArgumentException("model expects " + inputSize + " features, got " + features.length);
        }
        return predict(Tensor.of(Shape.of(1, inputSize), features, device())).toArray();
    }

    /**
     * Runs the network on several samples.
     *
     * @param samples one row of features per sample
     * @return one row of outputs per sample
     */
    public float[][] predict(float[][] samples) {
        float[] flat = predict(Tensor.of(samples)).toArray();
        float[][] out = new float[samples.length][outputSize];
        for (int i = 0; i < samples.length; i++) {
            System.arraycopy(flat, i * outputSize, out[i], 0, outputSize);
        }
        return out;
    }

    /**
     * Returns every trainable parameter, layer by layer.
     *
     * @return the parameters
     */
    public List<Var> parameters() {
        List<Var> params = new ArrayList<>();
        for (Layer layer : layers) {
            params.addAll(layer.parameters());
        }
        return params;
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
        return buffers;
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
     * Returns the device the parameters live on.
     *
     * @return the device; the default device for a model without parameters
     */
    public Device device() {
        List<Var> params = parameters();
        return params.isEmpty() ? Device.getDefault() : params.get(0).value().device();
    }

    /** @return the number of input features */
    public int inputSize() {
        return inputSize;
    }

    /** @return the number of output values per sample */
    public int outputSize() {
        return outputSize;
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
        sb.append("Input(").append(inputSize).append(")\n");
        for (Layer layer : layers) {
            long n = 0;
            for (Var p : layer.parameters()) {
                n += p.value().size();
            }
            sb.append(layer).append(n > 0 ? "  params: " + n : "").append('\n');
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

        private int inputSize = -1;
        private int width = -1;
        private Long seed;
        private Device device;
        private final List<Function<SplittableRandom, Layer>> pending = new ArrayList<>();

        private Builder() {
        }

        /**
         * Sets the number of input features. Must come first.
         *
         * @param features the input width
         * @return this builder
         */
        public Builder input(int features) {
            if (inputSize != -1) {
                throw new IllegalStateException("input(...) was already set");
            }
            if (features < 1) {
                throw new IllegalArgumentException("input needs at least 1 feature, got " + features);
            }
            inputSize = features;
            width = features;
            return this;
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
         * Adds a fully connected layer. Weights use He init for ReLU and Xavier init
         * otherwise.
         *
         * @param units the number of outputs
         * @param activation the activation
         * @return this builder
         */
        public Builder dense(int units, Activation activation) {
            return dense(units, activation, activation == Activation.RELU ? Init.HE : Init.XAVIER);
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
            int in = currentWidth();
            Objects.requireNonNull(activation, "activation");
            Objects.requireNonNull(init, "init");
            pending.add(random -> new Dense(in, units, activation, init, random));
            width = units;
            return this;
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
            currentWidth();
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
            int in = currentWidth();
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
            return layer(Objects.requireNonNull(activation, "activation"));
        }

        /**
         * Adds any layer, such as an extra {@link Activation} or a custom one.
         *
         * @param layer the layer
         * @return this builder
         */
        public Builder layer(Layer layer) {
            Objects.requireNonNull(layer, "layer");
            int in = currentWidth();
            pending.add(random -> layer);
            width = layer.outputSize(in);
            return this;
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
            List<Layer> layers = new ArrayList<>();
            for (var factory : pending) {
                layers.add(factory.apply(random));
            }
            Model model = new Model(inputSize, width, layers);
            if (device != null) {
                model.to(device);
            }
            return model;
        }

        private int currentWidth() {
            if (inputSize == -1) {
                throw new IllegalStateException("call input(features) before adding layers");
            }
            return width;
        }
    }
}
