package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Reads and writes the Brainiac model file format. All numbers are big endian except
 * tensor data, which is little-endian float32. Version 2:
 *
 * <pre>
 * int    magic "BRNC"   int version
 * int    inputSize      int outputSize
 * int    activationCount  per activation: utf name, config, int width (-1 unless bound)
 * int    layerCount     per layer: utf type, config
 * int    stepCount      per step: byte kind, int inputCount, int[] inputs, int layerIndex (-1 if none)
 * int    tensorCount    per tensor: int rank, int[] dims, float[] values
 * config: int count, per entry: utf key, byte tag, value (see {@link #writeConfig})
 * </pre>
 *
 * Layer types and activations are looked up by name in the {@link Registry}. Activations
 * are stored once in their own table and referenced by index, so an activation shared by
 * several layers stays shared. Tensors are every parameter, then every buffer, in model
 * order. Version 1 files (fixed layer types, activations by name) can still be read.
 */
final class ModelFile {

    private static final int MAGIC = 0x42524E43;
    private static final int VERSION = 3;

    private static final byte BOOLEAN = 0;
    private static final byte INT = 1;
    private static final byte LONG = 2;
    private static final byte FLOAT = 3;
    private static final byte DOUBLE = 4;
    private static final byte STRING = 5;
    private static final byte INTS = 6;
    private static final byte FLOATS = 7;
    private static final byte CONFIG = 8;
    private static final byte ACTIVATION = 9;

    private static final String HINT = "; register it before loading, or build the model in code and use"
            + " loadWeights(path) instead";

    private ModelFile() {
    }

    // ---------------------------------------------------------------- writing

    static void write(Model model, OutputStream stream) throws IOException {
        DataOutputStream out = new DataOutputStream(stream);
        out.writeInt(MAGIC);
        out.writeInt(VERSION);
        writeShape(model.inputShape(), out);
        writeShape(model.outputShape(), out);

        List<Layer> layers = model.layers();
        Map<Layer, Integer> layerIndex = new IdentityHashMap<>();
        List<String> types = new ArrayList<>();
        List<Config> configs = new ArrayList<>();
        for (Layer layer : layers) {
            layerIndex.put(layer, layerIndex.size());
            describe(layer, types, configs);
        }
        Map<Activation, Integer> activationIndex = new IdentityHashMap<>();
        List<Activation> activations = new ArrayList<>();
        for (Config c : configs) {
            intern(c, activationIndex, activations);
        }
        out.writeInt(activations.size());
        for (Activation a : activations) {
            out.writeUTF(a.name());
            writeConfig(a.config(), activationIndex, out);
            out.writeInt(a.boundWidth());
        }
        out.writeInt(layers.size());
        for (int i = 0; i < layers.size(); i++) {
            out.writeUTF(types.get(i));
            writeConfig(configs.get(i), activationIndex, out);
        }

        List<Model.Step> steps = model.steps();
        out.writeInt(steps.size());
        for (Model.Step step : steps) {
            out.writeByte(step.kind().ordinal());
            out.writeInt(step.inputs().length);
            for (int i : step.inputs()) {
                out.writeInt(i);
            }
            out.writeInt(step.layer() == null ? -1 : layerIndex.get(step.layer()));
        }
        List<Var> tensors = state(model);
        out.writeInt(tensors.size());
        for (Var v : tensors) {
            writeTensor(v.value(), out);
        }
        out.flush();
    }

    private static void describe(Layer layer, List<String> types, List<Config> configs) {
        if (layer instanceof Activation a) {
            types.add("activation");
            configs.add(Config.builder().put("activation", a).build());
            return;
        }
        Registry.LayerType<?> type = Registry.layerType(layer.getClass());
        if (type == null) {
            // still saved, so loadWeights works; load explains how to register it
            types.add("custom");
            configs.add(Config.builder().put("class", layer.getClass().getName())
                    .put("description", String.valueOf(layer)).build());
            return;
        }
        types.add(type.name());
        configs.add(Objects.requireNonNull(type.write(layer), "codec for " + type.name() + " returned null"));
    }

    /** Adds every activation a config refers to, dependencies first. */
    private static void intern(Config config, Map<Activation, Integer> index, List<Activation> table) {
        for (Object v : config.values().values()) {
            if (v instanceof Config nested) {
                intern(nested, index, table);
            } else if (v instanceof Activation a && !index.containsKey(a)) {
                intern(a.config(), index, table);
                index.put(a, table.size());
                table.add(a);
            }
        }
    }

    /**
     * Writes a config. Tags: 0 boolean, 1 int, 2 long, 3 float, 4 double, 5 utf string,
     * 6 int array, 7 float array (each: int length, values), 8 nested config,
     * 9 activation (int index into the activation table).
     */
    private static void writeConfig(Config config, Map<Activation, Integer> activations, DataOutputStream out)
            throws IOException {
        Map<String, Object> values = config.values();
        out.writeInt(values.size());
        for (var e : values.entrySet()) {
            out.writeUTF(e.getKey());
            switch (e.getValue()) {
                case Boolean b -> {
                    out.writeByte(BOOLEAN);
                    out.writeBoolean(b);
                }
                case Integer i -> {
                    out.writeByte(INT);
                    out.writeInt(i);
                }
                case Long l -> {
                    out.writeByte(LONG);
                    out.writeLong(l);
                }
                case Float f -> {
                    out.writeByte(FLOAT);
                    out.writeFloat(f);
                }
                case Double d -> {
                    out.writeByte(DOUBLE);
                    out.writeDouble(d);
                }
                case String str -> {
                    out.writeByte(STRING);
                    out.writeUTF(str);
                }
                case int[] a -> {
                    out.writeByte(INTS);
                    out.writeInt(a.length);
                    for (int x : a) {
                        out.writeInt(x);
                    }
                }
                case float[] a -> {
                    out.writeByte(FLOATS);
                    out.writeInt(a.length);
                    for (float x : a) {
                        out.writeFloat(x);
                    }
                }
                case Config c -> {
                    out.writeByte(CONFIG);
                    writeConfig(c, activations, out);
                }
                case Activation a -> {
                    out.writeByte(ACTIVATION);
                    out.writeInt(activations.get(a));
                }
                default -> throw new IllegalStateException("unsupported config value " + e.getValue());
            }
        }
    }

    // ---------------------------------------------------------------- reading

    static Model read(InputStream stream, Device device) throws IOException {
        DataInputStream in = new DataInputStream(stream);
        int version = readHeader(in);
        Shape inputShape = readShape(in, version);
        Shape outputShape = readShape(in, version);
        List<Layer> layers = version == 1 ? readLayersV1(in) : readLayers(in);
        List<Model.Step> steps = readSteps(in, layers);
        Model model = Model.fromSteps(inputShape, outputShape, steps);
        readState(model, in);
        return model.to(device);
    }

    static void readWeights(Model model, InputStream stream) throws IOException {
        DataInputStream in = new DataInputStream(stream);
        int version = readHeader(in);
        readShape(in, version);
        readShape(in, version);
        if (version == 1) {
            int layerCount = in.readInt();
            for (int i = 0; i < layerCount; i++) {
                skipLayerV1(in);
            }
        } else {
            int activationCount = in.readInt();
            for (int i = 0; i < activationCount; i++) {
                in.readUTF();
                readConfig(in, null);
                in.readInt();
            }
            int layerCount = in.readInt();
            for (int i = 0; i < layerCount; i++) {
                in.readUTF();
                readConfig(in, null);
            }
        }
        int stepCount = in.readInt();
        for (int s = 0; s < stepCount; s++) {
            in.readUnsignedByte();
            in.skipNBytes(4L * in.readInt());
            in.readInt();
        }
        Device device = model.device();
        readState(model, in);
        model.to(device);
    }

    private static List<Layer> readLayers(DataInputStream in) throws IOException {
        int activationCount = in.readInt();
        List<Activation> activations = new ArrayList<>(activationCount);
        for (int i = 0; i < activationCount; i++) {
            String name = in.readUTF();
            Config config = readConfig(in, activations);
            int width = in.readInt();
            Function<Config, Activation> factory = Registry.activation(name);
            if (factory == null) {
                throw new IOException("the file uses activation `" + name + "`, which is not registered with"
                        + " Activation.register(...)" + HINT);
            }
            Activation a = factory.apply(config);
            activations.add(width >= 0 ? a.bind(width) : a);
        }
        int layerCount = in.readInt();
        List<Layer> layers = new ArrayList<>(layerCount);
        for (int i = 0; i < layerCount; i++) {
            String type = in.readUTF();
            Config config = readConfig(in, activations);
            layers.add(switch (type) {
                case "activation" -> config.getActivation("activation");
                case "custom" -> throw new IOException("the file contains the custom layer `"
                        + config.getString("description") + "` (" + config.getString("class")
                        + "), which is not registered with Layer.register(...)" + HINT);
                default -> {
                    Registry.LayerType<?> t = Registry.layerType(type);
                    if (t == null) {
                        throw new IOException("the file uses layer type `" + type + "`, which is not registered"
                                + " with Layer.register(...)" + HINT);
                    }
                    yield t.codec().read(config);
                }
            });
        }
        return layers;
    }

    /** Reads a config; with {@code activations == null} activation references are skipped. */
    private static Config readConfig(DataInputStream in, List<Activation> activations) throws IOException {
        int count = in.readInt();
        if (count < 0) {
            throw new IOException("bad config size " + count);
        }
        Config.Builder b = Config.builder();
        for (int i = 0; i < count; i++) {
            String key = in.readUTF();
            byte tag = in.readByte();
            switch (tag) {
                case BOOLEAN -> b.put(key, in.readBoolean());
                case INT -> b.put(key, in.readInt());
                case LONG -> b.put(key, in.readLong());
                case FLOAT -> b.put(key, in.readFloat());
                case DOUBLE -> b.put(key, in.readDouble());
                case STRING -> b.put(key, in.readUTF());
                case INTS -> {
                    int[] a = new int[arrayLength(in)];
                    for (int k = 0; k < a.length; k++) {
                        a[k] = in.readInt();
                    }
                    b.put(key, a);
                }
                case FLOATS -> {
                    float[] a = new float[arrayLength(in)];
                    for (int k = 0; k < a.length; k++) {
                        a[k] = in.readFloat();
                    }
                    b.put(key, a);
                }
                case CONFIG -> b.put(key, readConfig(in, activations));
                case ACTIVATION -> {
                    int index = in.readInt();
                    if (activations != null) {
                        if (index < 0 || index >= activations.size()) {
                            throw new IOException("config `" + key + "` refers to activation " + index
                                    + ", which does not exist yet");
                        }
                        b.put(key, activations.get(index));
                    }
                }
                default -> throw new IOException("unknown config value tag " + tag);
            }
        }
        return b.build();
    }

    private static int arrayLength(DataInputStream in) throws IOException {
        int n = in.readInt();
        if (n < 0 || n > (1 << 24)) {
            throw new IOException("bad config array length " + n);
        }
        return n;
    }

    private static List<Model.Step> readSteps(DataInputStream in, List<Layer> layers) throws IOException {
        int stepCount = in.readInt();
        List<Model.Step> steps = new ArrayList<>(stepCount);
        Node.Kind[] kinds = Node.Kind.values();
        for (int s = 0; s < stepCount; s++) {
            int kind = in.readUnsignedByte();
            if (kind >= kinds.length) {
                throw new IOException("unknown step kind " + kind);
            }
            int[] inputs = new int[in.readInt()];
            for (int k = 0; k < inputs.length; k++) {
                inputs[k] = in.readInt();
                if (inputs[k] < 0 || inputs[k] > s) {
                    throw new IOException("step " + s + " reads value " + inputs[k] + ", which does not exist yet");
                }
            }
            int layer = in.readInt();
            if (layer >= layers.size()) {
                throw new IOException("step " + s + " uses layer " + layer + ", which does not exist");
            }
            steps.add(new Model.Step(kinds[kind], inputs, layer < 0 ? null : layers.get(layer)));
        }
        return steps;
    }

    private static List<Var> state(Model model) {
        List<Var> all = new ArrayList<>(model.parameters());
        all.addAll(model.buffers());
        return all;
    }

    private static void readState(Model model, DataInputStream in) throws IOException {
        List<Var> targets = state(model);
        int count = in.readInt();
        if (count != targets.size()) {
            throw new IOException("file has " + count + " tensors, model has " + targets.size());
        }
        List<Tensor> values = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            Tensor t = readTensor(in);
            if (!t.shape().equals(targets.get(i).shape())) {
                throw new IOException("tensor " + i + " has shape " + t.shape() + ", model expects " + targets.get(i).shape());
            }
            values.add(t);
        }
        for (int i = 0; i < count; i++) {
            targets.get(i).assign(values.get(i));
        }
    }

    private static int readHeader(DataInputStream in) throws IOException {
        if (in.readInt() != MAGIC) {
            throw new IOException("not a Brainiac model file");
        }
        int version = in.readInt();
        if (version < 1 || version > VERSION) {
            throw new IOException("unsupported model file version " + version + ", expected 1 to " + VERSION);
        }
        return version;
    }

    private static void writeShape(Shape shape, DataOutputStream out) throws IOException {
        out.writeInt(shape.rank());
        for (int d : shape.toArray()) {
            out.writeInt(d);
        }
    }

    private static Shape readShape(DataInputStream in, int version) throws IOException {
        if (version < 3) {
            int size = in.readInt();
            if (size < 1) {
                throw new IOException("bad size " + size);
            }
            return Shape.of(size);
        }
        int rank = in.readInt();
        if (rank < 1 || rank > 8) {
            throw new IOException("bad shape rank " + rank);
        }
        int[] dims = new int[rank];
        for (int i = 0; i < rank; i++) {
            dims[i] = in.readInt();
            if (dims[i] < 1) {
                throw new IOException("bad dimension " + dims[i]);
            }
        }
        return Shape.of(dims);
    }

    // ---------------------------------------------------------------- version 1

    private static List<Layer> readLayersV1(DataInputStream in) throws IOException {
        int layerCount = in.readInt();
        List<Layer> layers = new ArrayList<>(layerCount);
        for (int i = 0; i < layerCount; i++) {
            layers.add(readLayerV1(in));
        }
        return layers;
    }

    private static Activation activationV1(String name) throws IOException {
        Function<Config, Activation> factory = name.startsWith("custom:") ? null : Registry.activation(name);
        if (factory == null) {
            throw new IOException("the file uses activation `" + name + "`, which cannot be restored; build the model"
                    + " in code and use loadWeights(path) instead");
        }
        return factory.apply(Config.EMPTY);
    }

    private static Layer readLayerV1(DataInputStream in) throws IOException {
        String type = in.readUTF();
        return switch (type) {
            case "dense" -> {
                int inputs = in.readInt();
                int units = in.readInt();
                Activation activation = activationV1(in.readUTF());
                yield new Dense(inputs, units, activation, Tensor.zeros(Shape.of(inputs, units)), Tensor.zeros(Shape.of(units)));
            }
            case "dropout" -> new Dropout(in.readFloat(), new SplittableRandom());
            case "batchnorm" -> new BatchNorm(in.readInt(), in.readFloat(), in.readFloat());
            case "activation" -> activationV1(in.readUTF());
            case "custom" -> throw new IOException("the file contains the custom layer `" + in.readUTF()
                    + "`, which cannot be restored; build the model in code and use loadWeights(path) instead");
            default -> throw new IOException("unknown layer type " + type);
        };
    }

    private static void skipLayerV1(DataInputStream in) throws IOException {
        String type = in.readUTF();
        switch (type) {
            case "dense" -> {
                in.readInt();
                in.readInt();
                in.readUTF();
            }
            case "dropout" -> in.readFloat();
            case "batchnorm" -> {
                in.readInt();
                in.readFloat();
                in.readFloat();
            }
            case "activation", "custom" -> in.readUTF();
            default -> throw new IOException("unknown layer type " + type);
        }
    }

    // ---------------------------------------------------------------- tensors

    private static void writeTensor(Tensor t, DataOutputStream out) throws IOException {
        int[] dims = t.shape().toArray();
        out.writeInt(dims.length);
        for (int d : dims) {
            out.writeInt(d);
        }
        float[] values = t.toArray();
        ByteBuffer bytes = ByteBuffer.allocate(values.length * Float.BYTES).order(ByteOrder.LITTLE_ENDIAN);
        bytes.asFloatBuffer().put(values);
        out.write(bytes.array());
    }

    private static Tensor readTensor(DataInputStream in) throws IOException {
        int rank = in.readInt();
        if (rank < 0 || rank > 8) {
            throw new IOException("bad tensor rank " + rank);
        }
        int[] dims = new int[rank];
        long size = 1;
        for (int i = 0; i < rank; i++) {
            dims[i] = in.readInt();
            if (dims[i] < 1) {
                throw new IOException("bad tensor dimension " + dims[i]);
            }
            size *= dims[i];
            if (size > Integer.MAX_VALUE / Float.BYTES) {
                throw new IOException("tensor too large");
            }
        }
        byte[] raw = in.readNBytes((int) size * Float.BYTES);
        if (raw.length != size * Float.BYTES) {
            throw new IOException("file ends inside a tensor");
        }
        float[] values = new float[(int) size];
        ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(values);
        return Tensor.of(Shape.of(dims), values);
    }
}
