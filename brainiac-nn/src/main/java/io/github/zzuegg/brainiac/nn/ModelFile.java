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
import java.util.SplittableRandom;

/**
 * Reads and writes the Brainiac model file format (version 1). All numbers are big
 * endian except tensor data, which is little-endian float32.
 *
 * <pre>
 * int    magic "BRNC"   int version
 * int    inputSize      int outputSize
 * int    layerCount     layer specs (type name, then type-specific fields)
 * int    stepCount      per step: byte kind, int inputCount, int[] inputs, int layerIndex (-1 if none)
 * int    tensorCount    per tensor: int rank, int[] dims, float[] values
 * </pre>
 *
 * Tensors are every parameter, then every buffer, in model order.
 */
final class ModelFile {

    private static final int MAGIC = 0x42524E43;
    private static final int VERSION = 1;

    private static final List<Activation> ACTIVATIONS =
            List.of(Activation.IDENTITY, Activation.RELU, Activation.SIGMOID, Activation.TANH, Activation.SOFTMAX);

    private ModelFile() {
    }

    static void write(Model model, OutputStream stream) throws IOException {
        DataOutputStream out = new DataOutputStream(stream);
        out.writeInt(MAGIC);
        out.writeInt(VERSION);
        out.writeInt(model.inputSize());
        out.writeInt(model.outputSize());
        List<Layer> layers = model.layers();
        Map<Layer, Integer> layerIndex = new IdentityHashMap<>();
        out.writeInt(layers.size());
        for (Layer layer : layers) {
            layerIndex.put(layer, layerIndex.size());
            writeLayer(layer, out);
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

    static Model read(InputStream stream, Device device) throws IOException {
        DataInputStream in = new DataInputStream(stream);
        readHeader(in);
        int inputSize = in.readInt();
        int outputSize = in.readInt();
        int layerCount = in.readInt();
        List<Layer> layers = new ArrayList<>(layerCount);
        for (int i = 0; i < layerCount; i++) {
            layers.add(readLayer(in));
        }
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
            steps.add(new Model.Step(kinds[kind], inputs, layer < 0 ? null : layers.get(layer)));
        }
        Model model = Model.fromSteps(inputSize, outputSize, steps);
        readState(model, in);
        return model.to(device);
    }

    static void readWeights(Model model, InputStream stream) throws IOException {
        DataInputStream in = new DataInputStream(stream);
        readHeader(in);
        in.readInt();
        in.readInt();
        int layerCount = in.readInt();
        for (int i = 0; i < layerCount; i++) {
            skipLayer(in);
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

    private static void readHeader(DataInputStream in) throws IOException {
        if (in.readInt() != MAGIC) {
            throw new IOException("not a Brainiac model file");
        }
        int version = in.readInt();
        if (version != VERSION) {
            throw new IOException("unsupported model file version " + version + ", expected " + VERSION);
        }
    }

    private static void writeLayer(Layer layer, DataOutputStream out) throws IOException {
        switch (layer) {
            case Dense d -> {
                out.writeUTF("dense");
                out.writeInt(d.weight().shape().dim(0));
                out.writeInt(d.weight().shape().dim(1));
                out.writeUTF(activationName(d.activation()));
            }
            case Dropout d -> {
                out.writeUTF("dropout");
                out.writeFloat(d.rate());
            }
            case BatchNorm b -> {
                out.writeUTF("batchnorm");
                out.writeInt(b.features());
                out.writeFloat(b.momentum());
                out.writeFloat(b.epsilon());
            }
            case Activation a when ACTIVATIONS.contains(a) -> {
                out.writeUTF("activation");
                out.writeUTF(a.toString());
            }
            default -> {
                out.writeUTF("custom");
                out.writeUTF(layer.getClass().getName() + ": " + layer);
            }
        }
    }

    private static String activationName(Activation activation) {
        if (!ACTIVATIONS.contains(activation)) {
            return "custom:" + activation;
        }
        return activation.toString();
    }

    private static Activation activation(String name) throws IOException {
        for (Activation a : ACTIVATIONS) {
            if (a.toString().equals(name)) {
                return a;
            }
        }
        throw new IOException("the file uses activation `" + name + "`, which cannot be restored; build the model"
                + " in code and use loadWeights(path) instead");
    }

    private static Layer readLayer(DataInputStream in) throws IOException {
        String type = in.readUTF();
        return switch (type) {
            case "dense" -> {
                int inputs = in.readInt();
                int units = in.readInt();
                Activation activation = activation(in.readUTF());
                yield new Dense(inputs, units, activation, Tensor.zeros(Shape.of(inputs, units)), Tensor.zeros(Shape.of(units)));
            }
            case "dropout" -> new Dropout(in.readFloat(), new SplittableRandom());
            case "batchnorm" -> new BatchNorm(in.readInt(), in.readFloat(), in.readFloat());
            case "activation" -> activation(in.readUTF());
            case "custom" -> throw new IOException("the file contains the custom layer `" + in.readUTF()
                    + "`, which cannot be restored; build the model in code and use loadWeights(path) instead");
            default -> throw new IOException("unknown layer type " + type);
        };
    }

    private static void skipLayer(DataInputStream in) throws IOException {
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
