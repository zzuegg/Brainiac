package io.github.zzuegg.brainiac.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Writes a {@link Model} as an ONNX graph (opset 17) in float32, with a dynamic batch
 * dimension, for onnxruntime, DirectML or mobile runtimes. Each built-in layer becomes
 * the matching ONNX operators; layers without an ONNX form (recurrent, attention,
 * learnable activations) are refused by name.
 */
final class OnnxExport {

    static final int OPSET = 17;
    private static final int FLOAT = 1;
    private static final int INT64 = 7;

    /** A protocol buffers message being written. */
    static final class Pb {
        final ByteArrayOutputStream out = new ByteArrayOutputStream();

        Pb varint(long v) {
            while ((v & ~0x7FL) != 0) {
                out.write((int) ((v & 0x7F) | 0x80));
                v >>>= 7;
            }
            out.write((int) v);
            return this;
        }

        Pb tag(int field, int wire) {
            return varint(((long) field << 3) | wire);
        }

        Pb int64(int field, long v) {
            return tag(field, 0).varint(v);
        }

        Pb float32(int field, float v) {
            tag(field, 5);
            int bits = Float.floatToRawIntBits(v);
            for (int k = 0; k < 4; k++) {
                out.write(bits >>> (8 * k));
            }
            return this;
        }

        Pb bytes(int field, byte[] b) {
            tag(field, 2).varint(b.length);
            out.writeBytes(b);
            return this;
        }

        Pb string(int field, String s) {
            return bytes(field, s.getBytes(StandardCharsets.UTF_8));
        }

        Pb message(int field, Pb m) {
            return bytes(field, m.out.toByteArray());
        }

        Pb packed(int field, long[] values) {
            Pb p = new Pb();
            for (long v : values) {
                p.varint(v);
            }
            return message(field, p);
        }
    }

    private final List<Pb> nodes = new ArrayList<>();
    private final List<Pb> initializers = new ArrayList<>();
    private int counter;

    private OnnxExport() {
    }

    static void write(Model model, Path file) throws IOException {
        try (OutputStream out = Files.newOutputStream(file)) {
            out.write(bytes(model));
        }
    }

    static byte[] bytes(Model model) {
        OnnxExport e = new OnnxExport();
        List<Model.GraphStep> steps = model.graph();
        String[] names = new String[steps.size() + 1];
        Shape[] shapes = new Shape[steps.size() + 1];
        names[0] = "input";
        shapes[0] = model.inputShape();
        for (Model.GraphStep s : steps) {
            int i = s.index() + 1;
            String a = names[s.inputs().get(0)];
            Shape sa = shapes[s.inputs().get(0)];
            switch (s.kind()) {
                case LAYER -> {
                    names[i] = e.layer(s.layer(), a, sa);
                    shapes[i] = s.layer().outputShape(sa);
                }
                case ADD, MULTIPLY -> {
                    names[i] = e.node(s.kind() == Model.StepKind.ADD ? "Add" : "Mul", a, names[s.inputs().get(1)]);
                    shapes[i] = Shape.broadcast(sa, shapes[s.inputs().get(1)]);
                }
                case CONCAT -> {
                    String[] in = s.inputs().stream().map(k -> names[k]).toArray(String[]::new);
                    names[i] = e.node("Concat", in, attrInt("axis", 1));
                    int[] d = sa.toArray();
                    d[0] = s.inputs().stream().mapToInt(k -> shapes[k].dim(0)).sum();
                    shapes[i] = Shape.of(d);
                }
                default -> throw new UnsupportedOperationException("no ONNX form for a " + s.kind() + " step");
            }
        }
        e.nodes.add(new Pb().string(1, names[steps.size()]).string(2, "output").string(3, "output")
                .string(4, "Identity"));
        Pb graph = new Pb();
        e.nodes.forEach(n -> graph.message(1, n));
        graph.string(2, "brainiac");
        e.initializers.forEach(t -> graph.message(5, t));
        graph.message(11, valueInfo("input", shapes[0]));
        graph.message(12, valueInfo("output", shapes[steps.size()]));
        Pb model1 = new Pb();
        model1.int64(1, 8);
        model1.string(2, "brainiac");
        model1.message(7, graph);
        model1.message(8, new Pb().string(1, "").int64(2, OPSET));
        return model1.out.toByteArray();
    }

    private static Pb valueInfo(String name, Shape sample) {
        Pb shape = new Pb();
        shape.message(1, new Pb().string(2, "batch"));
        for (int d : sample.toArray()) {
            shape.message(1, new Pb().int64(1, d));
        }
        Pb tensor = new Pb().int64(1, FLOAT).message(2, shape);
        return new Pb().string(1, name).message(2, new Pb().message(1, tensor));
    }

    // ---------------------------------------------------------------- graph building

    private static Pb attrInt(String name, long v) {
        return new Pb().string(1, name).int64(3, v).int64(20, 2);
    }

    private static Pb attrFloat(String name, float v) {
        return new Pb().string(1, name).float32(2, v).int64(20, 1);
    }

    private static Pb attrString(String name, String v) {
        return new Pb().string(1, name).string(4, v).int64(20, 3);
    }

    private static Pb attrInts(String name, int[] v) {
        return new Pb().string(1, name).packed(8, Arrays.stream(v).asLongStream().toArray()).int64(20, 7);
    }

    private String fresh(String op) {
        return op.toLowerCase() + "_" + counter++;
    }

    private String node(String op, String input, Pb... attrs) {
        return node(op, new String[] {input}, attrs);
    }

    private String node(String op, String a, String b, Pb... attrs) {
        return node(op, new String[] {a, b}, attrs);
    }

    private String node(String op, String[] inputs, Pb... attrs) {
        String out = fresh(op);
        Pb n = new Pb();
        for (String in : inputs) {
            n.string(1, in);
        }
        n.string(2, out).string(3, out).string(4, op);
        for (Pb a : attrs) {
            n.message(5, a);
        }
        nodes.add(n);
        return out;
    }

    private String floats(String hint, int[] dims, float[] values) {
        String name = hint + "_" + counter++;
        Pb t = new Pb().packed(1, Arrays.stream(dims).asLongStream().toArray()).int64(2, FLOAT).string(8, name);
        ByteBuffer b = ByteBuffer.allocate(values.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        b.asFloatBuffer().put(values);
        t.bytes(9, b.array());
        initializers.add(t);
        return name;
    }

    private String weight(String hint, Var v) {
        Tensor t = v.value().to(DType.FLOAT32);
        return floats(hint, t.shape().toArray(), t.toArray());
    }

    private String weight(String hint, Tensor t, int[] dims) {
        return floats(hint, dims, t.to(DType.FLOAT32).toArray());
    }

    private String scalar(float v) {
        return floats("c", new int[0], new float[] {v});
    }

    private String longs(long... values) {
        String name = "i_" + counter++;
        Pb t = new Pb().packed(1, new long[] {values.length}).int64(2, INT64).string(8, name);
        ByteBuffer b = ByteBuffer.allocate(values.length * 8).order(ByteOrder.LITTLE_ENDIAN);
        b.asLongBuffer().put(values);
        t.bytes(9, b.array());
        initializers.add(t);
        return name;
    }

    private static long[] full(int[] sample, boolean keepBatch) {
        long[] r = new long[sample.length + 1];
        r[0] = keepBatch ? 0 : 1;
        for (int k = 0; k < sample.length; k++) {
            r[k + 1] = sample[k];
        }
        return r;
    }

    // ---------------------------------------------------------------- layers

    private String layer(Layer layer, String x, Shape in) {
        if (layer instanceof Activation a) {
            return activation(a, x, -1);
        }
        Registry.LayerType<?> type = Registry.layerType(layer.getClass());
        if (type == null) {
            throw new UnsupportedOperationException("no ONNX form for " + layer.getClass().getSimpleName());
        }
        Config c = type.write(layer);
        List<Var> p = layer.parameters();
        return switch (type.name()) {
            case "dense" -> {
                Dense d = (Dense) layer;
                String y = node("Add", node("MatMul", x, weight("weight", d.weight())), weight("bias", d.bias()));
                yield activation(d.activation(), y, -1);
            }
            case "dropout" -> x;
            case "batchnorm" -> {
                BatchNorm b = (BatchNorm) layer;
                int f = b.features();
                yield node("BatchNormalization", new String[] {x, weight("gamma", p.get(0).value(), new int[] {f}),
                        weight("beta", p.get(1).value(), new int[] {f}), weight("mean", b.runningMean(), new int[] {f}),
                        weight("var", b.runningVariance(), new int[] {f})}, attrFloat("epsilon", b.epsilon()));
            }
            case "conv" -> conv(c, p, x);
            case "pool" -> pool(c, x);
            case "global_pool" -> {
                String kind = c.getString("kind");
                String y = switch (kind) {
                    case "AVG" -> node("GlobalAveragePool", x);
                    case "MAX" -> node("GlobalMaxPool", x);
                    case "MIN" -> node("Neg", node("GlobalMaxPool", node("Neg", x)));
                    default -> node("Mul", node("GlobalAveragePool", x),
                            scalar(in.size() / in.dim(0)));
                };
                yield c.getBoolean("keepDims") ? y : node("Flatten", y, attrInt("axis", 1));
            }
            case "adaptive_pool" -> {
                int[] size = c.getInts("size");
                int[] kernel = new int[size.length];
                for (int k = 0; k < size.length; k++) {
                    int dim = in.dim(k + 1);
                    if (dim % size[k] != 0) {
                        throw new UnsupportedOperationException("ONNX export of adaptive pooling from " + dim
                                + " to " + size[k] + " cells");
                    }
                    kernel[k] = dim / size[k];
                }
                String kind = c.getString("kind");
                yield switch (kind) {
                    case "AVG" -> node("AveragePool", x, attrInts("kernel_shape", kernel), attrInts("strides", kernel));
                    case "MAX" -> node("MaxPool", x, attrInts("kernel_shape", kernel), attrInts("strides", kernel));
                    default -> throw new UnsupportedOperationException("ONNX export of adaptive " + kind + " pooling");
                };
            }
            case "pad" -> {
                int[] before = c.getInts("before");
                int[] after = c.getInts("after");
                yield pad(x, before, after, PadMode.valueOf(c.getString("mode")), c.getFloat("value"));
            }
            case "reshape" -> {
                int[] dims = c.getInts("dims");
                long[] target = new long[dims.length + 1];
                for (int k = 0; k < dims.length; k++) {
                    target[k + 1] = dims[k];
                }
                yield node("Reshape", x, longs(target));
            }
            case "permute" -> {
                int[] axes = c.getInts("axes");
                int[] perm = new int[axes.length + 1];
                for (int k = 0; k < axes.length; k++) {
                    perm[k + 1] = axes[k] + 1;
                }
                yield node("Transpose", x, attrInts("perm", perm));
            }
            case "resize" -> resize(c, layer.outputShape(in), x);
            case "pixel_shuffle" -> pixelShuffle(c.getInt("factor"), c.getBoolean("inverse"), in, x);
            case "layer_norm" -> layerNorm(c, p, x);
            case "group_norm" -> groupNorm(c, p, in, x);
            case "embedding" -> embedding((Embedding) layer, c, x);
            default -> throw new UnsupportedOperationException("no ONNX form for the " + type.name() + " layer");
        };
    }

    private String activation(Activation a, String x, int axisHint) {
        Config c = a.config();
        return switch (a.name()) {
            case "identity" -> x;
            case "relu" -> node("Relu", x);
            case "sigmoid" -> node("Sigmoid", x);
            case "tanh" -> node("Tanh", x);
            case "silu" -> node("Mul", x, node("Sigmoid", x));
            case "gelu" -> node("Mul", node("Mul", x, scalar(0.5f)),
                    node("Add", node("Erf", node("Mul", x, scalar((float) (1 / Math.sqrt(2))))), scalar(1f)));
            case "gelu_tanh" -> {
                String cube = node("Mul", node("Mul", x, x), x);
                String inner = node("Mul", node("Add", x, node("Mul", cube, scalar(0.044715f))),
                        scalar((float) Math.sqrt(2 / Math.PI)));
                yield node("Mul", node("Mul", x, scalar(0.5f)), node("Add", node("Tanh", inner), scalar(1f)));
            }
            case "quick_gelu" -> node("Mul", x, node("Sigmoid", node("Mul", x, scalar(1.702f))));
            case "softplus" -> node("Softplus", x);
            case "mish" -> node("Mul", x, node("Tanh", node("Softplus", x)));
            case "selu" -> node("Selu", x);
            case "relu6" -> node("Clip", new String[] {x, scalar(0f), scalar(6f)});
            case "hard_tanh" -> node("Clip", new String[] {x, scalar(-1f), scalar(1f)});
            case "hard_sigmoid" -> node("HardSigmoid", x, attrFloat("alpha", 1f / 6), attrFloat("beta", 0.5f));
            case "hard_swish" -> node("HardSwish", x);
            case "softsign" -> node("Softsign", x);
            case "softmax" -> node("Softmax", x, attrInt("axis", c.has("axis") ? c.getInt("axis") : -1));
            case "log_softmax" -> node("LogSoftmax", x, attrInt("axis", c.has("axis") ? c.getInt("axis") : -1));
            case "leaky_relu" -> node("LeakyRelu", x, attrFloat("alpha", c.getFloat("slope")));
            case "elu" -> node("Elu", x, attrFloat("alpha", c.getFloat("alpha")));
            default -> throw new UnsupportedOperationException("no ONNX form for the " + a.name() + " activation");
        };
    }

    private String conv(Config c, List<Var> p, String x) {
        boolean transposed = c.getBoolean("transposed");
        int[] stride = c.getInts("stride");
        int[] dilation = c.getInts("dilation");
        int[] before = c.getInts("padBefore");
        int[] after = c.getInts("padAfter");
        int r = stride.length;
        PadMode mode = PadMode.valueOf(c.getString("padMode"));
        float padValue = c.getFloat("padValue");
        if (!transposed && (mode != PadMode.CONSTANT || padValue != 0)) {
            x = pad(x, before, after, mode, padValue);
            before = new int[r];
            after = new int[r];
        }
        int[] pads = new int[2 * r];
        System.arraycopy(before, 0, pads, 0, r);
        System.arraycopy(after, 0, pads, r, r);
        List<String> in = new ArrayList<>(List.of(x, weight("weight", p.get(0))));
        if (c.getBoolean("bias")) {
            in.add(weight("bias", p.get(1)));
        }
        List<Pb> attrs = new ArrayList<>(List.of(attrInts("strides", stride), attrInts("dilations", dilation),
                attrInts("pads", pads), attrInt("group", c.getInt("groups"))));
        if (transposed) {
            attrs.add(attrInts("output_padding", c.getInts("outputPadding")));
        }
        String y = node(transposed ? "ConvTranspose" : "Conv", in.toArray(String[]::new), attrs.toArray(Pb[]::new));
        Activation a = c.getActivation("activation");
        if (!a.parameters().isEmpty()) {
            throw new UnsupportedOperationException("no ONNX form for a learnable activation in a convolution");
        }
        return activation(a, y, 1);
    }

    private String pool(Config c, String x) {
        String kind = c.getString("kind");
        int[] kernel = c.getInts("kernel");
        int r = kernel.length;
        int[] before = c.getInts("padBefore");
        int[] after = c.getInts("padAfter");
        int[] pads = new int[2 * r];
        System.arraycopy(before, 0, pads, 0, r);
        System.arraycopy(after, 0, pads, r, r);
        int[] dilation = c.getInts("dilation");
        boolean dilated = Arrays.stream(dilation).anyMatch(d -> d != 1);
        List<Pb> attrs = new ArrayList<>(List.of(attrInts("kernel_shape", kernel), attrInts("strides",
                c.getInts("stride")), attrInts("pads", pads), attrInt("ceil_mode", c.getBoolean("ceil") ? 1 : 0)));
        if (kind.equals("MAX") || kind.equals("MIN")) {
            attrs.add(attrInts("dilations", dilation));
        } else if (dilated) {
            throw new UnsupportedOperationException("ONNX export of dilated " + kind + " pooling");
        }
        Pb[] a = attrs.toArray(Pb[]::new);
        return switch (kind) {
            case "MAX" -> node("MaxPool", x, a);
            case "MIN" -> node("Neg", node("MaxPool", node("Neg", x), a));
            case "AVG" -> node("AveragePool", x, append(a, attrInt("count_include_pad",
                    c.getBoolean("countPadding") ? 1 : 0)));
            case "SUM" -> node("Mul", node("AveragePool", x, append(a, attrInt("count_include_pad", 1))),
                    scalar(Arrays.stream(kernel).reduce(1, (u, v) -> u * v)));
            case "LP" -> node("LpPool", x, append(a, attrInt("p", Math.round(c.getFloat("power")))));
            default -> throw new UnsupportedOperationException("ONNX export of " + kind + " pooling");
        };
    }

    private static Pb[] append(Pb[] a, Pb b) {
        Pb[] r = Arrays.copyOf(a, a.length + 1);
        r[a.length] = b;
        return r;
    }

    private String pad(String x, int[] before, int[] after, PadMode mode, float value) {
        int r = before.length;
        long[] pads = new long[2 * (r + 2)];
        for (int k = 0; k < r; k++) {
            pads[k + 2] = before[k];
            pads[r + 2 + k + 2] = after[k];
        }
        String m = switch (mode) {
            case CONSTANT -> "constant";
            case REFLECT -> "reflect";
            case REPLICATE -> "edge";
            case CIRCULAR -> "wrap";
            default -> throw new UnsupportedOperationException("ONNX export of " + mode + " padding");
        };
        return node("Pad", new String[] {x, longs(pads), scalar(value)}, attrString("mode", m));
    }

    private String resize(Config c, Shape out, String x) {
        String interpolation = c.getString("interpolation");
        boolean align = c.getBoolean("alignCorners");
        int[] spatial = Arrays.copyOfRange(out.toArray(), 1, out.rank());
        long[] tail = Arrays.stream(spatial).asLongStream().toArray();
        String head = node("Shape", x, attrInt("start", 0), attrInt("end", 2));
        String sizes = node("Concat", new String[] {head, longs(tail)}, attrInt("axis", 0));
        List<Pb> attrs = new ArrayList<>();
        switch (interpolation) {
            case "NEAREST" -> {
                attrs.add(attrString("mode", "nearest"));
                attrs.add(attrString("coordinate_transformation_mode", "asymmetric"));
                attrs.add(attrString("nearest_mode", "floor"));
            }
            case "LINEAR", "CUBIC" -> {
                attrs.add(attrString("mode", interpolation.equals("LINEAR") ? "linear" : "cubic"));
                attrs.add(attrString("coordinate_transformation_mode", align ? "align_corners" : "half_pixel"));
                attrs.add(attrFloat("cubic_coeff_a", -0.75f));
            }
            default -> throw new UnsupportedOperationException("ONNX export of " + interpolation + " resizing");
        }
        return node("Resize", new String[] {x, "", "", sizes}, attrs.toArray(Pb[]::new));
    }

    private String pixelShuffle(int factor, boolean inverse, Shape in, String x) {
        if (in.rank() != 3) {
            throw new UnsupportedOperationException("ONNX export of pixel shuffling outside 2D images");
        }
        int c = in.dim(0);
        int h = in.dim(1);
        int w = in.dim(2);
        if (!inverse) {
            return node("DepthToSpace", x, attrInt("blocksize", factor), attrString("mode", "CRD"));
        }
        String split = node("Reshape", x, longs(0, c, h / factor, factor, w / factor, factor));
        String moved = node("Transpose", split, attrInts("perm", new int[] {0, 1, 3, 5, 2, 4}));
        return node("Reshape", moved, longs(0, (long) c * factor * factor, h / factor, w / factor));
    }

    private String layerNorm(Config c, List<Var> p, String x) {
        int[] normalized = c.getInts("normalized");
        int k = normalized.length;
        float eps = c.getFloat("epsilon");
        boolean scale = c.getBoolean("scale");
        boolean shift = c.getBoolean("shift");
        int at = 0;
        String gamma = scale ? weight("gamma", p.get(at++)) : null;
        String beta = shift ? weight("beta", p.get(at)) : null;
        if (c.getBoolean("rms")) {
            int[] axes = new int[k];
            for (int i = 0; i < k; i++) {
                axes[i] = -k + i;
            }
            String ms = node("ReduceMean", node("Mul", x, x), attrInts("axes", axes), attrInt("keepdims", 1));
            String y = node("Div", x, node("Sqrt", node("Add", ms, scalar(eps))));
            if (gamma != null) {
                y = node("Mul", y, gamma);
            }
            return beta != null ? node("Add", y, beta) : y;
        }
        if (gamma == null) {
            float[] ones = new float[Math.toIntExact(Shape.of(normalized).size())];
            Arrays.fill(ones, 1f);
            gamma = floats("gamma", normalized, ones);
        }
        String[] in = beta != null ? new String[] {x, gamma, beta} : new String[] {x, gamma};
        return node("LayerNormalization", in, attrInt("axis", -k), attrFloat("epsilon", eps));
    }

    private String groupNorm(Config c, List<Var> p, Shape in, String x) {
        int groups = c.getInt("groups");
        int channels = c.getInt("channels");
        float[] ones = new float[groups];
        Arrays.fill(ones, 1f);
        String grouped = node("Reshape", x, longs(0, groups, -1));
        String normalized = node("InstanceNormalization", new String[] {grouped, floats("ones", new int[] {groups},
                ones), floats("zeros", new int[] {groups}, new float[groups])}, attrFloat("epsilon",
                c.getFloat("epsilon")));
        String y = node("Reshape", normalized, longs(full(in.toArray(), true)));
        if (!c.getBoolean("affine")) {
            return y;
        }
        int[] cs = new int[in.rank()];
        Arrays.fill(cs, 1);
        cs[0] = channels;
        return node("Add", node("Mul", y, weight("gamma", p.get(0).value(), cs)), weight("beta", p.get(1).value(),
                cs));
    }

    private String embedding(Embedding e, Config c, String x) {
        if (c.getFloat("maxNorm") > 0) {
            throw new UnsupportedOperationException("ONNX export of an embedding with a maximum norm");
        }
        String ids = node("Cast", x, attrInt("to", INT64));
        String y = node("Gather", weight("table", e.weights(), e.weights().shape().toArray()), ids,
                attrInt("axis", 0));
        int pad = c.getInt("paddingIndex");
        if (pad >= 0) {
            String keep = node("Cast", node("Not", node("Equal", ids, longs(pad))), attrInt("to", FLOAT));
            y = node("Mul", y, node("Unsqueeze", keep, longs(-1)));
        }
        if (c.getBoolean("scale")) {
            y = node("Mul", y, scalar((float) Math.sqrt(c.getInt("dimension"))));
        }
        return y;
    }
}
