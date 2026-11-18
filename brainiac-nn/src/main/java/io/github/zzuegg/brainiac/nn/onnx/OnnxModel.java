package io.github.zzuegg.brainiac.nn.onnx;

import io.github.zzuegg.brainiac.autograd.Einsum;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.PadMode;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.tensor.spi.ActivationOp;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.function.LongBinaryOperator;
import java.util.function.UnaryOperator;

/**
 * Runs an ONNX model: the graph's nodes are evaluated in order with Brainiac operations on
 * any device, so a model exported from PyTorch or another framework runs on the CPU or
 * the GPU, and its initializers can be fine-tuned through {@link #forward}.
 *
 * <p>Floating point values live on the device in the chosen precision. Integer and
 * boolean values that only describe shapes and indices stay on the host as whole numbers,
 * the way exporters use them. Control flow ({@code If}, {@code Loop}) and operators outside
 * the default domain are not supported; {@link #load} names any it meets.
 */
public final class OnnxModel {

    private static final Set<String> SUPPORTED = Set.of("Abs", "Acos", "Add", "And", "ArgMax", "ArgMin", "Asin",
            "Atan", "AveragePool", "BatchNormalization", "Cast", "CastLike", "Ceil", "Celu", "Clip", "Concat",
            "Constant", "ConstantOfShape", "Conv", "ConvTranspose", "Cos", "CumSum", "Div", "Dropout", "Einsum",
            "Elu", "Equal", "Erf", "Exp", "Expand", "Flatten", "Floor", "Gather", "GatherElements", "Gelu", "Gemm",
            "GlobalAveragePool", "GlobalMaxPool", "Greater", "GreaterOrEqual", "GroupNormalization", "HardSigmoid",
            "HardSwish", "Identity", "InstanceNormalization", "IsInf", "IsNaN", "LayerNormalization", "LeakyRelu",
            "Less", "LessOrEqual", "Log", "LogSoftmax", "LpPool", "MatMul", "Max", "MaxPool", "Mean", "Min", "Mish",
            "Mod", "Mul", "Neg", "Not", "Or", "Pad", "Pow", "PRelu", "Range", "Reciprocal", "ReduceL1", "ReduceL2",
            "ReduceLogSum", "ReduceLogSumExp", "ReduceMax", "ReduceMean", "ReduceMin", "ReduceProd", "ReduceSum",
            "ReduceSumSquare", "Relu", "Reshape", "Resize", "Round", "Selu", "Shape", "Sigmoid", "Sign", "Sin",
            "Size", "Slice", "Softmax", "Softplus", "Softsign", "Split", "Sqrt", "Squeeze", "Sub", "Sum", "Tan",
            "Tanh", "Tile", "Transpose", "Trilu", "Unsqueeze", "Upsample", "Where", "Xor");

    /** A value flowing through the graph: a device variable, or whole numbers on the host. */
    private record V(int type, int[] shape, Var var, long[] data) {
        static V of(Var v) {
            return new V(Graph.FLOAT, v.shape().toArray(), v, null);
        }

        static V of(int type, Var v) {
            return new V(type, v.shape().toArray(), v, null);
        }

        static V host(int type, int[] shape, long[] data) {
            return new V(type, shape, null, data);
        }

        boolean floating() {
            return var != null && Graph.isFloat(type);
        }

        int rank() {
            return shape.length;
        }
    }

    private final Graph graph;
    private final Device device;
    private final DType dtype;
    private final Device host;
    private final Map<String, V> constants = new LinkedHashMap<>();
    private final Map<String, Var> parameters = new LinkedHashMap<>();

    private OnnxModel(Graph graph, Device device, DType dtype) {
        this.graph = graph;
        this.device = device;
        this.dtype = dtype;
        Device h;
        try {
            h = Device.cpu();
        } catch (IllegalStateException e) {
            h = device;
        }
        this.host = h;
        graph.initializers.forEach((name, raw) -> {
            if (raw.floating()) {
                Var p = Var.param(tensor(raw));
                parameters.put(name, p);
                constants.put(name, V.of(p));
            } else {
                constants.put(name, value(raw));
            }
        });
    }

    /**
     * Reads an ONNX model; external data files are found next to it.
     *
     * @param file the {@code .onnx} file
     * @param device where to run the model
     * @param dtype the precision of floating point values, such as {@link DType#FLOAT32}
     * @return the model, with its weights on the device
     * @throws UnsupportedOperationException if the graph uses operators this runner lacks
     */
    public static OnnxModel load(Path file, Device device, DType dtype) {
        Graph g = Graph.read(file);
        Set<String> missing = new TreeSet<>();
        for (Graph.Node n : g.nodes) {
            boolean defaultDomain = n.domain().isEmpty() || n.domain().equals("ai.onnx");
            if (!defaultDomain || !SUPPORTED.contains(n.op())) {
                missing.add(defaultDomain ? n.op() : n.domain() + "." + n.op());
            }
        }
        if (!missing.isEmpty()) {
            throw new UnsupportedOperationException("unsupported ONNX operators in " + file.getFileName() + ": "
                    + String.join(", ", missing));
        }
        return new OnnxModel(g, device, dtype);
    }

    /** @return the operators this runner supports */
    public static Set<String> supportedOperators() {
        return new TreeSet<>(SUPPORTED);
    }

    /** @return the graph's inputs, without its initializers, in order */
    public List<String> inputNames() {
        return List.copyOf(graph.inputs.keySet());
    }

    /** @return the graph's outputs in order */
    public List<String> outputNames() {
        return List.copyOf(graph.outputs.keySet());
    }

    /** @return the operators the graph uses */
    public Set<String> operators() {
        Set<String> ops = new TreeSet<>();
        graph.nodes.forEach(n -> ops.add(n.op()));
        return ops;
    }

    /** @return the opset version of the default domain */
    public int opset() {
        return graph.opset;
    }

    /**
     * @return the floating point initializers by name, as trainable variables; integer
     *     initializers (shapes, indices) are left out
     */
    public Map<String, Var> parameters() {
        return Collections.unmodifiableMap(parameters);
    }

    /**
     * Runs the model without recording gradients.
     *
     * @param inputs a tensor for each input; integer inputs (token ids, shapes) hold whole numbers
     * @return each output by name; integer outputs come back as whole-number floats
     */
    public Map<String, Tensor> run(Map<String, Tensor> inputs) {
        return Var.noGrad(() -> {
            Map<String, Var> in = new LinkedHashMap<>();
            inputs.forEach((k, t) -> in.put(k, Var.of(t)));
            Map<String, Tensor> out = new LinkedHashMap<>();
            forward(in).forEach((k, v) -> out.put(k, v.value()));
            return out;
        });
    }

    /**
     * Runs the model with gradients flowing back to the inputs and {@link #parameters()}.
     *
     * @param inputs a variable for each input
     * @return each output by name
     */
    public Map<String, Var> forward(Map<String, Var> inputs) {
        Map<String, V> env = new HashMap<>(constants);
        graph.inputs.forEach((name, type) -> {
            Var v = inputs.get(name);
            if (v == null) {
                throw new IllegalArgumentException("missing input '" + name + "'; the model reads " + inputNames());
            }
            if (Graph.isFloat(type)) {
                env.put(name, V.of(cast(v.to(device))));
            } else {
                env.put(name, V.host(type, v.shape().toArray(), longs(v.value())));
            }
        });
        for (Graph.Node n : graph.nodes) {
            V[] in = new V[n.in().length];
            for (int k = 0; k < in.length; k++) {
                if (!n.in()[k].isEmpty()) {
                    in[k] = env.get(n.in()[k]);
                    if (in[k] == null) {
                        throw new IllegalStateException(n + " reads '" + n.in()[k] + "', which nothing produced");
                    }
                }
            }
            V[] out;
            try {
                out = node(n, in);
            } catch (RuntimeException e) {
                throw new IllegalStateException("ONNX " + n + " failed: " + e.getMessage(), e);
            }
            for (int k = 0; k < out.length && k < n.out().length; k++) {
                if (!n.out()[k].isEmpty()) {
                    env.put(n.out()[k], out[k]);
                }
            }
        }
        Map<String, Var> result = new LinkedHashMap<>();
        for (String name : graph.outputs.keySet()) {
            V v = env.get(name);
            if (v == null) {
                throw new IllegalStateException("nothing produced the output '" + name + "'");
            }
            result.put(name, v.var != null ? v.var : Var.of(Tensor.of(Shape.of(v.shape), floats(v.data), device)));
        }
        return result;
    }

    @Override
    public String toString() {
        return "OnnxModel[" + graph.nodes.size() + " nodes, opset " + graph.opset + ", inputs " + inputNames()
                + ", outputs " + outputNames() + "]";
    }

    // ---------------------------------------------------------------- conversions

    private Tensor tensor(Graph.Raw raw) {
        Tensor t = Tensor.of(Shape.of(raw.dims()), raw.f(), device);
        return dtype == DType.FLOAT32 ? t : t.to(dtype);
    }

    private V value(Graph.Raw raw) {
        if (raw.floating()) {
            return V.of(Var.of(tensor(raw)));
        }
        return V.host(raw.type() == Graph.BOOL ? Graph.BOOL : raw.type(), raw.dims(), raw.l());
    }

    private Var cast(Var v) {
        return v.dtype() == dtype ? v : v.to(dtype);
    }

    private static long[] longs(Tensor t) {
        float[] a = t.to(DType.FLOAT32).toArray();
        long[] l = new long[a.length];
        for (int k = 0; k < a.length; k++) {
            l[k] = (long) a[k];
        }
        return l;
    }

    private static float[] floats(long[] l) {
        float[] f = new float[l.length];
        for (int k = 0; k < l.length; k++) {
            f[k] = l[k];
        }
        return f;
    }

    /** The value as a floating point variable on the device. */
    private Var fl(V v) {
        if (v.var != null) {
            return cast(v.var);
        }
        return Var.of(tensor(new Graph.Raw(Graph.FLOAT, v.shape, floats(v.data), null)));
    }

    /** The value as whole numbers on the host. */
    private static long[] ls(V v) {
        return v.data != null ? v.data : longs(v.var.value());
    }

    private static int[] is(V v) {
        return Arrays.stream(ls(v)).mapToInt(Math::toIntExact).toArray();
    }

    private static float scalar(V v) {
        return v.data != null ? v.data[0] : v.var.value().to(DType.FLOAT32).toArray()[0];
    }

    /** Whole numbers lifted to host floats for a shape operation, exact below 2^24. */
    private Var lift(V v) {
        return Var.of(Tensor.of(Shape.of(v.shape), floats(v.data), host));
    }

    /** Applies a shape operation to a device value, or to host numbers through {@link #lift}. */
    private V shaped(V x, UnaryOperator<Var> f) {
        if (x.var != null) {
            return V.of(x.type, f.apply(x.var));
        }
        Var r = f.apply(lift(x));
        return V.host(x.type, r.shape().toArray(), longs(r.value()));
    }

    private static V reshape(V x, int[] shape) {
        if (x.var != null) {
            return V.of(x.type, x.var.reshape(shape));
        }
        return V.host(x.type, shape, x.data);
    }

    private Tensor condition(V v) {
        if (v.var != null) {
            return v.var.value();
        }
        return Tensor.of(Shape.of(v.shape), floats(v.data), device);
    }

    private V bool(Tensor t) {
        return V.of(Graph.BOOL, Var.of(t));
    }

    private static int axis(long a, int rank) {
        int r = (int) (a < 0 ? a + rank : a);
        if (r < 0 || r >= Math.max(rank, 1)) {
            throw new IllegalArgumentException("axis " + a + " out of range for rank " + rank);
        }
        return r;
    }

    private static long size(int[] shape) {
        long n = 1;
        for (int d : shape) {
            n *= d;
        }
        return n;
    }

    private static int[] broadcast(int[] a, int[] b) {
        return Shape.broadcast(Shape.of(a), Shape.of(b)).toArray();
    }

    /** Element-wise integer arithmetic with numpy broadcasting. */
    private static V hostBinary(V a, V b, int type, LongBinaryOperator f) {
        int[] out = broadcast(a.shape, b.shape);
        long[] x = ls(a);
        long[] y = ls(b);
        long[] r = new long[(int) size(out)];
        int[] at = new int[out.length];
        for (int k = 0; k < r.length; k++) {
            r[k] = f.applyAsLong(x[index(at, out, a.shape)], y[index(at, out, b.shape)]);
            for (int d = out.length - 1; d >= 0; d--) {
                if (++at[d] < out[d]) {
                    break;
                }
                at[d] = 0;
            }
        }
        return V.host(type, out, r);
    }

    private static int index(int[] at, int[] out, int[] shape) {
        int idx = 0;
        int off = out.length - shape.length;
        for (int d = 0; d < shape.length; d++) {
            idx = idx * shape[d] + (shape[d] == 1 ? 0 : at[d + off]);
        }
        return idx;
    }

    // ---------------------------------------------------------------- operators

    private V[] node(Graph.Node n, V[] in) {
        V x = in.length > 0 ? in[0] : null;
        return switch (n.op()) {
            case "Identity", "Dropout" -> new V[] {x};
            case "Constant" -> new V[] {constant(n)};
            case "Add" -> one(arith(in[0], in[1], Long::sum, Var::add));
            case "Sub" -> one(arith(in[0], in[1], (a, b) -> a - b, Var::sub));
            case "Mul" -> one(arith(in[0], in[1], (a, b) -> a * b, Var::mul));
            case "Div" -> one(arith(in[0], in[1], (a, b) -> a / b, Var::div));
            case "Mod" -> one(mod(n, in[0], in[1]));
            case "Pow" -> one(pow(in[0], in[1]));
            case "Max", "Min", "Sum", "Mean" -> one(variadic(n.op(), in));
            case "Equal", "Greater", "Less", "GreaterOrEqual", "LessOrEqual" -> one(compare(n.op(), in[0], in[1]));
            case "And" -> one(hostBinary(in[0], in[1], Graph.BOOL, (a, b) -> a != 0 && b != 0 ? 1 : 0));
            case "Or" -> one(hostBinary(in[0], in[1], Graph.BOOL, (a, b) -> a != 0 || b != 0 ? 1 : 0));
            case "Xor" -> one(hostBinary(in[0], in[1], Graph.BOOL, (a, b) -> (a != 0) != (b != 0) ? 1 : 0));
            case "Not" -> one(x.data != null
                    ? V.host(Graph.BOOL, x.shape, Arrays.stream(x.data).map(v -> v == 0 ? 1 : 0).toArray())
                    : bool(Tensor.full(Shape.of(), 1f, device).sub(x.var.value())));
            case "Where" -> one(where(in[0], in[1], in[2]));
            case "IsNaN" -> {
                Tensor t = fl(x).value();
                yield one(bool(Tensor.full(Shape.of(), 1f, device).sub(t.equalTo(t))));
            }
            case "IsInf" -> one(isInf(n, x));
            case "Neg" -> one(x.data != null ? V.host(x.type, x.shape, Arrays.stream(x.data).map(v -> -v).toArray())
                    : V.of(fl(x).neg()));
            case "Abs" -> one(x.data != null ? V.host(x.type, x.shape, Arrays.stream(x.data).map(Math::abs).toArray())
                    : V.of(fl(x).abs()));
            case "Sign" -> one(x.data != null
                    ? V.host(x.type, x.shape, Arrays.stream(x.data).map(Long::signum).toArray())
                    : V.of(Var.of(fl(x).value().sign())));
            case "Floor", "Ceil", "Round" -> one(rounding(n.op(), x));
            case "Sqrt" -> one(V.of(fl(x).sqrt()));
            case "Exp" -> one(V.of(fl(x).exp()));
            case "Log" -> one(V.of(fl(x).log()));
            case "Reciprocal" -> one(V.of(fl(x).pow(-1f)));
            case "Erf" -> one(V.of(fl(x).erf()));
            case "Sin" -> one(V.of(fl(x).sin()));
            case "Cos" -> one(V.of(fl(x).cos()));
            case "Tan" -> one(V.of(fl(x).sin().div(fl(x).cos())));
            case "Atan" -> one(V.of(fl(x).atan()));
            case "Asin" -> one(V.of(fl(x).atan2(fl(x).square().neg().add(1f).sqrt())));
            case "Acos" -> one(V.of(fl(x).square().neg().add(1f).sqrt().atan2(fl(x))));
            case "Tanh" -> one(V.of(fl(x).tanh()));
            case "Sigmoid" -> one(V.of(fl(x).sigmoid()));
            case "Relu" -> one(V.of(fl(x).relu()));
            case "Softplus" -> one(V.of(fl(x).activation(ActivationOp.SOFTPLUS, 0f)));
            case "Softsign" -> one(V.of(fl(x).activation(ActivationOp.SOFTSIGN, 0f)));
            case "Mish" -> one(V.of(fl(x).activation(ActivationOp.MISH, 0f)));
            case "HardSwish" -> one(V.of(fl(x).activation(ActivationOp.HARD_SWISH, 0f)));
            case "HardSigmoid" -> one(V.of(fl(x).mul(n.f("alpha", 0.2f)).add(n.f("beta", 0.5f)).clamp(0f, 1f)));
            case "LeakyRelu" -> one(V.of(fl(x).activation(ActivationOp.LEAKY_RELU, n.f("alpha", 0.01f))));
            case "Elu" -> one(V.of(fl(x).activation(ActivationOp.ELU, n.f("alpha", 1f))));
            case "Celu" -> {
                float alpha = n.f("alpha", 1f);
                yield one(V.of(fl(x).div(alpha).activation(ActivationOp.ELU, 1f).mul(alpha)));
            }
            case "Selu" -> one(selu(n, fl(x)));
            case "PRelu" -> {
                Var v = fl(x);
                yield one(V.of(v.relu().sub(fl(in[1]).mul(v.neg().relu()))));
            }
            case "Gelu" -> one(V.of(fl(x).activation(n.s("approximate", "none").equals("tanh")
                    ? ActivationOp.GELU_TANH : ActivationOp.GELU, 0f)));
            case "Clip" -> one(clip(n, in));
            case "Softmax", "LogSoftmax" -> one(V.of(softmax(n, fl(x))));
            case "MatMul" -> one(V.of(matmul(fl(in[0]), fl(in[1]))));
            case "Gemm" -> one(gemm(n, in));
            case "Conv" -> one(conv(n, in));
            case "ConvTranspose" -> one(convTranspose(n, in));
            case "MaxPool", "AveragePool", "LpPool" -> pool(n, x);
            case "GlobalAveragePool" -> one(V.of(Functional.globalPool(fl(x), Functional.PoolKind.AVG, true)));
            case "GlobalMaxPool" -> one(V.of(Functional.globalPool(fl(x), Functional.PoolKind.MAX, true)));
            case "BatchNormalization" -> one(batchNorm(n, in));
            case "LayerNormalization" -> one(layerNorm(n, in));
            case "InstanceNormalization" -> one(groupNorm(fl(x), x.shape[1], fl(in[1]), fl(in[2]),
                    n.f("epsilon", 1e-5f)));
            case "GroupNormalization" -> one(groupNorm(fl(x), (int) n.i("num_groups", 1), fl(in[1]), fl(in[2]),
                    n.f("epsilon", 1e-5f)));
            case "Shape" -> one(shapeOf(n, x));
            case "Size" -> one(V.host(Graph.INT64, new int[0], new long[] {size(x.shape)}));
            case "Reshape" -> one(reshape(n, x, ls(in[1])));
            case "Flatten" -> {
                int a = (int) n.i("axis", 1);
                a = a < 0 ? a + x.rank() : a;
                int outer = (int) size(Arrays.copyOfRange(x.shape, 0, a));
                yield one(reshape(x, new int[] {outer, (int) (size(x.shape) / Math.max(outer, 1))}));
            }
            case "Squeeze" -> one(squeeze(n, in));
            case "Unsqueeze" -> one(unsqueeze(n, in));
            case "Transpose" -> one(transpose(n, x));
            case "Concat" -> one(concat(n, in));
            case "Split" -> split(n, in);
            case "Slice" -> one(slice(n, in));
            case "Gather" -> one(gather((int) n.i("axis", 0), x, in[1]));
            case "GatherElements" -> one(gatherElements(n, x, in[1]));
            case "Expand" -> one(shaped(x, v -> v.expand(Shape.of(broadcast(x.shape, is(in[1]))))));
            case "Tile" -> one(tile(x, ls(in[1])));
            case "Cast" -> one(castTo((int) n.i("to", Graph.FLOAT), x));
            case "CastLike" -> one(castTo(in[1].type, x));
            case "ConstantOfShape" -> one(constantOfShape(n, x));
            case "Range" -> one(range(in[0], in[1], in[2]));
            case "ArgMax", "ArgMin" -> one(argmax(n, x));
            case "ReduceSum", "ReduceMean", "ReduceMax", "ReduceMin", "ReduceProd", "ReduceL1", "ReduceL2",
                    "ReduceSumSquare", "ReduceLogSum", "ReduceLogSumExp" -> one(reduce(n, in));
            case "CumSum" -> one(cumsum(n, x, in[1]));
            case "Einsum" -> {
                Var[] ops = new Var[in.length];
                for (int k = 0; k < in.length; k++) {
                    ops[k] = fl(in[k]);
                }
                yield one(V.of(Einsum.of(n.s("equation", ""), ops)));
            }
            case "Pad" -> one(pad(n, in));
            case "Resize", "Upsample" -> one(resize(n, in));
            case "Trilu" -> one(trilu(n, in));
            default -> throw new UnsupportedOperationException("unsupported operator " + n.op());
        };
    }

    private static V[] one(V v) {
        return new V[] {v};
    }

    private V constant(Graph.Node n) {
        Graph.Attr a;
        if ((a = n.attr("value")) != null) {
            return value(a.t);
        }
        if ((a = n.attr("value_float")) != null) {
            return V.of(Var.of(tensor(new Graph.Raw(Graph.FLOAT, new int[0], new float[] {a.f}, null))));
        }
        if ((a = n.attr("value_floats")) != null) {
            return V.of(Var.of(tensor(new Graph.Raw(Graph.FLOAT, new int[] {a.floats.length}, a.floats, null))));
        }
        if ((a = n.attr("value_int")) != null) {
            return V.host(Graph.INT64, new int[0], new long[] {a.i});
        }
        if ((a = n.attr("value_ints")) != null) {
            return V.host(Graph.INT64, new int[] {a.ints.length}, a.ints);
        }
        throw new UnsupportedOperationException("Constant without a value tensor or numbers");
    }

    private interface VarBinary {
        Var apply(Var a, Var b);
    }

    private V arith(V a, V b, LongBinaryOperator ints, VarBinary floats) {
        if (a.data != null && b.data != null) {
            return hostBinary(a, b, a.type, ints);
        }
        return V.of(floats.apply(fl(a), fl(b)));
    }

    private V mod(Graph.Node n, V a, V b) {
        boolean fmod = n.i("fmod", 0) == 1;
        if (a.data != null && b.data != null) {
            return hostBinary(a, b, a.type, fmod ? (p, q) -> p % q : Math::floorMod);
        }
        Var x = fl(a);
        Var y = fl(b);
        Tensor ratio = x.div(y).value();
        Tensor whole = fmod ? ratio.sign().mul(floor(ratio.abs())) : floor(ratio);
        return V.of(x.sub(y.mul(whole)));
    }

    /** Rounds down: rounding to nearest, then stepping back where that went up. */
    private static Tensor floor(Tensor x) {
        Tensor r = x.round();
        return Tensor.where(r.greater(x), r.sub(Tensor.full(Shape.of(), 1f, r.device())), r);
    }

    private V rounding(String op, V x) {
        if (x.data != null) {
            return x;
        }
        Tensor t = fl(x).value();
        Tensor r = switch (op) {
            case "Floor" -> floor(t);
            case "Ceil" -> floor(t.neg()).neg();
            default -> t.round();
        };
        return V.of(Var.of(r));
    }

    private V pow(V a, V b) {
        if (a.data != null && b.data != null) {
            return hostBinary(a, b, a.type, (p, q) -> (long) Math.pow(p, q));
        }
        if (size(b.shape) == 1) { // a fixed power, which also takes negative bases; the exponent is not trained
            float p = scalar(b);
            Var base = fl(a);
            Var r = p == 2f ? base.square() : p == 1f ? base : p == 0.5f ? base.sqrt() : base.pow(p);
            return V.of(b.rank() > a.rank() ? r.reshape(broadcast(a.shape, b.shape)) : r);
        }
        return V.of(fl(b).mul(fl(a).log()).exp());
    }

    private V variadic(String op, V[] in) {
        boolean ints = Arrays.stream(in).allMatch(v -> v.data != null);
        V acc = in[0];
        for (int k = 1; k < in.length; k++) {
            V b = in[k];
            acc = switch (op) {
                case "Max" -> ints ? hostBinary(acc, b, acc.type, Math::max) : V.of(fl(acc).maximum(fl(b)));
                case "Min" -> ints ? hostBinary(acc, b, acc.type, Math::min) : V.of(fl(acc).minimum(fl(b)));
                default -> ints ? hostBinary(acc, b, acc.type, Long::sum) : V.of(fl(acc).add(fl(b)));
            };
        }
        if (op.equals("Mean")) {
            acc = V.of(fl(acc).div(in.length));
        }
        return acc;
    }

    private V compare(String op, V a, V b) {
        if (a.data != null && b.data != null) {
            return hostBinary(a, b, Graph.BOOL, switch (op) {
                case "Equal" -> (p, q) -> p == q ? 1 : 0;
                case "Greater" -> (p, q) -> p > q ? 1 : 0;
                case "Less" -> (p, q) -> p < q ? 1 : 0;
                case "GreaterOrEqual" -> (p, q) -> p >= q ? 1 : 0;
                default -> (p, q) -> p <= q ? 1 : 0;
            });
        }
        Tensor x = fl(a).value();
        Tensor y = fl(b).value();
        Tensor one = Tensor.full(Shape.of(), 1f, device);
        return bool(switch (op) {
            case "Equal" -> x.equalTo(y);
            case "Greater" -> x.greater(y);
            case "Less" -> x.less(y);
            case "GreaterOrEqual" -> one.sub(x.less(y));
            default -> one.sub(x.greater(y));
        });
    }

    private V where(V c, V a, V b) {
        if (c.data != null && a.data != null && b.data != null) {
            Var r = Var.where(lift(c).value(), lift(a), lift(b));
            return V.host(a.type, r.shape().toArray(), longs(r.value()));
        }
        Var x = fl(a);
        Var y = fl(b);
        Tensor cond = condition(c);
        if (cond.dtype() != x.dtype()) {
            cond = cond.to(x.dtype());
        }
        return V.of(Graph.isFloat(a.type) || Graph.isFloat(b.type) ? Graph.FLOAT : a.type, Var.where(cond, x, y));
    }

    private V isInf(Graph.Node n, V x) {
        Tensor t = fl(x).value();
        Tensor big = Tensor.full(Shape.of(), Float.MAX_VALUE, device);
        Tensor pos = t.greater(big);
        Tensor neg = big.neg().greater(t);
        boolean detectPos = n.i("detect_positive", 1) == 1;
        boolean detectNeg = n.i("detect_negative", 1) == 1;
        Tensor r = detectPos && detectNeg ? pos.add(neg) : detectPos ? pos : neg;
        return bool(r);
    }

    private V selu(Graph.Node n, Var x) {
        float alpha = n.f("alpha", 1.6732632f);
        float gamma = n.f("gamma", 1.0507010f);
        return V.of(x.activation(ActivationOp.ELU, alpha).mul(gamma));
    }

    private V clip(Graph.Node n, V[] in) {
        float low = Float.NEGATIVE_INFINITY;
        float high = Float.POSITIVE_INFINITY;
        if (graph.opset < 11) {
            low = n.f("min", low);
            high = n.f("max", high);
        } else {
            if (n.input(1) != null) {
                low = scalar(in[1]);
            }
            if (n.input(2) != null) {
                high = scalar(in[2]);
            }
        }
        V x = in[0];
        if (x.data != null) {
            long lo = (long) Math.max(low, Long.MIN_VALUE);
            long hi = (long) Math.min(high, Long.MAX_VALUE);
            return V.host(x.type, x.shape, Arrays.stream(x.data).map(v -> Math.min(Math.max(v, lo), hi)).toArray());
        }
        return V.of(fl(x).clamp(low, high));
    }

    private Var softmax(Graph.Node n, Var x) {
        boolean log = n.op().equals("LogSoftmax");
        int rank = x.shape().rank();
        if (graph.opset < 13) { // the old definition flattens around the axis
            int a = axis(n.i("axis", 1), rank);
            int[] s = x.shape().toArray();
            int outer = (int) size(Arrays.copyOfRange(s, 0, a));
            Var flat = x.reshape(outer, (int) (size(s) / Math.max(outer, 1)));
            Var l = flat.sub(flat.logsumexp(1, true));
            return (log ? l : l.exp()).reshape(s);
        }
        int a = axis(n.i("axis", -1), rank);
        Var l = x.sub(x.logsumexp(a, true));
        return log ? l : l.exp();
    }

    private static Var matmul(Var a, Var b) {
        boolean vecA = a.shape().rank() == 1;
        boolean vecB = b.shape().rank() == 1;
        if (vecA) {
            a = a.reshape(1, a.shape().dim(0));
        }
        if (vecB) {
            b = b.reshape(b.shape().dim(0), 1);
        }
        Var r = a.matmul(b);
        int[] s = r.shape().toArray();
        if (vecA || vecB) {
            List<Integer> dims = new ArrayList<>();
            for (int d = 0; d < s.length; d++) {
                boolean drop = (vecA && d == s.length - 2) || (vecB && d == s.length - 1);
                if (!drop) {
                    dims.add(s[d]);
                }
            }
            r = r.reshape(dims.stream().mapToInt(Integer::intValue).toArray());
        }
        return r;
    }

    private V gemm(Graph.Node n, V[] in) {
        Var a = fl(in[0]);
        Var b = fl(in[1]);
        if (n.i("transA", 0) == 1) {
            a = a.transpose();
        }
        if (n.i("transB", 0) == 1) {
            b = b.transpose();
        }
        Var y = a.matmul(b);
        float alpha = n.f("alpha", 1f);
        if (alpha != 1f) {
            y = y.mul(alpha);
        }
        if (n.input(2) != null) {
            float beta = n.f("beta", 1f);
            Var c = fl(in[2]);
            y = y.add(beta == 1f ? c : c.mul(beta));
        }
        return V.of(y);
    }

    private static int[] ints(long[] l, int count, int otherwise) {
        int[] r = new int[count];
        Arrays.fill(r, otherwise);
        if (l != null) {
            for (int k = 0; k < count && k < l.length; k++) {
                r[k] = Math.toIntExact(l[k]);
            }
        }
        return r;
    }

    /** Padding before and after each spatial dimension, from {@code pads} or {@code auto_pad}. */
    private static int[][] pads(Graph.Node n, int[] in, int[] kernel, int[] stride, int[] dilation) {
        int r = kernel.length;
        int[] before = new int[r];
        int[] after = new int[r];
        String auto = n.s("auto_pad", "NOTSET");
        if (auto.startsWith("SAME")) {
            for (int d = 0; d < r; d++) {
                int out = (in[d] + stride[d] - 1) / stride[d];
                int total = Math.max(0, (out - 1) * stride[d] + (kernel[d] - 1) * dilation[d] + 1 - in[d]);
                int small = total / 2;
                before[d] = auto.equals("SAME_UPPER") ? small : total - small;
                after[d] = total - before[d];
            }
        } else if (!auto.equals("VALID")) {
            long[] p = n.ints("pads");
            if (p != null && p.length == 2 * r) {
                for (int d = 0; d < r; d++) {
                    before[d] = (int) p[d];
                    after[d] = (int) p[d + r];
                }
            }
        }
        return new int[][] {before, after};
    }

    private V conv(Graph.Node n, V[] in) {
        Var x = fl(in[0]);
        Var w = fl(in[1]);
        Var b = n.input(2) != null ? fl(in[2]) : null;
        int r = w.shape().rank() - 2;
        int[] kernel = Arrays.copyOfRange(w.shape().toArray(), 2, r + 2);
        int[] stride = ints(n.ints("strides"), r, 1);
        int[] dilation = ints(n.ints("dilations"), r, 1);
        int[] spatial = Arrays.copyOfRange(x.shape().toArray(), 2, r + 2);
        int[][] p = pads(n, spatial, kernel, stride, dilation);
        return V.of(Functional.conv(x, w, b, stride, p[0], p[1], dilation, (int) n.i("group", 1)));
    }

    private V convTranspose(Graph.Node n, V[] in) {
        Var x = fl(in[0]);
        Var w = fl(in[1]);
        Var b = n.input(2) != null ? fl(in[2]) : null;
        int r = w.shape().rank() - 2;
        int[] kernel = Arrays.copyOfRange(w.shape().toArray(), 2, r + 2);
        int[] stride = ints(n.ints("strides"), r, 1);
        int[] dilation = ints(n.ints("dilations"), r, 1);
        int[] outputPadding = ints(n.ints("output_padding"), r, 0);
        int[] before = new int[r];
        int[] after = new int[r];
        long[] outputShape = n.ints("output_shape");
        String auto = n.s("auto_pad", "NOTSET");
        int[] spatial = Arrays.copyOfRange(x.shape().toArray(), 2, r + 2);
        if (outputShape != null && outputShape.length > 0 || auto.startsWith("SAME")) {
            for (int d = 0; d < r; d++) {
                int want = outputShape != null && outputShape.length > 0
                        ? (int) outputShape[outputShape.length - r + d] : spatial[d] * stride[d];
                int total = stride[d] * (spatial[d] - 1) + outputPadding[d] + (kernel[d] - 1) * dilation[d] + 1
                        - want;
                total = Math.max(total, 0);
                before[d] = auto.equals("SAME_UPPER") ? total / 2 : total - total / 2;
                after[d] = total - before[d];
            }
        } else if (!auto.equals("VALID")) {
            long[] p = n.ints("pads");
            if (p != null && p.length == 2 * r) {
                for (int d = 0; d < r; d++) {
                    before[d] = (int) p[d];
                    after[d] = (int) p[d + r];
                }
            }
        }
        return V.of(Functional.convTranspose(x, w, b, stride, before, after, outputPadding, dilation,
                (int) n.i("group", 1)));
    }

    private V[] pool(Graph.Node n, V in) {
        if (n.op().equals("MaxPool") && n.out().length > 1 && !n.out()[1].isEmpty()) {
            throw new UnsupportedOperationException("MaxPool with an Indices output");
        }
        Var x = fl(in);
        long[] k = n.ints("kernel_shape");
        int r = k.length;
        int[] kernel = ints(k, r, 1);
        int[] stride = ints(n.ints("strides"), r, 1);
        int[] dilation = ints(n.ints("dilations"), r, 1);
        int[] spatial = Arrays.copyOfRange(x.shape().toArray(), 2, r + 2);
        int[][] p = pads(n, spatial, kernel, stride, dilation);
        Functional.PoolKind kind = switch (n.op()) {
            case "MaxPool" -> Functional.PoolKind.MAX;
            case "AveragePool" -> Functional.PoolKind.AVG;
            default -> Functional.PoolKind.LP;
        };
        boolean ceil = n.i("ceil_mode", 0) == 1;
        boolean countPadding = n.i("count_include_pad", 0) == 1;
        Var y = Functional.pool(x, kind, kernel, stride, p[0], p[1], dilation, ceil, countPadding,
                n.i("p", 2));
        return one(V.of(y));
    }

    private Var channels(Var v, int rank) {
        int[] s = new int[rank - 1];
        Arrays.fill(s, 1);
        s[0] = (int) v.shape().size();
        return v.reshape(s);
    }

    private V batchNorm(Graph.Node n, V[] in) {
        Var x = fl(in[0]);
        int rank = x.shape().rank();
        float eps = n.f("epsilon", 1e-5f);
        Var scale = channels(fl(in[1]), rank);
        Var bias = channels(fl(in[2]), rank);
        Var mean = channels(fl(in[3]), rank);
        Var var = channels(fl(in[4]), rank);
        Var y = x.sub(mean).div(var.add(eps).sqrt()).mul(scale).add(bias);
        return V.of(y);
    }

    private V layerNorm(Graph.Node n, V[] in) {
        Var x = fl(in[0]);
        int[] s = x.shape().toArray();
        int a = axis(n.i("axis", -1), s.length);
        int outer = (int) size(Arrays.copyOfRange(s, 0, a));
        int inner = (int) (size(s) / Math.max(outer, 1));
        Var flat = x.reshape(outer, inner);
        Var mean = flat.mean(1, true);
        Var centered = flat.sub(mean);
        Var var = centered.square().mean(1, true);
        Var y = centered.div(var.add(n.f("epsilon", 1e-5f)).sqrt()).reshape(s).mul(fl(in[1]));
        if (n.input(2) != null) {
            y = y.add(fl(in[2]));
        }
        return V.of(y);
    }

    private V groupNorm(Var x, int groups, Var scale, Var bias, float eps) {
        int[] s = x.shape().toArray();
        int batch = s[0];
        int c = s[1];
        Var flat = x.reshape(batch, groups, (int) (size(s) / batch / groups));
        Var centered = flat.sub(flat.mean(2, true));
        Var var = centered.square().mean(2, true);
        Var y = centered.div(var.add(eps).sqrt()).reshape(s);
        if (scale.shape().size() != c) { // opset 18 scales each group
            scale = scale.reshape(groups, 1).expand(Shape.of(groups, c / groups)).reshape(c);
            bias = bias.reshape(groups, 1).expand(Shape.of(groups, c / groups)).reshape(c);
        }
        return V.of(y.mul(channels(scale, s.length)).add(channels(bias, s.length)));
    }

    private static V shapeOf(Graph.Node n, V x) {
        int rank = x.rank();
        long start = n.i("start", 0);
        long end = n.i("end", rank);
        int a = (int) Math.min(Math.max(start < 0 ? start + rank : start, 0), rank);
        int b = (int) Math.min(Math.max(end < 0 ? end + rank : end, 0), rank);
        long[] dims = new long[Math.max(b - a, 0)];
        for (int d = a; d < b; d++) {
            dims[d - a] = x.shape[d];
        }
        return V.host(Graph.INT64, new int[] {dims.length}, dims);
    }

    private static V reshape(Graph.Node n, V x, long[] target) {
        boolean allowZero = n.i("allowzero", 0) == 1;
        int[] s = new int[target.length];
        int unknown = -1;
        long known = 1;
        for (int d = 0; d < s.length; d++) {
            long t = target[d];
            if (t == 0 && !allowZero) {
                t = x.shape[d];
            }
            if (t == -1) {
                unknown = d;
            } else {
                s[d] = (int) t;
                known *= t;
            }
        }
        if (unknown >= 0) {
            s[unknown] = (int) (size(x.shape) / Math.max(known, 1));
        }
        return reshape(x, s);
    }

    private long[] axesOf(Graph.Node n, V[] in, int index) {
        if (in.length > index && in[index] != null) {
            return ls(in[index]);
        }
        return n.ints("axes");
    }

    private V squeeze(Graph.Node n, V[] in) {
        V x = in[0];
        long[] axes = axesOf(n, in, 1);
        boolean[] drop = new boolean[x.rank()];
        if (axes == null || axes.length == 0) {
            for (int d = 0; d < drop.length; d++) {
                drop[d] = x.shape[d] == 1;
            }
        } else {
            for (long a : axes) {
                drop[axis(a, x.rank())] = true;
            }
        }
        int[] s = new int[x.rank()];
        int k = 0;
        for (int d = 0; d < drop.length; d++) {
            if (!drop[d]) {
                s[k++] = x.shape[d];
            }
        }
        return reshape(x, Arrays.copyOf(s, k));
    }

    private V unsqueeze(Graph.Node n, V[] in) {
        V x = in[0];
        long[] axes = axesOf(n, in, 1);
        int rank = x.rank() + axes.length;
        boolean[] added = new boolean[rank];
        for (long a : axes) {
            added[axis(a, rank)] = true;
        }
        int[] s = new int[rank];
        int k = 0;
        for (int d = 0; d < rank; d++) {
            s[d] = added[d] ? 1 : x.shape[k++];
        }
        return reshape(x, s);
    }

    private V transpose(Graph.Node n, V x) {
        long[] p = n.ints("perm");
        int[] perm = new int[x.rank()];
        for (int d = 0; d < perm.length; d++) {
            perm[d] = p != null && p.length == perm.length ? (int) p[d] : perm.length - 1 - d;
        }
        return shaped(x, v -> v.permute(perm));
    }

    private V concat(Graph.Node n, V[] in) {
        List<V> parts = new ArrayList<>();
        for (V v : in) {
            if (v != null && size(v.shape) > 0) {
                parts.add(v);
            }
        }
        if (parts.isEmpty()) {
            return in[0];
        }
        int a = axis(n.i("axis", 0), parts.get(0).rank());
        if (parts.size() == 1) {
            return parts.get(0);
        }
        if (parts.stream().allMatch(v -> v.data != null)) {
            Var[] vs = parts.stream().map(this::lift).toArray(Var[]::new);
            Var r = Var.concat(a, vs);
            return V.host(parts.get(0).type, r.shape().toArray(), longs(r.value()));
        }
        return V.of(parts.get(0).type, Var.concat(a, parts.stream().map(this::fl).toArray(Var[]::new)));
    }

    private V[] split(Graph.Node n, V[] in) {
        V x = in[0];
        int a = axis(n.i("axis", 0), x.rank());
        int length = x.shape[a];
        int outputs = n.out().length;
        long[] sizes = in.length > 1 && in[1] != null ? ls(in[1]) : n.ints("split");
        if (sizes == null || sizes.length == 0) {
            long parts = n.i("num_outputs", outputs);
            long each = (length + parts - 1) / parts;
            sizes = new long[(int) parts];
            for (int k = 0; k < parts; k++) {
                sizes[k] = Math.min(each, length - each * k);
            }
        }
        V[] out = new V[sizes.length];
        int start = 0;
        for (int k = 0; k < sizes.length; k++) {
            int from = start;
            int to = start + (int) sizes[k];
            out[k] = shaped(x, v -> v.slice(a, from, to));
            start = to;
        }
        return out;
    }

    private V slice(Graph.Node n, V[] in) {
        V x = in[0];
        long[] starts;
        long[] ends;
        long[] axes;
        long[] steps;
        if (graph.opset < 10) {
            starts = n.ints("starts");
            ends = n.ints("ends");
            axes = n.ints("axes");
            steps = null;
        } else {
            starts = ls(in[1]);
            ends = ls(in[2]);
            axes = in.length > 3 && in[3] != null ? ls(in[3]) : null;
            steps = in.length > 4 && in[4] != null ? ls(in[4]) : null;
        }
        V r = x;
        for (int k = 0; k < starts.length; k++) {
            int a = axis(axes != null && axes.length > 0 ? axes[k] : k, x.rank());
            long step = steps != null ? steps[k] : 1;
            int length = r.shape[a];
            long s = starts[k];
            long e = ends[k];
            s = s < 0 ? s + length : s;
            e = e < 0 ? e + length : e;
            if (step > 0) {
                s = Math.clamp(s, 0, length);
                e = Math.clamp(e, 0, length);
            } else {
                s = Math.clamp(s, -1, length - 1);
                e = Math.clamp(e, -1, length - 1);
            }
            if (step == 1) {
                int from = (int) s;
                int to = (int) Math.max(e, s);
                if (from == 0 && to == length) {
                    continue;
                }
                r = shaped(r, v -> v.slice(a, from, to));
            } else {
                List<Long> picks = new ArrayList<>();
                for (long i = s; step > 0 ? i < e : i > e; i += step) {
                    picks.add(i);
                }
                long[] idx = picks.stream().mapToLong(Long::longValue).toArray();
                r = gather(a, r, V.host(Graph.INT64, new int[] {idx.length}, idx));
            }
        }
        return r;
    }

    private V gather(int axisArg, V x, V indices) {
        int a = axis(axisArg, x.rank());
        long[] idx = ls(indices).clone();
        for (int k = 0; k < idx.length; k++) {
            if (idx[k] < 0) {
                idx[k] += x.shape[a];
            }
        }
        int[] is = indices.shape.length == 0 ? new int[] {1} : indices.shape;
        if (x.data != null) {
            // whole numbers stay exact: index the host array directly
            int outer = (int) size(Arrays.copyOfRange(x.shape, 0, a));
            int inner = (int) size(Arrays.copyOfRange(x.shape, a + 1, x.rank()));
            int length = x.shape[a];
            long[] r = new long[outer * idx.length * inner];
            int at = 0;
            for (int o = 0; o < outer; o++) {
                for (long i : idx) {
                    System.arraycopy(x.data, (int) ((o * (long) length + i) * inner), r, at, inner);
                    at += inner;
                }
            }
            return V.host(x.type, gatheredShape(x.shape, a, indices.shape), r);
        }
        Tensor t = Tensor.of(Shape.of(is), floats(idx), x.var.value().device());
        Var g = x.var.gather(a, t);
        return V.of(x.type, g.reshape(gatheredShape(x.shape, a, indices.shape)));
    }

    private static int[] gatheredShape(int[] shape, int a, int[] indices) {
        int[] r = new int[shape.length - 1 + indices.length];
        System.arraycopy(shape, 0, r, 0, a);
        System.arraycopy(indices, 0, r, a, indices.length);
        System.arraycopy(shape, a + 1, r, a + indices.length, shape.length - a - 1);
        return r;
    }

    private V gatherElements(Graph.Node n, V x, V indices) {
        int a = axis(n.i("axis", 0), x.rank());
        long[] idx = ls(indices).clone();
        for (int k = 0; k < idx.length; k++) {
            if (idx[k] < 0) {
                idx[k] += x.shape[a];
            }
        }
        int last = x.rank() - 1;
        int[] perm = new int[x.rank()];
        for (int d = 0; d < perm.length; d++) {
            perm[d] = d;
        }
        perm[a] = last;
        perm[last] = a;
        int[] is = indices.shape.clone();
        int tmp = is[a];
        is[a] = is[last];
        is[last] = tmp;
        // move the axis last, pick along it, move it back
        Tensor t = Tensor.of(Shape.of(indices.shape), floats(idx), x.var != null ? x.var.value().device() : host)
                .permute(perm);
        return shaped(x, v -> v.permute(perm).takeAlong(t).permute(perm));
    }

    private V tile(V x, long[] repeats) {
        return shaped(x, v -> {
            Var r = v;
            for (int d = 0; d < repeats.length; d++) {
                if (repeats[d] != 1) {
                    Var[] copies = new Var[(int) repeats[d]];
                    Arrays.fill(copies, r);
                    r = Var.concat(d, copies);
                }
            }
            return r;
        });
    }

    private V castTo(int to, V x) {
        if (Graph.isFloat(to)) {
            return V.of(fl(x));
        }
        if (to == Graph.BOOL) {
            if (x.data != null) {
                return V.host(to, x.shape, Arrays.stream(x.data).map(v -> v != 0 ? 1 : 0).toArray());
            }
            Tensor t = x.var.value();
            return bool(Tensor.full(Shape.of(), 1f, device).sub(t.equalTo(Tensor.full(Shape.of(), 0f, device))));
        }
        if (x.data != null) {
            return V.host(to, x.shape, x.data);
        }
        return V.host(to, x.shape, ls(x));
    }

    private V constantOfShape(Graph.Node n, V shape) {
        int[] s = is(shape);
        Graph.Attr a = n.attr("value");
        Graph.Raw v = a == null ? new Graph.Raw(Graph.FLOAT, new int[] {1}, new float[] {0f}, null) : a.t;
        if (v.floating()) {
            return V.of(Var.of(Tensor.full(Shape.of(s), v.f()[0], dtype, device)));
        }
        long[] r = new long[(int) size(s)];
        Arrays.fill(r, v.l()[0]);
        return V.host(v.type(), s, r);
    }

    private V range(V start, V limit, V delta) {
        if (start.data != null && limit.data != null && delta.data != null) {
            long s = start.data[0];
            long l = limit.data[0];
            long d = delta.data[0];
            int count = (int) Math.max(Math.ceilDiv(l - s, d), 0);
            long[] r = new long[count];
            for (int k = 0; k < count; k++) {
                r[k] = s + k * d;
            }
            return V.host(start.type, new int[] {count}, r);
        }
        float s = scalar(start);
        float l = scalar(limit);
        float d = scalar(delta);
        int count = (int) Math.max(Math.ceil((l - s) / d), 0);
        float[] r = new float[count];
        for (int k = 0; k < count; k++) {
            r[k] = s + k * d;
        }
        return V.of(Var.of(tensor(new Graph.Raw(Graph.FLOAT, new int[] {count}, r, null))));
    }

    private V argmax(Graph.Node n, V x) {
        Var v = x.data != null ? lift(x) : fl(x);
        if (n.op().equals("ArgMin")) {
            v = v.neg();
        }
        int a = axis(n.i("axis", 0), x.rank());
        boolean last = n.i("select_last_index", 0) == 1;
        Tensor t = last ? v.flip(a).value() : v.value();
        long[] idx = longs(t.argmax(a));
        if (last) {
            for (int k = 0; k < idx.length; k++) {
                idx[k] = x.shape[a] - 1 - idx[k];
            }
        }
        int[] s = x.shape.clone();
        if (n.i("keepdims", 1) == 1) {
            s[a] = 1;
        } else {
            int[] r = new int[s.length - 1];
            System.arraycopy(s, 0, r, 0, a);
            System.arraycopy(s, a + 1, r, a, s.length - a - 1);
            s = r;
        }
        return V.host(Graph.INT64, s, idx);
    }

    private V reduce(Graph.Node n, V[] in) {
        V x = in[0];
        long[] axes = axesOf(n, in, 1);
        boolean keep = n.i("keepdims", 1) == 1;
        if (axes == null || axes.length == 0) {
            if (n.i("noop_with_empty_axes", 0) == 1) {
                return x;
            }
            axes = new long[x.rank()];
            for (int d = 0; d < axes.length; d++) {
                axes[d] = d;
            }
        }
        boolean[] reduced = new boolean[x.rank()];
        for (long a : axes) {
            reduced[axis(a, x.rank())] = true;
        }
        String op = n.op();
        UnaryOperator<Var> f = v -> {
            Var r = switch (op) {
                case "ReduceL1" -> v.abs();
                case "ReduceL2", "ReduceSumSquare" -> v.square();
                case "ReduceMin" -> v.neg();
                default -> v;
            };
            for (int d = reduced.length - 1; d >= 0; d--) {
                if (!reduced[d]) {
                    continue;
                }
                r = switch (op) {
                    case "ReduceMean" -> r.mean(d, true);
                    case "ReduceMax", "ReduceMin" -> r.max(d, true);
                    case "ReduceLogSumExp" -> r.logsumexp(d, true);
                    case "ReduceProd" -> {
                        Var p = r.slice(d, 0, 1);
                        for (int i = 1; i < r.shape().dim(d); i++) {
                            p = p.mul(r.slice(d, i, i + 1));
                        }
                        yield p;
                    }
                    default -> r.sum(d, true);
                };
            }
            r = switch (op) {
                case "ReduceL2" -> r.sqrt();
                case "ReduceMin" -> r.neg();
                case "ReduceLogSum" -> r.log();
                default -> r;
            };
            if (!keep) {
                List<Integer> s = new ArrayList<>();
                for (int d = 0; d < reduced.length; d++) {
                    if (!reduced[d]) {
                        s.add(r.shape().dim(d));
                    }
                }
                r = r.reshape(s.stream().mapToInt(Integer::intValue).toArray());
            }
            return r;
        };
        if (x.data != null) {
            return shaped(x, f);
        }
        return V.of(f.apply(fl(x)));
    }

    private V cumsum(Graph.Node n, V x, V axisValue) {
        int a = axis(ls(axisValue)[0], x.rank());
        boolean exclusive = n.i("exclusive", 0) == 1;
        boolean reverse = n.i("reverse", 0) == 1;
        return shaped(x.data != null ? x : V.of(fl(x)), v -> {
            Var s = reverse ? v.flip(a) : v;
            Var c = s.cumsum(a);
            if (exclusive) {
                c = c.sub(s);
            }
            return reverse ? c.flip(a) : c;
        });
    }

    private V pad(Graph.Node n, V[] in) {
        V x = in[0];
        long[] pads;
        float value;
        long[] axes = null;
        if (graph.opset < 11) {
            pads = n.ints("pads");
            value = n.f("value", 0f);
        } else {
            pads = ls(in[1]);
            value = in.length > 2 && in[2] != null ? scalar(in[2]) : 0f;
            axes = in.length > 3 && in[3] != null ? ls(in[3]) : null;
        }
        PadMode mode = switch (n.s("mode", "constant")) {
            case "reflect" -> PadMode.REFLECT;
            case "edge" -> PadMode.REPLICATE;
            case "wrap" -> PadMode.CIRCULAR;
            default -> PadMode.CONSTANT;
        };
        int count = pads.length / 2;
        long[] ax = axes;
        float fill = value;
        return shaped(x.data != null ? x : V.of(fl(x)), v -> {
            Var r = v;
            for (int k = 0; k < count; k++) {
                int a = axis(ax != null ? ax[k] : k, v.shape().rank());
                int before = (int) pads[k];
                int after = (int) pads[k + count];
                if (before != 0 || after != 0) {
                    r = Functional.padAxis(r, a, before, after, mode, fill);
                }
            }
            return r;
        });
    }

    private V resize(Graph.Node n, V[] in) {
        Var x = fl(in[0]);
        int[] s = x.shape().toArray();
        int rank = s.length;
        float[] scales = null;
        int[] sizes = null;
        if (n.op().equals("Upsample") || graph.opset < 11) {
            V sc = n.op().equals("Upsample") && n.attr("scales") != null ? null : in[1];
            if (sc == null) {
                scales = n.attr("scales").floats;
            } else {
                scales = fl(sc).value().to(DType.FLOAT32).toArray();
            }
        } else {
            if (in.length > 3 && in[3] != null && size(in[3].shape) > 0) {
                sizes = is(in[3]);
            } else if (in.length > 2 && in[2] != null && size(in[2].shape) > 0) {
                scales = fl(in[2]).value().to(DType.FLOAT32).toArray();
            }
        }
        long[] axes = n.ints("axes");
        int[] target = s.clone();
        float[] scale = new float[rank];
        Arrays.fill(scale, 1f);
        int count = sizes != null ? sizes.length : scales.length;
        for (int k = 0; k < count; k++) {
            int a = axes != null && axes.length > 0 ? axis(axes[k], rank) : k;
            if (sizes != null) {
                target[a] = sizes[k];
                scale[a] = (float) sizes[k] / s[a];
            } else {
                scale[a] = scales[k];
                target[a] = (int) Math.floor(s[a] * scales[k]);
            }
        }
        if (target[0] != s[0] || target[1] != s[1]) {
            throw new UnsupportedOperationException("Resize of the batch or channel dimension");
        }
        String mode = n.s("mode", "nearest");
        String transform = n.s("coordinate_transformation_mode", graph.opset < 11 ? "asymmetric" : "half_pixel");
        if (mode.equals("nearest")) {
            String rounding = n.s("nearest_mode", "round_prefer_floor");
            Var r = x;
            for (int a = 2; a < rank; a++) {
                if (target[a] == s[a]) {
                    continue;
                }
                float[] idx = new float[target[a]];
                for (int o = 0; o < target[a]; o++) {
                    double src = source(transform, o, s[a], target[a], scale[a]);
                    double i = switch (rounding) {
                        case "floor" -> Math.floor(src);
                        case "ceil" -> Math.ceil(src);
                        case "round_prefer_ceil" -> Math.floor(src + 0.5);
                        default -> Math.ceil(src - 0.5);
                    };
                    idx[o] = (float) Math.clamp((long) i, 0, s[a] - 1);
                }
                r = r.gather(a, Tensor.of(Shape.of(target[a]), idx, device));
            }
            return V.of(r);
        }
        if (!transform.equals("half_pixel") && !transform.equals("pytorch_half_pixel")
                && !transform.equals("align_corners")) {
            throw new UnsupportedOperationException("Resize " + mode + " with " + transform + " coordinates");
        }
        Functional.Interpolation how = mode.equals("cubic") ? Functional.Interpolation.CUBIC
                : Functional.Interpolation.LINEAR;
        return V.of(Functional.resize(x, Arrays.copyOfRange(target, 2, rank), how, transform.equals("align_corners")));
    }

    private static double source(String transform, int o, int in, int out, float scale) {
        return switch (transform) {
            case "align_corners" -> out == 1 ? 0 : o * (double) (in - 1) / (out - 1);
            case "asymmetric" -> o / (double) scale;
            case "pytorch_half_pixel" -> out > 1 ? (o + 0.5) / scale - 0.5 : 0;
            case "tf_half_pixel_for_nn" -> (o + 0.5) / scale;
            default -> (o + 0.5) / scale - 0.5;
        };
    }

    private V trilu(Graph.Node n, V[] in) {
        V x = in[0];
        long k = in.length > 1 && in[1] != null ? ls(in[1])[0] : 0;
        boolean upper = n.i("upper", 1) == 1;
        int rows = x.shape[x.rank() - 2];
        int cols = x.shape[x.rank() - 1];
        float[] keep = new float[rows * cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                keep[i * cols + j] = (upper ? j - i >= k : j - i <= k) ? 1f : 0f;
            }
        }
        if (x.data != null) {
            long[] r = x.data.clone();
            for (int p = 0; p < r.length; p++) {
                if (keep[p % (rows * cols)] == 0f) {
                    r[p] = 0;
                }
            }
            return V.host(x.type, x.shape, r);
        }
        return V.of(fl(x).mul(Var.of(tensor(new Graph.Raw(Graph.FLOAT, new int[] {rows, cols}, keep, null)))));
    }
}
