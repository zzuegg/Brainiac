package io.github.zzuegg.brainiac.experimental.kernels;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * A small forward-graph language for graph rewriting, executed on core {@link Tensor}
 * operations. Nodes are inputs, constants (weights), {@code matmul}, broadcasting {@code add},
 * scalar {@code scale}, {@code concat} and {@code slice} along the last axis, a 2-D
 * {@code conv}, a per-channel {@code affine} (inference batch norm) and {@code relu}. The cost
 * of a graph ({@link #cost}) is its FLOPs plus a fixed overhead per executed kernel. Anything
 * computed from constants alone is free, since it can be folded ahead of time.
 */
public final class TensorExpressionGraph {

    /** Operations. */
    public enum Op {
        /** A named input. */
        INPUT,
        /** A constant tensor. */
        CONST,
        /** {@code [.., k] x [k, n]}. */
        MATMUL,
        /** Broadcasting addition. */
        ADD,
        /** Multiplication by the scalar payload. */
        SCALE,
        /** Concatenation along the last axis. */
        CONCAT,
        /** {@code [start, end)} of the last axis (payload {@code int[2]}). */
        SLICE,
        /** Convolution (payload {@code int[] {stride, pad}}). */
        CONV,
        /** {@code x * scale + shift} per channel of axis 1. */
        AFFINE,
        /** Rectifier. */
        RELU
    }

    /**
     * A node.
     *
     * @param op the operation
     * @param inputs the input node ids
     * @param payload the input name, constant tensor, scalar, slice range or conv settings
     * @param shape the output shape
     */
    public record Node(Op op, int[] inputs, Object payload, Shape shape) {
        @Override
        public boolean equals(Object o) {
            return o instanceof Node n && n.op == op && Arrays.equals(n.inputs, inputs)
                    && (op == Op.CONST ? n.payload == payload : Objects.deepEquals(n.payload, payload));
        }

        @Override
        public int hashCode() {
            return Objects.hash(op, Arrays.hashCode(inputs),
                    op == Op.CONST ? System.identityHashCode(payload)
                            : payload instanceof int[] a ? Arrays.hashCode(a) : Objects.hashCode(payload));
        }

        @Override
        public String toString() {
            return op + Arrays.toString(inputs) + (op == Op.INPUT || op == Op.SCALE ? "(" + payload + ")" : "");
        }
    }

    /** FLOP-equivalents charged per executed (non-constant) kernel. */
    public static final double KERNEL_OVERHEAD = 4096;

    private final List<Node> nodes = new ArrayList<>();
    private final List<Integer> outputs = new ArrayList<>();

    /** An empty graph. */
    public TensorExpressionGraph() {
        // nodes are added by the builder methods
    }

    /**
     * @param n a node
     * @return its id
     */
    public int add(Node n) {
        nodes.add(n);
        return nodes.size() - 1;
    }

    /**
     * @param name the name
     * @param shape the shape
     * @return the id
     */
    public int input(String name, Shape shape) {
        return add(new Node(Op.INPUT, new int[0], name, shape));
    }

    /**
     * @param value the tensor
     * @return the id
     */
    public int constant(Tensor value) {
        return add(new Node(Op.CONST, new int[0], value, value.shape()));
    }

    /**
     * @param op the operation
     * @param payload its payload or null
     * @param inputs the inputs
     * @return the id
     */
    public int op(Op op, Object payload, int... inputs) {
        Shape[] shapes = new Shape[inputs.length];
        for (int i = 0; i < inputs.length; i++) {
            shapes[i] = nodes.get(inputs[i]).shape();
        }
        return add(new Node(op, inputs, payload, shapeOf(op, payload, shapes)));
    }

    /**
     * Output shape inference.
     *
     * @param op the operation
     * @param payload its payload
     * @param in the input shapes
     * @return the output shape
     */
    public static Shape shapeOf(Op op, Object payload, Shape... in) {
        return switch (op) {
            case MATMUL -> {
                int[] d = in[0].toArray();
                d[d.length - 1] = in[1].dim(1);
                yield Shape.of(d);
            }
            case ADD -> in[0].size() >= in[1].size() ? in[0] : in[1];
            case SCALE, RELU, AFFINE -> in[0];
            case CONCAT -> {
                int[] d = in[0].toArray();
                int last = 0;
                for (Shape s : in) {
                    last += s.dim(-1);
                }
                d[d.length - 1] = last;
                yield Shape.of(d);
            }
            case SLICE -> {
                int[] r = (int[]) payload;
                int[] d = in[0].toArray();
                d[d.length - 1] = r[1] - r[0];
                yield Shape.of(d);
            }
            case CONV -> {
                int[] sp = (int[]) payload;
                int k = in[1].dim(2);
                int h = (in[0].dim(2) + 2 * sp[1] - k) / sp[0] + 1;
                int w = (in[0].dim(3) + 2 * sp[1] - k) / sp[0] + 1;
                yield Shape.of(in[0].dim(0), in[1].dim(0), h, w);
            }
            default -> throw new IllegalArgumentException(op + " has no inferred shape");
        };
    }

    /** @param id a node to return from {@link #evaluate} */
    public void output(int id) {
        outputs.add(id);
    }

    /** @return the nodes */
    public List<Node> nodes() {
        return List.copyOf(nodes);
    }

    /** @return the output ids */
    public List<Integer> outputs() {
        return List.copyOf(outputs);
    }

    /**
     * Applies one operation to values.
     *
     * @param n the node (for op and payload)
     * @param in the input values
     * @return the result
     */
    public static Tensor apply(Node n, Tensor... in) {
        return switch (n.op()) {
            case INPUT -> throw new IllegalArgumentException("inputs have no operation");
            case CONST -> (Tensor) n.payload();
            case MATMUL -> in[0].matmul(in[1]);
            case ADD -> in[0].add(in[1]);
            case SCALE -> in[0].mul(((Number) n.payload()).floatValue());
            case CONCAT -> Tensor.concat(in[0].rank() - 1, in);
            case SLICE -> {
                int[] r = (int[]) n.payload();
                yield in[0].slice(in[0].rank() - 1, r[0], r[1]);
            }
            case CONV -> {
                int[] sp = (int[]) n.payload();
                yield Var.noGrad(() -> Functional.conv(Var.of(in[0]), Var.of(in[1]), null, new int[] {sp[0], sp[0]},
                        new int[] {sp[1], sp[1]}, new int[] {sp[1], sp[1]}, new int[] {1, 1}, 1).value());
            }
            case AFFINE -> {
                int c = in[1].shape().size() > 0 ? (int) in[1].shape().size() : 1;
                yield in[0].mul(in[1].reshape(1, c, 1, 1)).add(in[2].reshape(1, c, 1, 1));
            }
            case RELU -> in[0].relu();
        };
    }

    /**
     * Runs the graph.
     *
     * @param inputs values by input name
     * @return the outputs
     */
    public List<Tensor> evaluate(Map<String, Tensor> inputs) {
        Map<Integer, Tensor> memo = new HashMap<>();
        List<Tensor> out = new ArrayList<>();
        for (int o : outputs) {
            out.add(eval(o, inputs, memo));
        }
        return out;
    }

    private Tensor eval(int id, Map<String, Tensor> inputs, Map<Integer, Tensor> memo) {
        Tensor t = memo.get(id);
        if (t != null) {
            return t;
        }
        Node n = nodes.get(id);
        if (n.op() == Op.INPUT) {
            t = inputs.get((String) n.payload());
        } else {
            Tensor[] in = new Tensor[n.inputs().length];
            for (int i = 0; i < in.length; i++) {
                in[i] = eval(n.inputs()[i], inputs, memo);
            }
            t = apply(n, in);
        }
        memo.put(id, t);
        return t;
    }

    /**
     * Folds every node whose inputs are all constants into a constant, so that
     * {@link #evaluate} runs only the kernels that {@link #cost} counts.
     *
     * @return the folded graph
     */
    public TensorExpressionGraph folded() {
        TensorExpressionGraph g = new TensorExpressionGraph();
        Map<Integer, Integer> map = new HashMap<>();
        for (int o : outputs) {
            g.output(copyFolded(o, g, map));
        }
        return g;
    }

    private int copyFolded(int id, TensorExpressionGraph g, Map<Integer, Integer> map) {
        Integer done = map.get(id);
        if (done != null) {
            return done;
        }
        Node n = nodes.get(id);
        int[] in = new int[n.inputs().length];
        boolean allConst = n.op() != Op.INPUT;
        for (int i = 0; i < in.length; i++) {
            in[i] = copyFolded(n.inputs()[i], g, map);
            allConst &= g.nodes.get(in[i]).op() == Op.CONST;
        }
        int r;
        if (allConst && n.op() != Op.CONST) {
            Tensor[] v = new Tensor[in.length];
            for (int i = 0; i < in.length; i++) {
                v[i] = (Tensor) g.nodes.get(in[i]).payload();
            }
            r = g.constant(apply(n, v));
        } else {
            r = g.add(new Node(n.op(), in, n.payload(), n.shape()));
        }
        map.put(id, r);
        return r;
    }

    /**
     * FLOPs of one node with these input shapes.
     *
     * @param op the operation
     * @param out the output shape
     * @param in the input shapes
     * @return the FLOPs
     */
    public static double flops(Op op, Shape out, Shape... in) {
        return switch (op) {
            case INPUT, CONST, SLICE, CONCAT -> 0;
            case MATMUL -> 2.0 * out.size() * in[0].dim(-1);
            case ADD, SCALE, RELU -> out.size();
            case AFFINE -> 2.0 * out.size();
            case CONV -> 2.0 * out.size() * in[1].dim(1) * in[1].dim(2) * in[1].dim(3);
        };
    }

    /** @return FLOPs of the non-constant nodes reachable from the outputs */
    public double flops() {
        return measure(false);
    }

    /** @return {@link #flops()} plus {@link #KERNEL_OVERHEAD} per non-constant kernel */
    public double cost() {
        return measure(true);
    }

    /** @return the number of non-constant kernels reachable from the outputs */
    public int kernels() {
        Set<Integer> seen = new HashSet<>();
        Set<Integer> constant = new HashSet<>();
        for (int o : outputs) {
            visit(o, seen, constant);
        }
        int k = 0;
        for (int id : seen) {
            Node n = nodes.get(id);
            if (!constant.contains(id) && n.op() != Op.INPUT && n.op() != Op.SLICE) {
                k++;
            }
        }
        return k;
    }

    private double measure(boolean overhead) {
        Set<Integer> seen = new HashSet<>();
        Set<Integer> constant = new HashSet<>();
        for (int o : outputs) {
            visit(o, seen, constant);
        }
        double total = 0;
        for (int id : seen) {
            Node n = nodes.get(id);
            if (constant.contains(id) || n.op() == Op.INPUT) {
                continue;
            }
            Shape[] in = new Shape[n.inputs().length];
            for (int i = 0; i < in.length; i++) {
                in[i] = nodes.get(n.inputs()[i]).shape();
            }
            total += flops(n.op(), n.shape(), in) + (overhead && n.op() != Op.SLICE ? KERNEL_OVERHEAD : 0);
        }
        return total;
    }

    private boolean visit(int id, Set<Integer> seen, Set<Integer> constant) {
        if (!seen.add(id)) {
            return constant.contains(id);
        }
        Node n = nodes.get(id);
        boolean c = n.op() != Op.INPUT;
        for (int i : n.inputs()) {
            c &= visit(i, seen, constant);
        }
        if (c) {
            constant.add(id);
        }
        return c;
    }

    @Override
    public String toString() {
        return "TensorExpressionGraph(" + nodes.size() + " nodes, " + outputs.size() + " outputs)";
    }
}
