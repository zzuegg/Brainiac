package io.github.zzuegg.brainiac.backend.opencl;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;

/**
 * Element-wise operations recorded instead of run, so that a chain of them becomes one
 * generated kernel.
 *
 * <p>An element-wise result is not computed when it is asked for. Its storage keeps the
 * expression ({@link ClStorage#pending}), and later element-wise operations that read it
 * build on that expression. Only when something else needs the values (a matmul, a
 * reduction, a download) is the whole expression turned into OpenCL C, compiled once per
 * distinct structure, and run as a single kernel. Shapes, strides and offsets are kernel
 * arguments, not part of the source, so one compiled kernel serves every batch size.
 *
 * <p>The generated code is plain OpenCL C 1.2 that calls the same element-wise functions as
 * the precompiled kernels, so results do not depend on the device or on whether fusion is
 * on. Only float32 and float64 are fused: they compute in their storage type, so keeping
 * intermediates in registers rounds exactly like storing them.
 */
final class Fused {

    /** Largest expression, in nodes, before inputs are computed on their own. */
    static final int MAX_NODES = 48;
    /** Kernel argument bytes allowed; OpenCL guarantees at least 1024. */
    static final int MAX_ARG_BYTES = 960;

    private Fused() {
    }

    /** A value per element of an index space; evaluated at one linear index. */
    sealed interface Expr permits Src, Un, Act, ActBack, Bin {
    }

    /**
     * Reads {@code storage} at {@code offset + sum(index_d * strides_d)}, where {@code index}
     * is the linear index decomposed over {@code dims}. A pending storage is inlined.
     */
    record Src(ClStorage storage, long offset, int[] dims, long[] strides, boolean direct) implements Expr {
    }

    record Un(int op, Expr x) implements Expr {
    }

    record Act(int op, float param, Expr x) implements Expr {
    }

    record ActBack(int op, float param, Expr x, Expr grad) implements Expr {
    }

    record Bin(int op, Expr a, Expr b) implements Expr {
    }

    /** A source reading all of {@code s} in order, as an index space of its own size. */
    static Src whole(ClStorage s) {
        return new Src(s, 0, new int[] {Math.toIntExact(s.size)}, new long[] {1}, true);
    }

    /** A source reading {@code s} through strides over {@code dims}. */
    static Src strided(ClStorage s, long offset, int[] dims, long[] strides) {
        long n = 1;
        for (int d : dims) {
            n *= d;
        }
        boolean direct = offset == 0 && n == s.size;
        long step = 1;
        for (int i = dims.length - 1; i >= 0 && direct; i--) {
            direct = dims[i] == 1 || strides[i] == step;
            step *= dims[i];
        }
        return new Src(s, offset, dims.clone(), strides.clone(), direct);
    }

    /** Nodes in the expression, counting inlined pending inputs. */
    static int nodes(Expr e) {
        return switch (e) {
            case Src s -> 1 + (s.storage().pending != null ? s.storage().pendingNodes : 0);
            case Un u -> 1 + nodes(u.x());
            case Act a -> 1 + nodes(a.x());
            case ActBack a -> 1 + nodes(a.x()) + nodes(a.grad());
            case Bin b -> 1 + nodes(b.a()) + nodes(b.b());
        };
    }

    /** The sources of an expression node, not looking into pending inputs. */
    static List<Src> sources(Expr e) {
        List<Src> out = new ArrayList<>();
        sources(e, out);
        return out;
    }

    private static void sources(Expr e, List<Src> out) {
        switch (e) {
            case Src s -> out.add(s);
            case Un u -> sources(u.x(), out);
            case Act a -> sources(a.x(), out);
            case ActBack a -> {
                sources(a.x(), out);
                sources(a.grad(), out);
            }
            case Bin b -> {
                sources(b.a(), out);
                sources(b.b(), out);
            }
        }
    }

    /** The direct sources of an expression node, not looking into pending inputs. */
    static List<ClStorage> inputs(Expr e) {
        List<ClStorage> out = new ArrayList<>();
        collect(e, out);
        return out;
    }

    private static void collect(Expr e, List<ClStorage> out) {
        switch (e) {
            case Src s -> out.add(s.storage());
            case Un u -> collect(u.x(), out);
            case Act a -> collect(a.x(), out);
            case ActBack a -> {
                collect(a.x(), out);
                collect(a.grad(), out);
            }
            case Bin b -> {
                collect(b.a(), out);
                collect(b.b(), out);
            }
        }
    }

    /** A generated kernel: its source and the arguments that follow {@code out, n}. */
    record Kernel(String source, List<Object> args, int argBytes, List<ClStorage> outputs) {
    }

    /** Generates the kernel computing {@code root} for every element of a result. */
    static Kernel generate(Expr root, boolean store) {
        Gen g = new Gen();
        g.root = store ? "i" : null;
        String v = g.eval(root, "i");
        StringBuilder src = new StringBuilder("__kernel void fused(__global T* out, long n");
        src.append(g.params);
        src.append(") {\n    long i = get_global_id(0);\n    if (i >= n) return;\n");
        src.append(g.body);
        src.append("    STORE(out, i, ").append(v).append(");\n}\n");
        return new Kernel(src.toString(), g.args, g.argBytes, g.outputs);
    }

    /**
     * Generates a kernel reducing {@code input} along one axis: per output, {@code len} values
     * {@code inner} apart. Operation codes are {@code ReduceOp} ordinals (sum, mean, max, min).
     * With {@code group > 1} a work-group of that many items shares each output (for long
     * rows); otherwise one work item loops over its row.
     */
    static Kernel generateReduce(Expr input, int op, int group) {
        Gen g = new Gen();
        g.argBytes += 16;
        String v = g.eval(input, "x");
        String init = op == 2 ? "(ACC) -INFINITY" : op == 3 ? "(ACC) INFINITY" : "(ACC) 0";
        String finish = op == 1 ? "acc / (ACC) len" : "acc";
        StringBuilder src = new StringBuilder();
        src.append("ACC combine_(ACC a, ACC b) {\n    return ")
                .append(op == 2 ? "isnan(a) || isnan(b) ? a + b : fmax(a, b)"
                        : op == 3 ? "isnan(a) || isnan(b) ? a + b : fmin(a, b)" : "a + b")
                .append(";\n}\n");
        src.append("__kernel void fused(__global T* out, long n, long len, long inner").append(g.params)
                .append(") {\n");
        if (group <= 1) {
            src.append("    long r = get_global_id(0);\n    if (r >= n) return;\n")
                    .append("    long base = (r / inner) * len * inner + r % inner;\n")
                    .append("    ACC acc = ").append(init).append(";\n")
                    .append("    for (long t = 0; t < len; t++) {\n    long x = base + t * inner;\n")
                    .append(g.body)
                    .append("    acc = combine_(acc, ").append(v).append(");\n    }\n")
                    .append("    STORE(out, r, ").append(finish).append(");\n}\n");
        } else {
            src.append("    __local ACC part[").append(group).append("];\n")
                    .append("    long r = get_group_id(0);\n    int lid = get_local_id(0);\n")
                    .append("    long base = (r / inner) * len * inner + r % inner;\n")
                    .append("    ACC acc = ").append(init).append(";\n")
                    .append("    for (long t = lid; t < len; t += ").append(group).append(") {\n")
                    .append("    long x = base + t * inner;\n")
                    .append(g.body)
                    .append("    acc = combine_(acc, ").append(v).append(");\n    }\n")
                    .append("    part[lid] = acc;\n    barrier(CLK_LOCAL_MEM_FENCE);\n")
                    .append("    for (int s = ").append(group / 2).append("; s > 0; s >>= 1) {\n")
                    .append("        if (lid < s) part[lid] = combine_(part[lid], part[lid + s]);\n")
                    .append("        barrier(CLK_LOCAL_MEM_FENCE);\n    }\n")
                    .append("    if (lid == 0) {\n        ACC acc = part[0];\n        STORE(out, r, ")
                    .append(finish).append(");\n    }\n}\n");
        }
        return new Kernel(src.toString(), g.args, g.argBytes, g.outputs);
    }

    private static final class Gen {
        final StringBuilder params = new StringBuilder();
        final StringBuilder body = new StringBuilder();
        final List<Object> args = new ArrayList<>();
        final Map<ClStorage, String> buffers = new IdentityHashMap<>();
        /** Pending inputs computed at the result's own index, written to their buffers too. */
        final List<ClStorage> outputs = new ArrayList<>();
        /** The index of the result, or null when the kernel writes nothing but its result. */
        String root;
        /** Values already computed in this kernel, by storage and index variable. */
        final Map<ClStorage, Map<String, String>> values = new IdentityHashMap<>();
        int argBytes = 16;
        int next;

        String eval(Expr e, String idx) {
            return switch (e) {
                case Src s -> source(s, idx);
                case Un u -> let("unary_op(" + u.op() + ", " + eval(u.x(), idx) + ")");
                case Act a -> let("activation_op(" + a.op() + ", " + literal(a.param()) + ", "
                        + eval(a.x(), idx) + ")");
                case ActBack a -> {
                    String x = eval(a.x(), idx);
                    String gr = eval(a.grad(), idx);
                    yield let(gr + " * activation_slope(" + a.op() + ", " + literal(a.param()) + ", " + x + ")");
                }
                case Bin b -> {
                    String x = eval(b.a(), idx);
                    String y = eval(b.b(), idx);
                    yield let("binary_op(" + b.op() + ", " + x + ", " + y + ")");
                }
            };
        }

        private String source(Src s, String idx) {
            String at = s.direct() ? idx : index(s, idx);
            ClStorage st = s.storage();
            Map<String, String> seen = values.computeIfAbsent(st, k -> new java.util.HashMap<>());
            String known = seen.get(at);
            if (known != null) {
                return known;
            }
            String v;
            if (st.pending != null) {
                v = eval(st.pending, at);
                if (at.equals(root)) {
                    // every element of the input is computed here exactly once: keep it
                    String w = "w" + outputs.size();
                    outputs.add(st);
                    params.append(", __global T* ").append(w);
                    args.add(st);
                    argBytes += 8;
                    body.append("    STORE(").append(w).append(", ").append(at).append(", ").append(v).append(");\n");
                }
            } else {
                String b = buffers.get(st);
                if (b == null) {
                    b = "b" + buffers.size();
                    buffers.put(st, b);
                    params.append(", __global const T* ").append(b);
                    args.add(st);
                    argBytes += 8;
                }
                v = let("LOAD(" + b + ", " + at + ")");
            }
            seen.put(at, v);
            return v;
        }

        private String index(Src s, String idx) {
            String j = "j" + next++;
            String off = arg(s.offset());
            body.append("    long ").append(j).append(" = ").append(off).append(";\n");
            List<Integer> kept = new ArrayList<>();
            for (int d = 0; d < s.dims().length; d++) {
                if (s.dims()[d] != 1) {
                    kept.add(d);
                }
            }
            if (!kept.isEmpty()) {
                String r = "r" + next++;
                body.append("    long ").append(r).append(" = ").append(idx).append(";\n");
                for (int k = kept.size() - 1; k >= 0; k--) {
                    int d = kept.get(k);
                    String dim = arg(s.dims()[d]);
                    String stride = arg(s.strides()[d]);
                    body.append("    ").append(j).append(" += (").append(r).append(" % ").append(dim)
                            .append(") * ").append(stride).append(";\n");
                    if (k > 0) {
                        body.append("    ").append(r).append(" /= ").append(dim).append(";\n");
                    }
                }
            }
            return j;
        }

        private String arg(long value) {
            String a = "a" + args.size();
            params.append(", long ").append(a);
            args.add(value);
            argBytes += 8;
            return a;
        }

        private String let(String expression) {
            String v = "v" + next++;
            body.append("    ACC ").append(v).append(" = ").append(expression).append(";\n");
            return v;
        }

        private static String literal(float f) {
            if (Float.isNaN(f)) {
                return "(ACC) NAN";
            }
            if (Float.isInfinite(f)) {
                return f > 0 ? "(ACC) INFINITY" : "(ACC) -INFINITY";
            }
            return "(ACC) " + Float.toHexString(f) + "f";
        }
    }
}
