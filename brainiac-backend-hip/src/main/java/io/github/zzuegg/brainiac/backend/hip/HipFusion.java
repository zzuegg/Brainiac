package io.github.zzuegg.brainiac.backend.hip;

import io.github.zzuegg.brainiac.tensor.DType;

import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * Element-wise operations recorded instead of run, so that a chain of them becomes one kernel
 * generated as HIP source and compiled by hipRTC: the Vulkan backend's fusion, ported.
 *
 * <p>An element-wise result is not computed when it is asked for. Its storage keeps the
 * expression ({@link HipStorage#pending}), and later element-wise operations that read it build
 * on that expression. Only when something else needs the values (a matrix product, a
 * reduction, a download) is the whole expression turned into a kernel, compiled once per
 * distinct structure, and run as one launch. A view (a transpose, a slice, a broadcast) is part
 * of the expression too: its source is read in place through its strides, never copied, and a
 * matrix product reads a transposed view in place.
 *
 * <p>Pointers, shapes, strides, offsets and constants are not part of the source but passed in
 * a table, so one compiled kernel serves every batch size and every learning rate. The
 * generated code calls the same functions as the kernels in kernels.hip; 16-bit chains compute
 * in float and round once, when they are stored.
 */
final class HipFusion {

    /** Largest expression, in nodes, before inputs are computed on their own. */
    static final int MAX_NODES = 48;

    private HipFusion() {
    }

    /** A value per element of an index space; evaluated at one linear index. */
    sealed interface Expr permits Src, Const, Random, Un, Act, ActBack, Bin {
    }

    /**
     * Reads {@code storage} at {@code offset + sum(index_d * strides_d)}, where {@code index} is
     * the linear index decomposed over {@code dims} (size-1 axes dropped, mergeable axes merged).
     * A pending storage is inlined. {@code direct}: element for element, the whole storage.
     */
    record Src(HipStorage storage, int offset, int[] dims, int[] strides, boolean direct) implements Expr {

        /** Whether the read index is the chain's own index. */
        boolean identity() {
            return offset == 0 && dims.length == 1 && strides[0] == 1;
        }

        long count() {
            long n = 1;
            for (int d : dims) {
                n *= d;
            }
            return n;
        }
    }

    /** The same value everywhere (a filled tensor, such as a scalar operand). */
    record Const(float value) implements Expr {
    }

    /** Uniform values in {@code [0, 1)} from a seed and the element index (CounterRandom); float32. */
    record Random(long seed) implements Expr {
    }

    record Un(int op, Expr x) implements Expr {
    }

    record Act(int op, float param, Expr x) implements Expr {
    }

    record ActBack(int op, float param, Expr x, Expr grad) implements Expr {
    }

    record Bin(int op, Expr a, Expr b) implements Expr {
    }

    /** A source reading all of {@code s} in order. */
    static Src whole(HipStorage s) {
        return new Src(s, 0, new int[] {Math.toIntExact(s.size)}, new int[] {1}, true);
    }

    /**
     * A source reading {@code s} through strides over {@code shape}; null if an offset or a
     * stride does not fit in an int.
     */
    static Src strided(HipStorage s, long offset, int[] shape, long[] strides) {
        long n = 1;
        int[] dims = new int[Math.max(1, shape.length)];
        long[] st = new long[dims.length];
        int r = 0;
        for (int d = 0; d < shape.length; d++) {
            n *= shape[d];
            if (shape[d] == 1) {
                continue;
            }
            if (r > 0 && st[r - 1] == strides[d] * shape[d]) {
                dims[r - 1] *= shape[d];
                st[r - 1] = strides[d];
            } else {
                dims[r] = shape[d];
                st[r] = strides[d];
                r++;
            }
        }
        if (r == 0) {
            dims[0] = 1;
            st[0] = 0;
            r = 1;
        }
        if (offset < 0 || offset > Integer.MAX_VALUE || n > Integer.MAX_VALUE) {
            return null;
        }
        int[] is = new int[r];
        for (int d = 0; d < r; d++) {
            if (Math.abs(st[d]) > Integer.MAX_VALUE) {
                return null;
            }
            is[d] = (int) st[d];
        }
        int[] ds = java.util.Arrays.copyOf(dims, r);
        boolean direct = offset == 0 && n == s.size && (r == 1 && (is[0] == 1 || n == 1));
        if (direct && n == 1) {
            is[0] = 1;   // a single value: the identity read
        }
        return new Src(s, (int) offset, ds, is, direct);
    }

    /** Nodes in the expression, counting inlined pending inputs at every read. */
    static int nodes(Expr e) {
        return switch (e) {
            case Src s -> 1 + (s.storage().pending != null ? s.storage().pendingNodes : 0);
            case Const c -> 1;
            case Random r -> 1;
            case Un u -> 1 + nodes(u.x());
            case Act a -> 1 + nodes(a.x());
            case ActBack a -> 1 + nodes(a.x()) + nodes(a.grad());
            case Bin b -> 1 + nodes(b.a()) + nodes(b.b());
        };
    }

    /**
     * The size of the kernel an expression becomes: its operations and loads, with a pending
     * input counted once however often it is read (the kernel computes it once per index).
     */
    static int cost(Expr e) {
        return cost(e, Collections.newSetFromMap(new IdentityHashMap<>()));
    }

    private static int cost(Expr e, Set<HipStorage> seen) {
        return switch (e) {
            case Src s -> s.storage().pending == null ? 1
                    : seen.add(s.storage()) ? 1 + cost(s.storage().pending, seen) : 0;
            case Const c -> 1;
            case Random r -> 1;
            case Un u -> 1 + cost(u.x(), seen);
            case Act a -> 1 + cost(a.x(), seen);
            case ActBack a -> 1 + cost(a.x(), seen) + cost(a.grad(), seen);
            case Bin b -> 1 + cost(b.a(), seen) + cost(b.b(), seen);
        };
    }

    /** The sources an expression reads itself, not looking through pending inputs. */
    static List<Src> sources(Expr e) {
        List<Src> out = new ArrayList<>();
        sources(e, out);
        return out;
    }

    private static void sources(Expr e, List<Src> out) {
        switch (e) {
            case Src s -> out.add(s);
            case Const c -> {
            }
            case Random r -> {
            }
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

    /** Whether evaluating {@code e} reads {@code s}, looking through pending inputs. */
    static boolean reads(Expr e, HipStorage s) {
        Set<HipStorage> seen = Collections.newSetFromMap(new IdentityHashMap<>());
        List<Expr> todo = new ArrayList<>(List.of(e));
        while (!todo.isEmpty()) {
            for (Src src : sources(todo.removeLast())) {
                HipStorage st = src.storage();
                if (st == s) {
                    return true;
                }
                if (st.pending != null && seen.add(st)) {
                    todo.add(st.pending);
                }
            }
        }
        return false;
    }

    // ------------------------------------------------------------------ generation

    /**
     * A generated kernel: its source after kernels.hip's shared part, the storages it loads, and
     * the table ({@code struct Meta}, passed by value) it reads its pointers, constants and views from.
     */
    record Kernel(String source, List<HipStorage> loads, List<Object> pointers, List<Float> constants,
            List<Integer> words) {

        /** The table's size: pointers, then floats, then ints, padded to 8 bytes as C++ lays it out. */
        int metaBytes() {
            int bytes = 8 * Math.max(1, pointers.size()) + 4 * Math.max(1, constants.size())
                    + 4 * Math.max(1, words.size());
            return (bytes + 7) & ~7;
        }

        /** Writes the table at {@code base}; the loaded storages must be computed by now. */
        void writeMeta(MemorySegment b, long base) {
            long at = base;
            for (Object p : pointers) {
                b.set(JAVA_LONG, at, p instanceof HipStorage s ? s.address() : (Long) p);
                at += 8;
            }
            at = base + 8L * Math.max(1, pointers.size());
            for (Float c : constants) {
                b.set(JAVA_FLOAT, at, c);
                at += 4;
            }
            at = base + 8L * Math.max(1, pointers.size()) + 4L * Math.max(1, constants.size());
            for (Integer w : words) {
                b.set(JAVA_INT, at, w);
                at += 4;
            }
        }
    }

    /** Generates the kernel computing {@code root} at every index of the output. */
    static Kernel generate(Expr root, DType type) {
        return generate(root, type, 0, 0);
    }

    /**
     * Generates the kernel computing {@code root}, storing index {@code i} at
     * {@code (i / inner) * outerStride + i % inner} of the output: a write into rows of a
     * larger array, such as a step of a key/value cache. An {@code inner} of 0 stores at {@code i}.
     */
    static Kernel generate(Expr root, DType type, int inner, int outerStride) {
        Gen g = new Gen(type);
        String v = g.eval(root, "i0");
        String at = "i0";
        if (inner > 0) {
            int w = g.words.size();
            g.words.add(inner);
            g.words.add(outerStride);
            at = "(i0 / (unsigned) m.w[" + w + "]) * (unsigned) m.w[" + (w + 1) + "] + i0 % (unsigned) m.w[" + w + "]";
        }
        StringBuilder src = new StringBuilder();
        src.append("typedef ").append(format(type)).append(" D;\ntypedef D::A ACC;\n");
        src.append("struct Meta { const void* p[").append(Math.max(1, g.pointers.size())).append("]; float c[")
                .append(Math.max(1, g.constants.size())).append("]; int w[").append(Math.max(1, g.words.size()))
                .append("]; };\n");
        if (g.random) {
            src.append("""
                    // CounterRandom.uniform(seed, i), as random_values computes it
                    __device__ __forceinline__ float rnd(const void* s, unsigned i) {
                        unsigned long long z = (unsigned long long) s + ((unsigned long long) i + 1ull) * 0x9E3779B97F4A7C15ull;
                        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
                        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
                        z ^= z >> 31;
                        return (float) (unsigned) (z >> 40) * (1.0f / 16777216.0f);
                    }
                    """);
        }
        src.append("extern \"C\" __global__ void fused(D::T* out_, unsigned n, Meta m) {\n");
        src.append("  for (unsigned i0 = blockIdx.x * blockDim.x + threadIdx.x; i0 < n; i0 += gridDim.x * blockDim.x) {\n");
        src.append(g.body);
        src.append("    D::st(out_, ").append(at).append(", ").append(v).append(");\n  }\n}\n");
        return new Kernel(src.toString(), List.copyOf(g.loads), List.copyOf(g.pointers), List.copyOf(g.constants),
                List.copyOf(g.words));
    }

    /** The struct of kernels.hip that loads and stores a format. */
    private static String format(DType type) {
        return switch (type) {
            case FLOAT32 -> "F32";
            case FLOAT64 -> "F64";
            case FLOAT16 -> "F16";
            case BFLOAT16 -> "BF16";
            default -> throw new IllegalStateException("no fused kernels for " + type);
        };
    }

    private static final class Gen {
        final StringBuilder body = new StringBuilder();
        final List<Object> pointers = new ArrayList<>();
        final List<HipStorage> loads = new ArrayList<>();
        final Map<HipStorage, Integer> pointerIndex = new IdentityHashMap<>();
        final List<Float> constants = new ArrayList<>();
        final List<Integer> words = new ArrayList<>();
        /** The variable holding a storage's value at an index variable. */
        final Map<HipStorage, Map<String, String>> values = new IdentityHashMap<>();
        final Map<Const, String> consts = new IdentityHashMap<>();
        final Set<HipStorage> inlining = new LinkedHashSet<>();
        boolean random;
        int vars = 1;   // i0 is the chain's own index
        /** The kernel's format; sources in another one are converted as they are read. */
        final DType type;

        Gen(DType type) {
            this.type = type;
        }

        String fresh(String type, String prefix, String value) {
            String name = prefix + (vars++);
            body.append("    ").append(type).append(' ').append(name).append(" = ").append(value).append(";\n");
            return name;
        }

        String constant(float c) {
            constants.add(c);
            return "(ACC) m.c[" + (constants.size() - 1) + "]";
        }

        String eval(Expr e, String idx) {
            return switch (e) {
                case Const c -> consts.computeIfAbsent(c, k -> fresh("ACC", "v", constant(k.value())));
                case Random r -> {
                    random = true;
                    pointers.add(r.seed());
                    yield fresh("ACC", "v", "(ACC) rnd(m.p[" + (pointers.size() - 1) + "], " + idx + ")");
                }
                case Src s -> load(s, idx);
                case Un u -> fresh("ACC", "v", "unary_op<ACC>(" + u.op() + ", " + eval(u.x(), idx) + ")");
                case Act a -> {
                    String x = eval(a.x(), idx);
                    yield fresh("ACC", "v", "activation_op<ACC>(" + a.op() + ", " + constant(a.param()) + ", " + x + ")");
                }
                case ActBack a -> {
                    String x = eval(a.x(), idx);
                    String gr = eval(a.grad(), idx);
                    yield fresh("ACC", "v",
                            gr + " * activation_slope<ACC>(" + a.op() + ", " + constant(a.param()) + ", " + x + ")");
                }
                case Bin b -> {
                    String x = eval(b.a(), idx);
                    String y = eval(b.b(), idx);
                    yield fresh("ACC", "v", "binary_op<ACC>(" + b.op() + ", " + x + ", " + y + ")");
                }
            };
        }

        String load(Src s, String idx) {
            String j = index(s, idx);
            HipStorage st = s.storage();
            Map<String, String> at = values.computeIfAbsent(st, k -> new HashMap<>());
            String v = at.get(j);
            if (v != null) {
                return v;
            }
            if (st.pending != null) {
                if (!inlining.add(st)) {
                    throw new IllegalStateException("a pending tensor reads itself");
                }
                v = eval(st.pending, j);
                inlining.remove(st);
            } else {
                Integer k = pointerIndex.get(st);
                if (k == null) {
                    k = pointers.size();
                    pointers.add(st);
                    loads.add(st);
                    pointerIndex.put(st, k);
                }
                String f = format(st.dtype);
                v = fresh("ACC", "v", "(ACC) " + f + "::ld((const " + f + "::T*) m.p[" + k + "], " + j + ")");
            }
            at.put(j, v);
            return v;
        }

        /** The index variable for reading a source at the chain's index {@code idx}. */
        String index(Src s, String idx) {
            if (s.identity()) {
                return idx;
            }
            int[] dims = s.dims();
            int[] st = s.strides();
            int r = dims.length;
            int a = words.size();
            words.add(s.offset());
            if (r == 1 && st[0] == 1) {
                return fresh("unsigned", "i", idx + " + (unsigned) m.w[" + a + "]");
            }
            String name = "i" + (vars++);
            body.append("    unsigned ").append(name).append(";\n    {\n        unsigned r = ").append(idx)
                    .append(";\n        int j = m.w[").append(a).append("];\n");
            for (int d = r - 1; d >= 1; d--) {
                int w = words.size();
                words.add(dims[d]);
                words.add(st[d]);
                body.append("        { unsigned d = (unsigned) m.w[").append(w).append("]; j += (int) (r % d) * m.w[")
                        .append(w + 1).append("]; r /= d; }\n");
            }
            int w = words.size();
            words.add(st[0]);
            body.append("        j += (int) r * m.w[").append(w).append("];\n        ").append(name)
                    .append(" = (unsigned) j;\n    }\n");
            return name;
        }
    }
}
