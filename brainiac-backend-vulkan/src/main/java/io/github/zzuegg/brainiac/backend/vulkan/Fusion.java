package io.github.zzuegg.brainiac.backend.vulkan;

import io.github.zzuegg.brainiac.tensor.spi.ActivationOp;
import io.github.zzuegg.brainiac.tensor.spi.BinaryOp;
import io.github.zzuegg.brainiac.tensor.spi.UnaryOp;

import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Element-wise operations recorded instead of run, so that a chain of them becomes one
 * generated compute shader.
 *
 * <p>An element-wise result is not computed when it is asked for. Its storage keeps the
 * expression ({@link VkStorage#pending}), and later element-wise operations that read it build
 * on that expression. Only when something else needs the values (a matmul, a reduction, a
 * download) is the whole expression turned into GLSL, compiled once per distinct structure,
 * and run as one dispatch. A view (a transpose, a slice, a broadcast) is part of the
 * expression too: its source is read in place through its strides, never copied.
 *
 * <p>Pointers, shapes, strides, offsets and constants are not part of the source but read from
 * a small table by device address, so one compiled shader serves every batch size and every
 * learning rate. The generated code calls the same functions as the precompiled shaders, and
 * only float32 and float64 are fused: they compute in their storage type, so keeping
 * intermediates in registers rounds like storing them.
 */
final class Fusion {

    /** Largest expression, in nodes, before inputs are computed on their own. */
    static final int MAX_NODES = 48;

    private Fusion() {
    }

    /** A value per element of an index space; evaluated at one linear index. */
    sealed interface Expr permits Src, Const, Random, Un, Act, ActBack, Bin {
    }

    /**
     * Reads {@code storage} at {@code offset + sum(index_d * strides_d)}, where {@code index} is
     * the linear index decomposed over {@code dims} (size-1 axes dropped, mergeable axes merged).
     * A pending storage is inlined. {@code direct}: element for element, the whole storage.
     */
    record Src(VkStorage storage, int offset, int[] dims, int[] strides, boolean direct) implements Expr {

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
    static Src whole(VkStorage s) {
        return new Src(s, 0, new int[] {Math.toIntExact(s.size)}, new int[] {1}, true);
    }

    /**
     * A source reading {@code s} through strides over {@code shape}; null if an offset or a
     * stride does not fit in an int.
     */
    static Src strided(VkStorage s, long offset, int[] shape, long[] strides) {
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
        if (offset < 0 || offset > Integer.MAX_VALUE) {
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
     * The size of the shader an expression becomes: its operations and loads, with a pending
     * input counted once however often it is read (the shader computes it once per index).
     */
    static int cost(Expr e) {
        return cost(e, Collections.newSetFromMap(new IdentityHashMap<>()));
    }

    private static int cost(Expr e, Set<VkStorage> seen) {
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

    /**
     * The functions built on the shaders' own exp and log (exp, log, tanh, sigmoid, erf, pow and
     * the smooth activations) in an expression, a pending input counted once. AMD's LLPC compiler
     * crashes on float64 shaders that combine several, so float64 chains hold at most one.
     */
    private static final Set<UnaryOp> TRANSCENDENTAL = java.util.EnumSet.of(UnaryOp.EXP, UnaryOp.LOG, UnaryOp.TANH,
            UnaryOp.SIGMOID, UnaryOp.ERF, UnaryOp.SOFTPLUS);

    static int transcendentals(Expr e) {
        return transcendentals(e, Collections.newSetFromMap(new IdentityHashMap<>()));
    }

    private static int transcendentals(Expr e, Set<VkStorage> seen) {
        return switch (e) {
            case Src s -> s.storage().pending != null && seen.add(s.storage())
                    ? transcendentals(s.storage().pending, seen) : 0;
            case Const c -> 0;
            case Random r -> 0;
            case Un u -> (TRANSCENDENTAL.contains(UnaryOp.values()[u.op()]) ? 1 : 0) + transcendentals(u.x(), seen);
            case Act a -> (ActivationOp.values()[a.op()].transcendental() ? 1 : 0) + transcendentals(a.x(), seen);
            case ActBack a -> (ActivationOp.values()[a.op()].transcendental() ? 1 : 0) + transcendentals(a.x(), seen)
                    + transcendentals(a.grad(), seen);
            case Bin b -> (b.op() == BinaryOp.POW.ordinal() ? 1 : 0) + transcendentals(b.a(), seen)
                    + transcendentals(b.b(), seen);
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
    static boolean reads(Expr e, VkStorage s) {
        Set<VkStorage> seen = Collections.newSetFromMap(new IdentityHashMap<>());
        List<Expr> todo = new ArrayList<>(List.of(e));
        while (!todo.isEmpty()) {
            for (Src src : sources(todo.removeLast())) {
                VkStorage st = src.storage();
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
     * A generated shader: its GLSL after the shared prelude, the storages it loads (whose slots
     * it reads), and the table it reads its pointers, constants and views from.
     */
    record Kernel(String source, List<VkStorage> loads, List<Object> pointers, List<Float> constants,
            List<Integer> words) {

        /** The table's size: pointers, then floats, then ints (std430, as the shader declares it). */
        int metaBytes() {
            return 8 * Math.max(1, pointers.size()) + 4 * Math.max(1, constants.size())
                    + 4 * Math.max(1, words.size());
        }

        /** Writes the table; the loaded storages must be stored by now. */
        void writeMeta(ByteBuffer b) {
            int at = 0;
            for (Object p : pointers) {
                b.putLong(at, p instanceof VkStorage s ? s.address() : (Long) p);
                at += 8;
            }
            at = 8 * Math.max(1, pointers.size());
            for (Float c : constants) {
                b.putFloat(at, c);
                at += 4;
            }
            at = 8 * Math.max(1, pointers.size()) + 4 * Math.max(1, constants.size());
            for (Integer w : words) {
                b.putInt(at, w);
                at += 4;
            }
        }
    }

    /** Generates the shader computing {@code root} at every index of the output. */
    static Kernel generate(Expr root) {
        Gen g = new Gen();
        String v = g.eval(root, "i0");
        StringBuilder src = new StringBuilder();
        src.append("layout(local_size_x = GROUP) in;\n");
        src.append("layout(buffer_reference, std430, buffer_reference_align = 8) restrict readonly buffer Meta {"
                + " uint64_t p[").append(Math.max(1, g.pointers.size())).append("]; float c[")
                .append(Math.max(1, g.constants.size())).append("]; int w[").append(Math.max(1, g.words.size()))
                .append("]; };\n");
        src.append("layout(push_constant) uniform P { Meta m; Buf out_; uint n; } pc;\n");
        if (g.random) {
            src.append("""
                    // CounterRandom.uniform(seed, i), as random.comp computes it
                    float rnd(uint64_t seed, uint i) {
                        uint64_t z = seed + (uint64_t(i) + 1ul) * 0x9E3779B97F4A7C15ul;
                        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ul;
                        z = (z ^ (z >> 27)) * 0x94D049BB133111EBul;
                        z ^= z >> 31;
                        return float(uint(z >> 40)) * (1.0 / 16777216.0);
                    }
                    """);
        }
        src.append("void main() {\n    uint i0 = gid();\n    if (i0 >= pc.n) return;\n    Meta m = pc.m;\n");
        src.append(g.body);
        src.append("    STORE(pc.out_, i0, ").append(v).append(");\n}\n");
        return new Kernel(src.toString(), List.copyOf(g.loads), List.copyOf(g.pointers), List.copyOf(g.constants),
                List.copyOf(g.words));
    }

    private static final class Gen {
        final StringBuilder body = new StringBuilder();
        final List<Object> pointers = new ArrayList<>();
        final List<VkStorage> loads = new ArrayList<>();
        final Map<VkStorage, Integer> pointerIndex = new IdentityHashMap<>();
        final List<Float> constants = new ArrayList<>();
        final List<Integer> words = new ArrayList<>();
        /** The variable holding a storage's value at an index variable. */
        final Map<VkStorage, Map<String, String>> values = new IdentityHashMap<>();
        final Map<Const, String> consts = new IdentityHashMap<>();
        final Set<VkStorage> inlining = new LinkedHashSet<>();
        boolean random;
        int vars = 1;   // i0 is the chain's own index

        String fresh(String type, String prefix, String value) {
            String name = prefix + (vars++);
            body.append("    ").append(type).append(' ').append(name).append(" = ").append(value).append(";\n");
            return name;
        }

        String constant(float c) {
            constants.add(c);
            return "ACC(m.c[" + (constants.size() - 1) + "])";
        }

        String eval(Expr e, String idx) {
            return switch (e) {
                case Const c -> consts.computeIfAbsent(c, k -> fresh("ACC", "v", constant(k.value())));
                case Random r -> {
                    random = true;
                    pointers.add(r.seed());
                    yield fresh("ACC", "v", "ACC(rnd(m.p[" + (pointers.size() - 1) + "], " + idx + "))");
                }
                case Src s -> load(s, idx);
                case Un u -> fresh("ACC", "v", "unary_op(" + u.op() + "u, " + eval(u.x(), idx) + ")");
                case Act a -> {
                    String x = eval(a.x(), idx);
                    yield fresh("ACC", "v", "activation_op(" + a.op() + "u, " + constant(a.param()) + ", " + x + ")");
                }
                case ActBack a -> {
                    String x = eval(a.x(), idx);
                    String gr = eval(a.grad(), idx);
                    yield fresh("ACC", "v",
                            gr + " * activation_slope(" + a.op() + "u, " + constant(a.param()) + ", " + x + ")");
                }
                case Bin b -> {
                    String x = eval(b.a(), idx);
                    String y = eval(b.b(), idx);
                    yield fresh("ACC", "v", "binary_op(" + b.op() + "u, " + x + ", " + y + ")");
                }
            };
        }

        String load(Src s, String idx) {
            String j = index(s, idx);
            VkStorage st = s.storage();
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
                v = fresh("ACC", "v", "LOAD(In(m.p[" + k + "]), " + j + ")");
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
                return fresh("uint", "i", idx + " + uint(m.w[" + a + "])");
            }
            String name = "i" + (vars++);
            body.append("    uint ").append(name).append(";\n    {\n        uint r = ").append(idx)
                    .append(";\n        int j = m.w[").append(a).append("];\n");
            for (int d = r - 1; d >= 1; d--) {
                int w = words.size();
                words.add(dims[d]);
                words.add(st[d]);
                body.append("        { uint d = uint(m.w[").append(w).append("]); j += int(r % d) * m.w[")
                        .append(w + 1).append("]; r /= d; }\n");
            }
            int w = words.size();
            words.add(st[0]);
            body.append("        j += int(r) * m.w[").append(w).append("];\n        ").append(name)
                    .append(" = uint(j);\n    }\n");
            return name;
        }
    }
}
