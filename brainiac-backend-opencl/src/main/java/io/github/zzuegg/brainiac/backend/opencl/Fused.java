package io.github.zzuegg.brainiac.backend.opencl;

import java.util.ArrayList;
import java.util.Collections;
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
    sealed interface Expr permits Src, Const, Un, Act, ActBack, Bin, MatMul, Reduce, Im2col, Col2im, Random {
    }

    /**
     * A reduction not yet run: {@code ReduceOp} ordinal {@code op} over {@code x}, per output
     * {@code length} values {@code inner} apart. Like a matrix product it is only ever the
     * value of a storage; when one is needed, the pending reductions of the same shape run
     * together in one kernel.
     */
    record Reduce(int op, ClStorage x, long outer, long length, long inner) implements Expr {
    }

    /**
     * The same value everywhere (a filled tensor, such as a scalar operand). It is a kernel
     * argument rather than part of the source, so a value that changes every step (a
     * learning rate, a bias correction) neither needs a buffer nor a new kernel.
     */
    record Const(float value) implements Expr {
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

    /**
     * A matrix product not yet run: the precompiled kernel's arguments after the output and
     * its work sizes. Only ever the value of a storage, never part of a larger expression;
     * an element-wise chain on it becomes the matmul kernel's epilogue. A non-null {@code a}
     * or {@code b} is an operand left pending (an unrolled image) for the kernel to compute.
     */
    record MatMul(Object[] args, long[] global, long[] local, ClStorage a, ClStorage b, int w) implements Expr {
    }

    /**
     * Every window of an image unrolled into a row ({@code [batch * positions, channels * ks]}):
     * the value of {@code x} at the cell {@code table[p * ks + k]} names, or {@code pad} where it
     * names none (-1). {@code owner} keeps the device table alive while this is pending.
     */
    record Im2col(Src x, java.lang.foreign.MemorySegment table, Object owner, int channels, int positions, int ks,
            int img, float pad) implements Expr {
    }

    /**
     * The inverse of {@link Im2col}: per image cell, the sum of the unrolled values that read it,
     * listed by {@code offsets} and {@code entries} (the {@code p * ks + k} of each).
     */
    record Col2im(Src cols, java.lang.foreign.MemorySegment offsets, java.lang.foreign.MemorySegment entries,
            Object owner, int channels, int positions, int ks, int img) implements Expr {
    }

    /** Uniform values in {@code [0, 1)} from a seed and the element index (CounterRandom). */
    record Random(long seed) implements Expr {
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
            case MatMul m -> 1;
            case Const c -> 1;
            case Random x -> 1;
            case Reduce r -> 1;
            case Im2col c -> 1 + nodes(c.x());
            case Col2im c -> 1 + nodes(c.cols());
            case Un u -> 1 + nodes(u.x());
            case Act a -> 1 + nodes(a.x());
            case ActBack a -> 1 + nodes(a.x()) + nodes(a.grad());
            case Bin b -> 1 + nodes(b.a()) + nodes(b.b());
        };
    }

    /**
     * The size of the kernel an expression becomes: its operations and the loads of its
     * stored inputs. A pending input is inlined, and counts once however often it is read,
     * as the kernel computes it once per index (such as the logits of a softmax over a few
     * candidates, each read by the maximum, the exponential and the sum). {@link #nodes}
     * counts every read, which the scheduling heuristics are tuned to.
     */
    static int cost(Expr e) {
        return cost(e, Collections.newSetFromMap(new IdentityHashMap<>()));
    }

    private static int cost(Expr e, java.util.Set<ClStorage> seen) {
        return switch (e) {
            case Src s -> s.storage().pending == null ? 1
                    : seen.add(s.storage()) ? cost(s.storage().pending, seen) : 0;
            case MatMul m -> 1;
            case Const c -> 1;
            case Random x -> 1;
            case Reduce r -> 1;
            case Im2col c -> 1 + cost(c.x(), seen);
            case Col2im c -> 1 + cost(c.cols(), seen);
            case Un u -> 1 + cost(u.x(), seen);
            case Act a -> 1 + cost(a.x(), seen);
            case ActBack a -> 1 + cost(a.x(), seen) + cost(a.grad(), seen);
            case Bin b -> 1 + cost(b.a(), seen) + cost(b.b(), seen);
        };
    }

    /**
     * The pending matrix products a chain reads, looking through pending inputs, each with
     * whether every read of it is element for element at the chain's own index.
     */
    static Map<ClStorage, Boolean> matmuls(Expr e) {
        Map<ClStorage, Boolean> out = new java.util.LinkedHashMap<>();   // identity, in reading order
        matmuls(e, true, out);
        return out;
    }

    private static void matmuls(Expr e, boolean direct, Map<ClStorage, Boolean> out) {
        switch (e) {
            case Src s -> {
                ClStorage st = s.storage();
                boolean d = direct && s.direct();
                if (st.pending instanceof MatMul) {
                    out.merge(st, d, Boolean::logicalAnd);
                } else if (st.pending != null) {
                    matmuls(st.pending, d, out);
                }
            }
            case MatMul m -> {
            }
            case Const c -> {
            }
            case Random x -> {
            }
            case Reduce r -> {
            }
            case Im2col c -> matmuls(c.x(), false, out);
            case Col2im c -> matmuls(c.cols(), false, out);
            case Un u -> matmuls(u.x(), direct, out);
            case Act a -> matmuls(a.x(), direct, out);
            case ActBack a -> {
                matmuls(a.x(), direct, out);
                matmuls(a.grad(), direct, out);
            }
            case Bin b -> {
                matmuls(b.a(), direct, out);
                matmuls(b.b(), direct, out);
            }
        }
    }

    /**
     * Generates a matmul kernel (the precompiled one's tiling and order) that applies
     * {@code root} to each product before storing it: the product is stored to {@code base}'s
     * buffer and the chain's value to the result. With {@code root} null the product is only
     * stored to the result. A non-null {@code a} or {@code b} is the pending value of that
     * operand, computed where the kernel loads it instead of read from its buffer (an unrolled
     * image, so the unrolled copy is never stored). Its layout is 1 when the operand's rows and
     * columns are the image's, 2 when transposed, 0 otherwise (read by linear index). Arguments after {@code out, total} are the
     * matmul's, then {@link Kernel#args}.
     */
    static Kernel generateMatMul(Expr root, ClStorage base, boolean store, Expr a, int aLayout, Expr b,
            int bLayout) {
        return generateMatMul(root, base, store, a, aLayout, b, bLayout, 0);
    }

    /**
     * As above; with {@code w} 4 or 2, in the tiling of the precompiled {@code matmul_big} or
     * {@code matmul_mid}: 16 x 16 work items each computing w x w outputs of a 16w x 16w tile.
     */
    static Kernel generateMatMul(Expr root, ClStorage base, boolean store, Expr a, int aLayout, Expr b,
            int bLayout, int w) {
        if (w > 0) {
            return generateTiledMatMul(root, base, store, a, aLayout, b, bLayout, w);
        }
        Gen g = new Gen();
        g.root = store ? "i" : null;
        g.base = base;
        g.argBytes += 8 * 3 + 4 * 3 + 8 * 9 + 4 * 2;
        String loadA = g.load(a, "ma", "row < M && ac < K", "aBase + (long) row * aRow + (long) ac * aCol",
                aLayout == 1 ? "row" : aLayout == 2 ? "ac" : null, aLayout == 1 ? "ac" : "row");
        String loadB = g.load(b, "mb", "br < K && col < N", "bBase + (long) br * bRow + (long) col * bCol",
                bLayout == 1 ? "br" : bLayout == 2 ? "col" : null, bLayout == 1 ? "col" : "br");
        String v = root == null ? null : g.eval(root, "i");
        String src = "__kernel void fused(__global T* out, long total, __global const T* ma, __global const T* mb,"
                + " __global T* mz, int M, int K, int N, long strideA, long strideB, long strideZ,"
                + " long aOff, long aRow, long aCol, long bOff, long bRow, long bCol, int splits, int chunk"
                + g.params + ") {\n"
                + "    int col = get_global_id(0);\n"
                + "    int row = get_global_id(1);\n"
                + "    long piece = get_global_id(2);\n"
                + "    long batch = piece / splits;\n"
                + "    int from = (int) (piece % splits) * chunk;\n"
                + "    int to = min(K, from + chunk);\n"
                + "    int lc = get_local_id(0);\n"
                + "    int lr = get_local_id(1);\n"
                + "    __local ACC tileA[TS][TS];\n"
                + "    __local ACC tileB[TS][TS];\n"
                + "    long aBase = aOff + batch * strideA;\n"
                + "    long bBase = bOff + batch * strideB;\n"
                + "    ACC acc = (ACC) 0;\n"
                + "    for (int t = from; t < to; t += TS) {\n"
                + "        int ac = t + lc;\n"
                + "        int br = t + lr;\n"
                + loadA.replace("TILE", "tileA[lr][lc]")
                + loadB.replace("TILE", "tileB[lr][lc]")
                + "        barrier(CLK_LOCAL_MEM_FENCE);\n"
                + "        for (int q = 0; q < TS; q++) {\n"
                + "            acc = fma(tileA[lr][q], tileB[q][lc], acc);\n"
                + "        }\n"
                + "        barrier(CLK_LOCAL_MEM_FENCE);\n"
                + "    }\n"
                + "    if (row < M && col < N) {\n"
                + "    long i = piece * strideZ + (long) row * N + col;\n"
                + (v == null ? "    STORE(out, i, acc);\n"
                        : "    STORE(mz, i, acc);\n" + g.body + "    STORE(out, i, " + v + ");\n")
                + "    }\n}\n";
        return new Kernel(src, g.args, g.argBytes, g.outputs);
    }

    private static Kernel generateTiledMatMul(Expr root, ClStorage base, boolean store, Expr a, int aLayout, Expr b,
            int bLayout, int w) {
        int tile = 16 * w;
        Gen g = new Gen();
        g.root = store ? "i" : null;
        g.base = base;
        g.argBytes += 8 * 3 + 4 * 3 + 8 * 9 + 4 * 2;
        String loadA = g.load(a, "ma", "row < M && ac < K", "aBase + (long) row * aRow + (long) ac * aCol",
                aLayout == 1 ? "row" : aLayout == 2 ? "ac" : null, aLayout == 1 ? "ac" : "row");
        String loadB = g.load(b, "mb", "br < K && col < N", "bBase + (long) br * bRow + (long) col * bCol",
                bLayout == 1 ? "br" : bLayout == 2 ? "col" : null, bLayout == 1 ? "col" : "br");
        String v = root == null ? null : g.eval(root, "i");
        String src = "__kernel void fused(__global T* out, long total, __global const T* ma, __global const T* mb,"
                + " __global T* mz, int M, int K, int N, long strideA, long strideB, long strideZ,"
                + " long aOff, long aRow, long aCol, long bOff, long bRow, long bCol, int splits, int chunk"
                + g.params + ") {\n"
                + "    int lc = get_local_id(0);\n"
                + "    int lr = get_local_id(1);\n"
                + "    int tid = lr * 16 + lc;\n"
                + "    int colBase = get_group_id(0) * " + tile + ";\n"
                + "    int rowBase = get_group_id(1) * " + tile + ";\n"
                + "    long piece = get_global_id(2);\n"
                + "    long batch = piece / splits;\n"
                + "    int from = (int) (piece % splits) * chunk;\n"
                + "    int to = min(K, from + chunk);\n"
                + "    __local ACC tileA[16][" + (tile + 1) + "];\n"
                + "    __local ACC tileB[16][" + tile + "];\n"
                + "    long aBase = aOff + batch * strideA;\n"
                + "    long bBase = bOff + batch * strideB;\n"
                + "    ACC accs[" + w + "][" + w + "];\n"
                + "    for (int wm = 0; wm < " + w + "; wm++) {\n"
                + "        for (int wn = 0; wn < " + w + "; wn++) {\n"
                + "            accs[wm][wn] = (ACC) 0;\n"
                + "        }\n"
                + "    }\n"
                + "    for (int t = from; t < to; t += 16) {\n"
                + "        for (int l = 0; l < " + w + "; l++) {\n"
                + "            int idx = l * 256 + tid;\n"
                + "            {\n"
                + "            int row = rowBase + (idx >> 4);\n"
                + "            int ac = t + (idx & 15);\n"
                + loadA.replace("TILE", "tileA[idx & 15][idx >> 4]")
                + "            }\n"
                + "            {\n"
                + "            int br = t + idx / " + tile + ";\n"
                + "            int col = colBase + idx % " + tile + ";\n"
                + loadB.replace("TILE", "tileB[idx / " + tile + "][idx % " + tile + "]")
                + "            }\n"
                + "        }\n"
                + "        barrier(CLK_LOCAL_MEM_FENCE);\n"
                + "        for (int q = 0; q < 16; q++) {\n"
                + "            ACC ra[" + w + "];\n"
                + "            ACC rb[" + w + "];\n"
                + "            for (int w = 0; w < " + w + "; w++) {\n"
                + "                ra[w] = tileA[q][lr + w * 16];\n"
                + "                rb[w] = tileB[q][lc + w * 16];\n"
                + "            }\n"
                + "            for (int wm = 0; wm < " + w + "; wm++) {\n"
                + "                for (int wn = 0; wn < " + w + "; wn++) {\n"
                + "                    accs[wm][wn] = fma(ra[wm], rb[wn], accs[wm][wn]);\n"
                + "                }\n"
                + "            }\n"
                + "        }\n"
                + "        barrier(CLK_LOCAL_MEM_FENCE);\n"
                + "    }\n"
                + "    for (int wm = 0; wm < " + w + "; wm++) {\n"
                + "    for (int wn = 0; wn < " + w + "; wn++) {\n"
                + "    int row = rowBase + lr + wm * 16;\n"
                + "    int col = colBase + lc + wn * 16;\n"
                + "    if (row < M && col < N) {\n"
                + "    ACC acc = accs[wm][wn];\n"
                + "    long i = piece * strideZ + (long) row * N + col;\n"
                + (v == null ? "    STORE(out, i, acc);\n"
                        : "    STORE(mz, i, acc);\n" + g.body + "    STORE(out, i, " + v + ");\n")
                + "    }\n    }\n    }\n}\n";
        return new Kernel(src, g.args, g.argBytes, g.outputs);
    }

    /**
     * Whether an expression sums unrolled values back into an image (a loop per element),
     * looking through pending inputs: worth computing once rather than at every read.
     */
    static boolean scatters(Expr e) {
        return switch (e) {
            case Col2im c -> true;
            case Src s -> s.storage().pending != null && !(s.storage().pending instanceof MatMul)
                    && !(s.storage().pending instanceof Reduce) && scatters(s.storage().pending);
            case MatMul m -> false;
            case Reduce r -> false;
            case Const c -> false;
            case Random x -> false;
            case Im2col c -> scatters(c.x());
            case Un u -> scatters(u.x());
            case Act x -> scatters(x.x());
            case ActBack x -> scatters(x.x()) || scatters(x.grad());
            case Bin x -> scatters(x.a()) || scatters(x.b());
        };
    }

    /** Whether an expression unrolls an image, looking through pending inputs. */
    static boolean unrolls(Expr e) {
        return switch (e) {
            case Im2col c -> true;
            case Col2im c -> false;
            case Src s -> s.storage().pending != null && !(s.storage().pending instanceof MatMul)
                    && !(s.storage().pending instanceof Reduce) && unrolls(s.storage().pending);
            case MatMul m -> false;
            case Reduce r -> false;
            case Const c -> false;
            case Random x -> false;
            case Un u -> unrolls(u.x());
            case Act x -> unrolls(x.x());
            case ActBack x -> unrolls(x.x()) || unrolls(x.grad());
            case Bin x -> unrolls(x.a()) || unrolls(x.b());
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
            case MatMul m -> {
            }
            case Const c -> {
            }
            case Random x -> {
            }
            case Reduce r -> {
            }
            case Im2col c -> out.add(c.x());
            case Col2im c -> out.add(c.cols());
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
            case MatMul m -> {
            }
            case Const c -> {
            }
            case Random x -> {
            }
            case Reduce r -> {
            }
            case Im2col c -> out.add(c.x().storage());
            case Col2im c -> out.add(c.cols().storage());
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
     * Generates one kernel computing several pending storages, each over its own index range
     * of a single launch: index {@code gi} in {@code [0, n)} falls into the range of one of
     * them. The first is written to {@code out}; inputs in {@code keep} that a chain computes
     * at its own index are written to their buffers too, so they need no range of their own.
     * Arguments after {@code out, n} are {@link Kernel#args}; {@link Kernel#outputs} lists every
     * storage the kernel writes besides the first.
     */
    static Many generateMany(List<ClStorage> roots, java.util.Set<ClStorage> keep, int maxArgBytes) {
        Gen g = new Gen();
        g.keep = keep;
        StringBuilder src = new StringBuilder();
        String start = null;
        long end = 0;
        boolean first = true;
        int used = 0;
        for (ClStorage r : roots) {
            if (g.outputs.contains(r) || r == g.first) {
                used++;
                continue;   // already written by an earlier range
            }
            // where to go back to if this range does not fit the arguments
            int params = g.params.length();
            int body = g.body.length();
            int args = g.args.size();
            int bytes = g.argBytes;
            int outputs = g.outputs.size();
            int buffers = g.bufferOrder.size();
            int constants = g.constantOrder.size();
            if (first) {
                g.first = r;
            }
            long rangeEnd = end + r.size;
            String bound = g.arg(rangeEnd);
            String idx = "i" + g.next++;
            g.body.append(first ? "    if (gi < " : "    else if (gi < ").append(bound).append(") {\n");
            g.body.append("    long ").append(idx).append(" = gi").append(start == null ? "" : " - " + start)
                    .append(";\n");
            g.root = idx;
            String v = g.eval(r.pending, idx);
            String o = "out";
            if (!first) {
                o = "w" + g.outputs.size();
                g.outputs.add(r);
                g.params.append(", __global T* ").append(o);
                g.args.add(r);
                g.argBytes += 8;
            }
            if (!first && g.argBytes > maxArgBytes) {
                g.params.setLength(params);
                g.body.setLength(body);
                g.args.subList(args, g.args.size()).clear();
                g.argBytes = bytes;
                g.outputs.subList(outputs, g.outputs.size()).clear();
                while (g.bufferOrder.size() > buffers) {
                    g.buffers.remove(g.bufferOrder.removeLast());
                }
                while (g.constantOrder.size() > constants) {
                    g.constants.remove(g.constantOrder.removeLast());
                }
                g.tables.values().removeIf(t -> Integer.parseInt(t.substring(1)) >= args);
                break;
            }
            g.body.append("    STORE(").append(o).append(", ").append(idx).append(", ").append(v).append(");\n    }\n");
            end = rangeEnd;
            start = bound;
            first = false;
            used++;
        }
        src.append("__kernel void fused(__global T* out, long n").append(g.params)
                .append(") {\n    long gi = get_global_id(0);\n    if (gi >= n) return;\n")
                .append(g.body).append("}\n");
        return new Many(new Kernel(src.toString(), g.args, g.argBytes, g.outputs), g.first, end, used);
    }

    /**
     * A kernel from {@link #generateMany}: the storage written to {@code out}, the work size,
     * and how many of the given roots it covers (the rest did not fit its arguments).
     */
    record Many(Kernel kernel, ClStorage first, long total, int used) {
    }

    /**
     * Generates one kernel for several reductions with the same shape: per output, {@code len}
     * values {@code inner} apart, read in one pass so that values the inputs share are loaded
     * once. Each keeps the order of a single reduction, so results do not depend on what was
     * reduced together. The reductions in {@code second} read those in {@code first} (each
     * at its own output, such as a variance reading its mean), so they run in a second pass
     * over the same row once the first are known. The first reduction is written to {@code out},
     * the others to the buffers listed first in {@link Kernel#args}. With {@code group > 1} a
     * work-group of that many items shares each output (for long rows); otherwise one work
     * item loops over its row.
     */
    static Kernel generateReduce(List<ClStorage> first, List<ClStorage> second, int group) {
        List<ClStorage> outputs = new ArrayList<>(first);
        outputs.addAll(second);
        Gen g = new Gen();
        g.argBytes += 16;
        int count = outputs.size();
        for (int k = 1; k < count; k++) {
            g.params.append(", __global T* o").append(k);
            g.args.add(outputs.get(k));
            g.argBytes += 8;
        }
        String[] v = new String[count];
        String[] init = new String[count];
        String[] finish = new String[count];
        java.util.Set<Integer> ops = new java.util.TreeSet<>();
        String[] bodies = new String[2];
        for (int k = 0; k < count; k++) {
            if (k == first.size()) {
                // the second pass: values of the first are known per row; nothing computed in
                // the first loop is in scope
                bodies[0] = g.body.toString();
                g.body.setLength(0);
                g.values.clear();
                g.stage = new IdentityHashMap<>();
                for (int f = 0; f < first.size(); f++) {
                    g.stage.put(first.get(f), "f" + f);
                }
            }
            Reduce r = (Reduce) outputs.get(k).pending;
            v[k] = g.eval(whole(r.x()), "x");
            init[k] = r.op() == 2 ? "(ACC) -INFINITY" : r.op() == 3 ? "(ACC) INFINITY" : "(ACC) 0";
            finish[k] = r.op() == 1 ? "acc" + k + " / (ACC) len" : r.op() == 4 ? "(ACC) best" + k : "acc" + k;
            if (r.op() != 4) {
                ops.add(r.op() == 1 ? 0 : r.op());
            } else if (group > 1) {
                throw new IllegalArgumentException("an argmax is reduced by one work item per output");
            }
        }
        bodies[second.isEmpty() ? 0 : 1] = g.body.toString();
        StringBuilder src = new StringBuilder();
        for (int op : ops) {
            src.append("ACC combine").append(op).append("(ACC a, ACC b) {\n    return ")
                    .append(op == 2 ? "isnan(a) || isnan(b) ? a + b : fmax(a, b)"
                            : op == 3 ? "isnan(a) || isnan(b) ? a + b : fmin(a, b)" : "a + b")
                    .append(";\n}\n");
        }
        src.append("__kernel void fused(__global T* out, long n, long len, long inner").append(g.params)
                .append(") {\n");
        if (group <= 1) {
            src.append("    long r = get_global_id(0);\n    if (r >= n) return;\n");
        } else {
            // one buffer, reused output after output: local memory does not grow with their count
            src.append("    __local ACC part[").append(group).append("];\n");
            src.append("    long r = get_group_id(0);\n    int lid = get_local_id(0);\n");
        }
        src.append("    long base = (r / inner) * len * inner + r % inner;\n");
        int stages = second.isEmpty() ? 1 : 2;
        for (int stage = 0; stage < stages; stage++) {
            int from = stage == 0 ? 0 : first.size();
            int to = stage == 0 && stages == 2 ? first.size() : count;
            StringBuilder accumulate = new StringBuilder();
            for (int k = from; k < to; k++) {
                src.append("    ACC acc").append(k).append(" = ").append(init[k]).append(";\n");
                if (((Reduce) outputs.get(k).pending).op() == 4) {
                    // the first largest value, as the precompiled argmax: NaN is never larger
                    src.append("    long best").append(k).append(" = 0;\n");
                    accumulate.append("    if (t == 0 || ").append(v[k]).append(" > acc").append(k).append(") {\n")
                            .append("        acc").append(k).append(" = ").append(v[k]).append(";\n        best")
                            .append(k).append(" = t;\n    }\n");
                    continue;
                }
                accumulate.append("    acc").append(k).append(" = ")
                        .append(combine(outputs.get(k), "acc" + k, v[k])).append(";\n");
            }
            src.append(group <= 1 ? "    for (long t = 0; t < len; t++) {\n"
                    : "    for (long t = lid; t < len; t += " + group + ") {\n")
                    .append("    long x = base + t * inner;\n")
                    .append(bodies[stage]).append(accumulate).append("    }\n");
            if (group > 1) {
                for (int k = from; k < to; k++) {
                    src.append("    part[lid] = acc").append(k).append(";\n")
                            .append("    barrier(CLK_LOCAL_MEM_FENCE);\n")
                            .append("    for (int s = ").append(group / 2).append("; s > 0; s >>= 1) {\n")
                            .append("        if (lid < s) {\n")
                            .append("            part[lid] = ")
                            .append(combine(outputs.get(k), "part[lid]", "part[lid + s]")).append(";\n")
                            .append("        }\n        barrier(CLK_LOCAL_MEM_FENCE);\n    }\n")
                            .append("    acc").append(k).append(" = part[0];\n")
                            .append("    barrier(CLK_LOCAL_MEM_FENCE);\n");
                }
            }
            if (stage == 0 && stages == 2) {
                for (int k = from; k < to; k++) {
                    src.append("    ACC f").append(k).append(" = ").append(finish[k]).append(";\n");
                }
            }
            src.append(group > 1 ? "    if (lid == 0) {\n" : "    {\n");
            for (int k = from; k < to; k++) {
                src.append("        STORE(").append(k == 0 ? "out" : "o" + k).append(", r, ")
                        .append(stage == 0 && stages == 2 ? "f" + k : finish[k]).append(");\n");
            }
            src.append("    }\n");
        }
        src.append("}\n");
        return new Kernel(src.toString(), g.args, g.argBytes, g.outputs);
    }

    /**
     * Whether every read of the reductions {@code targets} in {@code e} is at the output the
     * reading element is reduced into ({@code len} values {@code inner} apart per output), so a
     * second pass over the same row can use their values directly.
     */
    static boolean readsOwnRow(Expr e, java.util.Set<ClStorage> targets, long len, long inner) {
        return switch (e) {
            case Src s -> {
                ClStorage st = s.storage();
                if (targets.contains(st)) {
                    yield s.direct() ? false : ownRow(s, len, inner);
                }
                if (st.pending == null || st.pending instanceof MatMul || st.pending instanceof Reduce) {
                    yield true;
                }
                // through a pending input, only element for element (its index is the reader's)
                yield s.direct() ? readsOwnRow(st.pending, targets, len, inner)
                        : reduces(st.pending).stream().noneMatch(targets::contains);
            }
            case MatMul m -> true;
            case Reduce r -> true;
            case Const c -> true;
            case Random x -> true;
            case Im2col c -> reduces(c.x()).stream().noneMatch(targets::contains);
            case Col2im c -> reduces(c.cols()).stream().noneMatch(targets::contains);
            case Un u -> readsOwnRow(u.x(), targets, len, inner);
            case Act x -> readsOwnRow(x.x(), targets, len, inner);
            case ActBack x -> readsOwnRow(x.x(), targets, len, inner) && readsOwnRow(x.grad(), targets, len, inner);
            case Bin x -> readsOwnRow(x.a(), targets, len, inner) && readsOwnRow(x.b(), targets, len, inner);
        };
    }

    /** Whether a source maps element {@code (o * len + t) * inner + i} to {@code o * inner + i}. */
    private static boolean ownRow(Src s, long len, long inner) {
        if (s.offset() != 0) {
            return false;
        }
        long p = 1;
        long row = inner * len;
        for (int d = s.dims().length - 1; d >= 0; d--) {
            long size = s.dims()[d];
            long stride = s.strides()[d];
            if (size == 1) {
                continue;
            }
            if (p < inner) {
                if (p * size > inner || stride != p) {
                    return false;
                }
            } else if (p < row) {
                if (p * size > row || stride != 0) {
                    return false;
                }
            } else if (stride != inner * (p / row)) {
                return false;
            }
            p *= size;
        }
        return true;
    }

    private static String combine(ClStorage output, String a, String b) {
        int op = ((Reduce) output.pending).op();
        return "combine" + (op == 1 ? 0 : op) + "(" + a + ", " + b + ")";
    }

    /** The pending reductions an expression reads, looking through pending inputs. */
    static List<ClStorage> reduces(Expr e) {
        List<ClStorage> out = new ArrayList<>();
        reduces(e, out, java.util.Collections.newSetFromMap(new IdentityHashMap<>()));
        return out;
    }

    private static void reduces(Expr e, List<ClStorage> out, java.util.Set<ClStorage> seen) {
        switch (e) {
            case Src s -> {
                ClStorage st = s.storage();
                if (st.pending instanceof Reduce) {
                    if (seen.add(st)) {
                        out.add(st);
                    }
                } else if (st.pending != null && seen.add(st)) {
                    reduces(st.pending, out, seen);
                }
            }
            case MatMul m -> {
            }
            case Const c -> {
            }
            case Random x -> {
            }
            case Reduce r -> {
            }
            case Im2col c -> reduces(c.x(), out, seen);
            case Col2im c -> reduces(c.cols(), out, seen);
            case Un u -> reduces(u.x(), out, seen);
            case Act a -> reduces(a.x(), out, seen);
            case ActBack a -> {
                reduces(a.x(), out, seen);
                reduces(a.grad(), out, seen);
            }
            case Bin b -> {
                reduces(b.a(), out, seen);
                reduces(b.b(), out, seen);
            }
        }
    }

    private static final class Gen {
        final StringBuilder params = new StringBuilder();
        final StringBuilder body = new StringBuilder();
        final List<Object> args = new ArrayList<>();
        final Map<ClStorage, String> buffers = new IdentityHashMap<>();
        /** The argument holding each constant, by its bits. */
        final Map<Integer, String> constants = new java.util.HashMap<>();
        /** The argument holding each device table, by address. */
        final Map<Long, String> tables = new java.util.HashMap<>();
        /** The keys of {@link #constants} in the order they were added. */
        final List<Integer> constantOrder = new ArrayList<>();
        /** The keys of {@link #buffers} in the order they were added. */
        final List<ClStorage> bufferOrder = new ArrayList<>();
        /** Pending inputs computed at the result's own index, written to their buffers too. */
        final List<ClStorage> outputs = new ArrayList<>();
        /** The index of the result, or null when the kernel writes nothing but its result. */
        String root;
        /** The only pending inputs written at {@link #root}; null for all of them. */
        java.util.Set<ClStorage> keep;
        /** The storage written to {@code out}, when it can also appear as an input. */
        ClStorage first;
        /** Reductions of a first pass, by the variable holding their value for the current row. */
        Map<ClStorage, String> stage;
                /** A matrix product computed by the kernel itself, as {@code acc}. */
        ClStorage base;
        /** Values already computed in this kernel, by storage and index variable. */
        final Map<ClStorage, Map<String, String>> values = new IdentityHashMap<>();
        int argBytes = 16;
        int next;

        String eval(Expr e, String idx) {
            return switch (e) {
                case Src s -> source(s, idx);
                case MatMul m -> throw new IllegalStateException("a matrix product inside a chain");
                case Reduce r -> throw new IllegalStateException("a reduction inside a chain");
                case Im2col c -> {
                    String width = arg((long) c.channels() * c.ks());
                    String row = "q" + next++;
                    body.append("    int ").append(row).append(" = (int) (").append(idx).append(" / ").append(width)
                            .append(");\n");
                    yield window(c, row, "(int) (" + idx + " % " + width + ")");
                }
                case Random x -> let("(ACC) random_at(" + arg(x.seed()) + ", " + idx + ")");
                case Col2im c -> {
                    String offsets = buffer(c.offsets());
                    String entries = buffer(c.entries());
                    String img = arg(c.img());
                    String channels = arg(c.channels());
                    String ks = intArg(c.ks());
                    String plane = "q" + next++;
                    String cell = "q" + next++;
                    String ent = "q" + next++;
                    String at = "q" + next++;
                    String sum = "q" + next++;
                    body.append("    long ").append(plane).append(" = ").append(idx).append(" / ").append(img)
                            .append(";\n    int ").append(cell).append(" = (int) (").append(idx).append(" % ")
                            .append(img).append(");\n    ACC ").append(sum).append(" = (ACC) 0;\n    for (int ")
                            .append(ent).append(" = ").append(offsets).append("[").append(cell).append("]; ").append(ent)
                            .append(" < ").append(offsets).append("[").append(cell).append(" + 1]; ").append(ent)
                            .append("++) {\n    long ").append(at).append(" = ((").append(plane).append(" / ")
                            .append(channels).append(") * ").append(arg(c.positions())).append(" + ").append(entries)
                            .append("[").append(ent).append("] / ").append(ks).append(") * ").append(channels)
                            .append(" * ").append(ks).append(" + (").append(plane).append(" % ").append(channels)
                            .append(") * ").append(ks).append(" + ").append(entries).append("[").append(ent)
                            .append("] % ").append(ks).append(";\n");
                    // values computed in the loop are out of scope after it
                    Map<ClStorage, Map<String, String>> saved = new IdentityHashMap<>();
                    values.forEach((k, v) -> saved.put(k, new java.util.HashMap<>(v)));
                    String v = read(c.cols().storage(), at);
                    values.clear();
                    values.putAll(saved);
                    body.append("    ").append(sum).append(" += ").append(v).append(";\n    }\n");
                    yield sum;
                }
                case Const c -> {
                    // one argument and one value per distinct number in the kernel
                    int bits = Float.floatToRawIntBits(c.value());
                    String a = constants.get(bits);
                    if (a == null) {
                        a = "c" + args.size();
                        params.append(", float ").append(a);
                        args.add(c.value());
                        argBytes += 8;
                        constants.put(bits, a);
                        constantOrder.add(bits);
                    }
                    yield "((ACC) " + a + ")";
                }
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
            if (stage != null && stage.containsKey(s.storage())) {
                return stage.get(s.storage());   // a reduction of the first pass, at this row
            }
            if (s.storage().pending instanceof Const c && s.storage() != base) {
                return eval(c, idx);   // the same everywhere: no index to compute
            }
            return read(s.storage(), s.direct() ? idx : index(s, idx));
        }

        /** The value of {@code st} at the index held by {@code at}. */
        private String read(ClStorage st, String at) {
            if (st.pending instanceof Const c && st != base) {
                return eval(c, at);
            }
            if (st == base) {
                return "acc";
            }
            Map<String, String> seen = values.computeIfAbsent(st, k -> new java.util.HashMap<>());
            String known = seen.get(at);
            if (known != null) {
                return known;
            }
            String v;
            if (st.pending != null) {
                v = eval(st.pending, at);
                if (at.equals(root) && (keep == null || keep.contains(st)) && !outputs.contains(st) && st != first) {
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
                    bufferOrder.add(st);
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
            List<Integer> kept = new ArrayList<>();
            for (int d = 0; d < s.dims().length; d++) {
                if (s.dims()[d] != 1) {
                    kept.add(d);
                }
            }
            // 32-bit arithmetic when every index and position fits: a 64-bit division costs
            // several times as many instructions on GPUs
            boolean small = fitsInt(s);
            String off = small ? intArg((int) s.offset()) : arg(s.offset());
            body.append(small ? "    int " : "    long ").append(j).append(" = ").append(off).append(";\n");
            if (!kept.isEmpty()) {
                String r = "r" + next++;
                body.append(small ? "    uint " : "    long ").append(r).append(" = ")
                        .append(small ? "(uint) (" + idx + ")" : idx).append(";\n");
                for (int k = kept.size() - 1; k >= 0; k--) {
                    int d = kept.get(k);
                    String dim = small ? uintArg(s.dims()[d]) : arg(s.dims()[d]);
                    String stride = small ? intArg((int) s.strides()[d]) : arg(s.strides()[d]);
                    if (small) {
                        body.append("    ").append(j).append(" += (int) (").append(r).append(" % ").append(dim)
                                .append(") * ").append(stride).append(";\n");
                        if (k > 0) {
                            body.append("    ").append(r).append(" /= ").append(dim).append(";\n");
                        }
                        continue;
                    }
                    body.append("    ").append(j).append(" += (").append(r).append(" % ").append(dim)
                            .append(") * ").append(stride).append(";\n");
                    if (k > 0) {
                        body.append("    ").append(r).append(" /= ").append(dim).append(";\n");
                    }
                }
            }
            return j;
        }

        /**
         * The statements loading one matmul tile element into {@code TILE}: from {@code buffer},
         * or {@code value} computed at the element's index (within the tile loop, so nothing it
         * computes is visible to the epilogue).
         */
        String load(Expr value, String buffer, String inside, String index, String row, String col) {
            if (value == null) {
                return "        TILE = " + inside + " ? LOAD(" + buffer + ", " + index + ") : (ACC) 0;\n";
            }
            int mark = body.length();
            String at = "p" + next++;
            // an unrolled image read in its own row and column order needs no index arithmetic
            String v = row != null && value instanceof Im2col c ? window(c, row, col) : eval(value, at);
            String computed = body.substring(mark);
            body.setLength(mark);
            values.clear();   // values computed inside the loop are out of scope after it
            return "        TILE = (ACC) 0;\n        if (" + inside + ") {\n    long " + at + " = " + index + ";\n"
                    + computed + "        TILE = " + v + ";\n        }\n";
        }

        /**
         * The value of an unrolled image at row {@code row} and column {@code col} (int
         * expressions: rows, columns and image indices fit an int, and int division is
         * much cheaper than long on most devices).
         */
        String window(Im2col c, String row, String col) {
            String table = buffer(c.table());
            String positions = intArg(c.positions());
            String ks = intArg(c.ks());
            String r = "q" + next++;
            String k = "q" + next++;
            String cell = "q" + next++;
            String at = "q" + next++;
            body.append("    int ").append(r).append(" = ").append(row).append(";\n")
                    .append("    int ").append(k).append(" = ").append(col).append(";\n")
                    .append("    int ").append(cell).append(" = ").append(table).append("[(").append(r).append(" % ")
                    .append(positions).append(") * ").append(ks).append(" + ").append(k).append(" % ").append(ks)
                    .append("];\n")
                    .append("    int ").append(at).append(" = (").append(r).append(" / ").append(positions)
                    .append(" * ").append(intArg(c.channels())).append(" + ").append(k).append(" / ").append(ks)
                    .append(") * ").append(intArg(c.img())).append(" + max(").append(cell).append(", 0);\n");
            String v = read(c.x().storage(), at);
            return let(cell + " < 0 ? " + literal(c.pad()) + " : " + v);
        }

        /** Whether every position of a view, and every index into it, fits in an int. */
        private static boolean fitsInt(Src s) {
            long count = 1;
            long high = s.offset();
            for (int d = 0; d < s.dims().length; d++) {
                long size = s.dims()[d];
                long stride = s.strides()[d];
                if (size <= 0 || stride < 0) {
                    return false;
                }
                count = Math.multiplyHigh(count, size) != 0 ? Long.MAX_VALUE : count * size;
                high += (size - 1) * stride;
                if (count > Integer.MAX_VALUE || high > Integer.MAX_VALUE || high < 0) {
                    return false;
                }
            }
            return s.offset() >= 0;
        }

        private String uintArg(int value) {
            String a = "a" + args.size();
            params.append(", uint ").append(a);
            args.add(value);
            argBytes += 8;
            return a;
        }

        private String intArg(int value) {
            String a = "a" + args.size();
            params.append(", int ").append(a);
            args.add(value);
            argBytes += 8;
            return a;
        }

        /** A device table of ints, one argument per distinct table. */
        private String buffer(java.lang.foreign.MemorySegment table) {
            String t = tables.get(table.address());
            if (t == null) {
                t = "t" + args.size();
                params.append(", __global const int* ").append(t);
                args.add(table);
                argBytes += 8;
                tables.put(table.address(), t);
            }
            return t;
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
