package io.github.zzuegg.brainiac.backend.vulkan;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.spi.PackedMatrix;
import io.github.zzuegg.brainiac.tensor.spi.PackedStorage;
import io.github.zzuegg.brainiac.tensor.spi.Storage;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.SplittableRandom;

/**
 * Measures the kernel choices of {@link Tuning} on the device: each candidate runs a few
 * representative products, and the fastest is kept. Takes a few seconds, once per device and
 * driver.
 */
final class Tuner {

    /** Wall time a measurement aims for. */
    private static final long TARGET_NANOS = 6_000_000;

    private final VulkanBackend backend;
    private final Tuning tuning;
    private final StringBuilder log = new StringBuilder();
    private final List<Storage> owned = new ArrayList<>();

    Tuner(VulkanBackend backend, Tuning tuning) {
        this.backend = backend;
        this.tuning = tuning;
    }

    /** Runs every measurement not fixed by a property; @return what was measured, one line each */
    String run(boolean coopMatrix) {
        try {
            if (coopMatrix && !Tuning.fixed("coopShape") && !"false".equals(System.getProperty("brainiac.vulkan.coopmat"))) {
                coopShape();
                if (tuning.get("coopShape") >= 0 && !Tuning.fixed("coopRows")) {
                    coopRows();
                }
            }
            if (!Tuning.fixed("bigTileGroups")) {
                bigTiles();
            }
            if (!Tuning.fixed("splitBelow")) {
                splitBelow();
            }
            if (!Tuning.fixed("packedGroups") && !Tuning.fixed("packedPiece")) {
                packed();
            }
        } finally {
            for (Storage s : owned) {
                backend.release(s);
            }
            owned.clear();
        }
        return log.toString();
    }

    // ------------------------------------------------------------------ the measurements

    private void coopShape() {
        int[][] shapes = {{2048, 2048, 2048}, {512, 4096, 4096}};
        double[] times = new double[Tuning.COOP_SHAPES.size()];
        tuning.set("coopRows", 1);
        for (int c = 0; c < times.length; c++) {
            tuning.set("coopShape", c);
            for (int[] s : shapes) {
                times[c] += product(DType.FLOAT16, s[0], s[1], s[2]);
            }
            note("coopShape", name(c), times[c], flops(shapes) / times[c]);
        }
        tuning.set("coopShape", pick(times, 0));
        tuning.set("coopRows", Tuning.DEFAULTS.get("coopRows"));
    }

    /**
     * Whether and from how many rows the matrix cores beat the tiles, each size on its own: on
     * an RTX 4090 they lost at 2048 x 2048 x 2048 by half while a sum over sizes hid it.
     */
    private void coopRows() {
        int[][] shapes = {{2048, 2048, 2048}, {512, 4096, 4096}, {256, 4096, 4096}, {128, 4096, 4096},
                {64, 4096, 4096}, {32, 4096, 4096}, {16, 4096, 4096}};
        long chosen = -1;
        for (int i = 0; i < shapes.length; i++) {
            int[] s = shapes[i];
            tuning.set("coopRows", 1);
            double coop = product(DType.FLOAT16, s[0], s[1], s[2]);
            tuning.set("coopRows", Integer.MAX_VALUE);
            double tiles = product(DType.FLOAT16, s[0], s[1], s[2]);
            note("coop " + s[0] + "x" + s[1] + "x" + s[2], "matrix cores vs tiles", coop, tiles);
            if (coop > tiles) {
                break;
            }
            chosen = s[0];
        }
        if (chosen < 0) {
            // slower on the largest products: tiles throughout
            tuning.set("coopShape", -1);
            tuning.set("coopRows", Tuning.DEFAULTS.get("coopRows"));
        } else {
            tuning.set("coopRows", chosen);
        }
    }

    private void bigTiles() {
        int[][] shapes = {{128, 1024, 1024}, {256, 1024, 2048}, {512, 2048, 2048}};
        choose("bigTileGroups", new long[] {32, 128, 512}, () -> {
            double t = 0;
            for (int[] s : shapes) {
                t += product(DType.FLOAT32, s[0], s[1], s[2]);
            }
            return t;
        });
    }

    private void splitBelow() {
        choose("splitBelow", new long[] {1, 64, 256, 1024}, () -> product(DType.FLOAT32, 64, 4096, 512)
                + product(DType.FLOAT16, 32, 4096, 1024) + product(DType.FLOAT16, 128, 8192, 1024));
    }

    private void packed() {
        PackedStorage[] w = {packedMatrix(4096, 11008), packedMatrix(11008, 4096)};
        Storage[] x = {f16(4096, 1), f16(11008, 2)};
        Storage[] out = {backend.allocate(11008, DType.FLOAT16), backend.allocate(4096, DType.FLOAT16)};
        owned.addAll(List.of(out));
        double best = Double.MAX_VALUE;
        long[] choice = {tuning.get("packedGroups"), tuning.get("packedPiece")};
        for (long groups : new long[] {128, 256, 512, 1024}) {
            for (long piece : new long[] {32, 64, 128}) {
                tuning.set("packedGroups", groups);
                tuning.set("packedPiece", piece);
                double t = time(() -> {
                    backend.matmulPacked(x[0], w[0], 1, 4096, 11008, out[0]);
                    backend.matmulPacked(x[1], w[1], 1, 11008, 4096, out[1]);
                });
                note("packed " + groups + "/" + piece, "", t, 0);
                if (t < best) {
                    best = t;
                    choice = new long[] {groups, piece};
                }
            }
        }
        tuning.set("packedGroups", choice[0]);
        tuning.set("packedPiece", choice[1]);
        for (PackedStorage p : w) {
            backend.release(p);
        }
    }

    // ------------------------------------------------------------------ helpers

    private interface Measure {
        double seconds();
    }

    private void choose(String key, long[] candidates, Measure measure) {
        double[] times = new double[candidates.length];
        int current = -1;
        for (int i = 0; i < candidates.length; i++) {
            tuning.set(key, candidates[i]);
            times[i] = measure.seconds();
            note(key + " " + candidates[i], "", times[i], 0);
            if (candidates[i] == Tuning.DEFAULTS.get(key)) {
                current = i;
            }
        }
        tuning.set(key, candidates[pick(times, current)]);
    }

    /** Below this ratio to the default's time a candidate replaces it: measurements vary by a few percent. */
    private static final double MARGIN = 0.97;

    /** The fastest candidate if it beats the default (index {@code current}, -1: none) by the margin, else the default. */
    static int pick(double[] times, int current) {
        int best = 0;
        for (int i = 1; i < times.length; i++) {
            if (times[i] < times[best]) {
                best = i;
            }
        }
        return current < 0 || times[best] < times[current] * MARGIN ? best : current;
    }

    /** Seconds per product of an m x k and a k x n matrix in a format. */
    private double product(DType type, int m, int k, int n) {
        Storage a = type == DType.FLOAT16 ? f16((long) m * k, 3) : f32((long) m * k, 3);
        Storage b = type == DType.FLOAT16 ? f16((long) k * n, 4) : f32((long) k * n, 4);
        Storage z = backend.allocate((long) m * n, type);
        try {
            return time(() -> backend.matmulBatched(a, b, 1, m, k, n, z));
        } finally {
            backend.release(a);
            backend.release(b);
            backend.release(z);
        }
    }

    /** Seconds per run: the best of three rounds, each of enough runs for TARGET_NANOS. */
    private double time(Runnable op) {
        op.run();
        backend.finish();
        long start = System.nanoTime();
        op.run();
        backend.finish();
        long once = Math.max(1, System.nanoTime() - start);
        int reps = (int) Math.max(1, Math.min(500, TARGET_NANOS / once));
        double best = Double.MAX_VALUE;
        for (int round = 0; round < 3; round++) {
            start = System.nanoTime();
            for (int r = 0; r < reps; r++) {
                op.run();
            }
            backend.finish();
            best = Math.min(best, (System.nanoTime() - start) / 1e9 / reps);
        }
        return best;
    }

    private Storage f16(long n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment m = arena.allocate(n * 2, 2);
            for (long i = 0; i < n; i++) {
                m.setAtIndex(ValueLayout.JAVA_SHORT_UNALIGNED, i, Float.floatToFloat16((float) r.nextDouble(-1, 1)));
            }
            return backend.upload(m, DType.FLOAT16);
        }
    }

    private Storage f32(long n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment m = arena.allocate(n * 4, 4);
            for (long i = 0; i < n; i++) {
                m.setAtIndex(ValueLayout.JAVA_FLOAT_UNALIGNED, i, (float) r.nextDouble(-1, 1));
            }
            return backend.upload(m, DType.FLOAT32);
        }
    }

    /** A 4-bit matrix in groups of 32 rows, as a Q4 model's. */
    private PackedStorage packedMatrix(int rows, int cols) {
        SplittableRandom r = new SplittableRandom(rows);
        byte[] levels = new byte[rows * cols];
        for (int i = 0; i < levels.length; i++) {
            levels[i] = (byte) (r.nextInt(16) - 8);
        }
        float[] scales = new float[rows / 32 * cols];
        java.util.Arrays.fill(scales, 0.01f);
        return backend.pack(PackedMatrix.of(levels, scales, rows, cols, 4, 32));
    }

    private static double flops(int[][] shapes) {
        double f = 0;
        for (int[] s : shapes) {
            f += 2.0 * s[0] * s[1] * s[2];
        }
        return f;
    }

    private static String name(int coopShape) {
        Tuning.CoopShape s = Tuning.COOP_SHAPES.get(coopShape);
        return s.suffix().isEmpty() ? "128x128" : s.suffix();
    }

    private void note(String what, String detail, double seconds, double other) {
        log.append(String.format(Locale.ROOT, "%-24s %-22s %9.3f ms", what, detail, seconds * 1e3));
        if (other > 1e6) {
            log.append(String.format(Locale.ROOT, "  %6.2f TFLOPS", other / 1e12));
        } else if (other > 0) {
            log.append(String.format(Locale.ROOT, "  vs %9.3f ms", other * 1e3));
        }
        log.append('\n');
    }
}
