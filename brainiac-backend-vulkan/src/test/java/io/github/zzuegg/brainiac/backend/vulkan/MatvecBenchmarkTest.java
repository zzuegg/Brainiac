package io.github.zzuegg.brainiac.backend.vulkan;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
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

import org.junit.jupiter.api.Test;

/**
 * Decode speed of packed matrix-vector products, opt-in: {@code -Dbrainiac.bench=true}
 * (Gradle: {@code -Pbench}). Uses the shapes of a small chat model (Qwen2.5-0.5B: 24 layers,
 * 896 wide, 4864 hidden, 151936 tokens) with a distinct matrix per layer, so the weights don't
 * stay in the cache, and prints microseconds per product and the weight bandwidth reached.
 */
class MatvecBenchmarkTest {

    private static final int LAYERS = 24;
    /** k x n per product of one layer: q, k, v, o, gate, up, down. */
    private static final int[][] LAYER = {{896, 896}, {896, 128}, {896, 128}, {896, 896}, {896, 4864}, {896, 4864},
            {4864, 896}};

    @Test
    void decode() {
        assumeTrue(Boolean.getBoolean("brainiac.bench"), "benchmark: -Pbench");
        Device gpu = Device.available().stream().filter(d -> d.name().startsWith("vulkan")).findFirst().orElseThrow();
        VulkanBackend backend = VulkanBackend.loaded().orElseThrow();
        System.out.println(gpu.name() + "; " + backend.status());
        System.out.println("kernel tuning: " + backend.tuning());
        for (String wide : new String[] {"0", "1"}) {
            System.setProperty("brainiac.vulkan.packedWide", wide);
            System.out.println("packedWide " + wide + (wide.equals("1") ? " (16 columns per invocation)" : ""));
            products(backend);
        }
        System.clearProperty("brainiac.vulkan.packedWide");
    }

    /**
     * A 7B model's (Qwen2.5-7B) products in its GGUF formats, factored as Gguf.packedTransposed
     * keeps them: Q4_K (4-bit levels in groups of 32 with offsets) and Q6_K (6-bit levels, a
     * nibble and a 2-bit plane, in groups of 16), eight distinct matrices each.
     */
    @Test
    void ggufFormats() {
        assumeTrue(Boolean.getBoolean("brainiac.bench"), "benchmark: -Pbench");
        Device.available();   // loads the backends
        VulkanBackend backend = VulkanBackend.loaded().orElseThrow();
        System.out.println(backend.status() + "; " + backend.tuning());
        int[][] shapes = {{18944, 3584}, {3584, 3584}, {3584, 37888}, {3584, 4608}};
        for (int[] s : shapes) {
            for (int bits : new int[] {4, 6}) {
                List<PackedStorage> weights = new ArrayList<>();
                List<Runnable> calls = new ArrayList<>();
                Storage x = f16(backend, s[0], 3);
                Storage z = backend.allocate(s[1], DType.FLOAT16);
                for (int copy = 0; copy < 8; copy++) {
                    PackedStorage w = backend.pack(factored(s[0], s[1], bits, copy));
                    weights.add(w);
                    calls.add(() -> backend.matmulPacked(x, w, 1, s[0], s[1], z));
                }
                int group = bits == 4 ? 32 : 16;
                long bytes = (long) s[0] * s[1] * bits / 8 + (long) s[0] / group * s[1] * (bits == 4 ? 2 : 1);
                report(backend, s[0] + " x " + s[1] + (bits == 4 ? " Q4_K" : " Q6_K"), calls, bytes * 8, 8);
                weights.forEach(backend::release);
                backend.release(x);
                backend.release(z);
            }
        }
    }

    /** Random levels with Q4_K's or Q6_K's factored scales (and Q4_K's offsets). */
    private static PackedMatrix factored(int rows, int cols, int bits, long seed) {
        SplittableRandom r = new SplittableRandom(seed * 131 + rows + cols + bits);
        boolean q4 = bits == 4;
        int group = q4 ? 32 : 16;
        int shift = 1 << (bits - 1);
        byte[] levels = new byte[rows * cols];
        for (int i = 0; i < levels.length; i++) {
            levels[i] = (byte) (r.nextInt(2 * shift) - shift);
        }
        byte[] scaleCodes = new byte[rows / group * cols];
        for (int i = 0; i < scaleCodes.length; i++) {
            scaleCodes[i] = (byte) (q4 ? r.nextInt(64) : r.nextInt(-32, 32));
        }
        float[] superScales = new float[rows / 256 * cols];
        java.util.Arrays.fill(superScales, 0.001f);
        byte[] offsetCodes = null;
        float[] superOffsets = null;
        if (q4) {
            offsetCodes = new byte[scaleCodes.length];
            for (int i = 0; i < offsetCodes.length; i++) {
                offsetCodes[i] = (byte) r.nextInt(64);
            }
            superOffsets = new float[superScales.length];
            java.util.Arrays.fill(superOffsets, -0.002f);
        }
        return PackedMatrix.of(levels, new PackedMatrix.Factors(scaleCodes, superScales, offsetCodes, superOffsets, 256,
                q4 ? shift : 0), rows, cols, bits, group);
    }

    private static void products(VulkanBackend backend) {
        for (int bits : new int[] {8, 4}) {
            List<Storage> owned = new ArrayList<>();
            List<PackedStorage> weights = new ArrayList<>();
            List<Runnable> token = new ArrayList<>();
            long tokenBytes = 0;
            for (int[] s : LAYER) {
                List<Runnable> calls = new ArrayList<>();
                Storage x = f16(backend, s[0], s[0]);
                owned.add(x);
                for (int l = 0; l < LAYERS; l++) {
                    PackedStorage w = backend.pack(matrix(s[0], s[1], bits, l));
                    Storage z = backend.allocate(s[1], DType.FLOAT16);
                    weights.add(w);
                    owned.add(z);
                    calls.add(() -> backend.matmulPacked(x, w, 1, s[0], s[1], z));
                }
                long bytes = (long) s[0] * s[1] * bits / 8;
                tokenBytes += bytes * LAYERS;
                token.addAll(calls);
                report(backend, s[0] + " x " + s[1] + " q" + bits, calls, bytes * LAYERS, LAYERS);
            }
            int vocab = 151936;
            Storage x = f16(backend, 896, 7);
            PackedStorage head = backend.pack(matrix(896, vocab, bits, 99));
            Storage logits = backend.allocate(vocab, DType.FLOAT16);
            owned.add(x);
            owned.add(logits);
            weights.add(head);
            long headBytes = 896L * vocab * bits / 8;
            report(backend, "896 x " + vocab + " q" + bits, List.of(() -> backend.matmulPacked(x, head, 1, 896, vocab,
                    logits)), headBytes, 1);
            token.add(() -> backend.matmulPacked(x, head, 1, 896, vocab, logits));
            report(backend, "token q" + bits + " (" + token.size() + " products)", token, tokenBytes + headBytes,
                    token.size());
            weights.forEach(backend::release);
            owned.forEach(backend::release);
        }
    }

    /** Runs the calls back to back, best of five rounds, and prints the time per call and the bandwidth. */
    private static void report(VulkanBackend backend, String what, List<Runnable> calls, long bytes, int count) {
        calls.forEach(Runnable::run);
        backend.finish();
        double best = Double.MAX_VALUE;
        for (int round = 0; round < 5; round++) {
            long start = System.nanoTime();
            calls.forEach(Runnable::run);
            backend.finish();
            best = Math.min(best, (System.nanoTime() - start) / 1e9);
        }
        System.out.printf(Locale.ROOT, "%-34s %9.1f us per product %8.2f ms total %7.1f GB/s%n", what,
                best * 1e6 / count, best * 1e3, bytes / best / 1e9);
    }

    private static PackedMatrix matrix(int rows, int cols, int bits, long seed) {
        SplittableRandom r = new SplittableRandom(seed * 31 + rows + cols);
        byte[] levels = new byte[rows * cols];
        int half = 1 << (bits - 1);
        for (int i = 0; i < levels.length; i++) {
            levels[i] = (byte) (r.nextInt(2 * half) - half);
        }
        float[] scales = new float[rows / 32 * cols];
        java.util.Arrays.fill(scales, 0.01f);
        return PackedMatrix.of(levels, scales, rows, cols, bits, 32);
    }

    private static Storage f16(VulkanBackend backend, int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment m = arena.allocate(n * 2L, 2);
            for (int i = 0; i < n; i++) {
                m.setAtIndex(ValueLayout.JAVA_SHORT_UNALIGNED, i, Float.floatToFloat16((float) r.nextDouble(-1, 1)));
            }
            return backend.upload(m, DType.FLOAT16);
        }
    }
}
