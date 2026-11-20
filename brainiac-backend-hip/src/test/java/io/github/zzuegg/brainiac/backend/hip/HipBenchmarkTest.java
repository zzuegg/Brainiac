package io.github.zzuegg.brainiac.backend.hip;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.Locale;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

/**
 * Speeds on the first HIP device, opt-in: {@code -Dbrainiac.bench=true} (Gradle:
 * {@code -Pbench}). Prints matrix product TFLOPS for float32, float16 and bfloat16, and the
 * memory bandwidth of an element-wise operation.
 */
class HipBenchmarkTest {

    @Test
    void products() {
        assumeTrue(Boolean.getBoolean("brainiac.bench"), "benchmark: -Pbench");
        Device gpu = Device.available().stream().filter(d -> d.name().startsWith("hip")).findFirst().orElseThrow();
        HipBackend backend = HipBackend.loaded().orElseThrow();
        System.out.println(gpu.name() + "; " + backend.status());
        for (int[] s : List.of(new int[] {512, 512, 512}, new int[] {2048, 2048, 2048}, new int[] {4096, 4096, 4096},
                new int[] {8192, 8192, 8192}, new int[] {128, 4096, 4096}, new int[] {16, 4096, 4096},
                new int[] {1, 4096, 4096})) {
            for (DType type : List.of(DType.FLOAT32, DType.FLOAT16, DType.BFLOAT16)) {
                Tensor a = Tensor.uniform(Shape.of(s[0], s[1]), -1, 1, new SplittableRandom(1)).to(type).to(gpu);
                Tensor b = Tensor.uniform(Shape.of(s[1], s[2]), -1, 1, new SplittableRandom(2)).to(type).to(gpu);
                a.matmul(b).toArray();   // warm up: rocBLAS picks its kernels
                int reps = (int) Math.max(3, Math.min(500, 4e12 / (2.0 * s[0] * s[1] * s[2])));
                backend.finish();
                long start = System.nanoTime();
                for (int r = 0; r < reps; r++) {
                    a.matmul(b).release();
                }
                backend.finish();
                double seconds = (System.nanoTime() - start) / 1e9;
                System.out.printf(Locale.ROOT, "%5d x %5d x %5d %-9s %8.2f TFLOPS (%d reps, %.3f ms each)%n", s[0],
                        s[1], s[2], type, 2.0 * s[0] * s[1] * s[2] * reps / seconds / 1e12, reps, seconds * 1e3 / reps);
            }
        }
        for (DType type : List.of(DType.FLOAT32, DType.FLOAT16)) {
            int n = 1 << 26;
            Tensor a = Tensor.zeros(Shape.of(n), type, gpu);
            Tensor b = Tensor.zeros(Shape.of(n), type, gpu);
            added(a, b);
            int reps = 50;
            backend.finish();
            long start = System.nanoTime();
            for (int r = 0; r < reps; r++) {
                added(a, b);
            }
            backend.finish();
            double seconds = (System.nanoTime() - start) / 1e9;
            System.out.printf(Locale.ROOT, "add of %d %s: %.0f GB/s (%.3f ms each)%n", n, type,
                    3.0 * n * type.bytes() * reps / seconds / 1e9, seconds * 1e3 / reps);
        }
        System.out.println(backend.statistics());
    }

    /** {@code a + b} computed (not left pending for fusion) and released. */
    private static void added(Tensor a, Tensor b) {
        Tensor r = a.add(b);
        Tensor.evaluate(List.of(r));
        r.release();
    }

    /**
     * Decoding's small kernels, each run 1000 times back to back (the queue always full) and 200
     * times with the host pausing 100 us after each (the GPU idle between, as when decoding is
     * host-bound): if the second is much slower per kernel, the GPU lowers its clock when idle.
     */
    @Test
    void decodingKernels() {
        assumeTrue(Boolean.getBoolean("brainiac.bench"), "benchmark: -Pbench");
        Device gpu = Device.available().stream().filter(d -> d.name().startsWith("hip")).findFirst().orElseThrow();
        HipBackend backend = HipBackend.loaded().orElseThrow();
        Tensor logits = Tensor.uniform(Shape.of(1, 151936), -1, 1, new SplittableRandom(3)).to(gpu);
        Tensor x = Tensor.uniform(Shape.of(1, 896), -1, 1, new SplittableRandom(4)).to(DType.FLOAT16).to(gpu);
        Tensor y = Tensor.uniform(Shape.of(1, 896), -1, 1, new SplittableRandom(5)).to(DType.FLOAT16).to(gpu);
        Tensor down = Tensor.uniform(Shape.of(4864, 896), -1, 1, new SplittableRandom(6)).to(DType.FLOAT16).to(gpu);
        Tensor h = Tensor.uniform(Shape.of(1, 4864), -1, 1, new SplittableRandom(7)).to(DType.FLOAT16).to(gpu);
        Runnable[] work = {
            () -> logits.argmax(1).release(),
            () -> added(x, y),
            () -> h.matmul(down).release(),
        };
        String[] names = {"argmax of 151936 float32", "add of 896 float16", "[1, 4864] x [4864, 896] float16"};
        for (int w = 0; w < work.length; w++) {
            work[w].run();
            backend.finish();
            long start = System.nanoTime();
            for (int r = 0; r < 1000; r++) {
                work[w].run();
            }
            backend.finish();
            double busy = (System.nanoTime() - start) / 1e3 / 1000;
            start = System.nanoTime();
            for (int r = 0; r < 200; r++) {
                work[w].run();
                backend.finish();
                long until = System.nanoTime() + 100_000;
                while (System.nanoTime() < until) {
                    Thread.onSpinWait();
                }
            }
            double idle = (System.nanoTime() - start) / 1e3 / 200 - 100;
            System.out.printf(Locale.ROOT, "%-34s back to back %8.1f us each; with pauses %8.1f us each (incl. a wait)%n",
                    names[w], busy, idle);
        }
        System.out.println(backend.statistics());
    }

    @Test
    void graphs() {
        assumeTrue(Boolean.getBoolean("brainiac.bench"), "benchmark: -Pbench");
        Device.available().stream().filter(d -> d.name().startsWith("hip")).findFirst().orElseThrow();
        HipBackend backend = HipBackend.loaded().orElseThrow();
        for (int kernels : new int[] {100, 500}) {
            System.out.println(backend.graphBenchmark(kernels, 50));
        }
    }
}
