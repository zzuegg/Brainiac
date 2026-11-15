package io.github.zzuegg.brainiac.backend.cpu;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import io.github.zzuegg.brainiac.tensor.spi.BinaryOp;
import io.github.zzuegg.brainiac.tensor.spi.ReduceOp;
import io.github.zzuegg.brainiac.tensor.spi.Storage;
import io.github.zzuegg.brainiac.tensor.spi.UnaryOp;

import java.util.SplittableRandom;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.Test;

/**
 * Large, oddly sized inputs that take the blocked and multithreaded paths, compared
 * exactly with straightforward single-threaded loops in the same summation order.
 */
class ParallelKernelsTest {

    private final CpuBackend backend = new CpuBackend();

    private static float[] random(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = (float) r.nextDouble(-1, 1);
        }
        return v;
    }

    @Test
    void matmulMatchesNaiveExactly() {
        int m = 67;
        int k = 517;
        int n = 131;
        float[] a = random(m * k, 1);
        float[] b = random(k * n, 2);
        float[] expected = new float[m * n];
        for (int i = 0; i < m; i++) {
            for (int p = 0; p < k; p++) {
                for (int j = 0; j < n; j++) {
                    expected[i * n + j] += a[i * k + p] * b[p * n + j];
                }
            }
        }
        Storage out = backend.allocate(m * n);
        backend.fill(out, 7); // must be overwritten, not accumulated into
        backend.matmul(backend.upload(a), backend.upload(b), m, k, n, out);
        assertArrayEquals(expected, backend.download(out));
    }

    @Test
    void elementWiseOpsMatchSingleValues() {
        int size = 300_001;
        float[] x = random(size, 3);
        float[] y = random(size, 4);
        Storage out = backend.allocate(size);
        backend.unary(UnaryOp.TANH, backend.upload(x), out);
        float[] tanh = backend.download(out);
        backend.binary(BinaryOp.MUL, backend.upload(x), new long[] {1}, backend.upload(y), new long[] {1},
                new int[] {size}, out);
        float[] mul = backend.download(out);
        for (int i = 0; i < size; i += 997) {
            assertEquals((float) Math.tanh(x[i]), tanh[i]);
            assertEquals(x[i] * y[i], mul[i]);
        }
    }

    @Test
    void rowBroadcastAndScalarPaths() {
        int rows = 1000;
        int cols = 77;
        float[] x = random(rows * cols, 5);
        float[] bias = random(cols, 6);
        Storage out = backend.allocate(rows * cols);
        backend.binary(BinaryOp.SUB, backend.upload(x), new long[] {cols, 1}, backend.upload(bias), new long[] {0, 1},
                new int[] {rows, cols}, out);
        float[] z = backend.download(out);
        backend.binary(BinaryOp.DIV, backend.upload(new float[] {2}), new long[] {0, 0}, backend.upload(x),
                new long[] {cols, 1}, new int[] {rows, cols}, out);
        float[] w = backend.download(out);
        for (int i = 0; i < rows * cols; i += 101) {
            assertEquals(x[i] - bias[i % cols], z[i]);
            assertEquals(2 / x[i], w[i]);
        }
    }

    @Test
    void reductionsOverAnyAxisMatchNaive() {
        int outer = 3;
        int len = 501;
        int inner = 257;
        float[] x = random(outer * len * inner, 7);
        Storage in = backend.upload(x);
        for (ReduceOp op : ReduceOp.values()) {
            Storage out = backend.allocate(outer * inner);
            backend.reduce(op, in, outer, len, inner, out);
            float[] got = backend.download(out);
            for (int o = 0; o < outer; o++) {
                for (int j = 0; j < inner; j++) {
                    double sum = 0;
                    float max = Float.NEGATIVE_INFINITY;
                    float min = Float.POSITIVE_INFINITY;
                    int arg = 0;
                    for (int i = 0; i < len; i++) {
                        float v = x[(o * len + i) * inner + j];
                        sum += v;
                        min = Math.min(min, v);
                        if (v > max) {
                            max = v;
                            arg = i;
                        }
                    }
                    float expected = switch (op) {
                        case SUM -> (float) sum;
                        case MEAN -> (float) (sum / len);
                        case MAX -> max;
                        case MIN -> min;
                        case ARGMAX -> arg;
                    };
                    assertEquals(expected, got[o * inner + j], op + " at " + o + "," + j);
                }
            }
        }
    }

    @Test
    void forRangeCoversEveryIndexOnce() {
        int n = 100_003;
        AtomicInteger[] hits = new AtomicInteger[n];
        for (int i = 0; i < n; i++) {
            hits[i] = new AtomicInteger();
        }
        Parallel.forRange(n, 1_000, (from, to) -> {
            for (int i = from; i < to; i++) {
                hits[i].incrementAndGet();
            }
        });
        for (int i = 0; i < n; i++) {
            assertEquals(1, hits[i].get(), "index " + i);
        }
    }
}
