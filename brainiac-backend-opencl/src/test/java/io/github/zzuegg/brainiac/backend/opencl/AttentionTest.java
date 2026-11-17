package io.github.zzuegg.brainiac.backend.opencl;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Optional;
import java.util.SplittableRandom;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/** The device's fused attention kernels give what the CPU's do. */
class AttentionTest {

    private static Device gpu;

    @BeforeAll
    static void findDevice() {
        Optional<Device> found = Device.available().stream().filter(d -> d.name().startsWith("opencl")).findFirst();
        assumeTrue(found.isPresent(), "no OpenCL device");
        gpu = found.get();
    }

    @ParameterizedTest
    @CsvSource({
        "2, 4, 4, 5, 5, 6, 6, true",
        "2, 4, 2, 5, 7, 6, 3, true",
        "1, 3, 1, 1, 9, 4, 5, true",
        "2, 2, 2, 6, 4, 8, 8, false",
        "1, 8, 2, 70, 70, 64, 64, true",
    })
    void matchesTheCpu(int batch, int heads, int kvHeads, int tq, int tk, int dk, int dv, boolean causal) {
        SplittableRandom r = new SplittableRandom(batch * 31L + tq);
        Tensor q = Tensor.uniform(Shape.of(batch, heads, tq, dk), -1, 1, r);
        Tensor k = Tensor.uniform(Shape.of(batch, kvHeads, tk, dk), -1, 1, r);
        Tensor v = Tensor.uniform(Shape.of(batch, kvHeads, tk, dv), -1, 1, r);
        Tensor g = Tensor.uniform(Shape.of(batch, heads, tq, dv), -1, 1, r);
        float scale = (float) (1 / Math.sqrt(dk));
        Tensor[] cpu = Tensor.attention(q, k, v, scale, causal);
        Tensor[] dev = Tensor.attention(q.to(gpu), k.to(gpu), v.to(gpu), scale, causal);
        assertArrayEquals(cpu[0].toArray(), dev[0].toArray(), 1e-4f, "output");
        assertArrayEquals(cpu[1].toArray(), dev[1].toArray(), 1e-4f, "log-sum-exp");
        Tensor[] cg = Tensor.attentionBackward(q, k, v, cpu[0], cpu[1], g, scale, causal);
        Tensor[] dg = Tensor.attentionBackward(q.to(gpu), k.to(gpu), v.to(gpu), dev[0], dev[1], g.to(gpu), scale,
                causal);
        for (int i = 0; i < 3; i++) {
            assertArrayEquals(cg[i].toArray(), dg[i].toArray(), 1e-4f, "gradient " + i);
        }
    }

    @ParameterizedTest
    @CsvSource({
        "1, 4, 4, 1, 9, 8, 8",
        "2, 4, 2, 3, 300, 16, 12",
        "1, 14, 2, 1, 600, 64, 64",
        "1, 2, 2, 5, 5, 4, 4",
    })
    void withABiasMatchesTheUnfusedOne(int b, int h, int kvh, int tq, int tk, int dk, int dv) {
        SplittableRandom r = new SplittableRandom(b * 7L + tk);
        Tensor q = Tensor.uniform(Shape.of(b, h, tq, dk), -1, 1, r);
        Tensor k = Tensor.uniform(Shape.of(b, kvh, tk, dk), -1, 1, r);
        Tensor v = Tensor.uniform(Shape.of(b, kvh, tk, dv), -1, 1, r);
        float[] mask = new float[b * tq * tk];
        for (int i = 0; i < mask.length; i++) {
            mask[i] = r.nextInt(4) == 0 && i % tk != 0 ? -1e9f : (float) r.nextDouble(-1, 1);
        }
        Tensor bias = Tensor.of(Shape.of(b, 1, tq, tk), mask);
        float scale = 0.3f;
        // on the CPU: weights built, masked and normalized
        float[] expected = io.github.zzuegg.brainiac.nn.Functional.attention(
                io.github.zzuegg.brainiac.autograd.Var.of(q), io.github.zzuegg.brainiac.autograd.Var.of(k),
                io.github.zzuegg.brainiac.autograd.Var.of(v), bias, scale, 0, null).value().toArray();
        assumeTrue(Tensor.hasAttentionWithBias(q.to(gpu), k.to(gpu), v.to(gpu)));
        Tensor[] fused = Tensor.attention(q.to(gpu), k.to(gpu), v.to(gpu), bias.to(gpu), scale, false);
        assertArrayEquals(expected, fused[0].toArray(), 1e-5f, "output");
        // with no bias and few rows, plain attention takes the same kernel: as the CPU's
        Tensor[] cpu = Tensor.attention(q, k, v, scale, true);
        Tensor[] dev = Tensor.attention(q.to(gpu), k.to(gpu), v.to(gpu), scale, true);
        assertArrayEquals(cpu[0].toArray(), dev[0].toArray(), 1e-5f, "causal output");
        assertArrayEquals(cpu[1].toArray(), dev[1].toArray(), 1e-5f, "causal log-sum-exp");
    }
}
