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
}
