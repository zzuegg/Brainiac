package io.github.zzuegg.brainiac.backend.cuda;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Optional;
import java.util.SplittableRandom;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/** The fused attention's gradients (attention_dq, attention_dkv) give what the CPU's do. */
class AttentionBackwardTest {

    private static Device gpu;

    @BeforeAll
    static void findDevice() {
        Optional<Device> found = Device.available().stream().filter(d -> d.name().startsWith("cuda")).findFirst();
        assumeTrue(found.isPresent(), "no CUDA device");
        gpu = found.get();
    }

    @ParameterizedTest
    @CsvSource({
        "2, 4, 4, 5, 5, 6, 6, true",
        "2, 4, 2, 5, 7, 6, 3, true",
        "1, 3, 1, 1, 9, 4, 5, true",
        "2, 2, 2, 6, 4, 8, 8, false",
        "1, 8, 2, 70, 70, 64, 64, true",
        "1, 4, 2, 130, 150, 32, 16, true",
        "2, 2, 1, 100, 40, 16, 32, false",
        "1, 4, 2, 2, 600, 32, 32, true",
        "1, 2, 1, 3, 40, 256, 200, true",
    })
    void gradientsMatchTheCpu(int batch, int heads, int kvHeads, int tq, int tk, int dk, int dv, boolean causal) {
        SplittableRandom r = new SplittableRandom(batch * 31L + tq);
        Tensor q = Tensor.uniform(Shape.of(batch, heads, tq, dk), -1, 1, r);
        Tensor k = Tensor.uniform(Shape.of(batch, kvHeads, tk, dk), -1, 1, r);
        Tensor v = Tensor.uniform(Shape.of(batch, kvHeads, tk, dv), -1, 1, r);
        Tensor g = Tensor.uniform(Shape.of(batch, heads, tq, dv), -1, 1, r);
        float scale = (float) (1 / Math.sqrt(dk));
        Tensor[] cpu = Tensor.attention(q, k, v, scale, causal);
        Tensor[] cg = Tensor.attentionBackward(q, k, v, cpu[0], cpu[1], g, scale, causal);
        for (DType type : new DType[] {DType.FLOAT32, DType.FLOAT16, DType.BFLOAT16}) {
            if (!gpu.supports(type)) {
                continue;
            }
            float tolerance = type == DType.FLOAT32 ? 1e-4f : type == DType.FLOAT16 ? 3e-2f : 1.5e-1f;
            Tensor[] dev = Tensor.attention(q.to(type).to(gpu), k.to(type).to(gpu), v.to(type).to(gpu), scale,
                    causal);
            Tensor[] dg = Tensor.attentionBackward(q.to(type).to(gpu), k.to(type).to(gpu), v.to(type).to(gpu),
                    dev[0], dev[1], g.to(type).to(gpu), scale, causal);
            for (int i = 0; i < 3; i++) {
                assertArrayEquals(cg[i].toArray(), dg[i].toArray(), tolerance, type + " gradient " + i);
            }
        }
    }
}
