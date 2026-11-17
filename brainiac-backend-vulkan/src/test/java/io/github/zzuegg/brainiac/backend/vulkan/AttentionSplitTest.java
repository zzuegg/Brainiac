package io.github.zzuegg.brainiac.backend.vulkan;

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

/** Few query rows over many keys split the keys among work groups (flash decoding). */
class AttentionSplitTest {

    private static Device gpu;

    @BeforeAll
    static void findDevice() {
        Optional<Device> found = Device.available().stream().filter(d -> d.name().startsWith("vulkan")).findFirst();
        assumeTrue(found.isPresent(), "no Vulkan device");
        gpu = found.get();
    }

    @ParameterizedTest
    @CsvSource({
        "1, 4, 2, 1, 1000, 64, 64, true, false",
        "1, 4, 2, 1, 1000, 64, 64, false, true",
        "2, 2, 2, 3, 700, 32, 16, true, false",
        "1, 2, 1, 1, 257, 128, 128, false, false",
        "1, 1, 1, 4, 2000, 16, 16, true, true",
    })
    void matchesTheCpu(int batch, int heads, int kvHeads, int tq, int tk, int dk, int dv, boolean causal,
            boolean masked) {
        SplittableRandom r = new SplittableRandom(tk * 7L + tq);
        Tensor q = Tensor.uniform(Shape.of(batch, heads, tq, dk), -1, 1, r);
        Tensor k = Tensor.uniform(Shape.of(batch, kvHeads, tk, dk), -1, 1, r);
        Tensor v = Tensor.uniform(Shape.of(batch, kvHeads, tk, dv), -1, 1, r);
        float scale = (float) (1 / Math.sqrt(dk));
        // a cache-like mask: the buffer's unused end and a few padded keys
        float[] m = new float[batch * tq * tk];
        for (int i = 0; i < m.length; i++) {
            int key = i % tk;
            m[i] = masked && (key > tk * 3 / 5 || key % 97 == 5) ? -1e9f : 0;
        }
        Tensor bias = Tensor.of(Shape.of(batch, 1, tq, tk), m);
        // the CPU has no fused kernel with a bias: the reference then goes through the unfused path,
        // which gives no log-sum-exp
        Tensor[] cpu = masked ? new Tensor[] {io.github.zzuegg.brainiac.nn.Functional.attention(
                io.github.zzuegg.brainiac.autograd.Var.of(q), io.github.zzuegg.brainiac.autograd.Var.of(k),
                io.github.zzuegg.brainiac.autograd.Var.of(v),
                causal ? bias.add(io.github.zzuegg.brainiac.nn.Functional.causalMask(tq, tk, q.device())) : bias,
                scale, 0, null).value()} : Tensor.attention(q, k, v, scale, causal);
        for (DType type : new DType[] {DType.FLOAT32, DType.FLOAT16}) {
            if (!gpu.supports(type)) {
                continue;
            }
            float tolerance = type == DType.FLOAT32 ? 1e-4f : 2e-2f;
            Tensor qg = q.to(type).to(gpu);
            Tensor kg = k.to(type).to(gpu);
            Tensor vg = v.to(type).to(gpu);
            Tensor[] dev = masked ? Tensor.attention(qg, kg, vg, bias.to(type).to(gpu), scale, causal)
                    : Tensor.attention(qg, kg, vg, scale, causal);
            assertArrayEquals(cpu[0].toArray(), dev[0].toArray(), tolerance, type + " output");
            if (!masked) {
                assertArrayEquals(cpu[1].toArray(), dev[1].toArray(), tolerance * 10, type + " log-sum-exp");
            }
        }
    }
}
