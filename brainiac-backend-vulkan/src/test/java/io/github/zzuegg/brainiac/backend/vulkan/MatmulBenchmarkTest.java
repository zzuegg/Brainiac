package io.github.zzuegg.brainiac.backend.vulkan;

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
 * Matrix product speed on the first Vulkan device, opt-in: {@code -Dbrainiac.bench=true}
 * (Gradle: {@code -Pbench}). Prints TFLOPS for float32 and float16, the latter with and
 * without cooperative matrices.
 */
class MatmulBenchmarkTest {

    @Test
    void products() {
        assumeTrue(Boolean.getBoolean("brainiac.bench"), "benchmark: -Pbench");
        Device gpu = Device.available().stream().filter(d -> d.name().startsWith("vulkan")).findFirst().orElseThrow();
        System.out.println(gpu.name() + "; " + new VulkanBackend().status());
        for (int[] s : List.of(new int[] {512, 512, 512}, new int[] {2048, 2048, 2048}, new int[] {4096, 4096, 4096},
                new int[] {128, 4096, 4096}, new int[] {64, 4096, 4096}, new int[] {16, 4096, 4096})) {
            for (String mode : List.of("f16 coopmat", "f32", "f16")) {
                DType type = mode.equals("f32") ? DType.FLOAT32 : DType.FLOAT16;
                System.setProperty("brainiac.vulkan.coopmat", String.valueOf(mode.endsWith("coopmat")));
                Tensor a = Tensor.uniform(Shape.of(s[0], s[1]), -1, 1, new SplittableRandom(1)).to(type).to(gpu);
                Tensor b = Tensor.uniform(Shape.of(s[1], s[2]), -1, 1, new SplittableRandom(2)).to(type).to(gpu);
                a.matmul(b).toArray();   // warm up: pipelines
                int reps = (int) Math.max(3, Math.min(200, 2e11 / (2.0 * s[0] * s[1] * s[2])));
                long start = System.nanoTime();
                Tensor last = null;
                for (int r = 0; r < reps; r++) {
                    last = a.matmul(b);
                    Tensor.evaluate(List.of(last));
                }
                last.toArray();
                double seconds = (System.nanoTime() - start) / 1e9;
                System.out.printf(Locale.ROOT, "%5d x %5d x %5d %-12s %8.2f TFLOPS (%d reps, %.1f ms each)%n", s[0], s[1],
                        s[2], mode, 2.0 * s[0] * s[1] * s[2] * reps / seconds / 1e12, reps, seconds * 1e3 / reps);
            }
        }
        System.clearProperty("brainiac.vulkan.coopmat");
    }
}
