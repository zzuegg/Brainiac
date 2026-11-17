package io.github.zzuegg.brainiac.examples;

import io.github.zzuegg.brainiac.backend.vulkan.VulkanBackend;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Benchmark;
import io.github.zzuegg.brainiac.train.Benchmark.Report;
import io.github.zzuegg.brainiac.train.Benchmark.Task;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Lists the devices, then compares them: a large matrix multiplication, and training the
 * same CNN on the synthetic shapes task. Run with
 * {@code ./gradlew :brainiac-examples:run --args=gpu}; add
 * {@code -Dbrainiac.vulkan.devices=all} to the JVM to also try Vulkan CPU drivers
 * (such as lavapipe).
 */
public final class GpuExample {

    private GpuExample() {
    }

    /**
     * Runs the comparison.
     *
     * @param args optional matrix size (default 1024)
     */
    public static void main(String[] args) {
        int size = args.length > 0 ? Integer.parseInt(args[0]) : 1024;
        System.out.println("devices: " + Device.available());
        System.out.println("Vulkan: " + VulkanBackend.loaded().orElseGet(VulkanBackend::new).status());
        Tensor a = Tensor.uniform(Shape.of(size, size), -1, 1, new SplittableRandom(1));
        Tensor b = Tensor.uniform(Shape.of(size, size), -1, 1, new SplittableRandom(2));
        for (Device d : Device.available()) {
            Tensor x = a.to(d);
            Tensor y = b.to(d);
            x.matmul(y).sum().item();   // warm up: compile kernels, JIT
            long start = System.nanoTime();
            int runs = 5;
            for (int i = 0; i < runs; i++) {
                x.matmul(y).sum().item();
            }
            double seconds = (System.nanoTime() - start) / 1e9 / runs;
            System.out.printf("%-60s %dx%d matmul %.1f ms, %.1f GFLOP/s%n", d, size, size, seconds * 1e3,
                    2.0 * size * size * size / seconds / 1e9);
        }
        Task task = Tasks.shapes();
        List<Report> reports = new ArrayList<>();
        for (Device d : Device.available()) {
            reports.add(Benchmark.run(task, "cnn on " + d, BenchmarkExample.model("cnn", task).to(d),
                    t -> t.epochs(3)));
        }
        System.out.println();
        System.out.print(Benchmark.table(reports));
        VulkanBackend.loaded().ifPresent(v -> System.out.println("Vulkan: " + v.statistics()));
    }
}
