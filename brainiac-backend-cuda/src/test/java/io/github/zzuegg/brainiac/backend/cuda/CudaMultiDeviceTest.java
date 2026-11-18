package io.github.zzuegg.brainiac.backend.cuda;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

/**
 * Sums across devices: with NCCL between the machine's CUDA GPUs when there are two or more,
 * on the host between a GPU and the CPU.
 */
class CudaMultiDeviceTest {

    private static List<Device> gpus;

    @BeforeAll
    static void findDevices() {
        gpus = Device.available().stream().filter(d -> d.name().startsWith("cuda")).toList();
        assumeTrue(!gpus.isEmpty(), "no CUDA device");
        System.out.println("CUDA devices: " + gpus + "; " + CudaBackend.ncclStatus());
    }

    private static void sumsMatch(List<Device> devices, DType type) {
        int n = 3001;
        List<Tensor> parts = new ArrayList<>();
        Tensor expected = Tensor.zeros(Shape.of(n), type, Device.cpu());
        for (int i = 0; i < devices.size(); i++) {
            Tensor t = Tensor.uniform(Shape.of(n), -1, 1, new SplittableRandom(i)).to(type);
            expected = expected.add(t);
            parts.add(t.to(devices.get(i)));
        }
        List<Tensor> sums = Tensor.sumAcross(parts);
        assertEquals(devices.size(), sums.size());
        float tolerance = type == DType.FLOAT32 ? 1e-6f : type == DType.FLOAT16 ? 5e-3f : 3e-2f;
        for (int i = 0; i < devices.size(); i++) {
            assertEquals(devices.get(i), sums.get(i).device());
            assertEquals(type, sums.get(i).dtype());
            assertArrayEquals(expected.toArray(), sums.get(i).toArray(), tolerance * devices.size(),
                    type + " on " + devices.get(i));
        }
    }

    @ParameterizedTest
    @EnumSource(value = DType.class, names = {"FLOAT32", "FLOAT16", "BFLOAT16"})
    void sumsBetweenAGpuAndTheCpuMatch(DType type) {
        sumsMatch(List.of(gpus.get(0), Device.cpu()), type);
    }

    @ParameterizedTest
    @EnumSource(value = DType.class, names = {"FLOAT32", "FLOAT16", "BFLOAT16"})
    void sumsBetweenGpusMatch(DType type) {
        assumeTrue(gpus.size() >= 2, "one CUDA device");
        sumsMatch(gpus, type);
        sumsMatch(gpus.reversed(), type);   // another order: another communicator group
    }
}
