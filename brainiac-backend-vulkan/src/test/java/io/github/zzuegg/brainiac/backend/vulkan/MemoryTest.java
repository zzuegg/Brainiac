package io.github.zzuegg.brainiac.backend.vulkan;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.Optional;

import org.junit.jupiter.api.Test;

class MemoryTest {

    @Test
    void countsLiveTensorsAndThePeak() {
        Optional<Device> found = Device.available().stream().filter(d -> d.name().startsWith("vulkan")).findFirst();
        assumeTrue(found.isPresent(), "no Vulkan device");
        Device gpu = found.get();
        assertTrue(Device.cpu().memory().isEmpty());
        gpu.resetPeakMemory();
        Device.Memory before = gpu.memory().orElseThrow();
        Tensor t = Tensor.zeros(Shape.of(1 << 20), gpu);
        Tensor.evaluate(List.of(t));
        Device.Memory during = gpu.memory().orElseThrow();
        assertTrue(during.inUse() - before.inUse() >= 4L << 20, before + " -> " + during);
        assertTrue(during.peak() >= during.inUse());
        assertTrue(during.reserved() >= during.inUse() && during.budget() > 0, during.toString());
        t.release();
        Device.Memory after = gpu.memory().orElseThrow();
        assertTrue(after.inUse() < during.inUse(), during + " -> " + after);
        assertTrue(after.peak() >= during.inUse());
    }
}
