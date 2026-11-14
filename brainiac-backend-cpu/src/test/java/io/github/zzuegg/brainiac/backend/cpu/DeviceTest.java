package io.github.zzuegg.brainiac.backend.cpu;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.zzuegg.brainiac.tensor.Device;

import java.util.List;

import org.junit.jupiter.api.Test;

/** Device discovery through ServiceLoader. */
class DeviceTest {

    @Test
    void findsTheCpuBackend() {
        assertEquals(List.of(Device.cpu()), Device.available());
        assertEquals("cpu", Device.cpu().name());
        assertEquals(Device.Kind.CPU, Device.cpu().kind());
        assertEquals(Device.cpu(), Device.best());
        assertEquals(Device.cpu(), Device.getDefault());
    }

    @Test
    void explainsWhenThereIsNoGpu() {
        IllegalStateException e = assertThrows(IllegalStateException.class, Device::gpu);
        assertEquals("No GPU backend available. Available devices: [cpu]", e.getMessage());
    }
}
