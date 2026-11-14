package io.github.zzuegg.brainiac.backend.cpu;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class CpuBackendTest {

    @Test
    void isNamedCpu() {
        assertEquals("cpu", new CpuBackend().name());
    }
}
