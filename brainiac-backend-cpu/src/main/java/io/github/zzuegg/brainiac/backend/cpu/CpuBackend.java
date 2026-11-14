package io.github.zzuegg.brainiac.backend.cpu;

import io.github.zzuegg.brainiac.tensor.spi.Backend;

/**
 * Runs tensor operations on the CPU.
 */
public final class CpuBackend implements Backend {

    /** Creates the backend; called by {@link java.util.ServiceLoader}. */
    public CpuBackend() {
    }

    @Override
    public String name() {
        return "cpu";
    }
}
