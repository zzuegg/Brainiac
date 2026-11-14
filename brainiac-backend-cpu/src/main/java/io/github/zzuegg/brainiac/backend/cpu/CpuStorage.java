package io.github.zzuegg.brainiac.backend.cpu;

import io.github.zzuegg.brainiac.tensor.spi.Storage;

/** Storage backed by a Java float array. */
final class CpuStorage implements Storage {

    final float[] data;

    CpuStorage(float[] data) {
        this.data = data;
    }

    @Override
    public long size() {
        return data.length;
    }
}
