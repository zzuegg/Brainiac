package io.github.zzuegg.brainiac.backend.cpu;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import io.github.zzuegg.brainiac.tensor.spi.BinaryOp;
import io.github.zzuegg.brainiac.tensor.spi.Storage;

import org.junit.jupiter.api.Test;

/** Backend kernels called directly, as a GPU backend's tests would. */
class CpuBackendTest {

    private final CpuBackend backend = new CpuBackend();

    @Test
    void isNamedCpu() {
        assertEquals("cpu", backend.name());
    }

    @Test
    void uploadAndDownloadCopy() {
        float[] data = {1, 2};
        Storage s = backend.upload(data);
        data[0] = 9;
        assertArrayEquals(new float[] {1, 2}, backend.download(s));
    }

    @Test
    void binaryHandlesGeneralStrides() {
        // a viewed transposed ([2,2] with strides [1,2]) plus b broadcast down columns
        Storage a = backend.upload(new float[] {1, 2, 3, 4});
        Storage b = backend.upload(new float[] {10, 20});
        Storage out = backend.allocate(4);
        backend.binary(BinaryOp.ADD, a, new long[] {1, 2}, b, new long[] {1, 0}, new int[] {2, 2}, out);
        assertArrayEquals(new float[] {11, 13, 22, 24}, backend.download(out));
    }
}
