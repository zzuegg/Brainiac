package io.github.zzuegg.brainiac.backend.vulkan;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.zzuegg.brainiac.tensor.spi.Storage;

import org.junit.jupiter.api.Test;

class CloseTest {

    @Test
    void closingReleasesTheDeviceOnce() {
        VulkanBackend b = new VulkanBackend();
        assumeTrue(b.isAvailable(), "no Vulkan device");
        Storage s = b.upload(new float[] {1, 2, 3});
        Storage big = b.allocate(3 << 20);
        assertArrayEquals(new float[] {1, 2, 3}, b.download(s));
        b.release(big);
        b.close();
        assertFalse(b.isAvailable());
        assertTrue(b.status().contains("closed"), b.status());
        b.close();
    }
}
