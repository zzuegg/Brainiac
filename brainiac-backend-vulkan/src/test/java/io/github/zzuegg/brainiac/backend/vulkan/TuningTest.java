package io.github.zzuegg.brainiac.backend.vulkan;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class TuningTest {

    @Test
    void keptTuningsAreReadBackAndPropertiesWin(@TempDir Path dir) {
        Path file = dir.resolve("device.properties");
        assertNull(Tuning.load(file));
        Tuning t = new Tuning();
        t.set("coopShape", 3);
        t.set("splitBelow", 64);
        t.save(file, "test");
        Tuning back = Tuning.load(file);
        assertNotNull(back);
        assertEquals(3, back.get("coopShape"));
        assertEquals(64, back.get("splitBelow"));
        assertEquals(Tuning.DEFAULTS.get("coopRows"), back.get("coopRows"));
        System.setProperty("brainiac.vulkan.splitBelow", "512");
        try {
            assertTrue(Tuning.fixed("splitBelow"));
            assertEquals(512, back.get("splitBelow"));
        } finally {
            System.clearProperty("brainiac.vulkan.splitBelow");
        }
    }

    @Test
    void unknownShapesFallBackToTheDefault(@TempDir Path dir) throws Exception {
        Path file = dir.resolve("device.properties");
        Files.writeString(file, "coopShape=99\nbigTileGroups=x\n");
        Tuning t = Tuning.load(file);
        assertEquals(Tuning.DEFAULTS.get("coopShape"), t.get("coopShape"));
        assertEquals(Tuning.DEFAULTS.get("bigTileGroups"), t.get("bigTileGroups"));
    }

    @Test
    void theDefaultStaysUnlessClearlyBeaten() {
        assertEquals(1, Tuner.pick(new double[] {1.0, 0.99, 1.2}, 1));
        assertEquals(0, Tuner.pick(new double[] {0.9, 1.0, 1.2}, 1));
        assertEquals(2, Tuner.pick(new double[] {1.0, 1.1, 0.5}, -1));
    }

    /** A tuning measured on the device is kept in the tuning directory and used. */
    @Test
    void tuningMeasuresAndKeepsTheChoices(@TempDir Path dir) {
        // the device's own backend first: it stays VulkanBackend.loaded(), whose statistics other tests read
        io.github.zzuegg.brainiac.tensor.Device.available();
        System.setProperty("brainiac.vulkan.tuningDir", dir.toString());
        System.setProperty("brainiac.vulkan.tune", "false");
        try {
            VulkanBackend backend = new VulkanBackend();
            assumeTrue(backend.isAvailable(), "no Vulkan device");
            String choices = backend.retune();
            assertTrue(choices.contains("splitBelow"), choices);
            try (var files = Files.list(dir)) {
                assertEquals(1, files.filter(f -> f.toString().endsWith(".properties")).count());
            }
            assertTrue(backend.tuning().contains("tuned now"), backend.tuning());
        } catch (java.io.IOException e) {
            throw new java.io.UncheckedIOException(e);
        } finally {
            System.clearProperty("brainiac.vulkan.tuningDir");
            System.clearProperty("brainiac.vulkan.tune");
        }
    }
}
