package io.github.zzuegg.brainiac.io;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;

import java.io.IOException;
import java.net.URISyntaxException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Checkpoints from {@code pytorch/make_pytorch.py}, read against the same tensors as safetensors. */
class PyTorchCheckpointTest {

    private static Path resource(String name) throws URISyntaxException {
        return Path.of(PyTorchCheckpointTest.class.getResource("pytorch/" + name).toURI());
    }

    private static void sameTensors(WeightSource actual, WeightSource expected) {
        Device device = TestDevice.get();
        assertEquals(new java.util.TreeSet<>(expected.names()), new java.util.TreeSet<>(actual.names()));
        for (String n : expected.names()) {
            assertEquals(expected.shape(n), actual.shape(n), n);
            assertEquals(expected.format(n), actual.format(n), n);
            assertEquals(expected.naturalType(n), actual.naturalType(n), n);
            assertArrayEquals(expected.tensor(n, Device.cpu(), DType.FLOAT32).toArray(),
                    actual.tensor(n, device, DType.FLOAT32).toArray(), n);
        }
    }

    @Test
    void readsViewsOffsetsAndEveryDtype() throws Exception {
        try (WeightSource pt = WeightSource.open(resource("model.bin"));
                WeightSource st = WeightSource.open(resource("expected.safetensors"))) {
            assertEquals(List.of("a.weight", "a.weight_t", "b.half", "b.bf16", "c.ids", "c.slice", "d.scalar",
                    "e.flag", "f.param"), pt.names());
            sameTensors(pt, st);
            assertEquals(DType.FLOAT16, pt.tensor("b.half", TestDevice.get()).dtype());
        }
    }

    @Test
    void readsTheLegacyFormatBeforePyTorch16() throws Exception {
        try (WeightSource pt = WeightSource.open(resource("model-legacy.bin"));
                WeightSource st = WeightSource.open(resource("expected.safetensors"))) {
            sameTensors(pt, st);
        }
    }

    @Test
    void readsAShardedHuggingFaceFolder() throws Exception {
        try (WeightSource pt = WeightSource.open(resource("sharded"));
                WeightSource st = WeightSource.open(resource("sharded-expected.safetensors"))) {
            sameTensors(pt, st);
        }
    }

    @Test
    void refusesFilesThatAreNotCheckpoints(@TempDir Path dir) throws IOException {
        Path junk = dir.resolve("model.bin");
        Files.write(junk, new byte[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23});
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> WeightSource.open(junk));
        assertTrue(e.getMessage().contains("zip"), e.getMessage());
    }
}
