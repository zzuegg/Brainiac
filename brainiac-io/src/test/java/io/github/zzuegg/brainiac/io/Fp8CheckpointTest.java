package io.github.zzuegg.brainiac.io;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class Fp8CheckpointTest {

    private static final Device CPU = Device.cpu();

    @TempDir
    Path dir;

    /** A [4, 4] e4m3 weight of ones and twos with 2 x 2 blocks scaled 1, 2, 3 and 4, as Qwen3-FP8 lays it out. */
    private Path checkpoint(String quantization) throws IOException {
        String header = "{"
                + "\"proj.weight\":{\"dtype\":\"F8_E4M3\",\"shape\":[4,4],\"data_offsets\":[0,16]},"
                + "\"proj.weight_scale_inv\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[16,32]},"
                + "\"proj.input_scale\":{\"dtype\":\"F32\",\"shape\":[],\"data_offsets\":[32,36]},"
                + "\"norm.weight\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[36,44]}}";
        byte[] h = header.getBytes(StandardCharsets.UTF_8);
        ByteBuffer b = ByteBuffer.allocate(8 + h.length + 44).order(ByteOrder.LITTLE_ENDIAN);
        b.putLong(h.length).put(h);
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                b.put((byte) (c % 2 == 0 ? 0x38 : 0x40));   // 1 and 2
            }
        }
        b.putFloat(1).putFloat(2).putFloat(3).putFloat(4);
        b.putFloat(0.5f);
        b.putFloat(7).putFloat(8);
        Files.write(dir.resolve("model.safetensors"), b.array());
        Files.writeString(dir.resolve("config.json"), "{\"model_type\":\"qwen3\",\"quantization_config\":"
                + quantization + "}");
        return dir;
    }

    @Test
    void blockScalesMultiplyTheirTiles() throws IOException {
        try (WeightSource w = WeightSource.open(checkpoint(
                "{\"quant_method\":\"fp8\",\"fmt\":\"e4m3\",\"weight_block_size\":[2,2]}"))) {
            assertEquals(java.util.List.of("proj.weight", "norm.weight"), w.names());
            assertFalse(w.contains("proj.weight_scale_inv"));
            assertFalse(w.contains("proj.input_scale"));
            assertEquals(DType.BFLOAT16, w.naturalType("proj.weight"));
            assertEquals(Shape.of(4, 4), w.shape("proj.weight"));
            assertArrayEquals(new float[] {
                1, 2, 2, 4,
                1, 2, 2, 4,
                3, 6, 4, 8,
                3, 6, 4, 8}, w.tensor("proj.weight", CPU, DType.FLOAT32).toArray());
            assertArrayEquals(new float[] {7, 8}, w.tensor("norm.weight", CPU, DType.FLOAT32).toArray());
        }
    }

    @Test
    void tilesAreInferredWithoutABlockSize() throws IOException {
        try (WeightSource w = WeightSource.open(checkpoint("{\"quant_method\":\"fp8\"}"))) {
            assertArrayEquals(new float[] {1, 2, 2, 4}, java.util.Arrays.copyOf(
                    w.tensor("proj.weight", CPU, DType.FLOAT32).toArray(), 4));
        }
    }
}
