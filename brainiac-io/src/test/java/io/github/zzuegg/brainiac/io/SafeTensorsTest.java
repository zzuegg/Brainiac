package io.github.zzuegg.brainiac.io;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class SafeTensorsTest {

    @TempDir
    Path dir;

    private static final Device DEVICE = TestDevice.get();

    private static Tensor random(long seed, int... dims) {
        return Tensor.uniform(Shape.of(dims), -2, 2, new SplittableRandom(seed));
    }

    @Test
    void roundTripsEveryType() {
        Map<String, Tensor> out = new LinkedHashMap<>();
        for (DType t : DType.values()) {
            out.put("w_" + t, random(t.ordinal(), 3, 5).to(t));
        }
        out.put("scalar", Tensor.scalar(4.5f));
        Path file = dir.resolve("all.safetensors");
        SafeTensors.write(file, out, Map.of("format", "pt"));
        try (SafeTensors in = SafeTensors.open(file)) {
            assertEquals(List.copyOf(out.keySet()), in.names());
            assertEquals(Map.of("format", "pt"), in.metadata());
            assertEquals("BF16", in.format("w_BFLOAT16"));
            for (DType t : DType.values()) {
                String n = "w_" + t;
                assertEquals(Shape.of(3, 5), in.shape(n));
                assertEquals(t, in.naturalType(n));
                Tensor back = in.tensor(n, DEVICE);
                assertEquals(t, back.dtype());
                assertArrayEquals(out.get(n).toDoubleArray(), back.toDoubleArray(), n);
                // converted while loading
                assertArrayEquals(out.get(n).to(DType.FLOAT32).toArray(), in.tensor(n, DEVICE, DType.FLOAT32).toArray());
            }
            assertEquals(Shape.of(), in.shape("scalar"));
            assertEquals(4.5f, in.tensor("scalar", DEVICE).item());
        }
    }

    /** Builds a file by hand, as another library would, with types Brainiac never writes. */
    private Path handMade() throws IOException {
        String header = "{\"__metadata__\":{\"k\":\"v\"},"
                + "\"ids\":{\"dtype\":\"I64\",\"shape\":[3],\"data_offsets\":[0,24]},"
                + "\"mask\":{\"dtype\":\"BOOL\",\"shape\":[2],\"data_offsets\":[24,26]},"
                + "\"e4\":{\"dtype\":\"F8_E4M3\",\"shape\":[4],\"data_offsets\":[26,30]},"
                + "\"e5\":{\"dtype\":\"F8_E5M2\",\"shape\":[2],\"data_offsets\":[30,32]},"
                + "\"bf\":{\"dtype\":\"BF16\",\"shape\":[2,1],\"data_offsets\":[32,36]}}";
        byte[] h = header.getBytes(StandardCharsets.UTF_8);
        ByteBuffer b = ByteBuffer.allocate(8 + h.length + 36).order(ByteOrder.LITTLE_ENDIAN);
        b.putLong(h.length).put(h);
        b.putLong(-1).putLong(7).putLong(1L << 40);
        b.put((byte) 1).put((byte) 0);
        b.put((byte) 0x38).put((byte) 0xC0).put((byte) 0x01).put((byte) 0x7E);   // 1, -2, 2^-9, 448
        b.put((byte) 0x3C).put((byte) 0xC4);                                     // 1, -4
        b.putShort(DType.floatToBfloat16(1.5f)).putShort(DType.floatToBfloat16(-3f));
        Path file = dir.resolve("hand.safetensors");
        Files.write(file, b.array());
        return file;
    }

    @Test
    void decodesOtherStoredTypes() throws IOException {
        try (SafeTensors in = SafeTensors.open(handMade())) {
            assertEquals(Map.of("k", "v"), in.metadata());
            assertEquals(DType.FLOAT32, in.naturalType("ids"));
            assertArrayEquals(new float[] {-1, 7, 1L << 40}, in.tensor("ids", DEVICE).toArray());
            assertArrayEquals(new float[] {1, 0}, in.tensor("mask", DEVICE).toArray());
            assertArrayEquals(new float[] {1, -2, 0x1p-9f, 448}, in.tensor("e4", DEVICE).toArray());
            assertArrayEquals(new float[] {1, -4}, in.tensor("e5", DEVICE).toArray());
            assertArrayEquals(new double[] {1.5, -3}, in.tensor("bf", DEVICE, DType.FLOAT64).toDoubleArray());
            assertEquals(4, in.raw("bf").byteSize());
        }
    }

    @Test
    void rejectsBrokenFiles() throws IOException {
        Path tooShort = dir.resolve("short.safetensors");
        Files.write(tooShort, new byte[] {1, 2});
        assertThrows(IllegalArgumentException.class, () -> SafeTensors.open(tooShort));
        String header = "{\"w\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,8]}}";
        byte[] h = header.getBytes(StandardCharsets.UTF_8);
        ByteBuffer b = ByteBuffer.allocate(8 + h.length + 8).order(ByteOrder.LITTLE_ENDIAN);
        b.putLong(h.length).put(h);
        Path wrongSize = dir.resolve("size.safetensors");
        Files.write(wrongSize, b.array());
        assertThrows(IllegalArgumentException.class, () -> SafeTensors.open(wrongSize));
    }

    @Test
    void opensShardedModelsAndRenames() throws IOException {
        Tensor a = random(1, 2, 2);
        Tensor b = random(2, 3);
        Tensor c = random(3, 1, 4);
        Path model = Files.createDirectory(dir.resolve("model"));
        SafeTensors.write(model.resolve("model-00001-of-00002.safetensors"),
                Map.of("model.layers.0.self_attn.q_proj.weight", a));
        Map<String, Tensor> second = new LinkedHashMap<>();
        second.put("model.layers.1.self_attn.q_proj.weight", b);
        second.put("model.rotary_emb.inv_freq", c);
        SafeTensors.write(model.resolve("model-00002-of-00002.safetensors"), second);
        Files.writeString(model.resolve("model.safetensors.index.json"), """
                {"metadata": {"total_size": 1},
                 "weight_map": {
                   "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
                   "model.layers.1.self_attn.q_proj.weight": "model-00002-of-00002.safetensors",
                   "model.rotary_emb.inv_freq": "model-00002-of-00002.safetensors"}}""");

        NameMap map = NameMap.create()
                .drop(".*rotary_emb\\.inv_freq")
                .rule("model\\.layers\\.(\\d+)\\.self_attn\\.q_proj\\.weight", "block$1.query");
        try (WeightSource w = WeightSource.open(model).renamed(map)) {
            assertEquals(List.of("block0.query", "block1.query"), w.names());
            assertTrue(w.contains("block1.query"));
            assertFalse(w.contains("model.rotary_emb.inv_freq"));
            assertEquals(Shape.of(3), w.shape("block1.query"));
            assertEquals("F32", w.format("block1.query"));
            assertArrayEquals(a.toArray(), w.tensor("block0.query", DEVICE).toArray());
            assertArrayEquals(b.toArray(), w.tensor("block1.query", DEVICE).toArray());
            assertEquals(2, w.loadAll(DEVICE, DType.FLOAT32).size());
        }
        // a directory of shards without an index works too
        Files.delete(model.resolve("model.safetensors.index.json"));
        try (WeightSource w = WeightSource.open(model)) {
            assertEquals(3, w.names().size());
        }
    }

    @Test
    void nameMapKeepsOrDropsUnmatched() {
        NameMap keep = NameMap.create().rule("a(\\d)", "b$1");
        assertEquals("b1", keep.apply("a1"));
        assertEquals("zz", keep.apply("zz"));
        assertEquals(null, NameMap.create().dropUnmatched().apply("zz"));
    }
}
