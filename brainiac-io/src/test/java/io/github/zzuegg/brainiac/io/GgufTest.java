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
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * Reads fixture.gguf, written by llama.cpp's own gguf-py (make_fixture_gguf.py): random
 * blocks of every quantized format next to gguf-py's dequantization of them.
 */
class GgufTest {

    @TempDir
    static Path dir;

    private static Path file;
    private static final Device DEVICE = TestDevice.get();

    @BeforeAll
    static void copyFixture() throws IOException {
        file = dir.resolve("fixture.gguf");
        try (InputStream in = GgufTest.class.getResourceAsStream("fixture.gguf")) {
            Files.copy(in, file);
        }
    }

    @Test
    void readsMetadata() {
        try (Gguf g = Gguf.open(file)) {
            assertEquals(3, g.version());
            assertEquals("llama", g.architecture());
            assertEquals(42L, g.metadata().get("test.u32"));
            assertEquals(0.25, g.metadata().get("test.f32"));
            assertEquals("héllo", g.metadata().get("test.str"));
            assertEquals(true, g.metadata().get("test.bool"));
            assertEquals(List.of("a", "b"), g.metadata().get("test.arr"));
            assertEquals(List.of(1L, 2L, 3L), g.metadata().get("test.ints"));
            assertEquals(1L << 40, g.metadata().get("test.u64"));
            assertEquals(-3L, g.metadata().get("test.i8"));
        }
    }

    @ParameterizedTest
    @ValueSource(strings = {"Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_K"})
    void decodesQuantizedFormatsLikeLlamaCpp(String format) {
        try (WeightSource g = WeightSource.open(file)) {
            String name = "q." + format;
            assertEquals(format, g.format(name));
            Tensor expected = g.tensor("expected." + format, DEVICE);
            Tensor actual = g.tensor(name, DEVICE);
            assertEquals(expected.shape(), actual.shape());
            assertArrayEquals(expected.toArray(), actual.toArray(), 1e-6f, format);
        }
    }

    @Test
    void readsPlainTypesRowMajor() {
        try (Gguf g = Gguf.open(file)) {
            assertEquals(Shape.of(2, 3, 4), g.shape("plain.f32"));
            assertEquals(DType.FLOAT16, g.naturalType("plain.f16"));
            float[] f32 = g.tensor("plain.f32", DEVICE).toArray();
            float[] f16 = g.tensor("plain.f16", DEVICE, DType.FLOAT32).toArray();
            assertArrayEquals(f32, f16, 1e-3f);
            assertArrayEquals(g.tensor("expected.bf16", DEVICE).toArray(), g.tensor("plain.bf16", DEVICE, DType.FLOAT32)
                    .toArray());
            assertArrayEquals(new float[] {0, 1, 2, 3, 4}, g.tensor("plain.i32", DEVICE).toArray());
        }
    }

    @ParameterizedTest
    @ValueSource(strings = {"Q8_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"})
    void keepsQuantizedMatricesPacked(String format) {
        try (Gguf g = Gguf.open(file)) {
            String name = "q." + format;
            assertTrue(g.packable(name));
            Tensor packed = g.packedTransposed(name, DEVICE);
            assertTrue(packed.isPacked());
            Tensor dense = g.tensor("expected." + format, DEVICE);
            assertEquals(dense.transpose().shape(), packed.shape(), format);
            if (dense.shape().rank() != 2) {
                return;
            }
            assertArrayEquals(dense.transpose().toArray(), packed.unpacked().toArray(), 1e-6f, format);
            // multiplying by it reads the levels directly
            Tensor x = Tensor.uniform(Shape.of(2, packed.shape().dim(0)), -1, 1, new java.util.SplittableRandom(3))
                    .to(DEVICE);
            float[] expected = x.matmul(dense.transpose()).toArray();
            float largest = 1;
            for (float v : expected) {
                largest = Math.max(largest, Math.abs(v));
            }
            // sums in another order: rounding grows with the values
            assertArrayEquals(expected, x.matmul(packed).toArray(), 1e-5f * largest, format);
            assertFalse(g.packable("q.Q8_K"));
            assertFalse(g.packable("plain.f32"));
            assertThrows(IllegalArgumentException.class, () -> g.packedTransposed("q.Q8_K", DEVICE));
        }
    }

    @Test
    void rejectsOtherFiles() throws IOException {
        Path bad = dir.resolve("bad.gguf");
        Files.write(bad, new byte[64]);
        assertThrows(IllegalArgumentException.class, () -> Gguf.open(bad));
    }
}
