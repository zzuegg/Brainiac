package io.github.zzuegg.brainiac.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

class QuantizationTest {

    private static final Tensor X = Tensor.uniform(Shape.of(6, 16), -1, 1, new SplittableRandom(3));

    @TempDir
    Path dir;

    private static Tensor weights(int rows, int cols, long seed) {
        return Tensor.normal(Shape.of(rows, cols), 0, 0.3f, new SplittableRandom(seed));
    }

    /** The scale of each column: largest magnitude over the levels. */
    private static float[] columnScales(float[] w, int rows, int cols, int levels) {
        float[] s = new float[cols];
        for (int i = 0; i < rows * cols; i++) {
            s[i % cols] = Math.max(s[i % cols], Math.abs(w[i]));
        }
        for (int c = 0; c < cols; c++) {
            s[c] /= levels;
        }
        return s;
    }

    @ParameterizedTest
    @ValueSource(ints = {2, 3, 4, 8})
    void perChannelValuesLieOnTheGridAndKeepEachChannelsLargestMagnitude(int bits) {
        Quantization q = Quantization.bits(bits);
        Tensor w = weights(12, 5, 1);
        float[] before = w.toArray();
        float[] after = q.quantize(w, 1).toArray();
        float[] scales = columnScales(before, 12, 5, q.levels());
        float[] maxAfter = columnScales(after, 12, 5, q.levels());
        assertArrayEquals(scales, maxAfter, 1e-7f);
        for (int i = 0; i < before.length; i++) {
            float s = scales[i % 5];
            float level = after[i] / s;
            assertEquals(Math.rint(level), level, 1e-4, "value " + i + " is off the grid");
            assertTrue(Math.abs(level) <= q.levels() + 1e-4);
            assertTrue(Math.abs(after[i] - before[i]) <= s / 2 + 1e-6f, "error above half a step at " + i);
        }
    }

    @Test
    void int8ErrorIsSmall() {
        Tensor w = weights(64, 32, 2);
        Tensor err = Quantization.int8().quantize(w, 1).sub(w);
        float relative = (float) Math.sqrt(err.square().mean().item() / w.square().mean().item());
        assertTrue(relative < 0.02f, "relative error " + relative);
    }

    @Test
    void groupsHaveTheirOwnScales() {
        // the first 4 inputs are large, the last 4 small: per channel the small ones round away
        float[] values = new float[8 * 2];
        for (int r = 0; r < 8; r++) {
            for (int c = 0; c < 2; c++) {
                values[r * 2 + c] = (r < 4 ? 10f : 0.01f) * (1 + r + c);
            }
        }
        Tensor w = Tensor.of(Shape.of(8, 2), values);
        float[] channel = Quantization.int4().quantize(w, 1).toArray();
        float[] grouped = Quantization.int4().groups(4).quantize(w, 1).toArray();
        assertEquals(0f, channel[15]);
        assertEquals(values[15], grouped[15], values[15] / 7 / 2 + 1e-7f);
        // channel-first layout: groups along each row
        Tensor wt = w.transpose().reshape(2, 8);
        float[] groupedT = Quantization.int4().groups(4).quantize(wt, 0).toArray();
        assertEquals(grouped[15], groupedT[15], 1e-7f);
    }

    @Test
    void groupsThatDoNotDivideFallBackToChannels() {
        Tensor w = weights(10, 3, 4);
        assertArrayEquals(Quantization.int4().quantize(w, 1).toArray(),
                Quantization.int4().groups(4).quantize(w, 1).toArray());
    }

    @Test
    void perTensorUsesOneScale() {
        Tensor w = weights(6, 4, 5);
        float[] after = Quantization.int4().perTensor().quantize(w, 1).toArray();
        float scale = w.abs().max().item() / 7;
        for (float v : after) {
            assertEquals(Math.rint(v / scale), v / scale, 1e-4);
        }
    }

    @Test
    void zeroChannelsStayZero() {
        Tensor w = Tensor.zeros(Shape.of(4, 3));
        assertArrayEquals(new float[12], Quantization.int8().quantize(w, 1).toArray());
    }

    @ParameterizedTest
    @ValueSource(ints = {2, 3, 4, 5, 8})
    void packingReproducesTheRoundedValues(int bits) {
        for (Quantization q : List.of(Quantization.bits(bits), Quantization.bits(bits).groups(4),
                Quantization.bits(bits).perTensor())) {
            for (int axis : new int[] {0, 1}) {
                Tensor w = weights(8, 6, bits * 7L + axis);
                float[] rounded = q.quantize(w, axis).toArray();
                Shape shape = w.shape();
                Quantization.Packed packed = q.pack(rounded, shape, axis);
                assertEquals(q.dataBytes(shape.size()), packed.data().length);
                assertEquals(q.scaleCount(shape, axis), packed.scales().length);
                assertArrayEquals(rounded, q.unpack(packed, shape, axis), 0f, q + " axis " + axis);   // 0 == -0
                // packing unrounded weights rounds them the same way
                assertArrayEquals(rounded, q.unpack(q.pack(w.toArray(), shape, axis), shape, axis), 1e-6f);
            }
        }
    }

    @Test
    void choosesWeightMatricesButNotBiasesNormsOrActivationParameters() {
        Model m = Model.sequential().input(1, 8, 8)
                .conv2d(4, 3, Activation.RELU).flatten()
                .dense(16, Activation.mix(Activation.RELU, Activation.TANH)).layerNorm().dense(3).seed(1).build();
        List<Quantization.Weight> chosen = Quantization.weights(m);
        assertEquals(3, chosen.size());
        assertSame(((Conv) m.layers().get(0)).weight(), chosen.get(0).weight());
        assertEquals(0, chosen.get(0).axis());
        assertEquals(1, chosen.get(1).axis());
        Model embed = Model.sequential().input(5).embedding(20, 8).seed(1).build();
        assertEquals(0, Quantization.weights(embed).get(0).axis());
    }

    @Test
    void int8BarelyChangesPredictions() {
        Model m = Model.sequential().input(16).dense(32, Activation.RELU).dense(4).seed(7).build();
        Var bias = ((Dense) m.layers().get(0)).bias();
        bias.assign(Tensor.full(bias.shape(), 0.123456f));
        float[] before = m.predict(X).toArray();
        assertSame(m, m.quantize(Quantization.int8()));
        assertEquals(Quantization.int8(), m.quantization());
        assertArrayEquals(before, m.predict(X).toArray(), 0.05f);
        assertEquals(0.123456f, bias.value().toArray()[0]);
    }

    @Test
    void quantizedModelsSaveSmallerAndLoadExactly() throws IOException {
        Model m = Model.sequential().input(16).dense(256, Activation.RELU).dense(256, Activation.RELU).dense(4)
                .seed(7).build();
        Path full = dir.resolve("full.brainiac");
        m.save(full);
        List<Quantization> schemes = List.of(Quantization.int8(), Quantization.int4().groups(32),
                Quantization.bits(3).perTensor());
        double[] smaller = {3.5, 5, 8};
        for (int s = 0; s < schemes.size(); s++) {
            Quantization q = schemes.get(s);
            Model copy = Model.load(full).quantize(q);
            Path file = dir.resolve(q.bits() + "-" + q.groupSize() + ".brainiac");
            copy.save(file);
            double ratio = (double) Files.size(full) / Files.size(file);
            assertTrue(ratio > smaller[s], q + " is only " + ratio + "x smaller");
            Model loaded = Model.load(file);
            assertEquals(q, loaded.quantization());
            for (int i = 0; i < m.parameters().size(); i++) {
                assertArrayEquals(copy.parameters().get(i).value().toArray(),
                        loaded.parameters().get(i).value().toArray(), 0f, q + " parameter " + i);
            }
            assertArrayEquals(copy.predict(X).toArray(), loaded.predict(X).toArray());
            Model weightsOnly = Model.sequential().input(16).dense(256, Activation.RELU).dense(256, Activation.RELU)
                    .dense(4).seed(99).build().loadWeights(file);
            assertArrayEquals(copy.predict(X).toArray(), weightsOnly.predict(X).toArray());
            assertEquals(q, copy.copy(Device.cpu()).quantization());
        }
        assertNull(Model.load(full).quantization());
    }

    @Test
    void unquantizedFilesKeepTheOldVersion() throws IOException {
        Model m = Model.sequential().input(16).dense(4).seed(7).build();
        Path file = dir.resolve("plain.brainiac");
        m.save(file);
        assertEquals(4, java.nio.ByteBuffer.wrap(Files.readAllBytes(file), 4, 4).getInt());
        m.quantize(Quantization.int8()).save(file);
        assertEquals(5, java.nio.ByteBuffer.wrap(Files.readAllBytes(file), 4, 4).getInt());
    }

    @Test
    void halfPrecisionModelsKeepTheirFormat() throws IOException {
        Model m = Model.sequential().input(16).dense(8, Activation.TANH).dense(4).seed(7).build().to(DType.BFLOAT16);
        m.quantize(Quantization.int8());
        assertEquals(DType.BFLOAT16, m.dtype());
        Path file = dir.resolve("bf16.brainiac");
        m.save(file);
        Model loaded = Model.load(file);
        assertEquals(DType.BFLOAT16, loaded.dtype());
        assertArrayEquals(m.predict(X).toArray(), loaded.predict(X).toArray(), 0.02f);
    }

    @Test
    void rejectsBadSettings() {
        assertThrows(IllegalArgumentException.class, () -> Quantization.bits(1));
        assertThrows(IllegalArgumentException.class, () -> Quantization.bits(9));
        assertThrows(IllegalArgumentException.class, () -> Quantization.int4().groups(1));
        assertThrows(IllegalArgumentException.class, () -> Quantization.int8().quantize(Tensor.zeros(Shape.of(2, 3, 4)), 1));
    }
}
