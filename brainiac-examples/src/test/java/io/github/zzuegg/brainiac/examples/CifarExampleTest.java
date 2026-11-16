package io.github.zzuegg.brainiac.examples;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.data.Augmentation;
import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.train.Metric;
import io.github.zzuegg.brainiac.train.Optimizer;
import io.github.zzuegg.brainiac.train.Trainer;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;

import org.junit.jupiter.api.Test;

/** CIFAR-10 is read from its archive format, and the residual network learns. */
class CifarExampleTest {

    /** A tar entry: a 512-byte header with the name and octal size, then the padded data. */
    private static void entry(ByteArrayOutputStream tar, String name, byte[] data) {
        byte[] header = new byte[512];
        byte[] n = name.getBytes(StandardCharsets.US_ASCII);
        System.arraycopy(n, 0, header, 0, n.length);
        byte[] size = String.format("%011o", data.length).getBytes(StandardCharsets.US_ASCII);
        System.arraycopy(size, 0, header, 124, size.length);
        tar.writeBytes(header);
        tar.writeBytes(data);
        tar.writeBytes(new byte[(512 - data.length % 512) % 512]);
    }

    /** {@code count} records whose label is {@code first + i} and whose pixels all equal the label. */
    private static byte[] records(int first, int count) {
        byte[] out = new byte[count * 3073];
        for (int i = 0; i < count; i++) {
            int label = (first + i) % 10;
            out[i * 3073] = (byte) label;
            java.util.Arrays.fill(out, i * 3073 + 1, (i + 1) * 3073, (byte) (label * 25));
        }
        return out;
    }

    @Test
    void readsTheBinaryArchive() throws IOException {
        ByteArrayOutputStream tar = new ByteArrayOutputStream();
        entry(tar, "cifar-10-batches-bin/batches.meta.txt", "airplane\n".getBytes(StandardCharsets.US_ASCII));
        for (int b = 1; b <= 5; b++) {
            entry(tar, "cifar-10-batches-bin/data_batch_" + b + ".bin", records(b, 2));
        }
        entry(tar, "cifar-10-batches-bin/test_batch.bin", records(7, 3));
        tar.writeBytes(new byte[1024]);
        Dataset.Split split = CifarExample.read(new ByteArrayInputStream(tar.toByteArray()));
        assertEquals(10, split.train().size());
        assertEquals(3, split.test().size());
        assertEquals(3 * 32 * 32, split.test().featureCount());
        float[] target = split.test().targetsOf(1);   // label 8
        assertEquals(1f, target[8]);
        assertEquals(1f, java.util.Arrays.stream(new int[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 9})
                .mapToDouble(i -> target[i]).sum(), 1e-6);
        float[] pixels = split.test().featuresOf(1);
        assertEquals((200 / 255f - 0.4914f) / 0.2470f, pixels[0], 1e-5);           // red plane
        assertEquals((200 / 255f - 0.4465f) / 0.2616f, pixels[2 * 1024 + 5], 1e-5); // blue plane
    }

    @Test
    void refusesAnArchiveWithoutEveryBatch() {
        ByteArrayOutputStream tar = new ByteArrayOutputStream();
        entry(tar, "cifar-10-batches-bin/data_batch_1.bin", records(0, 1));
        tar.writeBytes(new byte[1024]);
        assertThrows(IOException.class, () -> CifarExample.read(new ByteArrayInputStream(tar.toByteArray())));
    }

    @Test
    void cropAndFlipShiftsAndMirrorsEachChannel() {
        // one 2-channel 3x3 image, values 1..9 and 11..19
        float[] image = new float[18];
        for (int i = 0; i < 9; i++) {
            image[i] = i + 1;
            image[9 + i] = i + 11;
        }
        Augmentation a = Augmentation.cropAndFlip(2, 3, 3, 1);
        java.util.SplittableRandom random = new java.util.SplittableRandom(3);
        boolean changed = false;
        for (int t = 0; t < 20; t++) {
            float[] copy = image.clone();
            a.apply(copy, 0, random);
            changed |= !java.util.Arrays.equals(copy, image);
            for (int i = 0; i < 9; i++) {
                // both channels move the same way: the second stays the first plus 10 (or both 0)
                assertTrue(copy[9 + i] == (copy[i] == 0 ? 0 : copy[i] + 10), "channel shift " + i);
            }
            // the centre value always comes from a neighbour within one pixel (or is itself)
            float centre = copy[4];
            assertTrue(centre >= 1 && centre <= 9, "centre " + centre);
        }
        assertTrue(changed);
    }

    @Test
    void aSmallResidualNetworkLearnsImageShapes() {
        Dataset train = MnistExample.shapes(600, 12, 1);
        Dataset test = MnistExample.shapes(200, 12, 2);
        Model model = CifarExample.resnet(1, 12, 3, 1);
        Trainer.of(model).loss(Loss.CROSS_ENTROPY).optimizer(Optimizer.adam(0.003f)).batchSize(32).epochs(4)
                .augment(Augmentation.cropAndFlip(1, 12, 12, 1)).fit(train);
        float accuracy = Trainer.of(model).loss(Loss.CROSS_ENTROPY).metric(Metric.ACCURACY).evaluate(test)
                .get(Metric.ACCURACY);
        assertTrue(accuracy >= 0.9f, "test accuracy " + accuracy);
        assertArrayEquals(new int[] {3}, new int[] {model.outputSize()});
    }
}
