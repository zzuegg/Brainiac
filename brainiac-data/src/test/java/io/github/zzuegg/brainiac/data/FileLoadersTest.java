package io.github.zzuegg.brainiac.data;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class FileLoadersTest {

    @Test
    void pngRoundTripAndFolders(@TempDir Path dir) throws IOException {
        float[] v = new float[3 * 2 * 4];
        for (int i = 0; i < v.length; i++) {
            v[i] = (i * 37 % 256) / 255f;
        }
        Tensor img = Tensor.of(Shape.of(3, 2, 4), v);
        Files.createDirectories(dir.resolve("cats"));
        Files.createDirectories(dir.resolve("dogs"));
        Images.write(img, dir.resolve("cats/a.png"));
        Images.write(img, dir.resolve("dogs/b.png"));
        Images.write(img, dir.resolve("dogs/c.png"));
        Tensor back = Images.read(dir.resolve("cats/a.png"), 3);
        assertEquals(Shape.of(3, 2, 4), back.shape());
        assertArrayEquals(v, back.toArray(), 1e-6f);
        assertEquals(Shape.of(1, 4, 8), Images.read(dir.resolve("cats/a.png"), 1, 4, 8).shape());
        Dataset d = Images.folder(dir, 3, 2, 4);
        assertEquals(3, d.size());
        assertEquals(List.of("cats", "dogs"), Images.classes(dir));
        assertThrows(IOException.class, () -> Images.read(Files.writeString(dir.resolve("x.png"), "no"), 3));
    }

    @Test
    void wavRoundTripAndFormats(@TempDir Path dir) throws IOException {
        float[] left = {0, 0.5f, -0.5f, 1};
        float[] right = {0.25f, -1, 0, 0.75f};
        Path f = dir.resolve("a.wav");
        Wav.write(new Wav.Audio(16000, new float[][] {left, right}), f);
        Wav.Audio a = Wav.read(f);
        assertEquals(16000, a.sampleRate());
        assertArrayEquals(left, a.channels()[0], 1e-4f);
        assertArrayEquals(right, a.channels()[1], 1e-4f);
        assertArrayEquals(new float[] {0.125f, -0.25f, -0.25f, 0.875f}, a.mono(), 1e-4f);
        // a mono float file
        ByteBuffer b = ByteBuffer.allocate(44 + 8).order(ByteOrder.LITTLE_ENDIAN);
        b.putInt(0x46464952).putInt(44).putInt(0x45564157).putInt(0x20746d66).putInt(16).putShort((short) 3)
                .putShort((short) 1).putInt(8000).putInt(32000).putShort((short) 4).putShort((short) 32)
                .putInt(0x61746164).putInt(8).putFloat(0.3f).putFloat(-0.7f);
        Files.write(f, b.array());
        assertArrayEquals(new float[] {0.3f, -0.7f}, Wav.read(f).channels()[0]);
    }
}
