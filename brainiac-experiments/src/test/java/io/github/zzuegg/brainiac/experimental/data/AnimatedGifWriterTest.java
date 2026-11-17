package io.github.zzuegg.brainiac.experimental.data;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.util.SplittableRandom;

import javax.imageio.ImageIO;
import javax.imageio.ImageReader;
import javax.imageio.stream.ImageInputStream;

import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

class AnimatedGifWriterTest {

    /** Decodes every frame with the JDK's own GIF reader. */
    static BufferedImage[] decode(byte[] gif) throws IOException {
        ImageReader reader = ImageIO.getImageReadersByFormatName("gif").next();
        try (ImageInputStream in = ImageIO.createImageInputStream(new ByteArrayInputStream(gif))) {
            reader.setInput(in);
            int n = reader.getNumImages(true);
            BufferedImage[] out = new BufferedImage[n];
            for (int i = 0; i < n; i++) {
                out[i] = reader.read(i);
            }
            return out;
        } finally {
            reader.dispose();
        }
    }

    @ParameterizedTest
    @ValueSource(ints = {2, 5, 16, 256})
    void framesRoundTripThroughTheJdkDecoder(int colours) throws IOException {
        int[] palette = new int[colours];
        for (int i = 0; i < colours; i++) {
            palette[i] = (i * 37 % 256) << 16 | (i * 91 % 256) << 8 | i * 13 % 256;
        }
        palette[colours - 1] = 0xFFFFFF;
        int w = 97;
        int h = 61;   // noisy frames this size overflow the 4096-code table and force clears
        SplittableRandom r = new SplittableRandom(colours);
        AnimatedGifWriter gif = AnimatedGifWriter.of(w, h, palette);
        byte[][] frames = new byte[3][w * h];
        for (int f = 0; f < 3; f++) {
            for (int i = 0; i < w * h; i++) {
                // runs and noise, so both long matches and new codes occur
                frames[f][i] = (byte) (f == 1 ? (i / 7) % colours : r.nextInt(colours));
            }
            gif.frame(frames[f], 10);
        }
        assertEquals(3, gif.frames());
        BufferedImage[] img = decode(gif.toBytes());
        assertEquals(3, img.length);
        for (int f = 0; f < 3; f++) {
            assertEquals(w, img[f].getWidth());
            for (int i = 0; i < w * h; i++) {
                assertEquals(palette[frames[f][i] & 0xFF], img[f].getRGB(i % w, i / w) & 0xFFFFFF,
                        "frame " + f + " pixel " + i);
            }
        }
    }
}
