package io.github.zzuegg.brainiac.llm;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import java.awt.image.BufferedImage;
import java.util.Map;

import org.junit.jupiter.api.Test;

class ImagePreprocessorTest {

    private static BufferedImage gradient(int w, int h) {
        BufferedImage image = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                image.setRGB(x, y, (x * 7 % 256) << 16 | (y * 11 % 256) << 8 | (x + y) * 3 % 256);
            }
        }
        return image;
    }

    @Test
    void aSingleSizeMeansTheShortestEdgeWhenACropFollows() {
        // openai/clip-vit-base-patch32 writes "size": 224 and "crop_size": 224
        Map<String, Object> old = Map.of("size", 16, "crop_size", 16, "do_center_crop", true, "resample", 3);
        Map<String, Object> current = Map.of("size", Map.of("shortest_edge", 16),
                "crop_size", Map.of("height", 16, "width", 16), "do_center_crop", true, "resample", 3);
        BufferedImage image = gradient(40, 20);
        int[] a = new int[2];
        int[] b = new int[2];
        float[] fromOld = new ImagePreprocessor(old, 224).apply(image, a);
        float[] fromCurrent = new ImagePreprocessor(current, 224).apply(image, b);
        assertArrayEquals(new int[] {16, 16}, a);
        assertArrayEquals(b, a);
        assertArrayEquals(fromCurrent, fromOld);
    }

    @Test
    void aSingleSizeWithoutACropIsASquare() {
        Map<String, Object> old = Map.of("size", 12);
        Map<String, Object> current = Map.of("size", Map.of("height", 12, "width", 12));
        BufferedImage image = gradient(30, 20);
        int[] a = new int[2];
        float[] fromOld = new ImagePreprocessor(old, 224).apply(image, a);
        assertArrayEquals(new int[] {12, 12}, a);
        assertArrayEquals(new ImagePreprocessor(current, 224).apply(image, new int[2]), fromOld);
    }
}
