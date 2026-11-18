package io.github.zzuegg.brainiac.llm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.io.WeightSource;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.Test;

/** Tiny CLIP, SigLIP, ViT and DINOv2 models against transformers (make_vision.py). */
class VisionEncoderTest {

    private static final int[][] CLIP_IDS = {{98, 5, 17, 42, 99, 0, 0, 0}, {98, 7, 3, 99, 0, 0, 0, 0}};
    private static final int[][] SIGLIP_IDS = {{5, 17, 42, 1, 0, 0, 0, 0}, {7, 3, 1, 0, 0, 0, 0, 0}};

    private static float[] expected(String model, String name) {
        try (WeightSource w = WeightSource.open(LlamaTest.resource(model).resolve("expected.safetensors"))) {
            return w.tensor(name, Device.cpu(), DType.FLOAT32).toArray();
        }
    }

    private static Tensor pixels(String model) {
        try (WeightSource w = WeightSource.open(LlamaTest.resource(model).resolve("expected.safetensors"))) {
            return w.tensor("pixel_values", LlamaTest.DEVICE, DType.FLOAT32);
        }
    }

    private static void close(float[] expected, float[][] actual, float tolerance, String what) {
        int d = actual[0].length;
        assertEquals(expected.length, actual.length * d, what);
        for (int i = 0; i < actual.length; i++) {
            for (int j = 0; j < d; j++) {
                assertEquals(expected[i * d + j], actual[i][j], tolerance, what + " [" + i + "][" + j + "]");
            }
        }
    }

    private static void close(float[] expected, float[] actual, float tolerance, String what) {
        assertEquals(expected.length, actual.length, what);
        for (int i = 0; i < actual.length; i++) {
            assertEquals(expected[i], actual[i], tolerance, what + " [" + i + "]");
        }
    }

    /** The processor's resize, crop and normalization, as transformers does them (to 8-bit rounding). */
    private static void preprocessesLikeTransformers(VisionEncoder model, String name) {
        Path image = LlamaTest.resource("vision-image.png");
        Tensor p = model.pixels(image);
        float[] e = expected(name, "pixel_values");
        assertEquals(e.length, p.size(), name);
        close(e, p.to(DType.FLOAT32).toArray(), 0.05f, name + " pixels");
    }

    @Test
    void clipMatchesTransformers() {
        VisionEncoder clip = VisionEncoder.load(LlamaTest.resource("vision-clip"), LlamaTest.DEVICE, DType.FLOAT32);
        assertEquals(VisionEncoder.Kind.CLIP, clip.kind());
        Tensor pixels = pixels("vision-clip");
        close(expected("vision-clip", "hidden"), clip.hidden(pixels).toArray(), 1e-4f, "hidden");
        close(expected("vision-clip", "image_embeds"), clip.embedImages(pixels, true), 1e-4f, "image");
        close(expected("vision-clip", "text_embeds"), clip.embedTexts(CLIP_IDS, true), 1e-4f, "text");
        close(expected("vision-clip", "logits"), clip.logits(pixels, CLIP_IDS), 1e-3f, "logits");
        preprocessesLikeTransformers(clip, "vision-clip");
    }

    @Test
    void siglipMatchesTransformers() {
        VisionEncoder siglip = VisionEncoder.load(LlamaTest.resource("vision-siglip"), LlamaTest.DEVICE, DType.FLOAT32);
        Tensor pixels = pixels("vision-siglip");
        close(expected("vision-siglip", "hidden"), siglip.hidden(pixels).toArray(), 1e-4f, "hidden");
        close(expected("vision-siglip", "image_embeds"), siglip.embedImages(pixels, true), 1e-4f, "image");
        close(expected("vision-siglip", "text_embeds"), siglip.embedTexts(SIGLIP_IDS, true), 1e-4f, "text");
        close(expected("vision-siglip", "logits"), siglip.logits(pixels, SIGLIP_IDS), 1e-3f, "logits");
        preprocessesLikeTransformers(siglip, "vision-siglip");
    }

    @Test
    void vitClassifiesLikeTransformers() {
        VisionEncoder vit = VisionEncoder.load(LlamaTest.resource("vision-vit"), LlamaTest.DEVICE, DType.FLOAT32);
        assertEquals(List.of("cat", "dog", "fish"), vit.labels());
        Tensor pixels = pixels("vision-vit");
        close(expected("vision-vit", "hidden"), vit.hidden(pixels).toArray(), 1e-4f, "hidden");
        close(expected("vision-vit", "logits"), vit.classify(pixels), 1e-4f, "logits");
        preprocessesLikeTransformers(vit, "vision-vit");
        assertThrows(IllegalStateException.class, () -> vit.embedTexts(CLIP_IDS, true));
    }

    @Test
    void dinov2InterpolatesItsPositionsLikeTransformers() {
        VisionEncoder dino = VisionEncoder.load(LlamaTest.resource("vision-dinov2"), LlamaTest.DEVICE, DType.FLOAT32);
        Tensor pixels = pixels("vision-dinov2");
        close(expected("vision-dinov2", "hidden"), dino.hidden(pixels).toArray(), 1e-4f, "hidden");
        close(expected("vision-dinov2", "image_embeds"), dino.embedImages(pixels, false), 1e-4f, "pooled");
        preprocessesLikeTransformers(dino, "vision-dinov2");
        assertTrue(dino.toString().contains("DINOV2"));
    }
}
