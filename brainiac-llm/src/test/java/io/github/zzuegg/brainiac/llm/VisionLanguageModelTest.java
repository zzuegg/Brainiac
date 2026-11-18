package io.github.zzuegg.brainiac.llm;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import io.github.zzuegg.brainiac.io.WeightSource;
import io.github.zzuegg.brainiac.data.CharacterTokenizer;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.List;

import javax.imageio.ImageIO;

import org.junit.jupiter.api.Test;

/** A tiny LLaVA against transformers (make_vlm.py). */
class VisionLanguageModelTest {

    private static float[] expected(String name) {
        return expected("vlm-llava", name);
    }

    private static float[] expected(String model, String name) {
        try (WeightSource w = WeightSource.open(LlamaTest.resource(model).resolve("expected.safetensors"))) {
            return w.tensor(name, Device.cpu(), DType.FLOAT32).toArray();
        }
    }

    private static void close(float[] expected, Tensor actual, float tolerance, String what) {
        float[] a = actual.to(DType.FLOAT32).toArray();
        assertEquals(expected.length, a.length, what);
        for (int i = 0; i < a.length; i++) {
            assertEquals(expected[i], a[i], tolerance * Math.max(1, Math.abs(expected[i])), what + " [" + i + "]");
        }
    }

    @Test
    void llavaMatchesTransformers() throws IOException {
        VisionLanguageModel llava = VisionLanguageModel.load(LlamaTest.resource("vlm-llava"), LlamaTest.DEVICE,
                DType.FLOAT32);
        BufferedImage image = ImageIO.read(LlamaTest.resource("vision-image.png").toFile());
        Tensor px = llava.pixels(image);
        close(expected("pixel_values"), px, 0.05f, "pixels");
        // transformers' own pixels, so that features compare exactly (ours round slightly differently)
        Tensor pixels = Tensor.of(io.github.zzuegg.brainiac.tensor.Shape.of(1, 3, 32, 32), expected("pixel_values"),
                LlamaTest.DEVICE);
        close(expected("features"), llava.imageFeatures(pixels), 1e-4f, "features");
        float[] ids = expected("ids");
        int[] prompt = new int[ids.length];
        for (int i = 0; i < ids.length; i++) {
            prompt[i] = (int) ids[i];
        }
        close(expected("logits"), llava.logits(new int[][] {prompt}, pixels), 1e-4f, "logits");

        // the same through a generator: one image token expands to 16
        int[] shortPrompt = {1, 5, 9, llava.imageToken(), 17, 23, 4};
        int[] expanded = llava.prompt(shortPrompt, List.of(image));
        assertArrayEquals(prompt, expanded);
        float[] want = expected("generated");
        Generator g = new Generator(llava, new CharacterTokenizer(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,"));
        Generator.Output out = g.generate(new int[][] {expanded}, Generator.Options.greedy(want.length), null)
                .getFirst();
        int[] got = out.tokens();
        for (int i = 0; i < want.length; i++) {
            assertEquals((int) want[i], got[i], "token " + i);
        }
    }

    @Test
    void gemma3MatchesTransformers() {
        VisionLanguageModel gemma = VisionLanguageModel.load(LlamaTest.resource("vlm-gemma3"), LlamaTest.DEVICE,
                DType.FLOAT32);
        assertEquals(VisionLanguageModel.Kind.GEMMA3, gemma.kind());
        Tensor pixels = Tensor.of(io.github.zzuegg.brainiac.tensor.Shape.of(1, 3, 32, 32),
                expected("vlm-gemma3", "pixel_values"), LlamaTest.DEVICE);
        Tensor features = gemma.imageFeatures(pixels);
        close(expected("vlm-gemma3", "features"), features, 1e-4f, "features");
        float[] ids = expected("vlm-gemma3", "ids");
        int[] prompt = new int[ids.length];
        for (int i = 0; i < ids.length; i++) {
            prompt[i] = (int) ids[i];
        }
        // the image's tokens see each other both ways, and the sliding layers' window of 4
        close(expected("vlm-gemma3", "logits"), gemma.logits(new int[][] {prompt}, pixels), 1e-4f, "logits");
        // the same with a cache, one prefill then decoding
        float[] want = expected("vlm-gemma3", "generated");
        Llama.State state = gemma.language().newState(1);
        Tensor logits = gemma.language().next(state, new int[][] {prompt}, gemma.imageToken(), features, true);
        for (int i = 0; i < want.length; i++) {
            int best = (int) logits.argmax(-1).toArray()[0];
            assertEquals((int) want[i], best, "token " + i);
            logits = gemma.language().next(state, new int[][] {{best}}, null);
        }
    }
}
