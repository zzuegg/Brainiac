package io.github.zzuegg.brainiac.llm;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import io.github.zzuegg.brainiac.autograd.Var;
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

    /**
     * The same models as a GGUF decoder with llama.cpp projector files (make_mmproj.py):
     * LLaVA in the older layout (last layer cut, feed-forward names swapped), Gemma 3 in
     * the current one (norm with its 1 added, projection transposed).
     */
    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(strings = {"vlm-llava", "vlm-gemma3"})
    void ggufWithAProjectorFileMatchesTransformers(String name, @org.junit.jupiter.api.io.TempDir java.nio.file.Path dir) {
        VisionLanguageModel hf = VisionLanguageModel.load(LlamaTest.resource(name), LlamaTest.DEVICE, DType.FLOAT32);
        java.nio.file.Path decoder = dir.resolve("decoder.gguf");
        hf.language().saveGguf(decoder, null, false);
        VisionLanguageModel vlm = VisionLanguageModel.load(decoder, LlamaTest.resource(name).resolve("mmproj.gguf"),
                LlamaTest.DEVICE, DType.FLOAT32);
        assertEquals(hf.kind(), vlm.kind());
        Tensor pixels = Tensor.of(io.github.zzuegg.brainiac.tensor.Shape.of(1, 3, 32, 32),
                expected(name, "pixel_values"), LlamaTest.DEVICE);
        close(expected(name, "features"), vlm.imageFeatures(pixels), 1e-4f, "features");
        float[] ids = expected(name, "ids");
        int[] prompt = new int[ids.length];
        for (int i = 0; i < ids.length; i++) {
            // a vocabulary without an image token: the model names its own placeholder
            prompt[i] = (int) ids[i] == hf.imageToken() ? vlm.imageToken() : (int) ids[i];
        }
        close(expected(name, "logits"), vlm.logits(new int[][] {prompt}, pixels), 1e-4f, "logits");
    }

    /** The server reads OpenAI image parts (data: URLs) and answers as the generator does. */
    @Test
    void serverAnswersAboutImages() throws Exception {
        VisionLanguageModel llava = VisionLanguageModel.load(LlamaTest.resource("vlm-llava"), LlamaTest.DEVICE,
                DType.FLOAT32);
        String chars = "abcdefghijklmnopqrstuvwxyz ?:.,!\nABCDEFGHIJKLMNOPQRSTUVWXYZ0";
        java.util.Map<String, Object> vocab = new java.util.LinkedHashMap<>();
        for (int i = 0; i < 60; i++) {
            vocab.put(String.valueOf(chars.charAt(i)), i);
        }
        java.util.Map<String, Object> root = java.util.Map.of("model", java.util.Map.of("type", "BPE", "vocab", vocab,
                "merges", List.of()), "added_tokens", List.of(
                java.util.Map.of("id", 60, "content", "<image>", "special", true),
                java.util.Map.of("id", 61, "content", "<end>", "special", true),
                java.util.Map.of("id", 62, "content", "<x>", "special", true),
                java.util.Map.of("id", 63, "content", "<y>", "special", true)));
        io.github.zzuegg.brainiac.io.PretrainedTokenizer tok = io.github.zzuegg.brainiac.io.PretrainedTokenizer.fromJson(
                io.github.zzuegg.brainiac.io.Json.write(root), java.util.Map.of("eos_token", "<end>", "chat_template",
                        "{% for m in messages %}{{ m['role'] }} {% if m['content'] is string %}{{ m['content'] }}"
                        + "{% else %}{% for p in m['content'] %}{% if p['type'] == 'image' %}<image>"
                        + "{% else %}{{ p['text'] }}{% endif %}{% endfor %}{% endif %}{% endfor %}"
                        + "{% if add_generation_prompt %}assistant {% endif %}"));
        Generator generator = new Generator(llava, tok);
        java.nio.file.Path file = LlamaTest.resource("vision-image.png");
        BufferedImage image = ImageIO.read(file.toFile());
        int[] ids = llava.prompt(generator.encode("user <image>what is it?assistant "), List.of(image));
        String want = generator.generate(new int[][] {ids}, Generator.Options.greedy(6), null).getFirst().text();
        String data = "data:image/png;base64," + java.util.Base64.getEncoder().encodeToString(
                java.nio.file.Files.readAllBytes(file));
        String body = "{\"max_tokens\":6,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":["
                + "{\"type\":\"image_url\",\"image_url\":{\"url\":\"" + data + "\"}},"
                + "{\"type\":\"text\",\"text\":\"what is it?\"}]}]}";
        for (int batch : new int[] {1, 4}) {
            try (OpenAiServer server = OpenAiServer.start(new java.net.InetSocketAddress("127.0.0.1", 0), "llava",
                    generator, tok, null, batch)) {
                java.net.HttpURLConnection c = (java.net.HttpURLConnection) java.net.URI.create(
                        "http://127.0.0.1:" + server.port() + "/v1/chat/completions").toURL().openConnection();
                c.setRequestMethod("POST");
                c.setDoOutput(true);
                try (java.io.OutputStream out = c.getOutputStream()) {
                    out.write(body.getBytes(java.nio.charset.StandardCharsets.UTF_8));
                }
                assertEquals(200, c.getResponseCode());
                String r = new String(c.getInputStream().readAllBytes(), java.nio.charset.StandardCharsets.UTF_8);
                java.util.Map<String, Object> choice = io.github.zzuegg.brainiac.io.Json.object(
                        io.github.zzuegg.brainiac.io.Json.array(io.github.zzuegg.brainiac.io.Json.object(
                                io.github.zzuegg.brainiac.io.Json.parse(r)).get("choices")).getFirst());
                Object message = io.github.zzuegg.brainiac.io.Json.object(choice.get("message")).get("content");
                assertEquals(want, message == null ? "" : message, "batch " + batch);
            }
        }
    }

    @Test
    void qwen25VlMatchesTransformers() throws IOException {
        VisionLanguageModel qwen = VisionLanguageModel.load(LlamaTest.resource("vlm-qwen25"), LlamaTest.DEVICE,
                DType.FLOAT32);
        assertEquals(VisionLanguageModel.Kind.QWEN_VL, qwen.kind());
        BufferedImage image = ImageIO.read(LlamaTest.resource("vision-image.png").toFile());
        // resized to 24 x 32 within the pixel budget: 6 x 8 patches of 4 pixels, 3 x 4 tokens
        QwenVision.Patches mine = qwen.qwen().patches(image);
        assertEquals(6, mine.height());
        assertEquals(8, mine.width());
        float[] px = expected("vlm-qwen25", "pixel_values");
        close(px, mine.values(), 0.05f, "patches");
        QwenVision.Patches theirs = new QwenVision.Patches(Tensor.of(io.github.zzuegg.brainiac.tensor.Shape.of(48, 96),
                px, LlamaTest.DEVICE), 6, 8);
        Tensor features = qwen.qwen().features(theirs);
        close(expected("vlm-qwen25", "features"), features, 1e-4f, "features");
        float[] ids = expected("vlm-qwen25", "ids");
        int[] prompt = new int[ids.length];
        for (int i = 0; i < ids.length; i++) {
            prompt[i] = (int) ids[i];
        }
        // 3D positions: the image at its start in time, rows and columns from there
        close(expected("vlm-qwen25", "logits"), Var.noGrad(() -> qwen.language().forward(new int[][] {prompt},
                qwen.imageToken(), features, new int[][] {{3, 4}})).value(), 1e-4f, "logits");
        int[] expanded = qwen.prompt(new int[] {1, 5, 9, 60, qwen.imageToken(), 62, 17, 23, 4}, List.of(image));
        assertArrayEquals(prompt, expanded);
        float[] want = expected("vlm-qwen25", "generated");
        Generator g = new Generator(qwen, new CharacterTokenizer(
                "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,"));
        int[] got = g.generate(new int[][] {expanded}, Generator.Options.greedy(want.length), null).getFirst().tokens();
        for (int i = 0; i < want.length; i++) {
            assertEquals((int) want[i], got[i], "token " + i);
        }
    }

    @Test
    void qwen2VlMatchesTransformers() {
        VisionLanguageModel qwen = VisionLanguageModel.load(LlamaTest.resource("vlm-qwen2"), LlamaTest.DEVICE,
                DType.FLOAT32);
        QwenVision.Patches theirs = new QwenVision.Patches(Tensor.of(io.github.zzuegg.brainiac.tensor.Shape.of(48, 96),
                expected("vlm-qwen2", "pixel_values"), LlamaTest.DEVICE), 6, 8);
        Tensor features = qwen.qwen().features(theirs);
        close(expected("vlm-qwen2", "features"), features, 1e-4f, "features");
        float[] ids = expected("vlm-qwen2", "ids");
        int[] prompt = new int[ids.length];
        for (int i = 0; i < ids.length; i++) {
            prompt[i] = (int) ids[i];
        }
        close(expected("vlm-qwen2", "logits"), Var.noGrad(() -> qwen.language().forward(new int[][] {prompt},
                qwen.imageToken(), features, new int[][] {{3, 4}})).value(), 1e-4f, "logits");
    }
}
