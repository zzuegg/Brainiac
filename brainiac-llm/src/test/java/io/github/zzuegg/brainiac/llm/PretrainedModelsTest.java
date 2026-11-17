package io.github.zzuegg.brainiac.llm;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.io.ChatTemplate;
import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.io.PretrainedTokenizer;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.stream.Stream;

import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Real models downloaded from Hugging Face, when {@code -PtestModels=<folder>} names a folder
 * of them: each model directory (with {@code config.json}) and each {@code .gguf} file in it.
 * It may also name one model directly. Skipped otherwise.
 *
 * <p>Each model answers a question it knows, greedily. When the folder has a {@code
 * <model>.expected.json} written by {@code reference_models.py} (transformers), the
 * prompt's next-token logits and the greedy tokens must match it.
 */
class PretrainedModelsTest {

    static final Device DEVICE = TestDevice.get();

    static Stream<Path> models() throws IOException {
        String dir = System.getProperty("brainiac.test.models", "");
        if (dir.isEmpty()) {
            return Stream.of((Path) null);   // one skipped run: JUnit wants at least one
        }
        Path folder = Path.of(dir);
        if (isModel(folder)) {
            return Stream.of(folder);   // a model itself rather than a folder of them
        }
        List<Path> models;
        try (Stream<Path> s = Files.list(folder)) {
            models = s.filter(PretrainedModelsTest::isModel).sorted().toList();
        }
        if (models.isEmpty()) {
            throw new IllegalArgumentException("no models in " + folder.toAbsolutePath()
                    + ": -PtestModels takes a model directory (with config.json), a .gguf file, or a folder of them");
        }
        return models.stream();
    }

    static boolean isModel(Path p) {
        return Files.isRegularFile(p.resolve("config.json")) || p.toString().endsWith(".gguf") && Files.isRegularFile(p);
    }

    static DType dtype() {
        return switch (System.getProperty("brainiac.test.dtype", "f32")) {
            case "f16" -> DType.FLOAT16;
            case "bf16" -> DType.BFLOAT16;
            default -> DType.FLOAT32;
        };
    }

    @ParameterizedTest
    @MethodSource("models")
    void answersAndMatchesTransformers(Path path) throws IOException {
        Assumptions.assumeTrue(path != null, "no -PtestModels folder");
        long t0 = System.nanoTime();
        Llama model = Llama.load(path, DEVICE, dtype());
        PretrainedTokenizer tok = PretrainedTokenizer.load(path);
        double load = (System.nanoTime() - t0) / 1e9;
        String question = "What is the capital of France? Answer in one word.";
        String prompt = tok.chatTemplate() != null
                ? tok.applyChatTemplate(List.of(ChatTemplate.message("user", question)), true)
                : "Q: " + question + "\nA:";
        Generator gen = new Generator(model, tok);
        Generator.Options options = Generator.Options.greedy(24).withStopTokens(Generator.stopTokens(path, tok));
        int[] ids = gen.encode(prompt);
        long t1 = System.nanoTime();
        Generator.Output out = gen.generate(new int[][] {ids}, options, null).getFirst();
        double seconds = (System.nanoTime() - t1) / 1e9;
        System.out.printf(Locale.ROOT, "%s on %s (%s): loaded in %.1f s, %d prompt + %d new tokens in %.2f s (%.1f/s)%n"
                + "  answer: %s%n", path.getFileName(), DEVICE, dtype(), load, ids.length, out.tokens().length, seconds,
                out.tokens().length / seconds, out.text().strip());
        assertTrue(out.text().toLowerCase(Locale.ROOT).contains("paris"), out.text());

        Path expectedFile = path.resolveSibling(path.getFileName() + ".expected.json");
        Assumptions.assumeTrue(Files.isRegularFile(expectedFile), "no transformers reference for " + path);
        Map<String, Object> e = Json.object(Json.read(expectedFile));
        // transformers' tokenizer(prompt) adds the special tokens as tok.encode(prompt, true) does
        assertArrayEquals(LlamaTest.ints(e.get("ids")), tok.encode(prompt, true), "the tokenizer differs from transformers");
        float[] logits = model.next(model.newState(1), new int[][] {ids}, null).toArray();
        float[] want = LlamaTest.rows(List.of(e.get("logits")));
        double worst = 0;
        double scale = 0;
        for (int i = 0; i < want.length; i++) {
            worst = Math.max(worst, Math.abs(logits[i] - want[i]));
            scale = Math.max(scale, Math.abs(want[i]));
        }
        System.out.printf(Locale.ROOT, "  logits: largest difference %.4f of %.1f%n", worst, scale);
        assertTrue(worst < (dtype() == DType.FLOAT32 && !path.toString().endsWith(".gguf") ? 2e-3 : 0.05) * scale,
                "logits differ from transformers by " + worst);
        assertEquals(Sampling.argmax(want), Sampling.argmax(logits));
        int[] greedy = LlamaTest.ints(e.get("greedy"));
        int[] mine = java.util.Arrays.copyOf(out.tokens(), Math.min(out.tokens().length, greedy.length));
        if (dtype() == DType.FLOAT32) {
            assertArrayEquals(java.util.Arrays.copyOf(greedy, mine.length), mine);
        }
    }
}
