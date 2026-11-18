package io.github.zzuegg.brainiac.studio.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.io.Json;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Chat models of this PC are merged into a new model folder, which is listed and answers. */
class LocalMergerTest {

    @TempDir
    Path models;

    private static Map<String, Object> job(LocalMerger m) {
        return Json.object(m.state().get("job"));
    }

    private static void await(LocalMerger m) throws InterruptedException {
        for (int i = 0; i < 1200 && m.running(); i++) {
            Thread.sleep(50);
        }
        assertFalse(m.running(), "the merge ends");
    }

    private static String answer(LocalProvider local, String model) throws IOException {
        StringBuilder text = new StringBuilder();
        local.chat(new ChatRequest(model, "", List.of(new ChatMessage("user", "What is this?")), 6, null, 0f), text::append);
        return text.toString();
    }

    @Test
    void modelsAreMergedIntoANewOneThatAnswers() throws Exception {
        LocalQuantizerTest.tinyLlama(models, "tiny-base");
        LocalQuantizerTest.tinyLlama(models, "tiny-a");
        LocalQuantizerTest.tinyLlama(models, "tiny-b");
        LocalProvider local = (LocalProvider) ChatService.localModels(List.of(models));
        LocalMerger m = new LocalMerger(local);
        assertFalse(m.state().containsKey("job"));
        assertEquals("linear", Json.object(Json.array(m.state().get("ways")).getFirst()).get("id"));

        List<String> two = List.of("tiny-a", "tiny-b");
        assertThrows(IllegalArgumentException.class, () -> m.start("magic", "", two, List.of(), 0.5, 0.5, "x"));
        assertThrows(IllegalArgumentException.class, () -> m.start("linear", "", List.of("tiny-a"), List.of(), 0.5, 0.5, "x"));
        assertThrows(IllegalArgumentException.class, () -> m.start("linear", "", List.of("tiny-a", "tiny-a"), List.of(), 0.5, 0.5, "x"));
        assertThrows(IllegalArgumentException.class, () -> m.start("linear", "", two, List.of(1.0), 0.5, 0.5, "x"));
        assertThrows(IllegalArgumentException.class, () -> m.start("linear", "", two, List.of(), 0.5, 0.5, "../x"));
        assertThrows(IllegalArgumentException.class, () -> m.start("ties", "", two, List.of(), 0.5, 0.5, "x"));
        assertThrows(IllegalArgumentException.class, () -> m.start("ties", "tiny-a", two, List.of(), 0.5, 0.5, "x"));
        assertThrows(IllegalArgumentException.class, () -> m.start("ties", "tiny-base", two, List.of(), 0.5, 0, "x"));
        assertThrows(IllegalArgumentException.class,
                () -> m.start("slerp", "", List.of("tiny-a", "tiny-b", "tiny-base"), List.of(), 0.5, 0.5, "x"));
        assertThrows(IOException.class, () -> m.start("linear", "", List.of("tiny-a", "no-such"), List.of(), 0.5, 0.5, "x"));
        IOException taken = assertThrows(IOException.class, () -> m.start("linear", "", two, List.of(), 0.5, 0.5, "tiny-base"));
        assertTrue(taken.getMessage().contains("there already"), taken.getMessage());
        assertFalse(m.state().containsKey("job"), "a refused merge leaves nothing behind");

        // the mean of two copies of one model is that model
        String before = answer(local, "tiny-a");
        m.start("linear", "", two, List.of(), 0.5, 0.5, "tiny-mean");
        await(m);
        assertEquals("done", job(m).get("state"), String.valueOf(job(m).get("error")));
        assertTrue(Files.isRegularFile(models.resolve("tiny-mean").resolve("config.json")));
        assertTrue(Files.isRegularFile(models.resolve("tiny-mean").resolve("tokenizer.json")));
        assertTrue(local.models().stream().anyMatch(x -> x.id().equals("tiny-mean")), local.models().toString());
        assertEquals(before, answer(local, "tiny-mean"));

        // with a base, and along the sphere
        m.start("ties", "tiny-base", two, List.of(1.0, 0.5), 0.5, 0.6, "tiny-ties");
        await(m);
        assertEquals("done", job(m).get("state"), String.valueOf(job(m).get("error")));
        assertEquals(before, answer(local, "tiny-ties"), "copies of the base have nothing to add to it");
        m.start("slerp", "", two, List.of(), 0.3, 0.5, "tiny-slerp");
        await(m);
        assertEquals("done", job(m).get("state"), String.valueOf(job(m).get("error")));
        try (var files = Files.list(models)) {
            assertTrue(files.noneMatch(p -> p.toString().endsWith(".part")));
        }

        // a single GGUF file cannot be merged
        Files.write(models.resolve("lone.gguf"), new byte[] {'G', 'G', 'U', 'F'});
        if (local.models().stream().anyMatch(x -> x.id().equals("lone.gguf"))) {
            IOException gguf = assertThrows(IOException.class,
                    () -> m.start("linear", "", List.of("tiny-a", "lone.gguf"), List.of(), 0.5, 0.5, "y"));
            assertTrue(gguf.getMessage().contains("GGUF"), gguf.getMessage());
        }
    }
}
