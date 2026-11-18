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

/** A chat model on this PC learns from examples with LoRA: an adapter and a merged copy come out. */
class LocalFineTuneTest {

    private static final String DATA = """
            {"prompt": "What is this?", "completion": "A box."}

            {"instruction": "What is that?", "input": "", "output": "A fox."}
            {"prompt": "Who is it?", "response": "Bob."}
            """;

    @TempDir
    Path models;

    private static Map<String, Object> job(LocalFineTune t) {
        return Json.object(t.state().get("job"));
    }

    private static Map<String, Object> finished(LocalFineTune t) throws InterruptedException {
        for (int i = 0; i < 1200 && t.running(); i++) {
            Thread.sleep(50);
        }
        assertFalse(t.running(), "the run ends");
        return job(t);
    }

    @Test
    void aModelLearnsFromExamplesAndItsMergedCopyIsListed() throws Exception {
        LocalQuantizerTest.tinyLlama(models, "tiny-llama");
        LocalVisionTest.tinyLlava(models, "tiny-eyes", null);
        LocalProvider local = (LocalProvider) ChatService.localModels(List.of(models));
        LocalFineTune teach = new LocalFineTune(local);
        LocalFineTune.Settings fast = new LocalFineTune.Settings(4, 8, 40, 0.02f, 64, 2);
        assertTrue(teach.state().isEmpty());

        assertThrows(IllegalArgumentException.class, () -> new LocalFineTune.Settings(0, 8, 1, 2e-4f, 64, 1));
        assertThrows(IllegalArgumentException.class, () -> new LocalFineTune.Settings(4, 8, 1, 0f, 64, 1));
        assertThrows(IllegalArgumentException.class, () -> teach.start("tiny-llama", "a name", DATA, fast));
        assertThrows(IllegalArgumentException.class, () -> teach.start("tiny-llama", "boxes", " ", fast));
        assertThrows(IOException.class, () -> teach.start("no-such-model", "boxes", DATA, fast));
        assertTrue(assertThrows(IOException.class, () -> teach.start("tiny-eyes", "boxes", DATA, fast))
                .getMessage().contains("looks at pictures"));

        teach.start("tiny-llama", "boxes", DATA, fast);
        assertTrue(assertThrows(IOException.class, () -> teach.start("tiny-llama", "other", DATA, fast))
                .getMessage().contains("is being taught"));
        Map<String, Object> done = finished(teach);
        assertEquals("done", done.get("state"), String.valueOf(done.get("error")));
        assertEquals(3, ((Number) done.get("examples")).intValue());
        assertEquals(40, ((Number) done.get("steps")).intValue(), "the three short examples fill one step an epoch");
        assertEquals(40, ((Number) done.get("step")).intValue());
        List<Object> losses = Json.array(done.get("losses"));
        double first = ((Number) Json.array(losses.getFirst()).get(1)).doubleValue();
        double last = ((Number) done.get("loss")).doubleValue();
        assertTrue(last < first * 0.6, "the loss falls: " + first + " to " + last);

        // the adapter as PEFT reads it, and a merged copy that is one of this PC's models now
        Path adapter = models.resolve("tiny-llama-boxes-lora");
        assertEquals(adapter.toString(), done.get("adapter"));
        assertTrue(Files.isRegularFile(adapter.resolve("adapter_config.json")));
        assertTrue(Files.isRegularFile(adapter.resolve("adapter_model.safetensors")));
        assertEquals("tiny-llama-boxes.gguf", done.get("mergedName"));
        assertTrue(local.models().stream().anyMatch(m -> m.id().equals("tiny-llama-boxes.gguf")), local.models().toString());
        assertFalse(local.models().stream().anyMatch(m -> m.id().contains("lora")), "an adapter is not a model");
        StringBuilder said = new StringBuilder();
        local.chat(new ChatRequest("tiny-llama-boxes.gguf", "", List.of(new ChatMessage("user", "What is this?")), 8,
                null, 0f), said::append);
        assertTrue(said.toString().strip().startsWith("A box."), "it answers as it was taught: " + said);

        assertTrue(assertThrows(IOException.class, () -> teach.start("tiny-llama", "boxes", DATA, fast))
                .getMessage().contains("already"));

        // a line that is no example fails the run and says which
        teach.start("tiny-llama", "broken", DATA + "{\"prompt\": \"only a prompt\"}\n", fast);
        Map<String, Object> failed = finished(teach);
        assertEquals("failed", failed.get("state"));
        assertTrue(String.valueOf(failed.get("error")).startsWith("line 5 needs"), String.valueOf(failed.get("error")));
        assertFalse(Files.exists(models.resolve("tiny-llama-broken-lora")));

        // stopping keeps what was learned so far
        teach.start("tiny-llama", "short", DATA, new LocalFineTune.Settings(4, 8, 100, 0.01f, 64, 1));
        for (int i = 0; i < 600 && ((Number) job(teach).get("step")).intValue() < 2; i++) {
            Thread.sleep(20);
        }
        assertTrue(teach.stop());
        Map<String, Object> stopped = finished(teach);
        assertEquals("stopped", stopped.get("state"), String.valueOf(stopped.get("error")));
        assertTrue(((Number) stopped.get("step")).intValue() < 300);
        assertTrue(Files.isDirectory(models.resolve("tiny-llama-short-lora")));
        assertFalse(teach.stop());
    }
}
