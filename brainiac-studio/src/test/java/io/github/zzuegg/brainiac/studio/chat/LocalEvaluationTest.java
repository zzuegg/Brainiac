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

/** A chat model on this PC is scored on a file of questions and timed, and the results are kept. */
class LocalEvaluationTest {

    private static final String CHOICES = """
            {"question": "What is a box?", "choices": ["a tin", "a crate", "a bird", "a song"], "answer": 1}
            {"question": "What is a wren?", "choices": ["a tin", "a crate", "a bird", "a song"], "answer": "C"}

            {"question": "What sings?", "choices": {"text": ["a tin", "a bird"], "label": ["A", "B"]}, "answerKey": "B"}
            {"ctx": "The bird sat in the tree and", "endings": ["sang", "sank", "melted", "typed"], "label": "0"}
            {"question": "What is a tune?", "choices": ["a tin", "a crate", "a bird", "a song"], "answer": "a song"}
            """;
    private static final String ANSWERS = """
            {"question": "What is 1 and 1?", "answer": "It is 1 + 1 = 2.\\n#### 2"}
            {"question": "What is 2 and 3?", "answer": "5"}
            {"prompt": "What is 4 and 4?", "answer": 8}
            """;

    @TempDir
    Path models;

    private static Map<String, Object> newest(LocalEvaluation e) {
        return Json.object(Json.array(e.state().get("runs")).getFirst());
    }

    private static void await(LocalEvaluation e) throws InterruptedException {
        for (int i = 0; i < 1200 && e.running(); i++) {
            Thread.sleep(50);
        }
        assertFalse(e.running(), "the measurement ends");
    }

    @Test
    void theLayoutsOfBenchmarkFilesAreRead() {
        var choice = LocalEvaluation.questions(CHOICES);
        assertEquals(5, choice.size());
        assertEquals(List.of(1, 2, 1, 0, 3), choice.stream().map(q -> q.answer()).toList());
        var open = LocalEvaluation.questions(ANSWERS);
        assertEquals(3, open.size());
        assertEquals(List.of("2", "5", "8"), open.stream().map(q -> q.target()).toList());

        IllegalArgumentException mixed = assertThrows(IllegalArgumentException.class,
                () -> LocalEvaluation.questions(CHOICES + ANSWERS));
        assertTrue(mixed.getMessage().startsWith("line 7 "), mixed.getMessage());
        IllegalArgumentException bad = assertThrows(IllegalArgumentException.class,
                () -> LocalEvaluation.questions("{\"question\": \"What?\"}\n"));
        assertTrue(bad.getMessage().startsWith("line 1 "), bad.getMessage());
        assertThrows(IllegalArgumentException.class, () -> LocalEvaluation.questions("not json"));
        assertThrows(IllegalArgumentException.class, () -> LocalEvaluation.questions("\n\n"));
        assertThrows(IllegalArgumentException.class, () -> LocalEvaluation.questions(
                "{\"question\": \"What?\", \"choices\": [\"a\", \"b\"], \"answer\": \"z\"}"));
    }

    @Test
    void aModelIsScoredAndTimedAndTheResultsAreKept() throws Exception {
        LocalQuantizerTest.tinyLlama(models, "tiny-llama");
        LocalProvider local = (LocalProvider) ChatService.localModels(List.of(models));
        Path file = models.resolve("kept").resolve("evaluations.json");
        LocalEvaluation e = new LocalEvaluation(local, file);
        assertEquals(List.of(), e.state().get("runs"));
        assertFalse(e.running());
        assertFalse(e.stop());

        assertThrows(IOException.class, () -> e.start("no-such-model", "mine", CHOICES, 0, 0));
        assertThrows(IllegalArgumentException.class, () -> e.start("tiny-llama", "mine", "", 0, 0));
        assertThrows(IllegalArgumentException.class, () -> e.start("tiny-llama", "mine", CHOICES, 0, 9));
        assertThrows(IllegalArgumentException.class, () -> e.start("tiny-llama", "mine", CHOICES, 0, 5));
        assertEquals(List.of(), e.state().get("runs"), "a refused measurement leaves nothing behind");

        // questions with choices: one is shown solved, three of the other four are asked
        e.start("tiny-llama", "mine", CHOICES, 3, 1);
        await(e);
        Map<String, Object> run = newest(e);
        assertEquals("done", run.get("state"), String.valueOf(run.get("error")));
        assertEquals("choice", run.get("kind"));
        assertEquals("mine", run.get("task"));
        assertEquals(3L, ((Number) run.get("total")).longValue());
        assertEquals(3L, ((Number) run.get("done")).longValue());
        assertEquals(1L, ((Number) run.get("shots")).longValue());
        long correct = ((Number) run.get("correct")).longValue();
        assertTrue(correct >= 0 && correct <= 3);
        assertEquals(correct / 3.0, ((Number) run.get("accuracy")).doubleValue(), 1e-9);
        assertTrue(run.containsKey("stderr") && run.containsKey("normalized"));

        // questions with one answer
        e.start("tiny-llama", "", ANSWERS, 0, 0);
        await(e);
        run = newest(e);
        assertEquals("done", run.get("state"), String.valueOf(run.get("error")));
        assertEquals("answer", run.get("kind"));
        assertEquals("questions", run.get("task"));
        assertEquals(3L, ((Number) run.get("done")).longValue());
        assertFalse(run.containsKey("normalized"));

        // speed
        e.startSpeed("tiny-llama");
        IOException busy = assertThrows(IOException.class, () -> e.startSpeed("tiny-llama"));
        assertTrue(busy.getMessage().contains("is being measured"), busy.getMessage());
        await(e);
        run = newest(e);
        assertEquals("done", run.get("state"), String.valueOf(run.get("error")));
        assertEquals("speed", run.get("kind"));
        assertTrue(((Number) run.get("readPerSecond")).doubleValue() > 0);
        assertTrue(((Number) run.get("writePerSecond")).doubleValue() > 0);

        // a stopped measurement keeps its score so far
        e.start("tiny-llama", "long", CHOICES.repeat(40), 0, 0);
        assertTrue(e.stop());
        await(e);
        run = newest(e);
        assertEquals("stopped", run.get("state"));
        assertTrue(((Number) run.get("done")).longValue() < 200);

        // they are kept, and one can be forgotten
        assertEquals(4, Json.array(e.state().get("runs")).size());
        LocalEvaluation again = new LocalEvaluation(local, file);
        assertEquals(4, Json.array(again.state().get("runs")).size());
        assertEquals("stopped", newest(again).get("state"));
        String id = (String) newest(again).get("id");
        assertTrue(again.remove(id));
        assertFalse(again.remove(id));
        assertEquals(3, Json.array(new LocalEvaluation(local, file).state().get("runs")).size());
        try (var files = Files.list(file.getParent())) {
            assertTrue(files.noneMatch(p -> p.toString().endsWith(".part")));
        }
    }
}
