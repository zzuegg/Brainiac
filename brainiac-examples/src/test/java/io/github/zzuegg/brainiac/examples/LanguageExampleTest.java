package io.github.zzuegg.brainiac.examples;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.train.Benchmark;
import io.github.zzuegg.brainiac.train.Benchmark.Task;
import io.github.zzuegg.brainiac.train.Optimizer;

import org.junit.jupiter.api.Test;

/** Text is cut into next-character windows, and a small causal transformer learns to write. */
class LanguageExampleTest {

    @Test
    void theVocabularyIsTheSortedCharacters() {
        assertEquals(" abc", LanguageExample.vocabulary("cab ba"));
    }

    @Test
    void windowsPredictEveryNextCharacter() {
        Dataset d = LanguageExample.windows("abcab", "abc", 2, 1);
        assertEquals(3, d.size());
        assertArrayEquals(new float[] {1, 2}, d.featuresOf(1));
        assertArrayEquals(new float[] {0, 0, 1, 1, 0, 0}, d.targetsOf(1));   // "bc" -> "ca"
        assertThrows(IllegalArgumentException.class, () -> LanguageExample.windows("ab", "ab", 2, 1));
        assertThrows(IllegalArgumentException.class, () -> LanguageExample.windows("abd", "ab", 1, 1));
    }

    @Test
    void aSmallTransformerLearnsToContinueText() {
        String text = "the cat sat on the mat.".repeat(120);   // 23 characters: windows start at every phase
        Task task = LanguageExample.task("cat", text, 16)
                .withDefaults(t -> t.optimizer(Optimizer.adam(0.01f)).clipGradientNorm(1).batchSize(16).epochs(30));
        Model model = LanguageExample.model(task.outputShape().dim(1), 16, 32, 2, 4);
        Benchmark.Report report = Benchmark.run(task, "transformer", model, t -> { });
        // Every character after the first few of a sentence is certain, so the loss gets small.
        assertTrue(report.test().loss() < 0.3f, "test loss per character " + report.test().loss());
        String vocabulary = LanguageExample.vocabulary(text);
        String written = LanguageExample.generate(model, vocabulary, "the c", 41, 0.1f, 7);
        assertEquals("the cat sat on the mat.the cat sat on the mat.", written);
    }
}
