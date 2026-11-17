package io.github.zzuegg.brainiac.llm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.data.CharacterTokenizer;
import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.tensor.DType;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class BenchmarkTest {

    static final String CHARS = "abcdefghijklmnopqrstuvwxyzABCDQ0123456789 .:?\n#$,";

    static Generator generator() {
        CharacterTokenizer tok = new CharacterTokenizer(CHARS);
        Llama model = Llama.random(LlamaConfig.small(tok.size(), 32, 2, 4, 2), 31, GeneratorTest.DEVICE,
                DType.FLOAT32);
        return new Generator(model, tok);
    }

    @Test
    void readsTheCommonLayouts(@TempDir Path dir) throws IOException {
        Path f = dir.resolve("mmlu.jsonl");
        Files.writeString(f, """
                {"question": "2 + 2?", "choices": ["3", "4", "5", "22"], "answer": 1}

                {"question": "Sky?", "choices": ["blue", "green"], "answer": 0}
                """);
        List<Benchmark.Example> mmlu = Benchmark.load(f, Benchmark::mmlu);
        assertEquals(2, mmlu.size());
        assertEquals("2 + 2?\nA. 3\nB. 4\nC. 5\nD. 22\nAnswer:", mmlu.getFirst().question());
        assertEquals(List.of("A", "B", "C", "D"), mmlu.getFirst().choices());
        assertEquals(1, mmlu.getFirst().answer());

        Benchmark.Example arc = Benchmark.arc(Json.object(Json.parse("""
                {"question": "Which is a gas?", "choices": {"text": ["ice", "steam"], "label": ["1", "2"]},
                 "answerKey": "2"}""")));
        assertEquals("Question: Which is a gas?\nAnswer:", arc.question());
        assertEquals(1, arc.answer());
        assertEquals("steam", arc.target());

        Benchmark.Example hs = Benchmark.hellaswag(Json.object(Json.parse("""
                {"activity_label": "Cooking", "ctx": "She cracks an egg", "endings": ["into a bowl.", "on the moon."],
                 "label": "0"}""")));
        assertEquals("Cooking: She cracks an egg", hs.question());
        assertEquals(0, hs.answer());

        Benchmark.Example gsm = Benchmark.gsm8k(Json.object(Json.parse("""
                {"question": "Tom has 3 apples and buys 4. How many?", "answer": "3 + 4 = 7\\n#### 7"}""")));
        assertEquals("7", gsm.target());
        assertEquals("3 + 4 = 7\n#### 7", gsm.solution());
        assertTrue(gsm.choices().isEmpty());
    }

    @Test
    void extractsAndNormalizesAnswers() {
        assertEquals("72", Benchmark.lastNumber("so she has 70 + 2 = 72 marbles."));
        assertEquals("1,234", Benchmark.lastNumber("work...\n#### 1,234\nmore 5"));
        assertEquals("-3.5", Benchmark.lastNumber("it drops to -3.5"));
        assertNull(Benchmark.lastNumber("no digits here"));
        assertEquals("1234", Benchmark.normalize(" $1,234.00. "));
        assertEquals("paris", Benchmark.normalize("Paris."));
        assertEquals("3.25", Benchmark.normalize("3.25"));
    }

    @Test
    void multipleChoicePicksTheLikeliestAnswer() {
        Generator gen = generator();
        List<Benchmark.Example> ex = List.of(
                Benchmark.Example.choice("Q: abc\nAnswer:", List.of("d", "zz", "abcd"), 0),
                Benchmark.Example.choice("Q: xyz\nAnswer:", List.of("1", "22", "333", "4"), 2),
                Benchmark.Example.choice("Q: what\nAnswer:", List.of("yes", "no"), 1));
        List<Benchmark.Example> shots = List.of(Benchmark.Example.choice("Q: a\nAnswer:", List.of("b", "c"), 0));
        Benchmark.Result r = Benchmark.multipleChoice(gen, ex, shots);
        assertEquals(3, r.total());
        String prefix = "Q: a\nAnswer: b\n\n";
        int correct = 0;
        for (int i = 0; i < ex.size(); i++) {
            Benchmark.Example e = ex.get(i);
            int best = -1;
            double score = Double.NEGATIVE_INFINITY;
            for (int c = 0; c < e.choices().size(); c++) {
                // a character tokenizer splits text exactly, so the context's tokens are its own
                double ll = Evaluation.logLikelihood(gen.model(), gen.encode(prefix + e.question()),
                        gen.encode(" " + e.choices().get(c)));
                if (ll > score) {
                    score = ll;
                    best = c;
                }
            }
            assertEquals(Integer.toString(best), r.predictions().get(i));
            correct += best == e.answer() ? 1 : 0;
        }
        assertEquals(correct, r.correct());
        assertTrue(r.stderr() >= 0 && r.toString().contains("3 examples"), r.toString());
    }

    @Test
    void generativeExtractsFromGreedyOutput() {
        Generator gen = generator();
        List<Benchmark.Example> ex = List.of(
                Benchmark.Example.open("Question: 1 plus 1?\nAnswer:", "2", null),
                Benchmark.Example.open("Question: 2 plus 3?\nAnswer:", "5", null),
                Benchmark.Example.open("Question: 4 plus 4?\nAnswer:", "8", null));
        List<Benchmark.Example> shots = List.of(Benchmark.Example.open("Question: 1 plus 2?\nAnswer:", "3",
                "1 + 2 = 3\n#### 3"));
        Benchmark.Result r = Benchmark.generative(gen, ex, shots, Benchmark::lastNumber, 12, 2);
        assertEquals(3, r.total());
        int correct = 0;
        for (int i = 0; i < ex.size(); i++) {
            String out = gen.generate("Question: 1 plus 2?\nAnswer: 1 + 2 = 3\n#### 3\n\n" + ex.get(i).question(),
                    Generator.Options.greedy(12).withStopStrings("\n\n", "Question:"));
            String got = Benchmark.lastNumber(out);
            assertEquals(got, r.predictions().get(i));
            correct += got != null && Benchmark.normalize(got).equals(ex.get(i).target()) ? 1 : 0;
        }
        assertEquals(correct, r.correct());
    }
}
