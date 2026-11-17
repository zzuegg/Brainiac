package io.github.zzuegg.brainiac.experimental.llm;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.data.CharacterTokenizer;
import io.github.zzuegg.brainiac.llm.Generator;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;

class WindowLmTest {

    @Test
    void learnsAndGeneratesThroughTheCoreGenerator() {
        CharacterTokenizer tok = new CharacterTokenizer("abc#");
        WindowLm lm = WindowLm.of(tok.size(), 4, 32, 1);
        List<int[]> prompts = List.of(tok.encode("a"), tok.encode("b"));
        List<int[]> completions = List.of(tok.encode("bc#"), tok.encode("ca#"));
        Optimizer opt = Optimizer.adam(1e-2f);
        float first = lm.supervisedStep(prompts, completions, opt);
        float last = first;
        for (int i = 0; i < 150; i++) {
            last = lm.supervisedStep(prompts, completions, opt);
        }
        assertTrue(last < first * 0.1, first + " -> " + last);
        assertArrayEquals(tok.encode("bc#"), lm.greedy(tok.encode("a"), 5, tok.encode("#")[0]));

        Generator gen = new Generator(lm, tok);
        Generator.Options options = Generator.Options.greedy(5).withStopTokens(Set.of(tok.encode("#")[0]));
        String out = gen.generate("b", options);
        assertTrue(out.startsWith("ca"), out);
    }

    @Test
    void sequenceLogProbMatchesTokenSum() {
        WindowLm lm = WindowLm.of(5, 3, 16, 2);
        int[] p = {1, 2};
        int[] c = {3, 4, 0};
        double total = lm.sequenceLogProb(p, c);
        double sum = 0;
        for (int t = 0; t < c.length; t++) {
            int[] prefix = new int[p.length + t];
            System.arraycopy(p, 0, prefix, 0, p.length);
            System.arraycopy(c, 0, prefix, p.length, t);
            double[] probs = LlmTensors.softmax(lm.logits(prefix), 1);
            sum += Math.log(probs[c[t]]);
        }
        assertEquals(sum, total, 1e-4);
    }
}
