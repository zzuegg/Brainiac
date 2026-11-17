package io.github.zzuegg.brainiac.llm;

import io.github.zzuegg.brainiac.data.CharacterTokenizer;
import io.github.zzuegg.brainiac.tensor.DType;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EvaluationTest {

    private final Llama model = Llama.random(LlamaConfig.small(26, 32, 2, 4, 2), 11, GeneratorTest.DEVICE,
            DType.FLOAT32);

    @Test
    void likelihoodsAddUpAndPerplexityFollowsThem() {
        int[] context = {1, 2, 3};
        int[] cont = {4, 5};
        double both = Evaluation.logLikelihood(model, context, cont);
        double first = Evaluation.logLikelihood(model, context, new int[] {4});
        double second = Evaluation.logLikelihood(model, new int[] {1, 2, 3, 4}, new int[] {5});
        assertEquals(both, first + second, 1e-4);
        assertTrue(both < 0);
        // one window: perplexity is exp(-mean log p) of every token after the first
        int[] text = {1, 2, 3, 4, 5};
        double expected = Math.exp(-Evaluation.logLikelihood(model, new int[] {1}, new int[] {2, 3, 4, 5}) / 4);
        assertEquals(expected, Evaluation.perplexity(model, text, 8), 1e-3);
        // an untrained model is close to uniform over 26 letters
        double ppl = Evaluation.perplexity(model, new int[] {3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5, 8, 9, 7, 9}, 4);
        assertTrue(ppl > 10 && ppl < 60, "perplexity " + ppl);
    }

    @Test
    void multipleChoicePicksTheMostLikelyAnswer() {
        int[] context = {7, 8, 9};
        int[][] choices = {{1, 2}, {3}, {4, 5, 6}};
        int pick = Evaluation.multipleChoice(model, context, false, choices);
        for (int i = 0; i < choices.length; i++) {
            assertTrue(Evaluation.logLikelihood(model, context, choices[pick])
                    >= Evaluation.logLikelihood(model, context, choices[i]));
        }
    }

    @Test
    void needleGridHasOneCellPerLengthAndDepth() {
        Llama spaced = Llama.random(LlamaConfig.small(28, 32, 2, 4, 2), 12, GeneratorTest.DEVICE, DType.FLOAT32);
        Generator gen = new Generator(spaced, new CharacterTokenizer(GeneratorTest.LETTERS + " \n"));
        boolean[][] grid = Evaluation.needleInHaystack(gen, "the sky is blue", "magicword", "whatisthemagicword",
                "zzzzzz", new int[] {40, 80}, new double[] {0, 0.5, 1}, 3);
        assertEquals(2, grid.length);
        assertEquals(3, grid[0].length);
    }
}
