package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.llm.LanguageModel;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class SelfSummarizingContextTest {

    static final int V = 64;
    static final int FACT = 40;          // 40..59 are facts, below is filler
    static final int SUMMARIZE = 60;
    static final int END = 61;
    static final int QUESTION = 62;

    /**
     * A scripted model: after the summary instruction it lists the facts it read, then
     * ends; after a question it answers with the latest fact it can see; else filler.
     */
    static final class Scripted implements LanguageModel {
        int longest;

        @Override
        public int vocabSize() {
            return V;
        }

        @Override
        public State newState(int batch) {
            return new Seen(new ArrayList<>());
        }

        @Override
        public Tensor next(State state, int[][] tokens, boolean[][] padding) {
            List<Integer> seen = ((Seen) state).tokens();
            for (int t : tokens[0]) {
                seen.add(t);
            }
            longest = Math.max(longest, seen.size());
            int s = seen.indexOf(SUMMARIZE);
            List<Integer> facts = (s < 0 ? seen : seen.subList(0, s)).stream().filter(t -> t >= FACT && t < SUMMARIZE)
                    .toList();
            int answer;
            if (s >= 0) {
                int written = seen.size() - s - 1;
                answer = written < facts.size() ? facts.get(written) : END;
            } else if (seen.getLast() == QUESTION) {
                answer = facts.isEmpty() ? 0 : facts.getLast();
            } else {
                answer = 0;
            }
            float[] logits = new float[V];
            logits[answer] = 1;
            return Tensor.of(Shape.of(1, V), logits);
        }

        record Seen(List<Integer> tokens) implements State {
            @Override
            public int length() {
                return tokens.size();
            }

            @Override
            public State select(int[] rows) {
                throw new UnsupportedOperationException();
            }
        }
    }

    static int ask(SelfSummarizingContext.Mode mode, List<int[]> summaries, int[] longest) {
        Scripted model = new Scripted();
        SelfSummarizingContext ctx = SelfSummarizingContext.of(model, 16, mode)
                .summarizeWith(new int[] {SUMMARIZE}, END, 4).keepRecent(4);
        SplittableRandom r = new SplittableRandom(1);
        ctx.append(43);
        for (int i = 0; i < 30; i++) {
            ctx.append(r.nextInt(FACT));
            assertTrue(ctx.context().length <= 16);
        }
        ctx.append(51);
        for (int i = 0; i < 30; i++) {
            ctx.append(r.nextInt(FACT), r.nextInt(FACT));
            assertTrue(ctx.context().length <= 16);
        }
        ctx.append(QUESTION);
        summaries.addAll(ctx.summaries());
        longest[0] = model.longest;
        return ctx.next();
    }

    @Test
    void factsSurviveCompressionThatTruncationLoses() {
        List<int[]> summaries = new ArrayList<>();
        int[] longest = new int[1];
        assertEquals(51, ask(SelfSummarizingContext.Mode.SUMMARIZE, summaries, longest));
        assertTrue(summaries.size() >= 5, "compressed repeatedly: " + summaries.size());
        // summaries are summarised again: the first fact is still in the last summary
        assertArrayEquals(new int[] {43, 51}, summaries.getLast());
        assertTrue(longest[0] <= 16 + 1 + 4, "the model never reads past the window, instruction and summary");

        List<int[]> none = new ArrayList<>();
        assertNotEquals(51, ask(SelfSummarizingContext.Mode.TRUNCATE, none, longest));
        assertTrue(none.isEmpty());
    }

    @Test
    void generationFillsAndCompressesTheWindow() {
        SelfSummarizingContext ctx = SelfSummarizingContext.of(new Scripted(), 8, SelfSummarizingContext.Mode.SUMMARIZE)
                .summarizeWith(new int[] {SUMMARIZE}, END, 2).keepRecent(2);
        ctx.append(44, QUESTION);
        assertEquals(44, ctx.next());
        int[] out = ctx.generate(20);
        assertEquals(44, out[0]);
        assertTrue(ctx.context().length <= 8);
        assertTrue(ctx.summaries().stream().allMatch(s -> s.length >= 1 && java.util.Arrays.stream(s).allMatch(t -> t == 44)), ctx.toString());
        assertTrue(ctx.modelTokens() > 22);
    }
}
