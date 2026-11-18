package io.github.zzuegg.brainiac.llm;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import io.github.zzuegg.brainiac.data.CharacterTokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;

import org.junit.jupiter.api.Test;

class BatcherTest {

    private final Generator gen = new Generator(GeneratorTest.letters(31), new CharacterTokenizer(GeneratorTest.LETTERS));

    @Test
    void joiningAndLeavingRequestsWriteWhatEachWouldAlone() throws Exception {
        String[] prompts = {"hello", "a", "thequickbrownfox", "zz", "lorem", "ipsumdolorsit", "x"};
        int[] lengths = {12, 3, 20, 7, 1, 15, 9};
        List<Generator.Output> expected = new ArrayList<>();
        for (int i = 0; i < prompts.length; i++) {
            expected.add(gen.generate(List.of(prompts[i]), options(lengths[i]), null).getFirst());
        }
        try (Generator.Batcher batcher = gen.batcher(3).warmUp(new int[] {1, 2}, 4)) {
            List<CompletableFuture<Generator.Output>> futures = new ArrayList<>();
            StringBuilder streamed = new StringBuilder();
            for (int i = 0; i < prompts.length; i++) {
                futures.add(batcher.submit(prompts[i], options(lengths[i]),
                        i == 2 ? (row, text) -> streamed.append(text) : null));
            }
            for (int i = 0; i < prompts.length; i++) {
                Generator.Output o = futures.get(i).get();
                assertArrayEquals(expected.get(i).tokens(), o.tokens(), prompts[i]);
                assertEquals(expected.get(i).text(), o.text());
                assertEquals(expected.get(i).finish(), o.finish());
            }
            assertEquals(expected.get(2).text(), streamed.toString());
        }
    }

    @Test
    void sampledRequestsKeepTheirOwnDraws() throws Exception {
        Generator.Options o = Generator.Options.sampled(10, Sampling.temperature(1.2f), 7).withStopTokens(Set.of());
        String alone = gen.generate("abc", o);
        try (Generator.Batcher batcher = gen.batcher(4)) {
            CompletableFuture<Generator.Output> other = batcher.submit("qrstuv", o, null);
            CompletableFuture<Generator.Output> mine = batcher.submit("abc", o, null);
            assertEquals(alone, mine.get().text());
            other.get();
        }
    }

    private static Generator.Options options(int n) {
        return Generator.Options.greedy(n).withStopTokens(Set.of(GeneratorTest.LETTERS.indexOf('q')));
    }

    /** A model whose first step runs out of memory, then works. */
    private static final class FailsOnce implements LanguageModel {
        private final LanguageModel inner;
        private boolean failed;

        FailsOnce(LanguageModel inner) {
            this.inner = inner;
        }

        @Override
        public int vocabSize() {
            return inner.vocabSize();
        }

        @Override
        public State newState(int batch) {
            return inner.newState(batch);
        }

        @Override
        public io.github.zzuegg.brainiac.tensor.Tensor next(State state, int[][] tokens, boolean[][] padding) {
            if (!failed) {
                failed = true;
                throw new OutOfMemoryError("test: device memory");
            }
            return inner.next(state, tokens, padding);
        }
    }

    @Test
    void anOutOfMemoryErrorFailsItsRequestsButNotTheBatcher() throws Exception {
        Generator gen = new Generator(new FailsOnce(GeneratorTest.letters(5)),
                new CharacterTokenizer(GeneratorTest.LETTERS));
        try (Generator.Batcher batcher = gen.batcher(2)) {
            CompletableFuture<Generator.Output> first = batcher.submit("abc", Generator.Options.greedy(4), null);
            java.util.concurrent.ExecutionException e = org.junit.jupiter.api.Assertions.assertThrows(
                    java.util.concurrent.ExecutionException.class, () -> first.get(30, java.util.concurrent.TimeUnit.SECONDS));
            assertEquals(OutOfMemoryError.class, e.getCause().getClass());
            Generator.Output second = batcher.submit("abc", Generator.Options.greedy(4), null)
                    .get(30, java.util.concurrent.TimeUnit.SECONDS);
            assertEquals(gen.generate("abc", Generator.Options.greedy(4)), second.text());
        }
    }
}
