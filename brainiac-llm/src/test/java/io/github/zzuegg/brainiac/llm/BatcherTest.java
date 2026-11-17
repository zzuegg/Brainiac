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
}
