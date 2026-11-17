package io.github.zzuegg.brainiac.llm;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.data.CharacterTokenizer;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import org.junit.jupiter.api.Test;

class PrefixCacheTest {

    private static final String SYSTEM = "youareahelpfulassistant";
    private static final String[] PROMPTS = {SYSTEM + "hello", SYSTEM + "howareyou", SYSTEM + "hello", "other",
        SYSTEM + "helloagain"};

    private static Generator.Options options() {
        return Generator.Options.greedy(10).withStopTokens(Set.of());
    }

    private static List<int[]> plain(Generator gen) {
        List<int[]> out = new ArrayList<>();
        for (String p : PROMPTS) {
            out.add(gen.generate(List.of(p), options(), null).getFirst().tokens());
        }
        return out;
    }

    @Test
    void cachedPromptsWriteWhatUncachedOnesDo() {
        Generator gen = new Generator(GeneratorTest.letters(5), new CharacterTokenizer(GeneratorTest.LETTERS));
        List<int[]> expected = plain(gen);
        gen.cachePrefixes(4);
        for (int round = 0; round < 2; round++) {
            for (int i = 0; i < PROMPTS.length; i++) {
                int[] got = gen.generate(List.of(PROMPTS[i]), options(), null).getFirst().tokens();
                assertArrayEquals(expected.get(i), got, PROMPTS[i] + " round " + round);
            }
        }
        PrefixCache cache = gen.prefixCache();
        assertTrue(cache.size() <= 4);
        // every prompt after the first reuses at least the system prompt, repeats all but the last token
        assertTrue(cache.reusedTokens() >= 6L * SYSTEM.length(), cache.toString());
        gen.cachePrefixes(0);
        assertEquals(0, cache.size());
    }

    @Test
    void theBatcherUsesTheCache() throws Exception {
        Generator gen = new Generator(GeneratorTest.letters(6), new CharacterTokenizer(GeneratorTest.LETTERS));
        List<int[]> expected = plain(gen);
        gen.cachePrefixes(8);
        try (Generator.Batcher batcher = gen.batcher(3)) {
            for (int round = 0; round < 2; round++) {
                List<CompletableFuture<Generator.Output>> futures = new ArrayList<>();
                for (String p : PROMPTS) {
                    futures.add(batcher.submit(p, options(), null));
                }
                for (int i = 0; i < PROMPTS.length; i++) {
                    assertArrayEquals(expected.get(i), futures.get(i).get().tokens(), PROMPTS[i]);
                }
            }
        }
        assertTrue(gen.prefixCache().reusedTokens() > 0);
    }

    @Test
    void forkedStatesReadOnSeparately() {
        Llama model = GeneratorTest.letters(7);
        LanguageModel.State a = model.newState(1);
        model.next(a, new int[][] {{1, 2, 3}}, null);
        LanguageModel.State b = a.fork();
        float[] fromA = model.next(a, new int[][] {{4}}, null).toArray();
        float[] fromB = model.next(b, new int[][] {{5}}, null).toArray();
        LanguageModel.State c = model.newState(1);
        float[] fresh = model.next(c, new int[][] {{1, 2, 3, 5}}, null).toArray();
        assertArrayEquals(fresh, fromB, 1e-4f);
        LanguageModel.State d = model.newState(1);
        assertArrayEquals(model.next(d, new int[][] {{1, 2, 3, 4}}, null).toArray(), fromA, 1e-4f);
        a.close();
        b.close();
    }
}
