package io.github.zzuegg.brainiac.llm;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.data.CharacterTokenizer;
import io.github.zzuegg.brainiac.data.Tokenizer;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;

import java.io.IOException;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

class GeneratorTest {

    static final Device DEVICE = TestDevice.get();
    static final String LETTERS = "abcdefghijklmnopqrstuvwxyz";

    static Llama letters(long seed) {
        return Llama.random(LlamaConfig.small(26, 32, 2, 4, 2), seed, DEVICE, DType.FLOAT32);
    }

    @ParameterizedTest
    @ValueSource(strings = {"llama", "llama3-rope", "qwen2"})
    void greedyAndBeamSearchMatchTransformers(String name) throws IOException {
        Map<String, Object> e = LlamaTest.expected(name + "/expected.json");
        Generator gen = new Generator(Llama.load(LlamaTest.resource(name), DEVICE, DType.FLOAT32), null);
        int[] ids = LlamaTest.ints(e.get("ids"));
        Generator.Output out = gen.generate(new int[][] {ids}, Generator.Options.greedy(6).withStopTokens(Set.of()),
                null).getFirst();
        assertArrayEquals(LlamaTest.ints(e.get("greedy")), out.tokens());
        assertEquals(Generator.Finish.LENGTH, out.finish());
        assertArrayEquals(LlamaTest.ints(e.get("beam3")), gen.beamSearch(ids, 3, 6, 1.0, Set.of()), name);
    }

    @Test
    void batchesOfDifferentLengthsMatchOneByOne() {
        Generator gen = new Generator(letters(1), new CharacterTokenizer(LETTERS));
        List<String> prompts = List.of("hello", "ab", "quickbrownfox");
        Generator.Options o = Generator.Options.sampled(12, Sampling.temperature(0.8f).withTopK(5), 7);
        List<Generator.Output> batch = gen.generate(prompts, Generator.Options.greedy(12), null);
        for (int i = 0; i < prompts.size(); i++) {
            assertEquals(gen.generate(prompts.get(i), Generator.Options.greedy(12)), batch.get(i).text());
            assertEquals(12, batch.get(i).tokens().length);
        }
        // the same seed gives the same text
        assertEquals(gen.generate(prompts, o, null).get(1).text(), gen.generate(prompts, o, null).get(1).text());
    }

    @Test
    void stopTokensEndOneRowAndNotTheOthers() {
        Generator gen = new Generator(letters(2), new CharacterTokenizer(LETTERS));
        int[][] prompts = {{1, 2, 3}, {4, 5}};
        List<Generator.Output> free = gen.generate(prompts, Generator.Options.greedy(10).withStopTokens(Set.of()),
                null);
        int stop = free.get(0).tokens()[3];
        int at = 0;
        while (free.get(0).tokens()[at] != stop) {
            at++;
        }
        List<Generator.Output> stopped = gen.generate(prompts,
                Generator.Options.greedy(10).withStopTokens(Set.of(stop)), null);
        assertArrayEquals(java.util.Arrays.copyOf(free.get(0).tokens(), at), stopped.get(0).tokens());
        assertEquals(Generator.Finish.STOP, stopped.get(0).finish());
        int[] other = free.get(1).tokens();
        int end = 0;
        while (end < other.length && other[end] != stop) {
            end++;
        }
        assertArrayEquals(java.util.Arrays.copyOf(other, end), stopped.get(1).tokens());
    }

    @Test
    void stopStringsCutTheTextAndAreNeverStreamed() {
        Generator gen = new Generator(letters(3), new CharacterTokenizer(LETTERS));
        String free = gen.generate("abc", Generator.Options.greedy(20));
        String stop = free.substring(6, 9);
        StringBuilder streamed = new StringBuilder();
        String text = gen.generate("abc", Generator.Options.greedy(20).withStopStrings(stop),
                (row, t) -> streamed.append(t));
        assertEquals(free.substring(0, free.indexOf(stop)), text);
        assertEquals(text, streamed.toString());
    }

    @Test
    void streamingNeverSplitsACharacter() {
        // two tokens make one character; a lone token decodes to U+FFFD, as a half UTF-8 sequence does
        Tokenizer pairs = new Tokenizer() {
            @Override
            public int[] encode(String text) {
                return text.chars().flatMap(c -> java.util.stream.IntStream.of(c - 'a', 0)).toArray();
            }

            @Override
            public String decode(int... ids) {
                StringBuilder s = new StringBuilder();
                for (int i = 0; i + 1 < ids.length; i += 2) {
                    s.append(LETTERS.charAt((ids[i] + ids[i + 1]) % 26));
                }
                return ids.length % 2 == 1 ? s.append('�').toString() : s.toString();
            }

            @Override
            public int size() {
                return 26;
            }

            @Override
            public String kind() {
                return "pairs";
            }

            @Override
            public void save(java.nio.file.Path path) {
                throw new UnsupportedOperationException();
            }
        };
        Generator gen = new Generator(letters(4), pairs);
        List<String> pieces = new java.util.ArrayList<>();
        String text = gen.generate("hi", Generator.Options.sampled(9, Sampling.temperature(1), 5),
                (row, t) -> pieces.add(t));
        assertEquals(4, text.length() - (text.endsWith("�") ? 1 : 0));
        for (String p : pieces.subList(0, pieces.size() - 1)) {
            assertFalse(p.contains("�"), p);
        }
        assertEquals(text, String.join("", pieces));
    }

    @Test
    void samplingCutsOffAsAsked() {
        float[] logits = {3, 2, 1, 0, -1};
        double[] all = Sampling.temperature(1).probabilities(logits);
        double sum = 0;
        for (double p : all) {
            sum += p;
            assertTrue(p > 0);
        }
        assertEquals(1, sum, 1e-9);
        double[] k2 = Sampling.temperature(1).withTopK(2).probabilities(logits);
        assertEquals(all[0] / (all[0] + all[1]), k2[0], 1e-9);
        assertEquals(0, k2[2]);
        // top-p keeps the fewest tokens reaching p: 0.665 alone does not reach 0.8
        double[] p8 = Sampling.temperature(1).withTopP(0.8f).probabilities(logits);
        assertTrue(p8[1] > 0);
        assertEquals(0, p8[2]);
        double[] m = Sampling.temperature(1).withMinP(0.3f).probabilities(logits);
        assertTrue(m[1] > 0); // e^-1 = 0.37 of the top
        assertEquals(0, m[2]); // e^-2 = 0.14
        assertEquals(1, Sampling.GREEDY.probabilities(logits)[0]);
    }

    @Test
    void repetitionPenaltyFollowsTransformers() {
        float[] logits = {2, -2, 1};
        Sampling s = Sampling.GREEDY.withRepetitionPenalty(2);
        s.penalize(logits, new int[] {0, 1, 0});
        assertArrayEquals(new float[] {1, -4, 1}, logits);
    }

    @Test
    void samplesFollowTheProbabilities() {
        float[] logits = {1, 0, -1};
        Sampling s = Sampling.temperature(1);
        double[] p = s.probabilities(logits);
        int[] counts = new int[3];
        SplittableRandom r = new SplittableRandom(1);
        int n = 20000;
        for (int i = 0; i < n; i++) {
            counts[s.sample(logits, new int[0], r)]++;
        }
        for (int i = 0; i < 3; i++) {
            assertEquals(p[i], counts[i] / (double) n, 0.015);
        }
        assertNotEquals(0, counts[2]);
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 3, 5})
    void greedySpeculativeDecodingWritesWhatGreedyDecodingWrites(int lookahead) {
        Llama model = letters(1);
        Generator gen = new Generator(model, new CharacterTokenizer(LETTERS));
        int[] prompt = gen.encode("speculate");
        Generator.Options o = Generator.Options.greedy(25).withStopTokens(Set.of());
        int[] expected = gen.generate(new int[][] {prompt}, o, null).getFirst().tokens();
        Generator.Speculation other = gen.speculative(prompt, letters(2), lookahead, o, null);
        assertArrayEquals(expected, other.output().tokens());
        Generator.Speculation same = gen.speculative(prompt, letters(1), lookahead, o, null);
        assertArrayEquals(expected, same.output().tokens());
        assertEquals(1.0, same.acceptance(), 1e-9);
        // a perfect draft: the model checks lookahead + 1 tokens per step
        assertTrue(same.modelSteps() <= 1 + Math.ceilDiv(25, lookahead + 1), same.toString());
    }

    @ParameterizedTest
    @ValueSource(ints = {1, 4})
    void promptLookupDraftsKeepGreedyDecodingExact(int lookahead) {
        Llama model = letters(1);
        Generator gen = new Generator(model, new CharacterTokenizer(LETTERS));
        int[] prompt = gen.encode("abcabcabcabc");
        Generator.Options o = Generator.Options.greedy(30).withStopTokens(Set.of());
        int[] expected = gen.generate(new int[][] {prompt}, o, null).getFirst().tokens();
        Generator.Speculation s = gen.speculative(prompt, new PromptLookup(model.vocabSize()), lookahead, o, null);
        assertArrayEquals(expected, s.output().tokens());
    }

    @Test
    void promptLookupGuessesWholeRunsOfCopiedText() {
        // a model that always continues the cycle 0 1 2 3 4 0 1 ...
        int v = 5;
        float[][] table = new float[v][v];
        for (int i = 0; i < v; i++) {
            java.util.Arrays.fill(table[i], -5);
            table[i][(i + 1) % v] = 5;
        }
        Generator gen = new Generator(new Bigram(table), null);
        Generator.Options o = Generator.Options.greedy(40).withStopTokens(Set.of());
        int[] prompt = {0, 1, 2, 3, 4, 0};
        int[] expected = gen.generate(new int[][] {prompt}, o, null).getFirst().tokens();
        Generator.Speculation s = gen.speculative(prompt, new PromptLookup(v), 7, o, null);
        assertArrayEquals(expected, s.output().tokens());
        assertEquals(1.0, s.acceptance(), 1e-9);
        assertTrue(s.modelSteps() <= 1 + Math.ceilDiv(40, 8), s.toString());
    }

    /** Next-token logits from the last token alone; goes back by forgetting tokens. */
    record Bigram(float[][] table) implements LanguageModel {
        @Override
        public int vocabSize() {
            return table.length;
        }

        @Override
        public State newState(int batch) {
            return new Seen(new java.util.ArrayList<>());
        }

        @Override
        public io.github.zzuegg.brainiac.tensor.Tensor next(State state, int[][] tokens, boolean[][] padding) {
            for (int t : tokens[0]) {
                ((Seen) state).tokens().add(t);
            }
            List<Integer> seen = ((Seen) state).tokens();
            return io.github.zzuegg.brainiac.tensor.Tensor.of(io.github.zzuegg.brainiac.tensor.Shape.of(1, table.length),
                    table[seen.getLast()], DEVICE);
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

            @Override
            public State truncate(int length) {
                tokens.subList(length, tokens.size()).clear();
                return this;
            }
        }
    }

    @Test
    void sampledSpeculativeDecodingFollowsTheModelNotTheDraft() {
        int v = 4;
        SplittableRandom r = new SplittableRandom(3);
        float[][] model = new float[v][v];
        float[][] draft = new float[v][v];
        for (int i = 0; i < v; i++) {
            for (int j = 0; j < v; j++) {
                model[i][j] = (float) r.nextDouble(-2, 2);
                draft[i][j] = (float) r.nextDouble(-2, 2);
            }
        }
        Sampling s = Sampling.temperature(1);
        Generator gen = new Generator(new Bigram(model), null);
        // the exact probability of each pair of first tokens after token 0
        double[][] pair = new double[v][v];
        double[] first = s.probabilities(model[0]);
        for (int a = 0; a < v; a++) {
            double[] second = s.probabilities(model[a]);
            for (int b = 0; b < v; b++) {
                pair[a][b] = first[a] * second[b];
            }
        }
        int n = 20000;
        int[][] counts = new int[v][v];
        for (int i = 0; i < n; i++) {
            Generator.Options o = Generator.Options.sampled(2, s, i).withStopTokens(Set.of());
            int[] t = gen.speculative(new int[] {0}, new Bigram(draft), 2, o, null).output().tokens();
            counts[t[0]][t[1]]++;
        }
        for (int a = 0; a < v; a++) {
            for (int b = 0; b < v; b++) {
                assertEquals(pair[a][b], counts[a][b] / (double) n, 0.012, a + "," + b);
            }
        }
    }
}
