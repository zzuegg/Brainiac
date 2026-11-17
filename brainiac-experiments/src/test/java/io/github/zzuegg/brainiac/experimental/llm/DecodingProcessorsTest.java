package io.github.zzuegg.brainiac.experimental.llm;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.data.CharacterTokenizer;
import io.github.zzuegg.brainiac.llm.Generator;
import io.github.zzuegg.brainiac.llm.Sampling;

import java.util.Map;
import java.util.Set;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class DecodingProcessorsTest {

    private static int kept(float[] l) {
        int n = 0;
        for (float x : l) {
            if (x != LogitMath.MASKED) {
                n++;
            }
        }
        return n;
    }

    /** A Zipf-like row: logit {@code -s ln(rank)}. */
    private static float[] zipf(int n, double s) {
        float[] l = new float[n];
        for (int i = 0; i < n; i++) {
            l[i] = (float) (-s * Math.log(i + 1));
        }
        return l;
    }

    @Test
    void truncationRulesKeepTheExpectedTokens() {
        float[] l = zipf(100, 1.2);
        double[] p = LogitMath.softmax(l);
        float[] mp = l.clone();
        LogitsProcessors.minP(0.1).process(new int[0], mp);
        int expectMin = 0;
        for (double q : p) {
            expectMin += q >= 0.1 * p[0] ? 1 : 0;
        }
        assertEquals(expectMin, kept(mp));
        float[] ta = l.clone();
        LogitsProcessors.topA(0.2).process(new int[0], ta);
        int expectA = 0;
        for (double q : p) {
            expectA += q >= 0.2 * p[0] * p[0] ? 1 : 0;
        }
        assertEquals(expectA, kept(ta));
        float[] tk = l.clone();
        LogitsProcessors.topK(5).process(new int[0], tk);
        assertEquals(5, kept(tk));
        float[] tp = l.clone();
        LogitsProcessors.topP(0.5).process(new int[0], tp);
        assertTrue(kept(tp) > 1 && kept(tp) < 100);
        float[] eps = l.clone();
        LogitsProcessors.epsilon(1e-3).process(new int[0], eps);
        float[] eta = l.clone();
        LogitsProcessors.eta(1e-3).process(new int[0], eta);
        assertTrue(kept(eta) <= kept(eps), "eta's floor is at least epsilon's when entropy is low");
        float[] typ = l.clone();
        LogitsProcessors.typical(0.5).process(new int[0], typ);
        assertTrue(kept(typ) >= 1 && kept(typ) < 100);
        // a peaked row: typical sampling keeps the top token
        float[] peaked = {10, 0, 0, 0};
        LogitsProcessors.typical(0.9).process(new int[0], peaked);
        assertEquals(1, kept(peaked));
    }

    @Test
    void penaltiesPushDownSeenTokens() {
        float[] l = {1, 1, 1, 1};
        LogitsProcessors.frequencyPresence(0.5f, 0.25f).process(new int[] {0, 0, 1}, l);
        assertArrayEquals(new float[] {-0.25f, 0.25f, 1, 1}, l, 1e-6f);
        float[] r = {2, -2, 1};
        LogitsProcessors.repetition(2).process(new int[] {0, 1}, r);
        assertArrayEquals(new float[] {1, -4, 1}, r, 1e-6f);
        float[] ng = {0, 0, 0, 0};
        LogitsProcessors.noRepeatNgram(2).process(new int[] {1, 2, 3, 1}, ng);
        assertEquals(LogitMath.MASKED, ng[2]);
        float[] b = {0, 0};
        LogitsProcessors.logitBias(Map.of(1, 3f)).process(new int[0], b);
        assertEquals(3f, b[1]);
    }

    @Test
    void dryPenalizesTheTokenThatExtendsARepeat() {
        // context "1 2 3 4 5 1 2 3": continuing with 4 would repeat a 3-token passage
        int[] c = {1, 2, 3, 4, 5, 1, 2, 3};
        float[] l = new float[8];
        LogitsProcessors.dry(1f, 2f, 2, Set.of()).process(c, l);
        assertEquals(-2f, l[4], 1e-6f, "match of 3, allowed 2: 1 * 2^(3-2)");
        for (int t = 0; t < 8; t++) {
            if (t != 4) {
                assertEquals(0f, l[t]);
            }
        }
        float[] broken = new float[8];
        LogitsProcessors.dry(1f, 2f, 2, Set.of(2)).process(c, broken);
        assertEquals(0f, broken[4], 1e-6f, "a breaker inside the match shortens it below the allowed length");
    }

    @Test
    void entropyTemperatureSharpensConfidentRows() {
        float[] sure = {4, 0, 0, 0};
        float[] unsure = {0.1f, 0, 0, 0};
        LogitsProcessor edt = LogitsProcessors.entropyTemperature(1f, 0.8f, 1f);
        float[] a = sure.clone();
        edt.process(new int[0], a);
        float[] b = unsure.clone();
        edt.process(new int[0], b);
        assertTrue(a[0] / sure[0] > b[0] / unsure[0], "lower entropy gets a lower temperature");
    }

    @Test
    void mirostatHoldsTheSurpriseNearTheTarget() {
        float[] l = zipf(200, 1.1);
        for (Mirostat m : new Mirostat[] {Mirostat.v2(4, 0.2), Mirostat.v1(4, 0.2, 100)}) {
            SplittableRandom r = new SplittableRandom(3);
            double sum = 0;
            int n = 600;
            for (int i = 0; i < n; i++) {
                m.sample(new int[0], l, r);
                if (i >= 100) {
                    sum += m.lastSurprise();
                }
            }
            double mean = sum / (n - 100);
            assertEquals(4, mean, 0.6, m + " mean surprise " + mean);
        }
    }

    @Test
    void processedModelRunsInTheCoreGenerator() {
        HookedDecoder m = TinyLm.trained();
        CharacterTokenizer tok = CharacterTokenizer.train("abcdefghijk");
        assertEquals(TinyLm.VOCAB, tok.size());
        Generator.Options opt = Generator.Options.greedy(6).withStopTokens(Set.of());
        int[] plain = new Generator(m, tok).generate(new int[][] {{0, 1, 2}}, opt, null).get(0).tokens();
        assertArrayEquals(new int[] {3, 4, 5, 6, 7, 8}, plain);
        // a no-repeat-1gram processor forbids every token already present
        ProcessedLanguageModel banned = new ProcessedLanguageModel(m, LogitsProcessors.noRepeatNgram(1));
        int[] out = new Generator(banned, tok).generate(new int[][] {{0, 1, 2}}, opt, null).get(0).tokens();
        assertEquals(6, Set.of(java.util.Arrays.stream(out).boxed().toArray(Integer[]::new)).size());
        // the pipeline with a core Sampling processor matches the core generator under greedy
        int[] piped = DecodingPipeline.of(m).processor(LogitsProcessor.of(Sampling.GREEDY))
                .generate(new int[] {0, 1, 2}, 6, new SplittableRandom(1));
        assertArrayEquals(plain, piped);
    }
}
