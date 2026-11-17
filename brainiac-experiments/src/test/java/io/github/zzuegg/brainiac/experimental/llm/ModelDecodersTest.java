package io.github.zzuegg.brainiac.experimental.llm;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.data.CharacterTokenizer;
import io.github.zzuegg.brainiac.llm.Generator;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.Arrays;
import java.util.Set;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class ModelDecodersTest {

    private static final int[] PROMPT = {1, 3, 5};

    @Test
    void jacobiAndLookaheadMatchGreedyInFewerPasses() {
        HookedDecoder m = TinyLm.trained();
        int[] greedy = TinyLm.greedy(m, PROMPT, 16);
        for (JacobiDecoding.Guess g : JacobiDecoding.Guess.values()) {
            JacobiDecoding.Result r = JacobiDecoding.generate(m, PROMPT, 16, 4, g);
            assertArrayEquals(greedy, r.tokens(), g.name());
            // plain Jacobi fixes at least one token per pass; the n-gram pool makes guesses hit
            int bound = g == JacobiDecoding.Guess.JACOBI ? 16 : 10;
            assertTrue(r.modelCalls() <= bound, g + " used " + r.modelCalls() + " passes");
        }
    }

    @Test
    void medusaHeadsLearnToGuessAheadAndKeepGreedyOutput() {
        HookedDecoder m = TinyLm.trained();
        MedusaHeads medusa = new MedusaHeads(m, 3, 4);
        Optimizer opt = Optimizer.adam(0.01f);
        SplittableRandom r = new SplittableRandom(5);
        for (int s = 0; s < 60; s++) {
            Var loss = medusa.loss(TinyLm.counting(r, 16, 12));
            loss.backward();
            opt.step(medusa.parameters());
            m.zeroGrad();
            for (Var p : medusa.parameters()) {
                p.zeroGrad();
            }
        }
        MedusaHeads.Result res = medusa.generate(PROMPT, 16);
        assertArrayEquals(TinyLm.greedy(m, PROMPT, 16), res.tokens());
        assertTrue(res.forwardPasses() <= 8, "passes " + res.forwardPasses());
    }

    @Test
    void earlyExitSavesBlocksAndSelfSpeculationIsExact() {
        HookedDecoder m = TinyLm.deeplySupervised();
        EarlyExit.Result calm = EarlyExit.generate(m, PROMPT, 12, 0.9);
        assertTrue(calm.meanDepth() < m.layers(), "mean depth " + calm.meanDepth());
        assertTrue(TinyLm.countingAccuracy(PROMPT, calm.tokens()) > 0.8);
        int[] greedy = TinyLm.greedy(m, PROMPT, 12);
        EarlyExit.Speculation spec = EarlyExit.selfSpeculative(m, PROMPT, 12, 2, 3);
        assertArrayEquals(greedy, spec.tokens());
        assertTrue(spec.fullPasses() < 12, "full passes " + spec.fullPasses());
        // skipping a block changes the computation but the supervised exits keep it usable
        m.skip(2);
        try {
            assertTrue(TinyLm.countingAccuracy(PROMPT, TinyLm.greedy(m, PROMPT, 8)) > 0.5);
        } finally {
            m.skip();
        }
    }

    @Test
    void dolaContrastsLayersAndStillCounts() {
        float[] fin = {3, 2, 0};
        float[][] early = {{3, 0, 0}, {3, 2, 0}};
        int[] chosen = new int[1];
        float[] s = DolaDecoding.scores(fin, early, 0.1, chosen);
        assertEquals(0, chosen[0], "the layer that differs most is chosen");
        assertEquals(1, LogitMath.argmax(s), "the token the late layer added wins");
        HookedDecoder m = TinyLm.deeplySupervised();
        DolaDecoding dola = new DolaDecoding(m, new int[] {1, 2}, 0.1);
        CharacterTokenizer tok = CharacterTokenizer.train("abcdefghijk");
        int[] out = new Generator(dola, tok).generate(new int[][] {PROMPT},
                Generator.Options.greedy(8).withStopTokens(Set.of()), null).get(0).tokens();
        assertTrue(TinyLm.countingAccuracy(PROMPT, out) >= 0.75, Arrays.toString(out));
    }

    @Test
    void contrastiveDecodingAndGuidanceWrapModels() {
        float[] e = {2, 1.9f, -5};
        float[] a = {2, 0, -5};
        float[] s = ContrastiveDecoding.scores(e, a, 0.1, 1);
        assertEquals(1, LogitMath.argmax(s), "what only the expert likes wins");
        assertEquals(LogitMath.MASKED, s[2], "implausible tokens are cut");
        HookedDecoder expert = TinyLm.trained();
        HookedDecoder amateur = HookedDecoder.create(HookedDecoder.Settings.tiny(TinyLm.VOCAB), 3);
        TinyLm.train(amateur, amateur.parameters(), Optimizer.adam(0.01f), 15, 12, 4);
        int[] cd = TinyLm.greedy(new ContrastiveDecoding(expert, amateur, 0.1, 0.5), PROMPT, 10);
        assertTrue(TinyLm.countingAccuracy(PROMPT, cd) >= 0.9, Arrays.toString(cd));

        float[] c = {1, 2, 3};
        float[] u = {3, 2, 1};
        double[] lc = LogitMath.logSoftmax(c);
        float[] g1 = ClassifierFreeGuidance.guide(c, u, 1);
        for (int i = 0; i < 3; i++) {
            assertEquals(lc[i], g1[i], 1e-5);
        }
        float[] g3 = ClassifierFreeGuidance.guide(c, u, 3);
        assertTrue(g3[2] - g3[0] > g1[2] - g1[0], "guidance sharpens towards the prompt");
        int[] guided = TinyLm.greedy(new ClassifierFreeGuidance(expert, new int[0], 1.5), PROMPT, 10);
        assertTrue(TinyLm.countingAccuracy(PROMPT, guided) >= 0.9, Arrays.toString(guided));
    }

    @Test
    void contrastiveSearchIsGreedyAtAlphaZeroAndAvoidsRepeats() {
        HookedDecoder m = TinyLm.trained();
        assertArrayEquals(TinyLm.greedy(m, PROMPT, 8), ContrastiveSearch.generate(m, PROMPT, 8, 4, 0));
        HookedDecoder random = HookedDecoder.create(HookedDecoder.Settings.tiny(24), 21);
        int[] greedy = TinyLm.greedy(random, new int[] {1, 2}, 20);
        int[] cs = ContrastiveSearch.generate(random, new int[] {1, 2}, 20, 5, 0.7);
        assertTrue(distinct(cs) > distinct(greedy), distinct(cs) + " vs greedy " + distinct(greedy));
    }

    private static long distinct(int[] t) {
        return Arrays.stream(t).distinct().count();
    }
}
