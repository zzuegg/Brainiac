package io.github.zzuegg.brainiac.experimental.nn;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class AdaptiveDepthTest {

    private static Layer block(int d, SplittableRandom r) {
        return TransformerBlock.of(2).causal(true).rmsNorm(true).create(Shape.of(1, d), r);
    }

    @Test
    void actReportsTicksAndPonderCostShortensThinking() {
        SplittableRandom r = new SplittableRandom(1);
        LoopedTransformer loop = new LoopedTransformer(block(8, r), 8, 6, LoopedTransformer.Halting.ACT, r);
        Tensor x = Tensor.uniform(Shape.of(2, 5, 8), -1, 1, new SplittableRandom(2));
        loop.forward(Var.of(x), true);
        float before = loop.meanTicks();
        for (float t : loop.lastTicks()) {
            assertTrue(t >= 1 && t <= 7, "ticks " + t);
        }
        Optimizer opt = Optimizer.adam(0.05f);
        for (int i = 0; i < 40; i++) {
            loop.forward(Var.of(x), true);
            if (!loop.ponderLoss().requiresGrad()) {
                break;   // every token halts at the first tick: nothing left to shorten
            }
            loop.ponderLoss().backward();
            opt.step(loop.parameters());
        }
        loop.forward(Var.of(x), true);
        assertTrue(loop.meanTicks() < before - 0.5, "ticks " + before + " -> " + loop.meanTicks());
    }

    @Test
    void ponderNetHaltingIsADistributionAndInferenceStopsEarly() {
        SplittableRandom r = new SplittableRandom(3);
        LoopedTransformer loop = new LoopedTransformer(block(8, r), 8, 5, LoopedTransformer.Halting.PONDER, r);
        Tensor x = Tensor.uniform(Shape.of(2, 4, 8), -1, 1, new SplittableRandom(4));
        loop.forward(Var.of(x), true);
        float[] total = new float[8];
        for (Var p : loop.haltingDistribution()) {
            float[] v = p.value().toArray();
            for (int i = 0; i < 8; i++) {
                total[i] += v[i];
            }
        }
        for (float s : total) {
            assertEquals(1, s, 1e-4f);
        }
        assertTrue(loop.ponderLoss().value().item() >= -1e-5);
        loop.forward(Var.of(x), false);
        for (float t : loop.lastTicks()) {
            assertTrue(t >= 1 && t <= 5);
            assertEquals(t, Math.round(t), 0f);
        }
    }

    @Test
    void loopedTransformerLearnsRecall() {
        SplittableRandom r = new SplittableRandom(5);
        LoopedTransformer loop = new LoopedTransformer(block(16, r), 16, 3, LoopedTransformer.Halting.ACT, r);
        double acc = MixerHarness.train(loop, MixerHarness.recall(6, 6, 3), 16, 200, 0.01f, 6,
                () -> loop.ponderLoss() == null ? null : loop.ponderLoss().mul(0.001f));
        assertTrue(acc > 0.8, "accuracy " + acc);
    }

    @Test
    void universalTransformerLearnsRecall() {
        UniversalTransformer ut = new UniversalTransformer(16, 2, true, 3, new SplittableRandom(7));
        double acc = MixerHarness.train(ut, MixerHarness.recall(6, 6, 3), 16, 200, 0.01f, 8);
        assertTrue(acc > 0.8, "accuracy " + acc);
    }

    @Test
    void mixtureOfRecursionsGivesTokensGeometricDepths() {
        SplittableRandom r = new SplittableRandom(9);
        MixtureOfRecursions mor = new MixtureOfRecursions(block(8, r), 8, 3, 0.5f, r);
        Tensor x = Tensor.uniform(Shape.of(2, 8, 8), -1, 1, new SplittableRandom(10));
        assertEquals(Shape.of(2, 8, 8), mor.forward(Var.of(x)).shape());
        int[] histogram = new int[4];
        for (int[] row : mor.lastDepths()) {
            for (int dpt : row) {
                histogram[dpt]++;
            }
        }
        // 4 tokens recurse at least once, 2 at least twice, 1 three times, per sequence
        assertEquals(2 * 4, histogram[0]);
        assertEquals(2 * 2, histogram[1]);
        assertEquals(2 * 1, histogram[2]);
        assertEquals(2 * 1, histogram[3]);
    }

    @Test
    void mixtureOfRecursionsLearnsRecall() {
        SplittableRandom r = new SplittableRandom(11);
        MixtureOfRecursions mor = new MixtureOfRecursions(block(16, r), 16, 2, 0.75f, r);
        double acc = MixerHarness.train(mor, MixerHarness.recall(6, 6, 3), 16, 250, 0.01f, 12);
        assertTrue(acc > 0.6, "accuracy " + acc);
    }

    @Test
    void feedbackMemoryTracksParityThatPlainAttentionCannot() {
        MixerHarness.Task task = MixerHarness.parity(5);
        double feedback = MixerHarness.train(new FeedbackTransformer(16, 2, 2, new SplittableRandom(13)), task, 16,
                600, 0.01f, 14);
        double plain = MixerHarness.train(new GatedAttention(16, 2, true, 10000f, new SplittableRandom(13)), task, 16,
                600, 0.01f, 14);
        assertTrue(feedback > 0.95, "feedback " + feedback);
        assertTrue(feedback > plain + 0.05, "feedback " + feedback + " vs attention " + plain);
    }
}
