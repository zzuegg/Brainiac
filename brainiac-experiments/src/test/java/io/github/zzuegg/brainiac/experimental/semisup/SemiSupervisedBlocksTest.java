package io.github.zzuegg.brainiac.experimental.semisup;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class SemiSupervisedBlocksTest {

    private static Tensor randn(int n, int d, long seed, double scale) {
        SplittableRandom r = new SplittableRandom(seed);
        float[] v = new float[n * d];
        for (int i = 0; i < v.length; i++) {
            v[i] = (float) (r.nextGaussian() * scale);
        }
        return Tensor.of(Shape.of(n, d), v);
    }

    @Test
    void sigregIsSmallForGaussianAndLargeForCollapsedFeatures() {
        double gauss = JepaMatch.sigreg(Var.of(randn(256, 8, 1, 1)), 16, 17, 3f, 7).value().item();
        double collapsed = JepaMatch.sigreg(Var.of(randn(256, 8, 2, 0.05)), 16, 17, 3f, 7).value().item();
        System.out.printf("SIGReg gaussian %.4f collapsed %.4f%n", gauss, collapsed);
        assertTrue(gauss < 5 && collapsed > 10 * gauss);
        Var z = Var.param(randn(64, 4, 3, 0.1));
        Var g = Var.gradients(JepaMatch.sigreg(z, 8, 9, 3f, 1), List.of(z), false).get(0);
        assertTrue(g.value().abs().sum().item() > 0, "differentiable");
    }

    @Test
    void consistencyRampAndPseudoLabelMask() {
        assertEquals(1f, ConsistencyTraining.rampUp(100, 100), 1e-6);
        assertTrue(ConsistencyTraining.rampUp(0, 100) < 0.01);
        ConsistencyTraining pl = new ConsistencyTraining(ConsistencyTraining.Mode.PSEUDO_LABEL, 1f, 0, 0.99f);
        Tensor logits = Tensor.of(Shape.of(2, 2), new float[] {0, 0, 0, 0});
        assertEquals(0, pl.loss(Var.of(logits), logits, 5).value().item(), 1e-6, "nothing above the threshold");
        ConsistencyTraining pi = new ConsistencyTraining(ConsistencyTraining.Mode.PI_MODEL, 1f, 0, 0f);
        assertEquals(0, pi.loss(Var.of(logits), logits, 5).value().item(), 1e-6, "identical passes agree");
    }

    @Test
    void fixMatchMasksAndAdaptiveThresholds() {
        float[][] probs = {{0.97f, 0.03f}, {0.6f, 0.4f}, {0.2f, 0.8f}};
        float[] w = FixMatch.fixed(0.95f).weights(probs, new int[] {0, 1, 2});
        assertEquals(1f, w[0]);
        assertEquals(0f, w[1]);
        AdaptiveThreshold flex = new AdaptiveThreshold(AdaptiveThreshold.Mode.FLEX, 2, 3, 0.95f, 0.9);
        float[] fw = flex.weights(probs, new int[] {0, 1, 2});
        assertEquals(3f, fw[0] + fw[1] + fw[2], 1e-6, "FlexMatch's warm-up keeps everything");
        AdaptiveThreshold soft = new AdaptiveThreshold(AdaptiveThreshold.Mode.SOFT, 2, 3, 0.95f, 0.5);
        float[] sw = soft.weights(probs, new int[] {0, 1, 2});
        assertTrue(sw[0] == 1f && sw[1] < 1f && sw[1] > 0f, "SoftMatch weighs low confidence softly");
    }

    @Test
    void mixMatchSharpensAndKeepsLabelledRowsDominant() {
        MixMatch mm = new MixMatch(0.5, 0.75f, 1f, 0, true, true);
        float[][] g = mm.guess(List.of(Tensor.of(Shape.of(1, 2), new float[] {1, 0})));
        assertTrue(g[0][0] > 0.73f, "sharpened beyond softmax");
        MixMatch.Mixed mixed = mm.mix(randn(4, 3, 1, 1), SoftTargets.oneHot(new int[] {0, 0, 0, 0}, 2), randn(4, 3, 2, 1),
                SoftTargets.oneHot(new int[] {1, 1, 1, 1}, 2), new SplittableRandom(3));
        float[] y = mixed.y().toArray();
        for (int i = 0; i < 4; i++) {
            assertTrue(y[i * 2] >= 0.5f, "lambda' >= 0.5 keeps labelled rows labelled-dominated");
        }
    }

    @Test
    void vatPerturbationHasTheRequestedNorm() {
        Var w = Var.param(randn(3, 2, 4, 1));
        Tensor x = randn(5, 3, 5, 1);
        VirtualAdversarial vat = new VirtualAdversarial(0.1f, 0.5f, 1, 0f);
        Tensor r = vat.perturbation(v -> v.matmul(w), x, x.matmul(w.value()), new SplittableRandom(1));
        float[] v = r.toArray();
        double norm = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
        assertEquals(0.5, norm, 1e-4);
        assertTrue(vat.loss(u -> u.matmul(w), x, new SplittableRandom(2)).value().item() >= 0);
    }
}
