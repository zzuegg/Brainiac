package io.github.zzuegg.brainiac.experimental.labels;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.LabelCorruption;
import io.github.zzuegg.brainiac.experimental.interp.AttentionRollout;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class LabelBlocksTest {

    private static final Tensor LOGITS = Tensor.of(Shape.of(2, 3), new float[] {2, 0, -1, 0.5f, 0.2f, 0.1f});
    private static final Tensor IDS = Tensor.of(Shape.of(2), new float[] {0, 2});

    @Test
    void robustLossesMatchTheirClosedForms() {
        double ce = Loss.crossEntropyFromLogits(0).compute(Var.of(LOGITS), IDS).value().item();
        double gceSmallQ = RobustLosses.gce(1e-3f).compute(Var.of(LOGITS), IDS).value().item();
        assertEquals(ce, gceSmallQ, 0.01, "GCE tends to CE as q -> 0");
        double mae = RobustLosses.mae().compute(Var.of(LOGITS), IDS).value().item();
        double gce1 = RobustLosses.gce(1f).compute(Var.of(LOGITS), IDS).value().item();
        assertEquals(mae / 2, gce1, 1e-5, "GCE at q = 1 is MAE / 2");
        double amse = RobustLosses.amse(Var.of(LOGITS), IDS, 1f).value().item();
        assertTrue(amse > 0 && amse < 1);
        double jal = RobustLosses.jointAsymmetric(1, 1, 10).compute(Var.of(LOGITS), IDS).value().item();
        assertTrue(Double.isFinite(jal));
    }

    @Test
    void forwardCorrectionWithIdentityIsCrossEntropy() {
        double[][] eye = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
        double ce = Loss.crossEntropyFromLogits(0).compute(Var.of(LOGITS), IDS).value().item();
        assertEquals(ce, TransitionCorrection.forwardLoss(Var.of(LOGITS), IDS, eye).value().item(), 1e-4);
        assertEquals(ce, TransitionCorrection.backwardLoss(Var.of(LOGITS), IDS, eye).value().item(), 1e-4);
        float[][] p = {{0.7f, 0.3f}, {0.1f, 0.9f}, {0.9f, 0.1f}, {0.2f, 0.8f}};
        double[][] t = TransitionCorrection.estimateMeanAnchors(p);
        assertEquals(0.8, t[0][0], 1e-6);
        assertEquals(0.85, t[1][1], 1e-6);
    }

    @Test
    void logitAdjustmentAndMargins() {
        int[] counts = {100, 10, 1};
        Tensor adj = LogitAdjustment.posthocAdditive(LOGITS, LogitAdjustment.priors(counts), 1);
        float[] v = adj.toArray();
        assertTrue(v[2] - v[0] > (-1 - 2), "the rare class is lifted");
        double bs = LogitAdjustment.balancedSoftmax(counts).compute(Var.of(LOGITS), IDS).value().item();
        assertTrue(bs > 0);
        double[] w = MarginAndDecoupling.drwWeights(counts, 0.999);
        assertTrue(w[2] > w[1] && w[1] > w[0]);
        double ldam = MarginAndDecoupling.ldam(counts, 1, 1, null).compute(Var.of(LOGITS), IDS).value().item();
        double ce = Loss.crossEntropyFromLogits(0).compute(Var.of(LOGITS), IDS).value().item();
        assertTrue(ldam > ce, "margins raise the loss");
        Tensor tau = MarginAndDecoupling.tauNormalize(Tensor.of(Shape.of(2, 2), new float[] {3, 0, 4, 1}), 1);
        float[] tv = tau.toArray();
        assertEquals(1, tv[0] * tv[0] + tv[2] * tv[2], 1e-5, "unit-norm columns at tau = 1");
    }

    @Test
    void sampleSelectionSplitsLossesWithAMixture() {
        SplittableRandom r = new SplittableRandom(1);
        double[] losses = new double[200];
        for (int i = 0; i < losses.length; i++) {
            losses[i] = i < 140 ? 0.1 + 0.05 * r.nextGaussian() : 2 + 0.3 * r.nextGaussian();
        }
        double[] clean = SampleSelection.cleanProbability(losses, 1);
        assertTrue(clean[0] > 0.9 && clean[199] < 0.1);
        assertArrayEquals(new int[] {1, 2}, SampleSelection.smallLoss(new double[] {3, 1, 2}, 0.5));
        assertEquals(0.6, SampleSelection.keepRate(100, 0.4, 50), 1e-9);
    }

    @Test
    void minCutSourceSideIsTheExposedEdmondsKarpCut() {
        double[][] cap = new double[4][4];
        cap[0][1] = 3;
        cap[1][3] = 1;
        cap[0][2] = 1;
        cap[2][3] = 5;
        assertEquals(2, AttentionRollout.maxFlow(cap, 0, 3), 1e-9);
        boolean[] side = AttentionRollout.minCutSourceSide(cap, 0, 3);
        assertArrayEquals(new boolean[] {true, true, false, false}, side);
    }

    @Test
    void puRisksAndPriorEstimate() {
        Var zp = Var.of(Tensor.of(Shape.of(2, 1), new float[] {5, 5}));
        Var zu = Var.of(Tensor.of(Shape.of(2, 1), new float[] {5, -5}));
        double upu = PositiveUnlabeled.unbiasedRisk(zp, zu, 0.5).value().item();
        PositiveUnlabeled.Step nn = PositiveUnlabeled.nonNegative(zp, zu, 0.5, 0, 1);
        assertEquals(upu, nn.risk(), 1e-6);
        assertTrue(nn.negativeRisk() >= -1e-6 || nn.corrective());
        assertEquals(0.8, PositiveUnlabeled.elkanNotoPrior(new double[] {0.2, 0.2}, 1.0 / 6), 1e-6);
        float[][] x = {{0, 0}, {0.1f, 0}, {0, 0.1f}, {5, 5}, {5.1f, 5}, {5, 5.1f}};
        boolean[] pred = PositiveUnlabeled.twoHnc(x, new boolean[] {true, false, false, false, false, false}, 0.4, 2, 10);
        assertArrayEquals(new boolean[] {true, true, true, false, false, false}, pred);
    }

    @Test
    void labelModelsRecoverPlantedAccuracies() {
        int[] y = new int[3000];
        SplittableRandom r = new SplittableRandom(1);
        for (int i = 0; i < y.length; i++) {
            y[i] = r.nextInt(3);
        }
        double[] acc = {0.9, 0.8, 0.7, 0.6, 0.6};
        double[] cov = {0.8, 0.8, 0.8, 0.8, 0.8};
        LabelCorruption.Votes v = LabelCorruption.labellingFunctions(y, 3, acc, cov, false, 2);
        LabelModel.Fit dp = LabelModel.dataProgramming(v.votes(), 3, 30, new int[0][]);
        LabelModel.Fit mc = LabelModel.matrixCompletion(v.votes(), 3);
        LabelModel.Fit ds = LabelModel.dawidSkene(v.votes(), 3, 30);
        for (int j = 0; j < acc.length; j++) {
            assertEquals(acc[j], dp.accuracies()[j], 0.05, "data programming " + j);
            assertEquals(acc[j], mc.accuracies()[j], 0.08, "matrix completion " + j);
        }
        double mv = LabelModel.majorityVote(v.votes(), 3).coveredAccuracy(v.votes(), y);
        assertTrue(dp.coveredAccuracy(v.votes(), y) >= mv);
        assertTrue(ds.coveredAccuracy(v.votes(), y) >= mv - 0.01);
    }

    @Test
    void attentionMilShapesAndWeights() {
        AttentionMil mil = new AttentionMil(4, 8, 6, 1, AttentionMil.Pooling.GATED, new SplittableRandom(1));
        Var out = mil.forward(Var.of(Tensor.of(Shape.of(2, 5, 4), new float[40])));
        assertEquals(Shape.of(2, 1), out.shape());
        float[] a = mil.attention().toArray();
        double s = 0;
        for (int i = 0; i < 5; i++) {
            s += a[i];
        }
        assertEquals(1, s, 1e-5);
        assertEquals(Shape.of(3, 1), AttentionMil.spec(8, 6, 1, AttentionMil.Pooling.MEAN).outputShape(Shape.of(3, 5, 4)));
    }

    @Test
    void partialLabelsAndProportions() {
        boolean[][] cand = {{true, true, false}};
        PartialLabels pl = new PartialLabels(cand);
        assertArrayEquals(new float[] {0.5f, 0.5f, 0}, pl.weights(0));
        pl.update(Tensor.of(Shape.of(1, 3), new float[] {2, 0, 5}), new int[] {0});
        float[] w = pl.weights(0);
        assertTrue(w[0] > 0.85f && w[2] == 0, "re-weighted within the candidate set");
        assertTrue(PartialLabels.ccLoss(Var.of(LOGITS.slice(0, 0, 1)), new int[] {0}, cand).value().item() > 0);
        int[] lab = LabelProportions.transportLabels(new float[][] {{0.6f, 0.4f}, {0.55f, 0.45f}}, new double[] {0.5, 0.5},
                0.05);
        assertTrue(lab[0] != lab[1], "the proportions are respected");
        double dllp = LabelProportions.dllpLoss(Var.of(Tensor.of(Shape.of(2, 2), new float[] {0, 0, 0, 0})),
                new double[] {0.5, 0.5}).value().item();
        assertEquals(Math.log(2), dllp, 1e-5);
        float[][] x = {{0, 0}, {1, 1}, {0, 0}, {1, 1}};
        LabelProportions.MeanMap mm = new LabelProportions.MeanMap(x, new int[][] {{0, 1}, {2, 3}, {0, 2}},
                new double[][] {{0.5, 0.5}, {0.5, 0.5}, {1, 0}});
        assertEquals(0, mm.predict(new float[] {0, 0}));
        assertEquals(1, mm.predict(new float[] {1, 1}));
    }

    @Test
    void labelCorruptionKeepsCleanLabelsAndRates() {
        int[] y = new int[2000];
        for (int i = 0; i < y.length; i++) {
            y[i] = i % 10;
        }
        LabelCorruption.Noisy sym = LabelCorruption.symmetric(y, 10, 0.4, 1);
        assertEquals(0.4, sym.rate(), 0.04);
        assertArrayEquals(y, sym.clean());
        LabelCorruption.Noisy pair = LabelCorruption.pairFlip(y, 10, 0.2, 1);
        for (int i = 0; i < y.length; i++) {
            assertTrue(pair.noisy()[i] == y[i] || pair.noisy()[i] == (y[i] + 1) % 10);
        }
        int[] tail = LabelCorruption.longTail(y, 10, 100, 100, 1);
        assertEquals(LabelCorruption.longTailCounts(10, 100, 100)[9], 1);
        assertTrue(tail.length < 400);
        LabelCorruption.Split split = LabelCorruption.labelledSplit(y, 10, 4, 1);
        assertEquals(40, split.labelled().length);
    }
}
