package io.github.zzuegg.brainiac.experiments.robustness;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.robust.TestTimeAdaptation;
import io.github.zzuegg.brainiac.experiments.techniques.CnnNormalizationShootout;
import io.github.zzuegg.brainiac.nn.Model;

import org.junit.jupiter.api.Test;

class ShiftAdaptationTest {

    @Test
    void bnStatisticsRecoverMostOfTheCorruptionGap() {
        Model bn = ShiftAdaptation.train(CnnNormalizationShootout.Norm.BATCH_NORM, false, 300, 1);
        Model gn = ShiftAdaptation.train(CnnNormalizationShootout.Norm.GROUP_NORM, false, 300, 1);
        Model aug = ShiftAdaptation.train(CnnNormalizationShootout.Norm.BATCH_NORM, true, 300, 1);
        double clean = ShiftAdaptation.evaluate(bn, TestTimeAdaptation.Method.NONE, ShiftAdaptation.Corruption.NOISE,
                0, 4, 64, false, 9);
        System.out.printf("Shift clean BN model accuracy %.3f%n", clean);
        double sumNone = 0;
        double sumStats = 0;
        double sumTent = 0;
        for (ShiftAdaptation.Corruption c : ShiftAdaptation.Corruption.values()) {
            for (int sev : new int[] {3, 5}) {
                double none = ShiftAdaptation.evaluate(bn, TestTimeAdaptation.Method.NONE, c, sev, 4, 64, false, 9);
                double stats = ShiftAdaptation.evaluate(bn, TestTimeAdaptation.Method.BN_STATS, c, sev, 4, 64, false, 9);
                double tent = ShiftAdaptation.evaluate(bn, TestTimeAdaptation.Method.TENT, c, sev, 4, 64, false, 9);
                double memo = ShiftAdaptation.evaluate(bn, TestTimeAdaptation.Method.MEMO, c, sev, 1, 24, false, 9);
                double group = ShiftAdaptation.evaluate(gn, TestTimeAdaptation.Method.NONE, c, sev, 4, 64, false, 9);
                double augmented = ShiftAdaptation.evaluate(aug, TestTimeAdaptation.Method.NONE, c, sev, 4, 64, false,
                        9);
                System.out.printf("Shift %-8s sev %d  none %.3f  BN stats %.3f  TENT %.3f  MEMO(bs1) %.3f  "
                        + "GroupNorm %.3f  aug-trained %.3f%n", c, sev, none, stats, tent, memo, group, augmented);
                sumNone += none;
                sumStats += stats;
                sumTent += tent;
            }
        }
        System.out.printf("Shift mean over corruptions: none %.3f  BN stats %.3f  TENT %.3f%n", sumNone / 8,
                sumStats / 8, sumTent / 8);
        assertTrue(clean > 0.8, "clean " + clean);
        assertTrue(sumStats > sumNone, "BN statistics help on average");
    }

    @Test
    void batchStatisticsFailUnderLabelShiftAndTentDriftsOnLongStreams() {
        Model bn = ShiftAdaptation.train(CnnNormalizationShootout.Norm.BATCH_NORM, false, 300, 1);
        ShiftAdaptation.Corruption c = ShiftAdaptation.Corruption.NOISE;
        double none = ShiftAdaptation.evaluate(bn, TestTimeAdaptation.Method.NONE, c, 3, 12, 32, true, 5);
        double stats = ShiftAdaptation.evaluate(bn, TestTimeAdaptation.Method.BN_STATS, c, 3, 12, 32, true, 5);
        double tent = ShiftAdaptation.evaluate(bn, TestTimeAdaptation.Method.TENT, c, 3, 12, 32, true, 5);
        System.out.printf("Shift label-shifted stream (one class per batch), noise sev 3: none %.3f  BN stats %.3f  "
                + "TENT %.3f%n", none, stats, tent);
        double tentShort = ShiftAdaptation.evaluate(bn, TestTimeAdaptation.Method.TENT, c, 3, 5, 64, false, 6);
        double tentLong = ShiftAdaptation.evaluate(bn, TestTimeAdaptation.Method.TENT, c, 3, 60, 64, false, 6);
        System.out.printf("Shift TENT stream noise sev 3: 5 batches %.3f  60 batches %.3f%n", tentShort, tentLong);
        assertTrue(stats < none, "test-batch statistics break when a batch holds one class");
    }
}
