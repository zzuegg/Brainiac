package io.github.zzuegg.brainiac.experiments.attention;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class RecallExtrapolationTest {

    @Test
    void scaledAndSigmoidAttentionKeepRecallingAtFourTimesTheKeys() {
        List<LengthGeneralisation.Row> rows = LengthGeneralisation.recall(List.of("softmax", "SSMax", "sigmoid"), 250,
                1);
        System.out.print(LengthGeneralisation.table("recall @ pairs", rows));
        System.out.printf("recall at 16 pairs (trained on 4): SSMax %.3f, sigmoid %.3f vs softmax %.3f%n",
                rows.get(1).accuracy()[2], rows.get(2).accuracy()[2], rows.get(0).accuracy()[2]);
        for (LengthGeneralisation.Row r : rows) {
            assertTrue(r.accuracy()[0] > 0.9, r.name());
            assertTrue(r.accuracy()[2] > 0.7, r.name());
        }
        assertTrue(rows.get(1).accuracy()[2] > rows.get(0).accuracy()[2] - 0.03);
    }

    @Test
    void interleavedRopeExtrapolatesBetterThanAllRope() {
        List<LengthGeneralisation.Row> rows = LengthGeneralisation.recall(List.of("all RoPE x2", "all NoPE x2",
                "iRoPE x2"), 250, 1);
        System.out.print(LengthGeneralisation.table("recall @ pairs", rows));
        System.out.printf("recall at 16 pairs (trained on 4): iRoPE %.3f vs all-RoPE %.3f vs all-NoPE %.3f%n",
                rows.get(2).accuracy()[2], rows.get(0).accuracy()[2], rows.get(1).accuracy()[2]);
        assertTrue(rows.get(2).accuracy()[0] > 0.9);
        assertTrue(rows.get(2).accuracy()[2] > rows.get(0).accuracy()[2] + 0.1);
    }
}
