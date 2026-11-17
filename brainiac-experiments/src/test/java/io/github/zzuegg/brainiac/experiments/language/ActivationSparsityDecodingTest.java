package io.github.zzuegg.brainiac.experiments.language;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class ActivationSparsityDecodingTest {

    @Test
    void fortyPercentSparsityKeepsTheLossAndSkipsWeightRows() {
        List<ActivationSparsityDecoding.Row> rows = ActivationSparsityDecoding.run(new float[] {0f, 0.25f, 0.4f, 0.5f},
                300, 3000, 1);
        System.out.print(ActivationSparsityDecoding.table(rows));
        ActivationSparsityDecoding.Row dense = rows.get(0);
        ActivationSparsityDecoding.Row forty = rows.get(2);
        ActivationSparsityDecoding.Row fifty = rows.get(3);
        System.out.printf("TEAL 40%%: val loss %.4f (+%.4f), density %.3f, %.2fx decode speed; 50%%: %.4f, %.2fx; "
                + "dense %.4f at %.0f tokens/s%n", forty.validationLoss(), forty.validationLoss() - dense.validationLoss(),
                forty.density(), forty.speedup(), fifty.validationLoss(), fifty.speedup(), dense.validationLoss(),
                dense.tokensPerSecond());
        assertEquals(1.0, dense.density(), 1e-9);
        assertTrue(forty.density() < 0.75);
        assertTrue(forty.validationLoss() < dense.validationLoss() + 0.1);
        assertTrue(rows.get(1).validationLoss() < dense.validationLoss() + 0.05);
    }
}
