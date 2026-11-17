package io.github.zzuegg.brainiac.experiments.attention;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class AugmentedResidualDepthTest {

    @Test
    void learnedResidualsMatchOrBeatThePlainResidualAtDepth() {
        List<AugmentedResidualDepth.Row> rows = AugmentedResidualDepth.run(List.of("plain residual", "LAuReL-RW+LR",
                "LAuReL-PA", "value residual"), 6, 16, 150, 1);
        System.out.print(AugmentedResidualDepth.table(rows));
        double plain = rows.get(0).validationLoss();
        double laurel = rows.get(1).validationLoss();
        double pa = rows.get(2).validationLoss();
        double vr = rows.get(3).validationLoss();
        System.out.printf("depth 6 val loss: LAuReL-RW+LR %.4f (+%d params), LAuReL-PA %.4f vs plain %.4f, value "
                + "residual %.4f%n", laurel, rows.get(1).parameters() - rows.get(0).parameters(), pa, plain, vr);
        assertTrue(laurel < plain + 0.02);
        assertTrue(laurel < Math.log(32));
        assertTrue(rows.get(1).parameters() - rows.get(0).parameters() < rows.get(0).parameters() / 8);
    }
}
