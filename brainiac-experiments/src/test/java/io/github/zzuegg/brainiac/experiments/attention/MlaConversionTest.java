package io.github.zzuegg.brainiac.experiments.attention;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class MlaConversionTest {

    @Test
    void convertedGqaRecoversWithAShortFineTune() {
        List<MlaConversion.Row> rows = MlaConversion.run(200, 40, new int[] {12, 8}, 1);
        StringBuilder line = new StringBuilder("GQA->MLA val loss (cached/token):");
        for (MlaConversion.Row r : rows) {
            line.append(String.format(" %s %.3f (%d);", r.name(), r.validationLoss(), r.cachedPerToken()));
        }
        System.out.println(line);
        double gqa = rows.get(0).validationLoss();
        for (int i = 2; i < rows.size(); i += 3) {
            MlaConversion.Row converted = rows.get(i);
            MlaConversion.Row tuned = rows.get(i + 1);
            MlaConversion.Row scratch = rows.get(i + 2);
            assertTrue(tuned.validationLoss() <= converted.validationLoss(), "fine-tuning helps " + tuned.name());
            assertTrue(tuned.validationLoss() < scratch.validationLoss() + 0.2, "close to from-scratch MLA");
        }
        assertTrue(rows.get(3).validationLoss() < gqa + 0.15, "same-cache conversion recovers GQA");
    }
}
