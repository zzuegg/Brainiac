package io.github.zzuegg.brainiac.experiments.language;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class SliceCompressionTest {

    @Test
    void slicingIsExactAtFullWidthAndUsableWhenCut() {
        List<SliceCompression.Row> rows = SliceCompression.run(200, new int[] {32, 24, 16}, 40, 1);
        StringBuilder line = new StringBuilder("SliceGPT vs Wanda vs block removal, val loss (kept):");
        for (SliceCompression.Row r : rows) {
            line.append(String.format(" %s %.3f (%.2f);", r.method(), r.validationLoss(), r.kept()));
        }
        System.out.println(line);
        double dense = rows.get(0).validationLoss();
        assertEquals(dense, rows.get(1).validationLoss(), 1e-3, "rotating without slicing is exact");
        for (int i = 6; i < rows.size(); i += 5) {
            SliceCompression.Row slice = rows.get(i);
            SliceCompression.Row wanda = rows.get(i + 1);
            SliceCompression.Row recovered = rows.get(i + 3);
            assertTrue(slice.validationLoss() < Math.log(32), "sliced model better than uniform: " + slice);
            assertTrue(recovered.validationLoss() < slice.validationLoss(), "recovery fine-tuning helps");
            assertTrue(wanda.validationLoss() < dense + 0.3, "Wanda keeps the model close: " + wanda);
        }
    }
}
