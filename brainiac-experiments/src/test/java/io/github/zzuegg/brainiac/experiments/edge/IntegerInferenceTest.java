package io.github.zzuegg.brainiac.experiments.edge;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class IntegerInferenceTest {

    @Test
    void integerEngineIsBitExactAndCloseToFloat() {
        List<EdgeDeployment.IntegerRow> rows = EdgeDeployment.integer(150, 1);
        rows.addAll(EdgeDeployment.integerKeywords(70, 1));
        for (EdgeDeployment.IntegerRow r : rows) {
            System.out.printf("9.6.13 %-16s %-56s acc %.3f, agree %.3f, bit-exact %.2f, %7d B, %9d MACs, %8.1f us%n",
                    r.model(), r.method(), r.accuracy(), r.agreement(), r.bitExact(), r.bytes(), r.macs(),
                    r.microsPerSample());
        }
        for (EdgeDeployment.IntegerRow r : rows) {
            if (r.method().contains("integer engine")) {
                assertEquals(1.0, r.bitExact(), 1e-12, r.toString());
            }
        }
        for (int m = 0; m < rows.size(); m += 5) {
            EdgeDeployment.IntegerRow fl = rows.get(m);
            EdgeDeployment.IntegerRow qat = rows.get(m + 4);
            assertTrue(qat.accuracy() > fl.accuracy() - 0.05, qat + " vs " + fl);
        }
        EdgeDeployment.IntegerRow fl = rows.get(0);
        EdgeDeployment.IntegerRow qat = rows.get(4);
        assertTrue(qat.bytes() * 3 < fl.bytes(), "int8 is about 4x smaller");
    }
}
