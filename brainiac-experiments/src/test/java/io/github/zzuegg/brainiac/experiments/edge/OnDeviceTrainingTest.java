package io.github.zzuegg.brainiac.experiments.edge;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class OnDeviceTrainingTest {

    @Test
    void sparseUpdateWithQasAdaptsUnderBudget() {
        List<EdgeDeployment.AdaptRow> rows = EdgeDeployment.adaptation(150, 60, 0.3, 4);
        for (EdgeDeployment.AdaptRow r : rows) {
            System.out.printf("9.6.16 %-52s %7d B, accuracy %.3f  [%s]%n", r.method(), r.trainingBytes(),
                    r.accuracy(), r.detail());
        }
        double none = rows.get(0).accuracy();
        double qas = rows.stream().filter(r -> r.method().startsWith("sparse update (")).findFirst().orElseThrow()
                .accuracy();
        assertTrue(qas > none + 0.1, "adaptation learns the new glyphs: " + qas + " vs " + none);
    }
}
