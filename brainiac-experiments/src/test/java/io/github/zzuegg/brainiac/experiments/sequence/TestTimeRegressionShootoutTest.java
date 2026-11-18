package io.github.zzuegg.brainiac.experiments.sequence;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Wave 6, 6.9: MesaNet and LaCT against the delta-rule family on associative recall. */
class TestTimeRegressionShootoutTest {

    @Test
    void optimalRegressionRecallsAsWellAsDeltaRules() {
        MixerShootout.Task task = MixerShootout.recall(8, 8, 4);
        List<String> names = new ArrayList<>(TestTimeRegressionShootout.lineup().keySet());
        List<MixerShootout.Result> results = DeltaFamilyShootout.run(TestTimeRegressionShootout.lineup(), names,
                List.of(task), 16, 300, 1);
        StringBuilder sb = new StringBuilder();
        for (MixerShootout.Result r : results) {
            sb.append(String.format(" %s %.2f (%d ms);", r.mixer(), r.accuracy(), r.millis()));
        }
        System.out.printf("recall(8,8,4) acc (chance 0.125):%s%n", sb);
        double mesa = DeltaFamilyShootout.accuracy(results, "MesaNet", task.name());
        double lact = DeltaFamilyShootout.accuracy(results, "LaCT", task.name());
        double gdn = DeltaFamilyShootout.accuracy(results, "Gated DeltaNet", task.name());
        assertTrue(mesa > 0.5, "MesaNet recalls: " + mesa);
        assertTrue(mesa >= gdn - 0.05, "MesaNet at least matches Gated DeltaNet");
        assertTrue(lact > 0.3, "LaCT recalls: " + lact);
    }
}
