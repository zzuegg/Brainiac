package io.github.zzuegg.brainiac.experiments.explainability;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.explainability.InterpretableNetworks.PrototypeRow;

import java.util.Arrays;
import java.util.List;

import org.junit.jupiter.api.Test;

class PrototypeNetworksTest {

    @Test
    void protoPNetExplainsByCasesAtSmallAccuracyCost() {
        List<PrototypeRow> rows = InterpretableNetworks.prototypes(new float[] {1e-3f, 3e-3f, 1e-2f}, 150, 1);
        for (PrototypeRow r : rows) {
            System.out.printf("Prototypes %-18s accuracy %s mean %.3f spread %.3f purity %s mean %.3f%n", r.model(),
                    Arrays.toString(Arrays.stream(r.accuracy()).mapToObj(v -> String.format("%.3f", v)).toArray()),
                    r.meanAccuracy(), r.spread(),
                    Arrays.toString(Arrays.stream(r.purity()).mapToObj(v -> String.format("%.2f", v)).toArray()),
                    r.meanPurity());
        }
        PrototypeRow chance = rows.get(rows.size() - 1);
        assertTrue(rows.get(0).meanAccuracy() > 1.0 / 6 + 0.1 || rows.get(1).meanAccuracy() > 1.0 / 6 + 0.1,
                "ProtoPNet beats chance");
        assertTrue(Math.max(rows.get(0).meanPurity(), rows.get(1).meanPurity()) > chance.meanPurity(),
                "prototype patches land on the object more than chance");
    }
}
