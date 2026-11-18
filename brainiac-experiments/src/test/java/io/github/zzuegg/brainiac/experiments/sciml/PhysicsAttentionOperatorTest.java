package io.github.zzuegg.brainiac.experiments.sciml;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.sciml.PhysicsAttention;

import java.util.List;

import org.junit.jupiter.api.Test;

class PhysicsAttentionOperatorTest {

    @Test
    void physicsAttentionLearnsBurgersOnRegularAndIrregularPoints() {
        OperatorLearning.Data data = OperatorLearning.burgers(120, 64, 2);
        List<OperatorLearning.Row> rows = OperatorLearning.physicsAttention(data, 100, 32, 15, 1);
        double[] t = OperatorLearning.attentionTime(PhysicsAttention.Mode.TRANSOLVER, new int[] {64, 256});
        StringBuilder line = new StringBuilder("8.3.5 Burgers, rel L2 regular/irregular (train ms):");
        rows.forEach(r -> line.append(String.format(" %s %.3f/%.3f (%.0f);", r.model(), r.onGrid(), r.offGrid(),
                r.millis())));
        line.append(String.format(" Transolver us/sample N=64 %.0f, N=256 %.0f", t[0], t[1]));
        System.out.println(line);
        for (OperatorLearning.Row r : rows) {
            assertTrue(r.onGrid() < 1.2, line.toString());
        }
    }
}
