package io.github.zzuegg.brainiac.experiments.sciml;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class OperatorSweepTest {

    @Test
    void sweepsWidthDataAndGridRegularity() {
        OperatorLearning.Data data = OperatorLearning.burgers(296, 64, 3);
        List<OperatorLearning.SweepRow> rows = OperatorLearning.sweep(data, 40, 32, new int[] {16, 48},
                new int[] {64, 256}, new double[] {0, 0.5}, 40, 1);
        StringBuilder line = new StringBuilder("9.5.16 Burgers sweep, rel L2 [width/train/drop]:");
        String cell = "";
        for (OperatorLearning.SweepRow r : rows) {
            String c = r.width() + "/" + r.train() + "/" + r.drop();
            if (!c.equals(cell)) {
                line.append(" [").append(c).append("]");
                cell = c;
            }
            line.append(String.format(" %s %.3f", r.model(), r.error()));
            assertTrue(Double.isFinite(r.error()) && r.error() < 2, r.toString());
        }
        System.out.println(line);
    }
}
