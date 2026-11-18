package io.github.zzuegg.brainiac.experiments.sciml;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class DeepOnetOperatorTest {

    @Test
    void deepOnetQueriesOffGridWhileFnoLeadsOnGrid() {
        StringBuilder line = new StringBuilder("8.3.4 operators, rel L2 on-grid/off-grid:");
        double[] fnoOn = new double[3];
        double[] onetOff = new double[3];
        OperatorLearning.Data[] data = {OperatorLearning.antiderivative(110, 64, 1),
            OperatorLearning.burgers(110, 64, 2), OperatorLearning.darcy(110, 64, 3)};
        String[] names = {"antiderivative", "burgers", "darcy"};
        for (int i = 0; i < 3; i++) {
            List<OperatorLearning.Row> rows = OperatorLearning.deepOnet(data[i], 90, 25, 1);
            line.append(' ').append(names[i]).append(':');
            for (OperatorLearning.Row r : rows) {
                line.append(String.format(" %s %.3f/%.3f", r.model(), r.onGrid(), r.offGrid()));
            }
            line.append(';');
            fnoOn[i] = rows.get(1).onGrid();
            onetOff[i] = rows.get(0).offGrid();
        }
        System.out.println(line);
        for (int i = 0; i < 3; i++) {
            assertTrue(fnoOn[i] < 0.6, line.toString());
            assertTrue(onetOff[i] < 1.0, line.toString());
        }
    }
}
