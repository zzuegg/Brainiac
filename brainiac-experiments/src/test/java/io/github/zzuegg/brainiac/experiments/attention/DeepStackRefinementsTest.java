package io.github.zzuegg.brainiac.experiments.attention;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class DeepStackRefinementsTest {

    @Test
    void valueResidualHelpsADeepStack() {
        List<DeepStackRefinements.Row> rows = DeepStackRefinements.run(5, 24, 120, 1, true);
        System.out.print(DeepStackRefinements.table(rows));
        System.out.printf("depth 5: value residual val loss %.4f vs plain %.4f vs core TransformerBlock %.4f%n",
                rows.get(0).validationLoss(), rows.get(1).validationLoss(), rows.get(2).validationLoss());
        assertTrue(rows.get(0).validationLoss() < rows.get(1).validationLoss() + 0.02);
        assertTrue(rows.get(0).validationLoss() < rows.get(2).validationLoss() + 0.02);
        assertTrue(rows.get(0).validationLoss() < Math.log(ArchitectureBench.VOCAB));
    }
}
