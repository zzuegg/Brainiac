package io.github.zzuegg.brainiac.experiments.attention;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class Gemma3nEfficiencyTest {

    @Test
    void altUpAndPerLayerEmbeddingsAddCapacityWithoutCompute() {
        List<Gemma3nBlocks.EfficiencyRow> rows = Gemma3nBlocks.efficiency(16, 200, 1);
        System.out.print(Gemma3nBlocks.efficiencyTable(rows));
        Gemma3nBlocks.EfficiencyRow plain = rows.get(0);
        Gemma3nBlocks.EfficiencyRow wide = rows.get(1);
        Gemma3nBlocks.EfficiencyRow altUp = rows.get(2);
        Gemma3nBlocks.EfficiencyRow ple = rows.get(3);
        System.out.printf("val loss / hot params: AltUp %.4f / %d, PLE %.4f / %d (+%d cold) vs plain d=16 %.4f / %d and "
                + "plain d=32 %.4f / %d%n", altUp.validationLoss(), altUp.hotParameters(), ple.validationLoss(),
                ple.hotParameters(), ple.coldParameters(), plain.validationLoss(), plain.hotParameters(),
                wide.validationLoss(), wide.hotParameters());
        assertTrue(altUp.hotParameters() < wide.hotParameters() / 2);
        assertTrue(altUp.validationLoss() < plain.validationLoss());
        assertTrue(ple.validationLoss() < plain.validationLoss() + 0.15);
        assertTrue(ple.hotParameters() < plain.hotParameters() * 1.2);
    }
}
