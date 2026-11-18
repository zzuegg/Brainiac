package io.github.zzuegg.brainiac.experiments.inference;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class AbcSummariesTest {

    @Test
    void summariesOnTheRickerModel() {
        Map<String, double[]> m = LikelihoodFree.summaries(3000, 10, 6);
        System.out.printf("ABC summaries, Ricker raw counts, n=3000 (13.5.5) {RMSE/width, sd/width, MI}: %s%n",
                NeuralLikelihoodTest.line(m));
        assertTrue(m.get("ABC hand")[0] < m.get("prior")[0], "hand-summary ABC beats the prior");
        assertTrue(m.get("ABC+adjust semi-auto MLP(raw)")[0] < m.get("prior")[0], "learned summaries beat the prior");
    }
}
