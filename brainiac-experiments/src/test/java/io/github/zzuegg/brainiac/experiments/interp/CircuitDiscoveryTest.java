package io.github.zzuegg.brainiac.experiments.interp;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class CircuitDiscoveryTest {

    @Test
    void edgeMethodsRecoverThePlantedCircuit() {
        List<CircuitDiscovery.EdgeScore> rows = CircuitDiscovery.edges(1);
        for (CircuitDiscovery.EdgeScore s : rows) {
            System.out.printf("Circuit discovery %-26s AUC %.3f saturated-edge rank %d passes %d (%.1f ms)%n",
                    s.method(), s.auc(), s.saturatedRank(), s.forwardPasses(), s.millis());
        }
        double best = rows.stream().filter(s -> s.method().startsWith("ACDC") || s.method().startsWith("EAP-IG"))
                .filter(s -> !s.method().contains("faithfulness")).mapToDouble(CircuitDiscovery.EdgeScore::auc).max()
                .orElse(0);
        assertTrue(best > 0.75, "ACDC or EAP-IG recovers the planted edges");
    }

    @Test
    void distributedAlignmentSearchFindsTheCarry() {
        CircuitDiscovery.AlignmentScores s = CircuitDiscovery.alignment(600, 300, 1);
        System.out.printf("DAS carry: task %.3f IIA k=1 %.3f k=2 %.3f probe-direction IIA %.3f (probe acc %.3f)"
                + " PCA1 %.3f PCA2 %.3f best neuron %.3f%n", s.taskAccuracy(), s.dasK1(), s.dasK2(), s.probe(),
                s.probeAccuracy(), s.pca1(), s.pca2(), s.bestNeuron());
        assertTrue(s.taskAccuracy() > 0.6, "the adder learns");
        double das = Math.max(s.dasK1(), s.dasK2());
        double baseline = Math.max(Math.max(s.probe(), s.bestNeuron()), Math.max(s.pca1(), s.pca2()));
        // a no-effect intervention scores 0; DAS stays well short of a perfect interchange at this budget
        assertTrue(das > baseline, "the learned rotation beats the probe, PCA and best-neuron directions");
    }
}
