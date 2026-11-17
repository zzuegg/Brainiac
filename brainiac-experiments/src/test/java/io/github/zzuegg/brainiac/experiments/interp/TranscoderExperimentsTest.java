package io.github.zzuegg.brainiac.experiments.interp;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class TranscoderExperimentsTest {

    @Test
    void transcodersExplainMlpOutputs() {
        List<TranscoderComparison.Point> points = TranscoderComparison.run(new int[] {1, 2, 4}, 800, 1);
        for (TranscoderComparison.Point p : points) {
            System.out.printf("Transcoder %-13s k=%d  EV: SAE-on-output %.3f  transcoder %.3f  skip-transcoder %.3f%n",
                    p.mlp(), p.k(), p.saeOnOutput(), p.transcoder(), p.skipTranscoder());
        }
        TranscoderComparison.Point k4 = points.get(2);
        TranscoderComparison.Point lm = points.get(3);
        assertTrue(k4.transcoder() > 0.7, k4.toString());
        assertTrue(k4.skipTranscoder() >= k4.transcoder() - 0.02, k4.toString());
        assertTrue(lm.skipTranscoder() > 0.7, lm.toString());
        assertTrue(points.get(0).transcoder() < k4.transcoder(), "sparser should explain less");
    }

    @Test
    void attributionGraphFindsThePlantedIntermediate() {
        List<CircuitTracing.Outcome> out = CircuitTracing.run(1500, 2);
        for (CircuitTracing.Outcome o : out) {
            System.out.printf("Circuit %-11s EV %.3f  replacement agreement %.3f  B-feature recovery %.2f  "
                    + "pruned nodes %.1f  completeness gap %.1e%n", o.transcoder(), o.explainedVariance(),
                    o.replacementAgreement(), o.bFeatureRecovery(), o.meanPrunedNodes(), o.completeness());
        }
        CircuitTracing.Outcome clt = out.get(0);
        assertTrue(clt.completeness() < 1e-3, clt.toString());
        assertTrue(clt.replacementAgreement() > 0.8, clt.toString());
        assertTrue(clt.bFeatureRecovery() >= 0.4, clt.toString());
    }
}
