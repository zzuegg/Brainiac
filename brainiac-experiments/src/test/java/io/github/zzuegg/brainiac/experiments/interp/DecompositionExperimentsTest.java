package io.github.zzuegg.brainiac.experiments.interp;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class DecompositionExperimentsTest {

    @Test
    void crosscoderFindsTheFineTuneOnlyFeature() {
        ModelDiffing.Outcome o = ModelDiffing.run(1200, 1);
        System.out.printf("Crosscoder diffing: planted %.2f  crosscoder exclusive %d (best trigger AUROC %.3f, "
                + "precision %.2f)  vs separate SAEs new %d (best AUROC %.3f, precision %.2f)%n", o.planted(),
                o.crosscoderExclusive(), o.crosscoderBestAuroc(), o.crosscoderPrecision(), o.saeNew(),
                o.saeBestAuroc(), o.saePrecision());
        assertTrue(o.planted() > 0.9, o.toString());
        assertTrue(o.crosscoderBestAuroc() > 0.75, o.toString());
    }

    @Test
    void parameterDecompositionStaysFaithfulAndSparse() {
        MechanismDecomposition.Outcome o = MechanismDecomposition.run(6000, 1);
        System.out.printf("SPD on TMS: recovered %d/%d mechanisms, %.2f active components per input, "
                + "faithfulness error %.3f  vs SAE on activations: %d/%d directions (TMS loss %.4f)%n",
                o.spdRecovered(), o.mechanisms(), o.spdActivePerInput(), o.faithfulness(), o.saeRecovered(),
                o.mechanisms(), o.tmsLoss());
        assertTrue(o.faithfulness() < 0.05, o.toString());
        assertTrue(o.spdActivePerInput() < 3, o.toString());
    }
}
