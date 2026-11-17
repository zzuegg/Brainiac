package io.github.zzuegg.brainiac.experiments.language;

import io.github.zzuegg.brainiac.nn.Model;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

class ElectraVsMaskedLmTest {

    @Test
    void replacedTokenDetectionAtEqualCompute() {
        int steps = 250;
        Model[] electra = ElectraVsMaskedLm.electra(steps, 1);
        int mlmSteps = ElectraVsMaskedLm.matchedSteps(steps, electra);
        Model mlm = ElectraVsMaskedLm.maskedLm(mlmSteps, 1);
        double e = ElectraVsMaskedLm.probe(electra[0], 5);
        double m = ElectraVsMaskedLm.probe(mlm, 5);
        double untrained = ElectraVsMaskedLm.probe(ElectraVsMaskedLm.encoder(32, 1, 1), 5);
        System.out.printf("stride probe on frozen encoders: ELECTRA %.3f (%d steps), masked LM %.3f (%d steps, equal "
                + "compute), untrained %.3f (chance 0.200)%n", e, steps, m, mlmSteps, untrained);
        assertTrue(e > untrained + 0.1, "ELECTRA pre-training helps the probe");
        assertTrue(e > m - 0.05, "ELECTRA keeps up with masked LM at equal compute");
    }
}
