package io.github.zzuegg.brainiac.experiments.explainability;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.explainability.InterpretableModels.RashomonReport;

import java.util.Arrays;

import org.junit.jupiter.api.Test;

class RashomonSetTest {

    @Test
    void theSetRevealsMoreMultiplicityThanRetraining() {
        RashomonReport r = InterpretableModels.rashomon(600, 0.03, 1);
        System.out.printf("Rashomon depth 2: enumerated %d (%.1f ms) brute force %d (%.1f ms) identical %s; depth 3 "
                + "set %d%n", r.setSize(), r.enumerateMillis(), r.bruteSize(), r.bruteMillis(), r.identical(),
                r.deepSize());
        System.out.printf("Rashomon ambiguity %.3f discrepancy %.3f vs bootstrap retraining ambiguity %.3f "
                + "discrepancy %.3f; members using a copy %.2f%n", r.ambiguity(), r.discrepancy(), r.retrainAmbiguity(),
                r.retrainDiscrepancy(), r.copyShare());
        System.out.printf("Rashomon reliance min %s max %s%n", fmt(r.minReliance()), fmt(r.maxReliance()));
        assertTrue(r.identical(), "the shared-list enumeration equals brute force");
        assertTrue(r.ambiguity() >= r.retrainAmbiguity(), "the set shows at least the retraining multiplicity");
        assertTrue(r.copyShare() > 0, "some near-optimal trees use the redundant copies");
    }

    private static String fmt(double[] v) {
        return Arrays.toString(Arrays.stream(v).mapToObj(d -> String.format("%.3f", d)).toArray());
    }
}
