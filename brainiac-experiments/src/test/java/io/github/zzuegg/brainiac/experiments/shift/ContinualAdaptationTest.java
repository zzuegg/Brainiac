package io.github.zzuegg.brainiac.experiments.shift;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.ood.FeatureClassifier;

import java.util.EnumMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class ContinualAdaptationTest {

    @Test
    void robustTestTimeAdaptationOnACorruptionStream() {
        FeatureClassifier src = DomainAdaptation.streamSource(800, 1);
        Map<DomainAdaptation.StreamMethod, DomainAdaptation.StreamResult> iid = new EnumMap<>(
                DomainAdaptation.StreamMethod.class);
        for (DomainAdaptation.StreamMethod m : DomainAdaptation.StreamMethod.values()) {
            DomainAdaptation.StreamResult r = DomainAdaptation.continual(src, m, 3, 2, 32, false, 1);
            DomainAdaptation.StreamResult one = DomainAdaptation.continual(src, m, 1, 2, 32, true, 2);
            iid.put(m, r);
            System.out.printf("Continual TTA %-8s acc %.3f  first round %.3f  last round %.3f  resets %d  adapted %.2f  | "
                    + "single-class batches %.3f%n", m, r.accuracy(), r.firstRound(), r.lastRound(), r.resets(),
                    r.adaptedShare(), one.accuracy());
        }
        assertTrue(iid.get(DomainAdaptation.StreamMethod.BN_STATS).accuracy()
                > iid.get(DomainAdaptation.StreamMethod.NONE).accuracy(), "BN statistics help on corruptions");
        assertTrue(iid.get(DomainAdaptation.StreamMethod.EATA).accuracy() > 0.3, "EATA stays above chance");
    }
}
