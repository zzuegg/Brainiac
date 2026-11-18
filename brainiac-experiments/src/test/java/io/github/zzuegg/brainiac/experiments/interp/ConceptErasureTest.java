package io.github.zzuegg.brainiac.experiments.interp;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class ConceptErasureTest {

    @Test
    void leaceRemovesTheColourForLinearProbes() {
        List<ConceptErasure.Scores> rows = ConceptErasure.run(1);
        for (ConceptErasure.Scores s : rows) {
            System.out.printf("Erasure %-45s linear probe %.3f MLP probe %.3f task %.3f worst group %.3f change %.3f%n",
                    s.method(), s.linearProbe(), s.mlpProbe(), s.taskAccuracy(), s.worstGroup(), s.change());
        }
        ConceptErasure.Scores none = rows.get(0);
        ConceptErasure.Scores leace = rows.get(1);
        assertTrue(none.linearProbe() > 0.75, "the colour is linearly decodable before erasure");
        assertTrue(leace.linearProbe() < none.linearProbe() - 0.15, "LEACE lowers the linear probe");
    }
}
