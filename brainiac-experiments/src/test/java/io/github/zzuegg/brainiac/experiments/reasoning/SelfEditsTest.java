package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class SelfEditsTest {

    @Test
    void learnedSelfEditsInjectFactsTheRawPassageCannot() {
        long t0 = System.nanoTime();
        SelfEdits.Result r = SelfEdits.run(25, 1);
        System.out.printf("SEAL recall after update: self-edit %.3f vs raw passage %.3f (untrained editor %.3f, ideal rewrite %.3f), %.1f s%n",
                r.selfEdit(), r.rawPassage(), r.untrainedEdit(), r.oracleEdit(), (System.nanoTime() - t0) / 1e9);
        assertTrue(r.selfEdit() > r.rawPassage(), r.toString());
    }
}
