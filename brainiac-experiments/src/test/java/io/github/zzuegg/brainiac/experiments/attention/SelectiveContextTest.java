package io.github.zzuegg.brainiac.experiments.attention;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

class SelectiveContextTest {

    @Test
    void selectivePruningKeepsAccuracyWhereH2oLosesIt() {
        SelectiveContext.Result r = SelectiveContext.run(1, 300, 2f, 1);
        System.out.printf("recent recall, %.2f of the context kept: selective pruned %.3f (unpruned %.3f) vs "
                + "softmax with H2O %.3f (full %.3f)%n", r.keptShare(), r.selectivePruned(), r.selectiveFull(),
                r.h2o(), r.full());
        assertTrue(r.keptShare() < 0.85);
        assertTrue(r.selectiveFull() > r.full() + 0.1, "selective attention solves recent recall better");
        assertTrue(r.selectivePruned() > r.selectiveFull() - 0.03, "pruning by the penalty is nearly free");
        assertTrue(r.selectivePruned() > r.h2o() + 0.1);
    }
}
