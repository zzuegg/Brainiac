package io.github.zzuegg.brainiac.experiments.labels;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.labels.SemiSupervisedShootout.Method;
import io.github.zzuegg.brainiac.experiments.labels.SemiSupervisedShootout.Result;

import org.junit.jupiter.api.Test;

class ImbalancedPoolSemiSupervisedTest {

    @Test
    void fixedAgainstAdaptiveThresholdOnAnImbalancedPool() {
        for (Method m : new Method[] {Method.FIXMATCH, Method.FLEXMATCH}) {
            Result r = SemiSupervisedShootout.digits(m, new SemiSupervisedShootout.Setting(4, 10, 400), 1);
            System.out.printf("SemiSup imbalanced pool (ratio 10) %-10s acc %.3f worst %.3f kept %.2f precision %.2f%n",
                    m, r.accuracy(), r.worstClass(), r.kept(), r.precision());
            assertTrue(r.accuracy() > 0.2, r.toString());
        }
    }
}
