package io.github.zzuegg.brainiac.experiments.labels;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.labels.SemiSupervisedShootout.Method;
import io.github.zzuegg.brainiac.experiments.labels.SemiSupervisedShootout.Result;

import org.junit.jupiter.api.Test;

class JepaMatchShootoutTest {

    @Test
    void sigregAndVicregOnTopOfFlexMatch() {
        for (Method m : new Method[] {Method.JEPAMATCH, Method.FLEX_VICREG}) {
            Result r = SemiSupervisedShootout.digits(m, new SemiSupervisedShootout.Setting(4, 1, 400), 1);
            System.out.printf("SemiSup thresholds %-14s acc %.3f worst %.3f half %.3f kept %.2f precision %.2f  %5.0f ms%n",
                    m, r.accuracy(), r.worstClass(), r.halfway(), r.kept(), r.precision(), r.millis());
            assertTrue(r.accuracy() > 0.2, "trains: " + r);
        }
    }
}
