package io.github.zzuegg.brainiac.experiments.labels;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.labels.SemiSupervisedShootout.Method;
import io.github.zzuegg.brainiac.experiments.labels.SemiSupervisedShootout.Result;

import java.util.EnumMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class ThresholdedSemiSupervisedTest {

    @Test
    void fixMatchFamilyOnDigits() {
        Map<Method, Result> rows = new EnumMap<>(Method.class);
        for (Method m : new Method[] {Method.SUPERVISED_AUG, Method.FIXMATCH, Method.FIXMATCH_07, Method.FLEXMATCH,
                Method.FREEMATCH, Method.SOFTMATCH}) {
            Result r = SemiSupervisedShootout.digits(m, new SemiSupervisedShootout.Setting(4, 1, 400), 1);
            rows.put(m, r);
            System.out.printf("SemiSup thresholds %-14s acc %.3f worst %.3f half %.3f kept %.2f precision %.2f  %5.0f ms%n",
                    m, r.accuracy(), r.worstClass(), r.halfway(), r.kept(), r.precision(), r.millis());
        }
        double sup = rows.get(Method.SUPERVISED_AUG).accuracy();
        assertTrue(rows.get(Method.FIXMATCH).accuracy() > sup, "FixMatch beats supervised-only");
        assertTrue(rows.get(Method.FLEXMATCH).kept() > rows.get(Method.FIXMATCH).kept(),
                "FlexMatch uses more unlabelled rows than FixMatch at 0.95");
        for (Result r : rows.values()) {
            assertTrue(r.accuracy() > 0.2, "every method trains: " + r);
        }
    }
}
