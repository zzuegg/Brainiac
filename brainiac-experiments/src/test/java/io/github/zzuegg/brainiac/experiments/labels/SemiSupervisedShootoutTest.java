package io.github.zzuegg.brainiac.experiments.labels;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.labels.SemiSupervisedShootout.Method;
import io.github.zzuegg.brainiac.experiments.labels.SemiSupervisedShootout.Result;

import java.util.EnumMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class SemiSupervisedShootoutTest {

    @Test
    void classicConsistencyMethodsOnDigitsWithFourLabelsPerClass() {
        Map<Method, Result> rows = new EnumMap<>(Method.class);
        for (Method m : new Method[] {Method.SUPERVISED, Method.SUPERVISED_AUG, Method.ORACLE, Method.PSEUDO_LABEL,
                Method.PI_MODEL, Method.MEAN_TEACHER, Method.VAT, Method.MIXMATCH}) {
            Result r = SemiSupervisedShootout.digits(m, new SemiSupervisedShootout.Setting(4, 1, 400), 1);
            rows.put(m, r);
            System.out.printf("SemiSup 4 labels/class %-14s acc %.3f worst %.3f half %.3f  %5.0f ms%n", m,
                    r.accuracy(), r.worstClass(), r.halfway(), r.millis());
        }
        double sup = rows.get(Method.SUPERVISED_AUG).accuracy();
        assertTrue(rows.get(Method.ORACLE).accuracy() > sup + 0.1, "the oracle is well above supervised-only");
        assertTrue(rows.get(Method.SUPERVISED).accuracy() > 0.2, "supervised-only is above chance");
        assertTrue(rows.get(Method.MEAN_TEACHER).accuracy() > sup - 0.05, "Mean Teacher " + rows.get(Method.MEAN_TEACHER));
        assertTrue(rows.get(Method.PI_MODEL).accuracy() > sup - 0.05, "Pi-model " + rows.get(Method.PI_MODEL));
        for (Result r : rows.values()) {
            assertTrue(r.accuracy() > 0.15, "every method trains: " + r);
        }
    }
}
