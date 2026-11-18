package io.github.zzuegg.brainiac.experiments.labels;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.labels.SemiSupervisedShootout.Method;
import io.github.zzuegg.brainiac.experiments.labels.SemiSupervisedShootout.Tabular;

import java.util.EnumMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class SemiSupervisedTabularTest {

    @Test
    void vatAndConsistencyWithoutImageAugmentations() {
        Map<Method, Double> spirals = new EnumMap<>(Method.class);
        for (Method m : new Method[] {Method.SUPERVISED, Method.PI_MODEL, Method.MEAN_TEACHER, Method.VAT,
                Method.ORACLE}) {
            double s = SemiSupervisedShootout.tabular(Tabular.SPIRALS, m, 400, 1);
            double t = SemiSupervisedShootout.tabular(Tabular.STRESS_SUITE, m, 400, 1);
            spirals.put(m, s);
            System.out.printf("SemiSup 20 labels/class %-13s spirals %.3f  stress suite %.3f%n", m, s, t);
        }
        assertTrue(spirals.get(Method.VAT) > spirals.get(Method.SUPERVISED), "VAT beats supervised on spirals");
        assertTrue(spirals.get(Method.ORACLE) > 0.95, "oracle");
    }
}
