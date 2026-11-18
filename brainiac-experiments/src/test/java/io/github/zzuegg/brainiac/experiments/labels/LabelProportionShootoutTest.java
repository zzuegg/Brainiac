package io.github.zzuegg.brainiac.experiments.labels;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.labels.WeakLabelShootout.ProportionMethod;

import java.util.EnumMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class LabelProportionShootoutTest {

    @Test
    void dllpAgainstBagSize() {
        for (int size : new int[] {8, 128}) {
            Map<ProportionMethod, Double> rows = new EnumMap<>(ProportionMethod.class);
            for (ProportionMethod m : ProportionMethod.values()) {
                double acc = WeakLabelShootout.proportions(m, size, false, 400, 1);
                rows.put(m, acc);
                System.out.printf("LLP bag %3d random %-12s acc %.3f%n", size, m, acc);
            }
            if (size == 8) {
                assertTrue(rows.get(ProportionMethod.DLLP) > rows.get(ProportionMethod.BAG_MAJORITY),
                        "DLLP beats bag-majority labels");
                assertTrue(rows.get(ProportionMethod.DLLP) > 0.5, "DLLP learns at bag size 8");
            }
        }
    }
}
