package io.github.zzuegg.brainiac.experiments.labels;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.labels.WeakLabelShootout.PuMethod;
import io.github.zzuegg.brainiac.experiments.labels.WeakLabelShootout.PuResult;

import java.util.EnumMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class PositiveUnlabelledTest {

    @Test
    void nnPuFixesUpuOnEvenAgainstOddDigits() {
        Map<PuMethod, PuResult> rows = new EnumMap<>(PuMethod.class);
        for (PuMethod m : PuMethod.values()) {
            PuResult r = WeakLabelShootout.pu(m, new int[] {0, 2, 4, 6, 8}, 600, 1);
            rows.put(m, r);
            System.out.printf("PU even/odd %-22s acc %.3f auc %.3f risk %7.3f prior %.2f%n", m, r.accuracy(), r.auc(),
                    r.finalRisk(), r.prior());
        }
        assertTrue(rows.get(PuMethod.PN_ORACLE).accuracy() > 0.8, "oracle");
        assertTrue(rows.get(PuMethod.UPU).finalRisk() < 0, "uPU's risk estimate goes negative");
        assertTrue(rows.get(PuMethod.NNPU).accuracy() > rows.get(PuMethod.UPU).accuracy(), "nnPU beats uPU");
        assertTrue(rows.get(PuMethod.NNPU).accuracy() > 0.6, "nnPU learns");
        assertTrue(Math.abs(rows.get(PuMethod.NNPU_ESTIMATED_PRIOR).prior() - rows.get(PuMethod.NNPU).prior()) < 0.3,
                "the Elkan-Noto prior is in range");
        assertTrue(rows.get(PuMethod.TWO_HNC).accuracy() > 0.4, "2-HNC runs");
    }
}
