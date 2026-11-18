package io.github.zzuegg.brainiac.experiments.labels;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.labels.WeakLabelShootout.MilMethod;
import io.github.zzuegg.brainiac.experiments.labels.WeakLabelShootout.MilResult;

import java.util.EnumMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class AttentionMilShootoutTest {

    @Test
    void gatedAttentionLocalisesTheKeyInstance() {
        Map<MilMethod, MilResult> rows = new EnumMap<>(MilMethod.class);
        for (MilMethod m : new MilMethod[] {MilMethod.MEAN, MilMethod.MAX, MilMethod.ATTENTION, MilMethod.GATED}) {
            MilResult r = WeakLabelShootout.mil(m, 200, 6, 1);
            rows.put(m, r);
            System.out.printf("MIL 200 bags %-16s bag AUC %.3f instance AUC %.3f%n", m, r.bagAuc(), r.instanceAuc());
        }
        assertTrue(rows.get(MilMethod.GATED).bagAuc() > 0.8, "gated attention MIL");
        assertTrue(rows.get(MilMethod.GATED).instanceAuc() > 0.8, "attention localises the 9s");
        assertTrue(rows.get(MilMethod.GATED).bagAuc() > rows.get(MilMethod.MEAN).bagAuc(), "gated beats mean pooling");
    }
}
