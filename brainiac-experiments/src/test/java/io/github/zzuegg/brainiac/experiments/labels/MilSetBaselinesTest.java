package io.github.zzuegg.brainiac.experiments.labels;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.labels.WeakLabelShootout.MilMethod;
import io.github.zzuegg.brainiac.experiments.labels.WeakLabelShootout.MilResult;

import org.junit.jupiter.api.Test;

class MilSetBaselinesTest {

    @Test
    void setEncodersAndTheInstanceOracle() {
        for (MilMethod m : new MilMethod[] {MilMethod.DEEP_SETS, MilMethod.SET_TRANSFORMER, MilMethod.INSTANCE_ORACLE}) {
            MilResult r = WeakLabelShootout.mil(m, 200, 6, 1);
            System.out.printf("MIL 200 bags %-16s bag AUC %.3f instance AUC %.3f%n", m, r.bagAuc(), r.instanceAuc());
            assertTrue(r.bagAuc() > (m == MilMethod.INSTANCE_ORACLE ? 0.9 : 0.4), r.toString());
        }
    }
}
