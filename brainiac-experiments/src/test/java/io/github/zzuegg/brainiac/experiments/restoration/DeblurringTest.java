package io.github.zzuegg.brainiac.experiments.restoration;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.restoration.RestorationShootout.DeblurMethod;
import io.github.zzuegg.brainiac.experiments.restoration.RestorationShootout.DenoiseMethod;
import io.github.zzuegg.brainiac.experiments.restoration.RestorationShootout.NoiseKind;
import io.github.zzuegg.brainiac.experiments.restoration.RestorationShootout.Row;
import io.github.zzuegg.brainiac.experiments.restoration.RestorationShootout.SelfMethod;
import io.github.zzuegg.brainiac.experiments.restoration.RestorationShootout.Source;
import io.github.zzuegg.brainiac.experiments.restoration.RestorationShootout.SrMethod;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

class DeblurringTest {

    @Test
    void plugAndPlayDeblurring() {
        Map<String, List<Row>> r = RestorationShootout.deblurring(16, 0.01f, 200, 6, 1, DeblurMethod.BLURRED,
                DeblurMethod.WIENER, DeblurMethod.PNP_HQS);
        System.out.printf("deblurring 1%% noise: %s%n", r);
        for (List<Row> rows : r.values()) {
            assertTrue(rows.get(2).psnr() > rows.get(0).psnr(), "PnP-HQS improves on the blurred input: " + rows);
        }
    }
}
