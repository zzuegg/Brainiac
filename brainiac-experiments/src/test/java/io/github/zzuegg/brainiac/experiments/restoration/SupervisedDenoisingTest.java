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

class SupervisedDenoisingTest {

    @Test
    void residualDenoisersBeatFilters() {
        Map<String, List<Row>> r = RestorationShootout.denoising(16, 120, 6, 1, DenoiseMethod.GAUSSIAN,
                DenoiseMethod.NLM, DenoiseMethod.DNCNN, DenoiseMethod.FFDNET);
        System.out.printf("denoising sigma 25: %s; varying: %s%n", r.get("25"), r.get("varying"));
        assertTrue(r.get("25").get(2).psnr() > r.get("25").get(0).psnr(), "DnCNN beats Gaussian filtering");
    }
}
