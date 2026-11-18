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

class SelfSupervisedDenoisingTest {

    @Test
    void selfSupervisedDenoising() {
        List<Row> iid = RestorationShootout.selfSupervised(NoiseKind.GAUSSIAN, 16, 80, 40, 2, 1, SelfMethod.MEDIAN,
                SelfMethod.SUPERVISED, SelfMethod.N2N, SelfMethod.NE2NE, SelfMethod.ZS_N2N);
        List<Row> corr = RestorationShootout.selfSupervised(NoiseKind.CORRELATED, 16, 0, 40, 2, 1, SelfMethod.MEDIAN,
                SelfMethod.M2M);
        System.out.printf("self-supervised iid: %s; correlated: %s%n", iid, corr);
        assertTrue(iid.get(2).psnr() > iid.get(0).psnr(), "Noise2Noise beats the median filter");
        assertTrue(corr.get(1).psnr() > 0, "Median2Median runs");
    }
}
