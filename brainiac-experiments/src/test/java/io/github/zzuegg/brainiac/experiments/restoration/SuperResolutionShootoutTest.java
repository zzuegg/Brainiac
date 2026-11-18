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

class SuperResolutionShootoutTest {

    @Test
    void superResolutionBeatsInterpolation() {
        List<Row> r = RestorationShootout.superResolution(Source.SCENES, 16, 2, 120, 16, 1, SrMethod.NEAREST,
                SrMethod.BICUBIC, SrMethod.ESPCN, SrMethod.EDSR);
        System.out.printf("SR x2 scenes 16px, 120 steps: %s%n", r);
        assertTrue(r.get(2).psnr() > r.get(0).psnr(), "ESPCN beats nearest");
        assertTrue(r.get(3).psnr() > r.get(1).psnr() - 0.5, "EDSR is at least near bicubic");
    }
}
