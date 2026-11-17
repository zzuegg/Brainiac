package io.github.zzuegg.brainiac.experiments.generative;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

class ThermodynamicDenoisingTest {

    @Test
    void chainedPBitStagesLearnBarsAndStripes() {
        ThermodynamicDenoising.Score dtm = ThermodynamicDenoising.pbits(4, 40, 600, 1);
        ThermodynamicDenoising.Score single = ThermodynamicDenoising.pbits(1, 40, 600, 1);
        ThermodynamicDenoising.Score ddpm = ThermodynamicDenoising.ddpm(1500, 10, 1);
        System.out.printf("bars and stripes 4x4 (valid share / distinct of 30 / MMD): %s %.3f/%d/%.4f vs %s "
                + "%.3f/%d/%.4f vs %s %.3f/%d/%.4f%n", dtm.method(), dtm.validShare(), dtm.distinct(), dtm.mmd(),
                single.method(), single.validShare(), single.distinct(), single.mmd(), ddpm.method(),
                ddpm.validShare(), ddpm.distinct(), ddpm.mmd());
        assertTrue(dtm.validShare() > 0.8 && dtm.distinct() >= 25, "the DTM covers the valid patterns");
        assertTrue(dtm.mmd() < 1.5 * single.mmd(), "the DTM is on par with a single machine");
        assertTrue(dtm.mmd() < ddpm.mmd(), "the p-bit models beat a thresholded continuous DDPM");
    }
}
