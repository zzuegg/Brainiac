package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.generative.NovelViewSynthesis;
import org.junit.jupiter.api.Test;

class NerfVsSplatsTest {

    @Test
    void nerfRendersSlowerThanSplats() {
        SceneFields.Score splats = SceneFields.splats(64, 400, 1);
        SceneFields.Score nerf = SceneFields.nerf(SceneFields.Encoding.POSITIONAL, false, 64, 250, 1);
        double black = NovelViewSynthesis.blackPsnr();
        System.out.printf("novel views, held-out PSNR / train s / render ms per view: NeRF %.2f dB / %.1f s / %.1f ms,"
                + " 3DGS %.2f dB / %.1f s / %.1f ms (all-black %.2f dB)%n", nerf.psnr(), nerf.trainSeconds(),
                nerf.renderMillis(), splats.psnr(), splats.trainSeconds(), splats.renderMillis(), black);
        assertTrue(nerf.psnr() > black + 3, "the radiance field learns the scene");
        assertTrue(nerf.renderMillis() > 5 * splats.renderMillis(), "ray marching renders much slower than splatting");
    }
}
