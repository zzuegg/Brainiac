package io.github.zzuegg.brainiac.experiments.generative;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

class NovelViewSynthesisTest {

    @Test
    void splatsRenderHeldOutViews() {
        NovelViewSynthesis.Score splats = NovelViewSynthesis.splats(64, 400, 1);
        NovelViewSynthesis.Score nerf = NovelViewSynthesis.nerf(128, 400, 1);
        double black = NovelViewSynthesis.blackPsnr();
        System.out.printf("novel views, 400 steps: %s held-out PSNR %.2f dB (train %.2f) in %.1f s vs %s %.2f dB "
                + "(train %.2f) in %.1f s (all-black %.2f dB)%n", splats.method(), splats.psnr(), splats.trainPsnr(),
                splats.seconds(), nerf.method(), nerf.psnr(), nerf.trainPsnr(), nerf.seconds(), black);
        assertTrue(splats.psnr() > black + 8, "the splats fit the scene");
        assertTrue(splats.psnr() > nerf.psnr() - 1, "splats are on par with the NeRF field or better");
    }
}
