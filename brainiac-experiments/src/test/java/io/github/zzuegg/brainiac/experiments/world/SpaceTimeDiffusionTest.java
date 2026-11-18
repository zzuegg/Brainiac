package io.github.zzuegg.brainiac.experiments.world;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class SpaceTimeDiffusionTest {

    private static String q(VideoGeneration.Quality q) {
        return q == null ? "-" : String.format("%.3f/%.3f/%.2f/%.2f", q.frameMmd(), q.clipMmd(), q.countError(),
                q.velocityError());
    }

    @Test
    void factorisedAttentionAgainstFrameWiseAndFull3d() {
        List<VideoGeneration.DiffusionRow> rows = VideoGeneration.spaceTime(30, 6, 8, 1);
        StringBuilder line = new StringBuilder("9.5.1 video diffusion, frameMMD/clipMMD/count err/velocity err"
                + " uncond | first-frame cond (ms/step):");
        for (VideoGeneration.DiffusionRow r : rows) {
            line.append(String.format(" %s %s | %s (%.0f);", r.model(), q(r.unconditional()), q(r.conditional()),
                    r.millisPerStep()));
            assertTrue(Double.isFinite(r.conditional().clipMmd()), r.toString());
        }
        System.out.println(line);
    }
}
