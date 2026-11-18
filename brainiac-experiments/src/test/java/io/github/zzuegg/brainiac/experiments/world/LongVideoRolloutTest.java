package io.github.zzuegg.brainiac.experiments.world;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class LongVideoRolloutTest {

    @Test
    void selfForcingAndHistoryGuidanceOnFourTimesTheTrainingLength() {
        List<VideoGeneration.RolloutRow> rows = VideoGeneration.longRollouts(20, 2, 2, new float[] {4f}, 1);
        StringBuilder line = new StringBuilder("9.5.3/9.5.4 32-frame rollouts, count err frames 1-8/8-16/16-32;"
                + " frame MMD 16-32; velocity err:");
        for (VideoGeneration.RolloutRow r : rows) {
            line.append(String.format(" %s %.2f/%.2f/%.2f; %.3f; %.2f;", r.model(), r.countError()[0],
                    r.countError()[1], r.countError()[2], r.frameMmd()[2], r.velocityError()));
            assertTrue(Double.isFinite(r.frameMmd()[2]), line.toString());
        }
        System.out.println(line);
    }
}
