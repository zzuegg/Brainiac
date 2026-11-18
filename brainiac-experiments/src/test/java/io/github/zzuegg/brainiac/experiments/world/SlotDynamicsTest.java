package io.github.zzuegg.brainiac.experiments.world;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class SlotDynamicsTest {

    @Test
    void slotTransformerAgainstPerSlotMlp() {
        List<WorldModelShootout.SlotRow> rows = WorldModelShootout.slotVideo(150, 100, 8, 1);
        StringBuilder line = new StringBuilder("9.5.8 slot video, PSNR@10/20, pos err@10/20, FG-ARI:");
        for (WorldModelShootout.SlotRow r : rows) {
            line.append(String.format(" %s %.1f/%.1f %.2f/%.2f %.2f;", r.model(), r.psnr10(), r.psnr20(),
                    r.position10(), r.position20(), r.fgAri()));
            assertTrue(Double.isFinite(r.psnr10()), line.toString());
        }
        System.out.println(line);
    }
}
