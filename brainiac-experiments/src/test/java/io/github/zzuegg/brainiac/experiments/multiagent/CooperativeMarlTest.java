package io.github.zzuegg.brainiac.experiments.multiagent;

import io.github.zzuegg.brainiac.experimental.marl.OtherPlay;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

class CooperativeMarlTest {

    @Test
    void meanField() {
        Map<String, CooperativeMarl.Lattice> anti = CooperativeMarl.meanField(4, -1f, 1200, 1);
        System.out.printf("11.6.14 antiferromagnetic Ising 4x4 return/order (ground 10/1): %s%n",
                CooperativeMarl.line(anti));
        Map<String, CooperativeMarl.Lattice> ferro = CooperativeMarl.meanField(10, 1f, 600, 1);
        System.out.printf("11.6.14 ferromagnetic Ising 10x10 (100 agents) return/order (ground 10/1): %s%n",
                CooperativeMarl.line(ferro));
        assertTrue(anti.get("IQL").teamReturn() > anti.get("random").teamReturn());
        assertTrue(ferro.get("MF-Q").order() > 0.5);
    }

    @Test
    void otherPlay() {
        Map<String, OtherPlay.Result> r = CooperativeMarl.otherPlay(3000, 4);
        System.out.printf("11.6.15 self-play/cross-play: %s%n", CooperativeMarl.line(r));
        assertTrue(r.get("levers(10) Q_LEARNING OTHER_PLAY").crossPlay() > 0.5);
        assertTrue(r.get("levers(10) Q_LEARNING SELF_PLAY").crossPlay() < 0.5);
    }
}
