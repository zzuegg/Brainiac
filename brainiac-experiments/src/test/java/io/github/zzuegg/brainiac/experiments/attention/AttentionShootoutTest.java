package io.github.zzuegg.brainiac.experiments.attention;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.sequence.MixerShootout;

import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

class AttentionShootoutTest {

    @Test
    void everyVariantTrainsThroughItsSpec() {
        List<MixerShootout.Result> results = MixerShootout.run(AttentionShootout.lineup(),
                List.of(MixerShootout.recall(4, 4, 2)), 16, 2, 0.01f, 1);
        System.out.println("attention lineup, 2 training steps (smoke run):\n" + MixerShootout.table(results));
        assertEquals(AttentionShootout.lineup().size(), results.size());
        for (MixerShootout.Result r : results) {
            assertTrue(r.parameters() > 0, r.mixer());
        }
    }

    @Test
    void attentionVariantsSolveAssociativeRecall() {
        Map<String, io.github.zzuegg.brainiac.nn.LayerSpec> all = AttentionShootout.lineup();
        Map<String, io.github.zzuegg.brainiac.nn.LayerSpec> some = Map.of("gated NoPE", all.get("gated NoPE"),
                "differential", all.get("differential"));
        List<MixerShootout.Result> r = MixerShootout.run(some, List.of(MixerShootout.recall(6, 6, 3)), 16, 250,
                0.01f, 2);
        System.out.println(MixerShootout.table(r));
        for (MixerShootout.Result x : r) {
            assertTrue(x.accuracy() > 0.9, x.mixer() + " " + x.accuracy());
        }
    }
}
