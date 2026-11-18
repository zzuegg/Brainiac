package io.github.zzuegg.brainiac.experiments.compression;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class ExpertCompressionTest {

    @Test
    void expertPruningAndMergingHalveTheExperts() {
        List<CompactRow> rows = PruningShootout.moe(new int[] {4, 8}, 300, 1);
        StringBuilder line = new StringBuilder("8.5.7 MoE expert pruning/merging, val MSE:");
        for (CompactRow r : rows) {
            line.append(String.format(" %s %.4f;", r.method(), r.metric()));
        }
        System.out.println(line);
        for (int i = 0; i < rows.size(); i += 6) {
            CompactRow full = rows.get(i);
            CompactRow search = rows.get(i + 1);
            CompactRow drop = rows.get(i + 2);
            assertTrue(search.metric() <= drop.metric() + 1e-9,
                    "the subset search is at least as good as dropping the least used: " + search + " vs " + drop);
            assertTrue(full.metric() < search.metric() + 1e-9, "halving costs something: " + full);
        }
    }
}
