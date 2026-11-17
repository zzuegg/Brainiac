package io.github.zzuegg.brainiac.experiments.techniques;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class DataDependentPositionsTest {

    @Test
    void pathTracksFlipFlopsAndExtrapolates() {
        List<DataDependentPositions.Row> rows = DataDependentPositions.run(DataDependentPositions.flipFlopLineup(),
                List.of("PaTH", "PaTH-FoX", "RoPE", "NoPE", "FoX"), DataDependentPositions::flipFlop,
                new int[] {16, 32, 48}, 200, 1);
        System.out.print(DataDependentPositions.table("flip-flop @ len", rows));
        double[] path = rows.get(0).accuracy();
        double[] rope = rows.get(2).accuracy();
        System.out.printf("flip-flop at 16/48: PaTH %.3f/%.3f, PaTH-FoX %.3f/%.3f vs RoPE %.3f/%.3f, NoPE %.3f/%.3f, "
                + "FoX %.3f/%.3f (chance 0.5)%n", path[0], path[2], rows.get(1).accuracy()[0],
                rows.get(1).accuracy()[2], rope[0], rope[2], rows.get(3).accuracy()[0], rows.get(3).accuracy()[2],
                rows.get(4).accuracy()[0], rows.get(4).accuracy()[2]);
        assertTrue(path[0] > 0.9);
        assertTrue(path[2] > rope[2]);
    }

    @Test
    void contextualPositionsCountWords() {
        List<DataDependentPositions.Row> rows = DataDependentPositions.run(DataDependentPositions.countingLineup(),
                List.of("CoPE", "RoPE", "stick-breaking"), DataDependentPositions::kthWord, new int[] {16, 24}, 250,
                1);
        System.out.print(DataDependentPositions.table("k-th word @ len", rows));
        double[] cope = rows.get(0).accuracy();
        double[] rope = rows.get(1).accuracy();
        double[] sb = rows.get(2).accuracy();
        System.out.printf("k-th last word at 16/24: CoPE %.3f/%.3f vs RoPE %.3f/%.3f vs stick-breaking %.3f/%.3f "
                + "(chance 0.125)%n", cope[0], cope[1], rope[0], rope[1], sb[0], sb[1]);
        assertTrue(cope[0] > rope[0]);
        assertTrue(cope[1] > rope[1]);
    }
}
