package io.github.zzuegg.brainiac.experiments.symbolic;

import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ConstraintSolvingTest {

    private static ConstraintSolving.Row find(List<ConstraintSolving.Row> rows, String method) {
        return rows.stream().filter(r -> r.method().startsWith(method)).findFirst().orElseThrow();
    }

    private static String brief(List<ConstraintSolving.Row> rows) {
        return rows.stream().map(ConstraintSolving.Row::toString).collect(Collectors.joining("; "));
    }

    @Test
    void sudokuAndQueens() {
        List<ConstraintSolving.Row> s = ConstraintSolving.sudoku(4, 55, 1);
        System.out.printf("13.2.7 sudoku: %s%n", brief(s));
        ConstraintSolving.Row mac = find(s, "MAC MRV");
        assertEquals(mac.instances(), mac.solved());
        assertTrue(mac.work() < 2 * 55, "MAC + MRV needs under two nodes per blank cell on Sudoku");
        assertTrue(find(s, "FC MRV").solved() == mac.instances());
        List<ConstraintSolving.Row> q = ConstraintSolving.queens(10, 200, 1);
        System.out.printf("13.2.7 queens: %s%n", brief(q));
        ConstraintSolving.Row mc = q.stream().filter(r -> r.method().startsWith("min-conflicts")).findFirst()
            .orElseThrow();
        assertEquals(1, mc.solved(), "min-conflicts solves 200-queens");
        assertTrue(mc.work() < 1000);
    }

    @Test
    void phaseTransition() {
        Map<Double, List<ConstraintSolving.Row>> pt = ConstraintSolving.phaseTransition(25,
            new double[] {0.45, 0.55, 0.6, 0.65, 0.7, 0.8}, 3, 1);
        pt.forEach((p, rows) -> System.out.printf("13.2.7 %s%n", brief(rows)));
        double peakFc = 0;
        double edgeFc = find(pt.get(0.45), "FC").work();
        for (List<ConstraintSolving.Row> rows : pt.values()) {
            peakFc = Math.max(peakFc, find(rows, "FC").work());
            assertTrue(find(rows, "MAC").work() <= find(rows, "FC").work() + 1e-9, "MAC never needs more nodes");
        }
        assertTrue(peakFc > edgeFc, "search cost peaks away from the easy side");
    }

    @Test
    void anyCsp() {
        List<ConstraintSolving.Row> rows = ConstraintSolving.learned(20, 40, 250, 5, 1000, 1);
        System.out.printf("13.2.8 ANYCSP: %s%n", brief(rows));
        assertTrue(find(rows, "ANYCSP (").solved() >= find(rows, "ANYCSP untrained").solved());
        assertEquals(5, find(rows, "MAC").solved(), "MAC is complete on planted instances");
    }
}
