package io.github.zzuegg.brainiac.experiments.combinatorial;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.combopt.AttentionModel;
import io.github.zzuegg.brainiac.experimental.data.CombinatorialInstances.Tsp;
import io.github.zzuegg.brainiac.experiments.combinatorial.RoutingShootout.Row;

import java.util.List;

import org.junit.jupiter.api.Test;

/** Item 11.2.3: POMO's shared baseline and x8 augmentation, and Sym-NCO on rotated instances. */
class PomoRoutingTest {

    @Test
    void pomoAgainstRolloutBaseline() {
        List<Tsp> test = RoutingShootoutTest.TSP10;
        List<Tsp> rotated = RoutingShootout.rotated(test, 9);
        // equal training rollouts: 100 x 32 for the attention model, 40 x 8 x 10 starts for POMO
        AttentionModel am = RoutingShootout.trainAttentionModel(10, 2, 50, 32, 4);
        Row amRow = RoutingShootout.greedy(am, "AM rollout baseline", test);
        RoutingShootout.Curve pomo = RoutingShootout.curve(10, 40, 20, 8, null, 0, test, 4);
        List<Row> rows = RoutingShootout.pomo(pomo.pomo(), "POMO", test);
        List<Row> rot = RoutingShootout.pomo(pomo.pomo(), "POMO rotated", rotated);
        // Sym-NCO: one rotation per instance, so half the instances per step at equal rollouts
        RoutingShootout.Curve ps = RoutingShootout.curve(10, 40, 40, 4, null, 0, 0, 1, test, 4);
        RoutingShootout.Curve sym = RoutingShootout.curve(10, 40, 40, 4, null, 0, 0.1, 1, test, 4);
        List<Row> psRot = RoutingShootout.pomo(ps.pomo(), "problem symmetry rotated", rotated);
        List<Row> symRot = RoutingShootout.pomo(sym.pomo(), "Sym-NCO rotated", rotated);
        double twoOpt = RoutingShootoutTest.find(RoutingShootout.heuristics(test, 1), "2-opt", "random").gap();
        System.out.println("POMO vs rollout baseline (3200 rollouts, TSP-10): " + amRow + "; POMO half budget "
                + String.format("%.1f%%", pomo.gaps()[0]) + "; " + RoutingShootoutTest.line(rows) + "; Sym-NCO "
                + String.format("%.1f%%", sym.last()) + "; " + RoutingShootoutTest.line(rot) + "; "
                + RoutingShootoutTest.line(psRot) + "; " + RoutingShootoutTest.line(symRot));
        System.out.printf("11.2.3 TSP-10 gap: POMO %.1f%% (x8 %.1f%%, half budget %.1f%%), AM rollout %.1f%%, 2-opt %.1f%%;"
                + " rotated set x8: POMO %.1f%%, problem symmetry %.1f%%, Sym-NCO %.1f%%%n", rows.get(0).gap(),
                rows.get(1).gap(), pomo.gaps()[0], amRow.gap(), twoOpt, rot.get(1).gap(), psRot.get(1).gap(),
                symRot.get(1).gap());
        assertTrue(rows.get(1).gap() <= rows.get(0).gap() + 1e-9, "x8 augmentation never hurts (it includes x1)");
        assertTrue(rows.get(0).gap() < amRow.gap() + 5, "POMO is at least close to the rollout baseline at equal rollouts");
    }
}
