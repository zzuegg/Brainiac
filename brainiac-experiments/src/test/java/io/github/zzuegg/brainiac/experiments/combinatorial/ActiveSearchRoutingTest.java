package io.github.zzuegg.brainiac.experiments.combinatorial;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.combopt.Pomo;
import io.github.zzuegg.brainiac.experimental.data.CombinatorialInstances.Tsp;
import io.github.zzuegg.brainiac.experiments.combinatorial.RoutingShootout.Row;

import java.util.List;

import org.junit.jupiter.api.Test;

/** Item 11.2.5: active search and efficient active search at equal per-instance rollouts. */
class ActiveSearchRoutingTest {

    @Test
    void activeSearchOnUnseenSize() {
        Pomo pomo = RoutingShootout.curve(10, 40, 40, 8, null, 0, RoutingShootoutTest.TSP10, 6).pomo();
        List<Tsp> test = RoutingShootoutTest.TSP20.subList(0, 6);
        List<Row> rows = RoutingShootout.activeSearch(pomo.model(), test, 16, 10, 6);
        List<Row> aug = RoutingShootout.pomo(pomo, "POMO", test);
        List<Row> heur = RoutingShootout.heuristics(test, 1);
        System.out.println("Active search, policy trained on TSP-10, tested on TSP-20: " + RoutingShootoutTest.line(rows)
                + "; " + RoutingShootoutTest.line(aug));
        double sampling = rows.get(0).gap();
        double best = rows.stream().skip(1).mapToDouble(Row::gap).min().orElseThrow();
        System.out.printf("11.2.5 TSP-20 gap at 160 rollouts: sampling %.1f%%, FULL %.1f%%, EAS-Emb %.1f%%, EAS-Lay %.1f%%,"
                + " EAS-Tab %.1f%%, POMO x8 %.1f%%, 2-opt %.1f%%, Or-opt %.1f%%%n", sampling, rows.get(1).gap(),
                rows.get(2).gap(), rows.get(3).gap(), rows.get(4).gap(), aug.get(1).gap(),
                RoutingShootoutTest.find(heur, "2-opt", "NN").gap(), RoutingShootoutTest.find(heur, "Or-opt", "").gap());
        assertTrue(best < sampling + 3, "some adaptation is within noise of sampling or better");
    }
}
