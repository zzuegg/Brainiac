package io.github.zzuegg.brainiac.experiments.combinatorial;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.combopt.EdgeHeatmapGcn;
import io.github.zzuegg.brainiac.experimental.combopt.RoutingEnv;
import io.github.zzuegg.brainiac.experimental.combopt.SoftPermutationTsp;
import io.github.zzuegg.brainiac.experimental.data.CombinatorialInstances;
import io.github.zzuegg.brainiac.experiments.combinatorial.RoutingShootout.Row;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

/** Item 11.2.7: supervised edge heatmaps and the unsupervised soft-permutation model. */
class HeatmapRoutingTest {

    @Test
    void heatmapsAgainstNearestNeighbour() {
        SplittableRandom r = new SplittableRandom(8);
        EdgeHeatmapGcn gcn = EdgeHeatmapGcn.of(32, 2, 5, 8);
        Optimizer opt = Optimizer.adam(2e-3f);
        for (int s = 0; s < 60; s++) {
            List<RoutingEnv> envs = RoutingShootout.randomBatch(16, 10, r);
            int[][] tours = new int[16][];
            for (int i = 0; i < 16; i++) {
                tours[i] = CombinatorialInstances.heldKarp(envs.get(i).distances());
            }
            gcn.step(envs, tours, opt);
        }
        SoftPermutationTsp soft = SoftPermutationTsp.of(32, 2, 2, 0.1f, 8);
        soft.train(10, 100, 32, Optimizer.adam(2e-3f), r);
        List<Row> rows = RoutingShootout.heatmaps(gcn, 8, soft, 16, RoutingShootoutTest.TSP10, 8);
        List<Row> gen = RoutingShootout.heatmaps(gcn, 8, soft, 16, RoutingShootoutTest.TSP20, 8);
        List<Row> heur = RoutingShootout.heuristics(RoutingShootoutTest.TSP10, 1);
        double nn = RoutingShootoutTest.find(heur, "nearest neighbour", "").gap();
        System.out.println("Heatmaps (trained on TSP-10): " + RoutingShootoutTest.line(rows) + "; "
                + RoutingShootoutTest.line(gen));
        System.out.printf("11.2.7 TSP-10 gap: edge GCN beam 8 %.1f%%, soft permutation best of 16 %.1f%% (single %.1f%%),"
                + " NN %.1f%%, random %.1f%%%n", rows.get(1).gap(), rows.get(3).gap(), rows.get(2).gap(), nn,
                RoutingShootoutTest.find(heur, "random", "").gap());
        assertTrue(rows.get(1).gap() <= rows.get(0).gap() + 1, "beam search is about as good as greedy or better");
        assertTrue(rows.get(3).gap() < RoutingShootoutTest.find(heur, "random", "").gap(),
                "the unsupervised model beats random tours");
    }
}
