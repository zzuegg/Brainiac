package io.github.zzuegg.brainiac.experiments.combinatorial;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.combopt.AttentionModel;
import io.github.zzuegg.brainiac.experiments.combinatorial.RoutingShootout.Row;

import java.util.List;

import org.junit.jupiter.api.Test;

/** Item 11.2.2: the attention model with a greedy-rollout baseline. */
class AttentionModelRoutingTest {

    @Test
    void attentionModelLearnsTsp10() {
        AttentionModel model = RoutingShootout.trainAttentionModel(10, 4, 50, 32, 2);
        List<Row> rows = RoutingShootout.attentionModel(model, RoutingShootoutTest.TSP10, 32, 3);
        List<Row> gen = RoutingShootout.attentionModel(model, RoutingShootoutTest.TSP20, 32, 3);
        List<Row> heur = RoutingShootout.heuristics(RoutingShootoutTest.TSP10, 1);
        double nn = RoutingShootoutTest.find(heur, "nearest neighbour", "").gap();
        double random = RoutingShootoutTest.find(heur, "random", "").gap();
        System.out.println("Attention model (200 steps x 32 TSP-10): " + RoutingShootoutTest.line(rows) + "; "
                + RoutingShootoutTest.line(gen));
        System.out.printf("11.2.2 TSP-10 gap: AM greedy %.1f%%, AM x32 sampled %.1f%%, NN %.1f%%, 2-opt %.1f%%, random %.1f%%%n",
                rows.get(0).gap(), rows.get(1).gap(), nn, RoutingShootoutTest.find(heur, "2-opt", "NN").gap(), random);
        assertTrue(rows.get(0).gap() < random / 2, "the attention model halves the random-tour gap");
        assertTrue(rows.get(1).gap() < rows.get(0).gap() + 2, "sampling 32 is about as good as greedy or better");
    }
}
