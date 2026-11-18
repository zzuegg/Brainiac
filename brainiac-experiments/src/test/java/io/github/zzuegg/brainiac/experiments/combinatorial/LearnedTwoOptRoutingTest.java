package io.github.zzuegg.brainiac.experiments.combinatorial;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.combopt.LearnedTwoOpt;
import io.github.zzuegg.brainiac.experimental.data.CombinatorialInstances;
import io.github.zzuegg.brainiac.experimental.data.CombinatorialInstances.Tsp;
import io.github.zzuegg.brainiac.experiments.combinatorial.RoutingShootout.Row;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

/** Item 11.2.6: a learned 2-opt move selector against classical improvement at equal moves. */
class LearnedTwoOptRoutingTest {

    @Test
    void learnedSelectorAgainstTwoOpt() {
        LearnedTwoOpt policy = LearnedTwoOpt.of(32, 7);
        policy.train(20, 8, 16, 40, 8, Optimizer.adam(1e-3f), new SplittableRandom(7));
        List<Tsp> test = CombinatorialInstances.tspSet(10, 20, false, 707);
        List<Row> rows = RoutingShootout.improvement(policy, test, new int[] {50, 100, 200}, 7);
        System.out.println("Improvement from random TSP-20 tours: " + RoutingShootoutTest.line(rows));
        Row best100 = rows.get(5);
        Row learned100 = rows.get(7);
        Row learned200 = rows.get(11);
        Row anneal200 = rows.get(10);
        System.out.printf("11.2.6 TSP-20 gap: learned %.1f%% (100 moves) / %.1f%% (200), best 2-opt %.1f%%, annealing (200) %.1f%%%n",
                learned100.gap(), learned200.gap(), best100.gap(), anneal200.gap());
        assertTrue(learned200.gap() < 60, "the learned selector improves random tours (random is about 200%)");
    }
}
