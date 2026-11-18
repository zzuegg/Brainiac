package io.github.zzuegg.brainiac.experiments.combinatorial;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.data.CombinatorialInstances;

import org.junit.jupiter.api.Test;

/** Item 11.2.10: GNN imitation of strong branching against classic branching rules. */
class LearnedBranchingTest {

    @Test
    void learnedBranchingOnKnapsacks() {
        SolverLearning.Branching b = SolverLearning.branching(s -> CombinatorialInstances.multiKnapsack(15, 3, s),
                s -> CombinatorialInstances.multiKnapsack(25, 4, s), 10, 8, 4, 1);
        SolverLearning.Branching cover = SolverLearning.branching(s -> CombinatorialInstances.setCover(15, s),
                s -> CombinatorialInstances.setCover(25, s), 6, 6, 4, 2);
        System.out.println("Multi-knapsack 15x3 -> 25x4, " + b.samples() + " samples, agreement "
                + String.format("%.2f", b.agreement()) + ": " + b.rows());
        System.out.println("Set cover 15 -> 25, " + cover.samples() + " samples: " + cover.rows());
        System.out.printf("11.2.10 MKP-25 nodes: learned %.1f vs strong %.1f, pseudocost %.1f, most-fractional %.1f;"
                + " ms learned %.1f, strong %.1f, pseudocost %.1f; set cover-25 nodes learned %.1f vs most-fractional %.1f%n",
                b.row("LEARNED").nodes(), b.row("STRONG").nodes(), b.row("PSEUDOCOST").nodes(),
                b.row("MOST_FRACTIONAL").nodes(), b.row("LEARNED").millis(), b.row("STRONG").millis(),
                b.row("PSEUDOCOST").millis(), cover.row("LEARNED").nodes(), cover.row("MOST_FRACTIONAL").nodes());
        assertTrue(b.row("LEARNED").solved() == 1.0, "the learned rule solves every instance to optimality");
        assertTrue(b.samples() > 100, "strong branching yields training samples");
    }
}
