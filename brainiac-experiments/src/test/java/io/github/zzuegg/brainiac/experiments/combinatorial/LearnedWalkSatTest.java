package io.github.zzuegg.brainiac.experiments.combinatorial;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.combinatorial.SolverLearning.SatRow;

import java.util.List;

import org.junit.jupiter.api.Test;

/** Item 11.2.11: a GNN flip policy against WalkSAT, a random walk and DPLL on random 3-SAT. */
class LearnedWalkSatTest {

    static SatRow find(List<SatRow> rows, String method, int n) {
        return rows.stream().filter(r -> r.method().equals(method) && r.n() == n).findFirst().orElseThrow();
    }

    @Test
    void learnedFlipsAgainstWalkSat() {
        List<SatRow> rows = SolverLearning.walkSat(new int[] {10, 20}, 15, 6, 100, new int[] {20, 40}, 20, 500, 4.26, 1);
        rows.forEach(System.out::println);
        System.out.printf("11.2.11 median flips n=20: learned %.0f vs WalkSAT %.0f, random walk %.0f; n=40: learned %.0f"
                + " vs WalkSAT %.0f; solved n=40 learned %.2f vs WalkSAT %.2f%n",
                find(rows, "learned WalkSAT", 20).medianFlips(), find(rows, "WalkSAT p=0.5", 20).medianFlips(),
                find(rows, "random walk", 20).medianFlips(), find(rows, "learned WalkSAT", 40).medianFlips(),
                find(rows, "WalkSAT p=0.5", 40).medianFlips(), find(rows, "learned WalkSAT", 40).solved(),
                find(rows, "WalkSAT p=0.5", 40).solved());
        assertTrue(find(rows, "learned WalkSAT", 20).solved() > 0.5, "the learned policy solves most training-size formulas");
    }
}
