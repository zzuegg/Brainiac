package io.github.zzuegg.brainiac.experiments.combinatorial;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.combinatorial.GraphCoShootout.Row;

import java.util.List;
import java.util.StringJoiner;

import org.junit.jupiter.api.Test;

/** Item 11.2.8: PI-GNN and Erdős goes neural against Ising solvers and greedy rules. */
class GraphCoShootoutTest {

    static String line(List<Row> rows) {
        StringJoiner j = new StringJoiner("; ");
        rows.forEach(r -> j.add(r.toString()));
        return j.toString();
    }

    static double ratio(List<Row> rows, String method) {
        return rows.stream().filter(r -> r.method().equals(method)).findFirst().orElseThrow().ratio();
    }

    @Test
    void maxCutAndIndependentSet() {
        List<double[][]> train = GraphCoShootout.graphs(16, 60, 0.1, 500);
        List<Row> er = GraphCoShootout.maxCut(GraphCoShootout.graphs(3, 60, 0.1, 1), train, 300, 1000, 1);
        List<Row> reg = GraphCoShootout.maxCut(GraphCoShootout.graphs(2, 60, 0, 2), train, 300, 1000, 2);
        List<double[][]> misTrain = GraphCoShootout.graphs(16, 50, 0.1, 600);
        List<Row> mis = GraphCoShootout.independentSet(GraphCoShootout.graphs(3, 50, 0.1, 3), misTrain, 300, 1000, 3);
        System.out.println("MaxCut ER n=60 p=0.1, ratio to best known: " + line(er));
        System.out.println("MaxCut 3-regular n=60: " + line(reg));
        System.out.println("MIS ER n=50 p=0.1, ratio to the exact optimum: " + line(mis));
        System.out.printf("11.2.8 MaxCut ER-60: bSB %.3f, PI-GNN %.3f, Erdos CE %.3f, greedy %.3f; MIS ER-50: Erdos CE %.3f,"
                + " PI-GNN CE %.3f, greedy %.3f, SA %.3f%n", ratio(er, "SB_BALLISTIC"), ratio(er, "PI-GNN threshold"),
                ratio(er, "Erdos cond. expectation"), ratio(er, "greedy+local search"), ratio(mis, "Erdos cond. expectation"),
                ratio(mis, "PI-GNN cond. expectation"), ratio(mis, "greedy MIS"), ratio(mis, "SIMULATED_ANNEALING"));
        assertTrue(ratio(er, "SB_BALLISTIC") > 0.9, "ballistic SB is near the best known cut");
        assertTrue(ratio(er, "PI-GNN threshold") > 0.6, "PI-GNN finds a sizeable cut");
        assertTrue(ratio(mis, "Erdos cond. expectation") > 0.5, "conditional expectation yields a good independent set");
    }
}
