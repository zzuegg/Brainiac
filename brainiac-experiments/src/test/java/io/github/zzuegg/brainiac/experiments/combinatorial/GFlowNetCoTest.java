package io.github.zzuegg.brainiac.experiments.combinatorial;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.combopt.GFlowNetCo;
import io.github.zzuegg.brainiac.experimental.data.CombinatorialInstances;

import org.junit.jupiter.api.Test;

/** Item 11.2.9: forward-looking detailed balance against trajectory balance on MIS and MaxCut. */
class GFlowNetCoTest {

    @Test
    void forwardLookingAgainstTrajectoryBalance() {
        double[][] g = CombinatorialInstances.erdosRenyi(50, 0.1, 9);
        double mis = CombinatorialInstances.maxIndependentSet(g).value();
        GraphCoShootout.GflowResult a = GraphCoShootout.gflownets(GFlowNetCo.independentSet(g, 0.5), -mis, 150, 16, 9);
        double[][] w = CombinatorialInstances.erdosRenyi(16, 0.3, 10);
        double cut = CombinatorialInstances.maxCut(w, 10).value();
        GraphCoShootout.GflowResult b = GraphCoShootout.gflownets(GFlowNetCo.maxCut(w, 0.5), -cut, 200, 16, 10);
        System.out.println("GFlowNets, MIS on ER n=50 (optimum " + (int) mis + "): " + a);
        System.out.println("GFlowNets, MaxCut on ER n=16 (optimum " + (int) cut + "): " + b);
        System.out.printf("11.2.9 best of 32 / optimum: MIS FL-DB %.3f vs TB %.3f; MaxCut FL-DB %.3f vs TB %.3f%n",
                a.bestOf32()[1], a.bestOf32()[0], b.bestOf32()[1], b.bestOf32()[0]);
        assertTrue(a.bestOf32()[1] > 0.7 && b.bestOf32()[1] > 0.7, "FL-DB samples good solutions");
    }
}
