package io.github.zzuegg.brainiac.experiments.combinatorial;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.combopt.PreferenceOptimization.PreferenceModel;
import io.github.zzuegg.brainiac.experimental.data.CombinatorialInstances.Tsp;

import java.util.Arrays;
import java.util.List;

import org.junit.jupiter.api.Test;

/** Item 11.2.4: preference optimisation against POMO's REINFORCE at equal rollouts. */
class PreferenceRoutingTest {

    @Test
    void preferenceOptimisationAgainstReinforce() {
        List<Tsp> test = RoutingShootoutTest.TSP10;
        // 40 steps x 8 instances x 10 starts = 3200 rollouts each
        RoutingShootout.Curve reinforce = RoutingShootout.curve(10, 40, 10, 8, null, 0, test, 5);
        RoutingShootout.Curve bt = RoutingShootout.curve(10, 40, 10, 8, PreferenceModel.BRADLEY_TERRY, 0, test, 5);
        RoutingShootout.Curve th = RoutingShootout.curve(10, 40, 40, 8, PreferenceModel.THURSTONE, 0, test, 5);
        RoutingShootout.Curve ls = RoutingShootout.curve(10, 40, 40, 8, PreferenceModel.BRADLEY_TERRY, 3, test, 5);
        System.out.println("PO vs REINFORCE, greedy POMO gap every 800 rollouts (TSP-10): REINFORCE "
                + Arrays.toString(round(reinforce.gaps())) + ", PO Bradley-Terry " + Arrays.toString(round(bt.gaps())));
        System.out.printf("11.2.4 TSP-10 gap after 3200 rollouts: PO-BT %.1f%%, PO-Thurstone %.1f%%, PO-BT+2-opt pairs %.1f%%,"
                + " POMO REINFORCE %.1f%%%n", bt.last(), th.last(), ls.last(), reinforce.last());
        assertTrue(Arrays.stream(bt.gaps()).min().orElseThrow() < 60, "preference optimisation trains (random tours are about 87%)");
        assertTrue(Double.isFinite(th.last()) && Double.isFinite(ls.last()));
    }

    private static double[] round(double[] v) {
        return Arrays.stream(v).map(x -> Math.round(x * 10) / 10.0).toArray();
    }
}
