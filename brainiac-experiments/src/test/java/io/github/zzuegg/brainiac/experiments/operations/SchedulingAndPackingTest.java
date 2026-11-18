package io.github.zzuegg.brainiac.experiments.operations;

import io.github.zzuegg.brainiac.experimental.data.OperationsSims;
import org.junit.jupiter.api.Test;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

class SchedulingAndPackingTest {

    @Test
    void jobShop() {
        Map<String, Double> r = SchedulingAndPacking.jobShop(6, 6, 6, 4, 8, 50_000, 1);
        System.out.printf("12.6.8 job shop 6x6 mean gap to branch and bound: %s%n", InventoryAndPricingTest.row(r));
        double bestRule = Math.min(Math.min(r.get("SPT"), r.get("MWKR")), Math.min(r.get("MOPNR"), r.get("FDD_MWKR")));
        assertTrue(bestRule < r.get("RANDOM"), "a priority rule beats random dispatch");
        assertTrue(r.get("L2D (learned)") <= r.get("RANDOM") + 1e-9, "the learned dispatcher is no worse than random");
    }

    @Test
    void binPacking() {
        Map<String, Double> exact = SchedulingAndPacking.binPacking(OperationsSims.ItemDistribution.DISCRETE, 2_000, 0,
                3, 0.7);
        Map<String, Double> noisy = SchedulingAndPacking.binPacking(OperationsSims.ItemDistribution.DISCRETE, 2_000,
                0.3, 3, 0.7);
        Map<String, Double> adv = SchedulingAndPacking.binPacking(OperationsSims.ItemDistribution.ADVERSARIAL_FF, 2_000,
                0.3, 3, 0.7);
        System.out.printf("12.6.9 bins/lower bound discrete exact predictions: %s | 30%% error: %s | adversarial: %s%n",
                InventoryAndPricingTest.row(exact), InventoryAndPricingTest.row(noisy), InventoryAndPricingTest.row(adv));
        // genuine negative result: on random-order discrete streams First Fit is already within 5% of the
        // bound, so profile packing (built on an FFD profile) does not beat it; the hybrid limits the damage
        assertTrue(exact.get("PROFILE_PACKING") < exact.get("FIRST_FIT") * 1.1, "profile packing is near First Fit");
        assertTrue(noisy.get("HYBRID") < noisy.get("PROFILE_PACKING"), "the hybrid is more robust to bad predictions");
    }

    @Test
    void rlPacker() {
        Map<String, double[]> r = SchedulingAndPacking.rlPacker(OperationsSims.ItemDistribution.UNIFORM,
                OperationsSims.ItemDistribution.DISCRETE, 200, 3_000, 1);
        System.out.printf("12.6.9 RL packer bins/lower bound train (uniform) / shift (discrete): %s%n",
                InventoryAndPricingTest.table(r));
        assertTrue(r.get("RL packer (DQN)")[0] < 2, "the RL packer packs");
    }
}
