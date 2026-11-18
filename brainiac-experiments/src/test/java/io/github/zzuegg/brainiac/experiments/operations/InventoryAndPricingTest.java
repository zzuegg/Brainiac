package io.github.zzuegg.brainiac.experiments.operations;

import io.github.zzuegg.brainiac.experimental.data.OperationsSims;
import org.junit.jupiter.api.Test;

import java.util.Locale;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertTrue;

class InventoryAndPricingTest {

    static String table(Map<String, double[]> r) {
        StringBuilder sb = new StringBuilder();
        r.forEach((k, v) -> {
            sb.append(k).append('=');
            for (int i = 0; i < v.length; i++) {
                sb.append(String.format(Locale.ROOT, i == 0 ? "%.4g" : "/%.4g", v[i]));
            }
            sb.append(' ');
        });
        return sb.toString().trim();
    }

    static String row(Map<String, Double> r) {
        StringBuilder sb = new StringBuilder();
        r.forEach((k, v) -> sb.append(k).append('=').append(String.format(Locale.ROOT, "%.4g ", v)));
        return sb.toString().trim();
    }

    @Test
    void newsvendor() {
        Map<String, Double> light = InventoryAndPricing.newsvendor(2_000, 0.9, false, 1);
        Map<String, Double> heavy = InventoryAndPricing.newsvendor(2_000, 0.9, true, 1);
        System.out.printf("12.6.6 newsvendor tau=0.9 cost/oracle log-normal: %s | Pareto: %s%n", row(light), row(heavy));
        assertTrue(heavy.get("ERM linear") < heavy.get("predict-then-optimise"),
                "ERM beats predict-then-optimise under heavy-tailed demand");
        assertTrue(light.get("ERM linear") < 1.5);
    }

    @Test
    void lostSalesInventory() {
        Map<String, double[]> l1 = InventoryAndPricing.inventory(1, 9, 3_000, 0, 1);
        Map<String, double[]> l3 = InventoryAndPricing.inventory(3, 9, 3_000, 0, 1);
        System.out.printf("12.6.6 lost sales cost/(cost/DP) lead 1: %s | lead 3: %s%n", table(l1), table(l3));
        assertTrue(l1.get("best base-stock")[1] < 1.1, "base-stock is near optimal at lead time 1");
        assertTrue(l3.get("DP optimal")[0] <= l3.get("order the mean demand")[0]);
    }

    @Test
    void pricing() {
        Map<String, double[]> free = InventoryAndPricing.pricing(OperationsSims.DemandModel.LOGIT, 10, 2_000, -1, 4);
        Map<String, double[]> stock = InventoryAndPricing.pricing(OperationsSims.DemandModel.LINEAR, 10, 2_000, 400, 4);
        Map<String, Double> ctx = InventoryAndPricing.contextualPricing(3, 2_000, 3);
        System.out.printf("12.6.7 pricing regret/stock-out logit no inventory: %s | linear with inventory 400: %s | "
                + "contextual regret: %s%n", table(free), table(stock), row(ctx));
        assertTrue(free.get("THOMPSON_PARAMETRIC")[0] < free.get("UCB1")[0], "parametric TS beats UCB1");
        assertTrue(ctx.get("LINEAR_VALUATION") < ctx.get("LINUCB_GRID"), "structured contextual pricing wins");
    }
}
