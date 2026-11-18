package io.github.zzuegg.brainiac.experiments.graphs;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

/** Wave 7 items 7.1.9 and 7.1.10. */
class AggregationAndPoolingTest {

    @Test
    void pnaCombinesAggregators() {
        Map<String, Double> r = GraphStructureShootout.principalAggregation(30, 80, 1);
        System.out.printf("7.1.9 PNA vs GIN / GCN / SAGE: trees/cycles %.3f vs %.3f / %.3f / %.3f; max-neighbour"
                + " log10 MSE %.3f vs %.3f / %.3f / %.3f; diameter MAE %.3f vs %.3f / %.3f / %.3f (mean %.3f)%n",
                r.get("trees/cycles PNA"), r.get("trees/cycles GIN"), r.get("trees/cycles GCN"), r.get("trees/cycles SAGE"),
                r.get("max-neighbour logMSE PNA"), r.get("max-neighbour logMSE GIN"), r.get("max-neighbour logMSE GCN"),
                r.get("max-neighbour logMSE SAGE"), r.get("diameter MAE PNA"), r.get("diameter MAE GIN"),
                r.get("diameter MAE GCN"), r.get("diameter MAE SAGE"), r.get("diameter MAE predict-mean"));
        assertTrue(r.get("max-neighbour logMSE PNA") < r.get("max-neighbour logMSE GIN") - 0.5, r.toString());
        assertTrue(r.get("diameter MAE PNA") < r.get("diameter MAE GIN"), r.toString());
    }

    @Test
    void topKPoolingSeparatesTreesFromCycles() {
        Map<String, Double> r = GraphStructureShootout.hierarchicalPooling(30, 80, 1);
        System.out.printf("7.1.10 trees/cycles: TopK %.3f, SAGPool %.3f, DiffPool %.3f; GIN+sum %.3f, GCN+mean %.3f."
                + " Communities: TopK %.3f, SAGPool %.3f, DiffPool %.3f; GIN+sum %.3f, GCN+mean %.3f%n",
                r.get("trees/cycles TopK"), r.get("trees/cycles SAGPool"), r.get("trees/cycles DiffPool"),
                r.get("trees/cycles GIN + sum"), r.get("trees/cycles GCN + mean"), r.get("communities TopK"),
                r.get("communities SAGPool"), r.get("communities DiffPool"), r.get("communities GIN + sum"),
                r.get("communities GCN + mean"));
        assertTrue(r.get("trees/cycles SAGPool") > r.get("trees/cycles GCN + mean"), r.toString());
    }
}
