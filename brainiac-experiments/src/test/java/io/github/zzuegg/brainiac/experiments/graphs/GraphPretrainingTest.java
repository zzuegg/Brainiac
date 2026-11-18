package io.github.zzuegg.brainiac.experiments.graphs;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

/** Wave 7 item 7.1.11. */
class GraphPretrainingTest {

    @Test
    void pretrainedEncodersBeatRandomOnesUnderALinearProbe() {
        Map<String, Double> r = GraphPretraining.linearProbe(new double[] {0.01, 0.05}, 100, 2, 1);
        System.out.printf("7.1.11 probe @1%% / @5%%: DGI %.3f / %.3f, GraphCL %.3f / %.3f, GraphMAE %.3f / %.3f; random"
                + " encoder %.3f / %.3f, GAE %.3f / %.3f, supervised GCN %.3f / %.3f%n", r.get("DGI @ 0.01"),
                r.get("DGI @ 0.05"), r.get("GRAPH_CL @ 0.01"), r.get("GRAPH_CL @ 0.05"), r.get("GRAPH_MAE @ 0.01"),
                r.get("GRAPH_MAE @ 0.05"), r.get("random encoder @ 0.01"), r.get("random encoder @ 0.05"),
                r.get("GAE @ 0.01"), r.get("GAE @ 0.05"), r.get("supervised GCN @ 0.01"),
                r.get("supervised GCN @ 0.05"));
        for (String m : new String[] {"DGI", "GRAPH_CL", "GRAPH_MAE"}) {
            assertTrue(r.get(m + " @ 0.05") > r.get("random encoder @ 0.05"), r.toString());
        }
    }
}
