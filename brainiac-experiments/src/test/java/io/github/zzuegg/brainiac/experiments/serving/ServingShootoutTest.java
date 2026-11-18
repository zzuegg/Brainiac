package io.github.zzuegg.brainiac.experiments.serving;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import org.junit.jupiter.api.Test;

class ServingShootoutTest {

    @Test
    void draftGuidedSelectionBeatsQuestionOnlyEviction() {
        double[] budgets = {0.1, 0.2, 0.3, 0.5};
        Map<String, ServingShootout.Cell[]> r = ServingShootout.approximateInference(128, 16, budgets, 40,
                new long[] {1, 2, 3});
        for (Map.Entry<String, ServingShootout.Cell[]> e : r.entrySet()) {
            StringBuilder sb = new StringBuilder();
            for (ServingShootout.Cell c : e.getValue()) {
                sb.append(String.format(" %.2f(%.0f/%.0f)", c.accuracy(), c.prefillTokens(), c.draftTokens()));
            }
            System.out.printf("8.4.1/8.4.2 %-22s acc(prefill/draft) at budgets 10/20/30/50%%:%s%n", e.getKey(), sb);
        }
        assertTrue(r.get("full cache")[0].accuracy() > 0.9);
        assertTrue(r.get("SpecKV")[0].accuracy() > r.get("SnapKV")[0].accuracy() + 0.5, "SpecKV at 10%");
        assertTrue(r.get("SpecPC")[1].accuracy() > r.get("LLMLingua")[1].accuracy() + 0.5, "SpecPC at 20%");
        assertTrue(r.get("LongLLMLingua")[3].accuracy() > r.get("truncation")[3].accuracy() + 0.3, "at 50%");
    }
}
