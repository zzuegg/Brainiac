package io.github.zzuegg.brainiac.experiments.context;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;

import org.junit.jupiter.api.Test;

class HeadBudgetComparisonTest {

    private static String line(String title, Map<String, HeadBudgetComparison.Outcome> r) {
        StringBuilder s = new StringBuilder(title);
        r.forEach((k, v) -> s.append(String.format(" %s %.2f/%.3f/%.0f;", k, v.recall(), v.error(), v.entries())));
        return s.toString();
    }

    @Test
    void headLevelBudgetsKeepNeedlesAndCutError() {
        Map<String, HeadBudgetComparison.Outcome> r = HeadBudgetComparison.run(192, 0.25, 1);
        System.out.println(line("Head-level KV at 25% budget, recall/error/entries:", r));
        HeadBudgetComparison.Outcome uniform = r.get("snapkv (uniform)");
        assertTrue(r.get("quest").recall() >= 0.95 && r.get("quest").error() < 2 * uniform.error(), r.toString());
        assertTrue(r.get("pyramidkv").error() < uniform.error(), r.toString());
        assertTrue(r.get("ada-kv").error() < uniform.error(), r.toString());
        assertTrue(r.get("duoattention").recall() >= 0.95, r.toString());
        assertTrue(r.get("streaming (all heads)").recall() < 0.2, r.toString());
        assertTrue(r.get("duoattention").entries() < 0.5 * r.get("full").entries());
    }

    @Test
    void reconstructionScoresServeQuestionsTheCompressionNeverSaw() {
        Map<String, HeadBudgetComparison.Outcome> r = HeadBudgetComparison.otherQuestions(192, 0.25, 2);
        System.out.println(line("Head-level KV, other questions, recall/error/entries:", r));
        assertTrue(r.get("kvzip").recall() >= 0.9, r.toString());
        assertTrue(r.get("kvzip").recall() > r.get("snapkv (uniform)").recall() + 0.4, r.toString());
        // Quest evicts nothing: it reads a quarter of the cache per query and still finds every needle
        assertTrue(r.get("quest").recall() >= 0.95 && r.get("h2o").recall() < 0.2, r.toString());
        assertTrue(r.get("quest").entries() < 0.3 * r.get("full").entries());
    }

    @Test
    void questReadsFewPagesOfARealDecoderAndBeatsTheWindow() {
        Map<String, double[]> r = HeadBudgetComparison.questOnLlama(256, 32, 4, 16, 4);
        System.out.printf("Quest on PagedLlama, top-1 agreement/bytes read share: quest %.2f/%.2f, recent window "
                + "%.2f/%.2f, all %.2f/%.2f%n", r.get("quest")[0], r.get("quest")[1], r.get("recent")[0],
                r.get("recent")[1], r.get("all")[0], r.get("all")[1]);
        assertTrue(r.get("quest")[1] < 0.35, "Quest reads a fraction of the cache");
        assertTrue(r.get("quest")[0] > r.get("recent")[0], "at equal bytes Quest beats the recent window");
    }
}
