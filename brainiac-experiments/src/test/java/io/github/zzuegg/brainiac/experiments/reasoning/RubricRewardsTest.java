package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Map;
import org.junit.jupiter.api.Test;

/** Wave 6, 6.14: rubric rewards against a scalar reward model and a Likert judge. */
class RubricRewardsTest {

    @Test
    void rubricRewardsBeatHolisticJudges() {
        Map<String, RubricRewards.Arm> arms = RubricRewards.run(60, 1);
        StringBuilder sb = new StringBuilder();
        arms.forEach((k, v) -> sb.append(String.format(" %s %.2f (win %.2f);", k, v.satisfaction(), v.winRate())));
        System.out.printf("held-out rubric satisfaction (win rate vs SFT):%s%n", sb);
        double rubric = arms.get("rubric reward").satisfaction();
        assertTrue(rubric > arms.get("SFT").satisfaction() + 0.1, "rubric reward improves on SFT");
        assertTrue(rubric > arms.get("scalar reward model").satisfaction(), "beats the scalar reward model");
        assertTrue(rubric > arms.get("Likert judge").satisfaction(), "beats the Likert judge");
    }
}
