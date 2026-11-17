package io.github.zzuegg.brainiac.experiments.control;

import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

import java.util.Map;

class RewardMachineTasksTest {

    @Test
    void counterfactualRewardMachineLearningSolvesTheOfficeTask() {
        Map<String, Double> r = RewardMachineTasks.run(new int[] {5000, 20000, 60000}, 3);
        System.out.printf("W6 3.15 office coffee-mail-office greedy success at 5k/20k/60k steps: CRM %.2f/%.2f/%.2f vs"
                + " product Q %.2f/%.2f/%.2f, history-free Q %.2f/%.2f/%.2f%n", r.get("CRM, 5000"), r.get("CRM, 20000"),
                r.get("CRM, 60000"), r.get("product Q-learning, 5000"), r.get("product Q-learning, 20000"),
                r.get("product Q-learning, 60000"), r.get("history-free Q-learning (core), 5000"),
                r.get("history-free Q-learning (core), 20000"), r.get("history-free Q-learning (core), 60000"));
        assertTrue(r.get("CRM, 20000") > r.get("product Q-learning, 20000"), r.toString());
        assertTrue(r.get("CRM, 20000") == 1.0, r.toString());
    }
}
