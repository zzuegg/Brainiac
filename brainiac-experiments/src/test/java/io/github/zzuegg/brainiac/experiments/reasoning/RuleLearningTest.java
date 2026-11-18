package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.reasoning.RuleLearning.Learner;
import io.github.zzuegg.brainiac.experiments.reasoning.RuleLearning.Scores;
import io.github.zzuegg.brainiac.experiments.reasoning.RuleLearning.Task;

import org.junit.jupiter.api.Test;

/** Item 7.6.13: dILP and Neural Logic Machines against a GNN, trained on 10-node and tested on 50-node worlds. */
class RuleLearningTest {

    @Test
    void learnedRulesRunOnLargerWorlds() {
        StringBuilder line = new StringBuilder("Rule learning, 50-node pair acc / recall:");
        double dilpRecall = 0;
        double gnnRecall = 0;
        String programs = "";
        for (Task task : Task.values()) {
            line.append(" | ").append(task);
            for (Learner l : Learner.values()) {
                Scores s = RuleLearning.run(task, l, l == Learner.DILP ? 150 : 300, 1);
                line.append(String.format(" %s %.3f/%.2f", l, s.testAccuracy(), s.positiveRecall()));
                dilpRecall += l == Learner.DILP ? s.positiveRecall() : 0;
                gnnRecall += l == Learner.GNN ? s.positiveRecall() : 0;
                programs += l == Learner.DILP ? " [" + task + ": " + s.program() + "]" : "";
            }
        }
        System.out.println(line + programs);
        assertTrue(dilpRecall > gnnRecall, "dILP's extracted rules recall more target pairs on large worlds");
    }
}
