package io.github.zzuegg.brainiac.experiments.metric;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.metric.MetricLearningShootout.Objective;
import io.github.zzuegg.brainiac.experiments.metric.MetricLearningShootout.Scores;

import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;

/** Item 10.3.7: SupCon (and SupCon with KoLeo, item 10.3.5) against cross-entropy, SimCLR and prototypical networks on unseen-class retrieval. */
class SupervisedContrastiveTest {

    private static List<Scores> run(Objective... objectives) {
        List<Scores> out = new ArrayList<>();
        for (Objective o : objectives) {
            Scores s = MetricLearningShootout.run(o, 600, 1);
            System.out.println(s);
            out.add(s);
        }
        return out;
    }

    @Test
    void supConRetrievesUnseenClasses() {
        Scores untrained = MetricLearningShootout.untrained(1);
        System.out.println("untrained " + untrained);
        List<Scores> s = run(Objective.CROSS_ENTROPY, Objective.SIMCLR, Objective.PROTOTYPICAL, Objective.SUPCON,
                Objective.SUPCON_KOLEO);
        Scores supcon = s.get(3);
        assertTrue(supcon.recall1() > 0.4, "SupCon retrieves unseen classes above chance (0.25)");
        assertTrue(supcon.seenRecall1() > untrained.seenRecall1(), "SupCon improves on the training classes");
    }
}
