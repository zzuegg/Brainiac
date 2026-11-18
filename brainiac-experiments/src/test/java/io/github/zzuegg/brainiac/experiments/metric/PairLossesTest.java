package io.github.zzuegg.brainiac.experiments.metric;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.metric.MetricLearningShootout.Objective;
import io.github.zzuegg.brainiac.experiments.metric.MetricLearningShootout.Scores;

import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;

/** Item 10.3.8: contrastive, triplet with random, hard and semi-hard mining, and multi-similarity, under one protocol. */
class PairLossesTest {

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
    void pairLossesWithMining() {
        List<Scores> s = run(Objective.CONTRASTIVE, Objective.TRIPLET_RANDOM, Objective.TRIPLET_HARD,
                Objective.TRIPLET_SEMI_HARD, Objective.MULTI_SIMILARITY);
        for (Scores x : s) {
            assertTrue(x.recall1() > 0.4, x.objective() + " beats chance (0.25) on unseen classes");
        }
    }
}
