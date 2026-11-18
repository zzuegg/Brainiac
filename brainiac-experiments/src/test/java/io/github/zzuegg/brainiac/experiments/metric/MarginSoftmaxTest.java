package io.github.zzuegg.brainiac.experiments.metric;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.metric.MetricLearningShootout.Objective;
import io.github.zzuegg.brainiac.experiments.metric.MetricLearningShootout.Scores;

import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;

/** Item 10.3.9: CosFace, ArcFace and sub-center ArcFace against normalised softmax and cross-entropy, by verification EER. */
class MarginSoftmaxTest {

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
    void angularMarginsTightenClusters() {
        List<Scores> s = run(Objective.CROSS_ENTROPY, Objective.NORM_SOFTMAX, Objective.COSFACE, Objective.ARCFACE,
                Objective.SUBCENTER_ARCFACE);
        Scores arc = s.get(3);
        assertTrue(arc.eer() < 0.5 && arc.recall1() > 0.4, "ArcFace beats chance");
        assertTrue(arc.intraAngle() < arc.interAngle(), "same-class pairs are closer in angle");
    }
}
