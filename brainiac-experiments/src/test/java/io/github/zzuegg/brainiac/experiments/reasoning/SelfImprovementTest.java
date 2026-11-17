package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.llm.DpoTrainer;
import org.junit.jupiter.api.Test;

class SelfImprovementTest {

    private final ArithmeticTask task = ArithmeticTask.addition(4);

    @Test
    void dpoLearnsFromPreferences() {
        SelfImprovement.Result r = SelfImprovement.preference(task, DpoTrainer.Objective.DPO, 120, 1);
        System.out.println("DPO " + r);
        assertTrue(r.after() > 0.8 && r.after() > r.before() + 0.5, r.toString());
    }

    @Test
    void ipoAndSimpoLearnToo() {
        SelfImprovement.Result ipo = SelfImprovement.preference(task, DpoTrainer.Objective.IPO, 120, 2);
        SelfImprovement.Result simpo = SelfImprovement.preference(task, DpoTrainer.Objective.SIMPO, 120, 3);
        System.out.println("IPO " + ipo + " SimPO " + simpo);
        assertTrue(ipo.after() > 0.6, ipo.toString());
        assertTrue(simpo.after() > 0.6, simpo.toString());
    }

    @Test
    void spinImprovesWithoutNewLabels() {
        SelfImprovement.Result r = SelfImprovement.spin(task, 25, 10, 4);
        System.out.println("SPIN " + r);
        assertTrue(r.after() > r.before() + 0.2, r.toString());
    }

    @Test
    void selfRewardingLiftsTheActor() {
        SelfImprovement.Result r = SelfImprovement.selfRewarding(task, 180, 10, 5);
        System.out.println("Self-rewarding " + r);
        assertTrue(r.after() > r.before() + 0.15, r.toString());
    }
}
