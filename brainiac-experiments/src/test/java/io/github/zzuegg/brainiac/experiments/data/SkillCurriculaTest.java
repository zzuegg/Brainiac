package io.github.zzuegg.brainiac.experiments.data;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.data.SkillTasks;

import java.util.EnumMap;
import java.util.Map;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class SkillCurriculaTest {

    @Test
    void skillTasksEncodeTheAnswers() {
        SkillTasks.Batch b = SkillTasks.sample(SkillTasks.Skill.CARRY, 20, new SplittableRandom(1));
        float[] y = b.y().toArray();
        for (int i = 0; i < 20; i++) {
            assertEquals(1, (int) y[i * SkillTasks.OUT + 1], "a carry writes a tens digit of 1");
            assertEquals(10, (int) y[i * SkillTasks.OUT + 2], "then blanks");
        }
        assertTrue(SkillTasks.prerequisites()[SkillTasks.Skill.CARRY.ordinal()][SkillTasks.Skill.ADD_MULTI.ordinal()]);
    }

    @Test
    void skillItAgainstUniformAndTargetOnly() {
        DataCuration.SkillGraph g = DataCuration.skillGraph(100, 60, 1);
        System.out.println("Skill-It graph from probe runs: " + g);
        DataCuration.SkillRun targetOnly = DataCuration.skillSelection(DataCuration.SkillSampling.TARGET_ONLY, 400,
                0, null, 2);
        double threshold = targetOnly.targetLoss() * 1.1;
        Map<DataCuration.SkillSampling, DataCuration.SkillRun> runs = new EnumMap<>(DataCuration.SkillSampling.class);
        for (DataCuration.SkillSampling m : DataCuration.SkillSampling.values()) {
            runs.put(m, DataCuration.skillSelection(m, 400, threshold, g.graph(), 2));
            System.out.println("skill sampling, 400 steps, threshold " + String.format("%.3f", threshold) + ": "
                    + runs.get(m));
        }
        DataCuration.SkillRun skillIt = runs.get(DataCuration.SkillSampling.SKILL_IT);
        System.out.printf("Skill-It: target loss %.4f, steps to threshold %d vs target-only %d, uniform %d; graph "
                + "recovered %d/%d edges%n", skillIt.targetLoss(), skillIt.stepsToTarget(),
                runs.get(DataCuration.SkillSampling.TARGET_ONLY).stepsToTarget(),
                runs.get(DataCuration.SkillSampling.UNIFORM).stepsToTarget(), g.recovered(), g.planted());
        assertTrue(skillIt.skillShares()[SkillTasks.Skill.REVERSE.ordinal()] < 0.2, "the decoy is avoided");
        assertTrue(skillIt.targetLoss() < runs.get(DataCuration.SkillSampling.UNIFORM).targetLoss(),
                "Skill-It beats uniform on the target");
    }
}
