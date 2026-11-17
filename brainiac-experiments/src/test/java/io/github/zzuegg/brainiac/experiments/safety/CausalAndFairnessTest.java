package io.github.zzuegg.brainiac.experiments.safety;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.safety.CausalAndFairness.EffectMethod;
import io.github.zzuegg.brainiac.experiments.safety.CausalAndFairness.EffectScores;
import io.github.zzuegg.brainiac.experiments.safety.CausalAndFairness.FairnessMethod;
import io.github.zzuegg.brainiac.experiments.safety.CausalAndFairness.FairnessScores;
import io.github.zzuegg.brainiac.experiments.safety.CausalAndFairness.RobustMethod;
import io.github.zzuegg.brainiac.experiments.safety.CausalAndFairness.RobustScores;

import java.util.EnumMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

class CausalAndFairnessTest {

    @Test
    void estimatorsRemoveConfoundingBias() {
        Map<EffectMethod, EffectScores> all = new EnumMap<>(EffectMethod.class);
        for (EffectMethod m : EffectMethod.values()) {
            EffectScores s = CausalAndFairness.effects(m, 1);
            all.put(m, s);
            System.out.printf("Causal %-10s ATE %.3f (truth 1) error %.3f PEHE %.3f%n", m, s.ate(), s.ateError(),
                    s.pehe());
        }
        double naive = all.get(EffectMethod.NAIVE).ateError();
        assertTrue(all.get(EffectMethod.AIPW).ateError() < naive / 2, "AIPW removes most of the confounding bias");
        assertTrue(all.get(EffectMethod.DRAGONNET).ateError() < naive / 2, "DragonNet too");
        assertTrue(all.get(EffectMethod.TARNET).pehe() < 1, "TARNet estimates the conditional effect");
    }

    @Test
    void notearsRecoversTheGraph() {
        CausalAndFairness.StructureScores s = CausalAndFairness.structure(1);
        System.out.printf("Structure: NOTEARS SHD %d skeleton %d, PC skeleton %d, empty graph %d%n", s.notearsShd(),
                s.notearsSkeleton(), s.pcSkeleton(), s.edges());
        assertTrue(s.notearsShd() < s.edges(), "NOTEARS beats the empty graph");
    }

    @Test
    void fairnessInterventionsShrinkTheGaps() {
        Map<FairnessMethod, FairnessScores> all = new EnumMap<>(FairnessMethod.class);
        for (FairnessMethod m : FairnessMethod.values()) {
            FairnessScores s = CausalAndFairness.fairness(m, 1);
            all.put(m, s);
            System.out.printf("Fairness %-20s accuracy %.3f DP gap %.3f EO gap %.3f%n", m, s.accuracy(), s.dpGap(),
                    s.eoGap());
        }
        FairnessScores erm = all.get(FairnessMethod.ERM);
        assertTrue(all.get(FairnessMethod.REWEIGHING).dpGap() < erm.dpGap(), "reweighing narrows the DP gap");
        assertTrue(all.get(FairnessMethod.EQUALIZED_ODDS_POST).eoGap() < erm.eoGap(), "thresholds narrow the EO gap");
    }

    @Test
    void groupRobustTrainingResistsTheSpuriousColour() {
        Map<RobustMethod, RobustScores> all = new EnumMap<>(RobustMethod.class);
        for (RobustMethod m : RobustMethod.values()) {
            RobustScores s = CausalAndFairness.spurious(m, 1);
            all.put(m, s);
            System.out.printf("Spurious colour %-10s average %.3f worst group %.3f%n", m, s.average(), s.worstGroup());
        }
        assertTrue(all.get(RobustMethod.GROUP_DRO).worstGroup() > all.get(RobustMethod.ERM).worstGroup(),
                "GroupDRO lifts the worst group");
    }
}
