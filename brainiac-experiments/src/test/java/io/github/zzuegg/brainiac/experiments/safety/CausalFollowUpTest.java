package io.github.zzuegg.brainiac.experiments.safety;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.safety.CausalAndFairness.RobustMethod;
import io.github.zzuegg.brainiac.experiments.safety.CausalAndFairness.RobustScores;
import io.github.zzuegg.brainiac.experiments.safety.CausalAndFairness.StructureMethod;
import io.github.zzuegg.brainiac.experiments.safety.CausalAndFairness.StructureRow;

import java.util.EnumMap;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

/** Wave 7 follow-ups of the IRM and NOTEARS negatives (items 7.6.15 and 7.6.16). */
class CausalFollowUpTest {

    @Test
    void vrexAndJustTrainTwiceAgainstErm() {
        Map<RobustMethod, RobustScores> all = new EnumMap<>(RobustMethod.class);
        for (RobustMethod m : new RobustMethod[] {RobustMethod.ERM, RobustMethod.VREX, RobustMethod.JTT}) {
            all.put(m, CausalAndFairness.spurious(m, 1));
        }
        System.out.printf("Spurious colour worst group / average: ERM %.3f/%.3f, V-REx %.3f/%.3f, JTT %.3f/%.3f"
                + " (wave 6: IRM 0.268, GroupDRO 0.462 with group labels)%n",
                all.get(RobustMethod.ERM).worstGroup(), all.get(RobustMethod.ERM).average(),
                all.get(RobustMethod.VREX).worstGroup(), all.get(RobustMethod.VREX).average(),
                all.get(RobustMethod.JTT).worstGroup(), all.get(RobustMethod.JTT).average());
        double erm = all.get(RobustMethod.ERM).worstGroup();
        assertTrue(Math.max(all.get(RobustMethod.VREX).worstGroup(), all.get(RobustMethod.JTT).worstGroup()) > erm,
                "a follow-up lifts the worst group above ERM");
    }

    @Test
    void dagmaAndGesAgainstNotearsAndPc() {
        StringBuilder line = new StringBuilder("Structure shootout SHD/skeleton/TPR/FPR/ms:");
        Map<StructureMethod, Integer> shd = new EnumMap<>(StructureMethod.class);
        Map<StructureMethod, Integer> skeleton = new EnumMap<>(StructureMethod.class);
        int edges = 0;
        for (int d : new int[] {10, 20}) {
            for (boolean nonlinear : new boolean[] {false, true}) {
                List<StructureRow> rows = CausalAndFairness.structureShootout(d, nonlinear, 3);
                line.append(String.format(" | d=%d %s (%d edges)", d, nonlinear ? "tanh" : "linear",
                        rows.get(0).edges()));
                for (StructureRow r : rows) {
                    line.append(String.format(" %s %d/%d/%.2f/%.2f/%d", r.method(), r.shd(), r.skeleton(), r.tpr(),
                            r.fpr(), r.millis()));
                    if (!nonlinear) {
                        shd.merge(r.method(), r.shd(), Integer::sum);
                        skeleton.merge(r.method(), r.skeleton(), Integer::sum);
                    }
                }
                edges += nonlinear ? 0 : rows.get(0).edges();
            }
        }
        System.out.println(line);
        assertTrue(shd.get(StructureMethod.DAGMA) < shd.get(StructureMethod.NOTEARS), "DAGMA beats NOTEARS on SHD");
        assertTrue(skeleton.get(StructureMethod.GES) < edges, "GES's skeleton beats the empty graph");
    }
}
