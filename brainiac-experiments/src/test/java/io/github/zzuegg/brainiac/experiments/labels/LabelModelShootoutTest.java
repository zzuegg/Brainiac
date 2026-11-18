package io.github.zzuegg.brainiac.experiments.labels;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.labels.WeakLabelShootout.LabelMethod;
import io.github.zzuegg.brainiac.experiments.labels.WeakLabelShootout.LabelResult;

import java.util.EnumMap;
import java.util.List;
import java.util.Map;

import org.junit.jupiter.api.Test;

class LabelModelShootoutTest {

    @Test
    void learnedLabelModelsBeatMajorityVote() {
        List<LabelResult> rows = WeakLabelShootout.labelModel(true, 300, 1);
        Map<LabelMethod, LabelResult> by = new EnumMap<>(LabelMethod.class);
        double gold = 0;
        for (LabelResult r : rows) {
            if (r.method() == null) {
                gold = r.endModel();
                System.out.printf("LabelModel %-28s end model %.3f%n", "GOLD_5_PER_CLASS", gold);
                continue;
            }
            by.put(r.method(), r);
            System.out.printf("LabelModel %-28s covered %.3f acc err %.3f end model %.3f%n", r.method(),
                    r.coveredAccuracy(), r.accuracyError(), r.endModel());
        }
        double mv = by.get(LabelMethod.MAJORITY_VOTE).coveredAccuracy();
        assertTrue(by.get(LabelMethod.DATA_PROGRAMMING_CORRELATED).coveredAccuracy() > mv + 0.03,
                "modelling the duplicated weak function beats MV");
        assertTrue(by.get(LabelMethod.DATA_PROGRAMMING_CORRELATED).accuracyError()
                < by.get(LabelMethod.DATA_PROGRAMMING).accuracyError(), "and estimates accuracies better");
        assertTrue(by.get(LabelMethod.MATRIX_COMPLETION).accuracyError() < 0.1, "triplet accuracies are close");
        assertTrue(by.get(LabelMethod.DATA_PROGRAMMING_CORRELATED).endModel() > gold, "weak labels beat 5 gold/class");
        for (LabelResult r : WeakLabelShootout.labelModel(false, 300, 1)) {
            if (r.method() != null) {
                System.out.printf("LabelModel strong pair %-28s covered %.3f acc err %.3f end model %.3f%n", r.method(),
                        r.coveredAccuracy(), r.accuracyError(), r.endModel());
            }
        }
    }
}
