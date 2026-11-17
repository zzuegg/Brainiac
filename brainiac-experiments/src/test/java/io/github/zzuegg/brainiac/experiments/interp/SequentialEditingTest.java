package io.github.zzuegg.brainiac.experiments.interp;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class SequentialEditingTest {

    @Test
    void nullSpaceEditsSurviveASequence() {
        List<SequentialEditing.Result> out = SequentialEditing.run(160, 80, 1);
        for (SequentialEditing.Result r : out) {
            System.out.printf("Editing %-9s %3d edits  efficacy %.2f  final success %.2f  specificity %.2f%n",
                    r.method(), r.edits(), r.efficacy(), r.success(), r.specificity());
        }
        SequentialEditing.Result trained = out.get(0);
        SequentialEditing.Result rome = out.get(1);
        SequentialEditing.Result alpha = out.get(2);
        assertTrue(trained.specificity() > 0.95, trained.toString());
        assertTrue(alpha.success() > 0.7, alpha.toString());
        assertTrue(alpha.specificity() >= rome.specificity() && alpha.specificity() > 0.9, alpha + " vs " + rome);
    }
}
