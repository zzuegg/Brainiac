package io.github.zzuegg.brainiac.experiments.uncertainty;

import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertTrue;

class EnsembleShootoutStackingTest {

    @Test
    void superLearnerAgainstItsLibrary() {
        List<EnsembleShootout.StackRow> rows = EnsembleShootout.stacking(300, 1);
        StringBuilder b = new StringBuilder("11.4.8 test MSE / log loss:");
        rows.forEach(r -> b.append(String.format(" %s %s %.3f;", r.task(), r.method(), r.loss())));
        System.out.println(b);
        for (String task : List.of("regression", "classification")) {
            double best = rows.stream().filter(r -> r.task().equals(task) && !r.method().startsWith("super"))
                    .mapToDouble(EnsembleShootout.StackRow::loss).min().orElseThrow();
            double convex = rows.stream().filter(r -> r.task().equals(task) && r.method().equals("super learner convex"))
                    .findFirst().orElseThrow().loss();
            assertTrue(convex < best * 1.15, task + ": the super learner is near the best member");
        }
    }
}
