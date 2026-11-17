package io.github.zzuegg.brainiac.experiments.language;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;

import org.junit.jupiter.api.Test;

class HyperfittingTest {

    @Test
    void hyperfittedGreedyRepeatsLessDespiteWorseValidationLoss() {
        List<Hyperfitting.Outcome> rows = Hyperfitting.run(200, 120, 1);
        StringBuilder line = new StringBuilder(
                "Hyperfitting (3-gram repetition, distinct-2, copied 8-grams, val loss, train loss):");
        for (Hyperfitting.Outcome o : rows) {
            line.append(String.format(" %s %.3f/%.3f/%.3f/%.3f/%.3f;", o.name(), o.repetition(), o.distinct2(),
                    o.copied(), o.validationLoss(), o.trainLoss()));
        }
        System.out.println(line);
        Hyperfitting.Outcome greedy = rows.get(0);
        Hyperfitting.Outcome hyper = rows.get(2);
        assertTrue(hyper.trainLoss() < 0.1, "hyperfitted");
        assertTrue(hyper.repetition() < greedy.repetition(), "less repetition");
        assertTrue(hyper.distinct2() > greedy.distinct2(), "more diverse");
        assertTrue(hyper.validationLoss() > greedy.validationLoss(), "worse validation loss");
    }
}
