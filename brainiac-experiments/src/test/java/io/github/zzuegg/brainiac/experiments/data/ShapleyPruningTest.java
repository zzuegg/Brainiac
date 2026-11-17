package io.github.zzuegg.brainiac.experiments.data;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.data.ShapleyPruning.Method;
import io.github.zzuegg.brainiac.experiments.data.ShapleyPruning.Result;

import org.junit.jupiter.api.Test;

class ShapleyPruningTest {

    @Test
    void inRunShapleyFlagsMislabelledExamples() {
        Result random = ShapleyPruning.run(Method.RANDOM, 0.2, 0.2, 1);
        Result el2n = ShapleyPruning.run(Method.EL2N, 0.2, 0.2, 1);
        Result first = ShapleyPruning.run(Method.IN_RUN_SHAPLEY, 0.2, 0.2, 1);
        Result second = ShapleyPruning.run(Method.IN_RUN_SHAPLEY_2, 0.2, 0.2, 1);
        System.out.printf("In-Run Shapley pruning (20%% flipped, 20%% removed): flipped share removed / test acc "
                + "first-order %.2f/%.3f, second-order %.2f/%.3f vs EL2N %.2f/%.3f, random %.2f/%.3f%n",
                first.flippedRemoved(), first.accuracy(), second.flippedRemoved(), second.accuracy(),
                el2n.flippedRemoved(), el2n.accuracy(), random.flippedRemoved(), random.accuracy());
        assertTrue(first.flippedRemoved() > 0.3 && second.flippedRemoved() > first.flippedRemoved(),
                "the lowest-valued examples are mislabelled well above chance (0.2), more so with the interactions");
        assertTrue(second.accuracy() > random.accuracy() - 0.03, "and removing them costs no accuracy");
    }
}
