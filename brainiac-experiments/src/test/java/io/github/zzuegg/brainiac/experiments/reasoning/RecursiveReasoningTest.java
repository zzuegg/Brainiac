package io.github.zzuegg.brainiac.experiments.reasoning;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.HierarchicalReasoningModel;
import io.github.zzuegg.brainiac.experimental.nn.TinyRecursiveModel;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class RecursiveReasoningTest {

    private final List<List<LatinSquarePuzzles.Puzzle>> data = LatinSquarePuzzles.split(6, 400, 1);

    @Test
    void thereAre576LatinSquares() {
        assertEquals(576, LatinSquarePuzzles.allSquares().size());
    }

    @Test
    void tinyRecursiveModelSolvesLatinSquares() {
        TinyRecursiveModel trm = TinyRecursiveModel.of(LatinSquarePuzzles.INPUTS, 64, LatinSquarePuzzles.OUTPUTS,
                3, 2, 3, 1);
        LatinSquarePuzzles.train(trm::supervised, trm.parameters(), data.get(0), 300, 64, 1);
        LatinSquarePuzzles.Evaluation e = LatinSquarePuzzles.evaluate(trm::supervised, data.get(1));
        System.out.println("TRM " + Arrays.toString(e.blankAccuracy()) + " solved " + Arrays.toString(e.solved()));
        double last = e.blankAccuracy()[e.blankAccuracy().length - 1];
        assertTrue(last > 0.6, "well above the 25% of guessing: " + last);
        assertTrue(Arrays.stream(e.blankAccuracy()).allMatch(a -> a > 0.6), "every supervision step is trained");
    }

    @Test
    void hierarchicalReasoningModelSolvesLatinSquares() {
        HierarchicalReasoningModel hrm = HierarchicalReasoningModel.of(LatinSquarePuzzles.INPUTS, 64,
                LatinSquarePuzzles.OUTPUTS, 3, 2, 3, 2);
        LatinSquarePuzzles.train(hrm::supervised, hrm.parameters(), data.get(0), 300, 64, 2);
        LatinSquarePuzzles.Evaluation e = LatinSquarePuzzles.evaluate(hrm::supervised, data.get(1));
        System.out.println("HRM " + Arrays.toString(e.blankAccuracy()) + " solved " + Arrays.toString(e.solved()));
        double last = e.blankAccuracy()[e.blankAccuracy().length - 1];
        assertTrue(last > 0.6, "well above the 25% of guessing: " + last);
        assertTrue(Arrays.stream(e.blankAccuracy()).allMatch(a -> a > 0.6), "every supervision step is trained");
    }

    @Test
    void recursiveModelsAreLayers() {
        Model m = Model.sequential().input(LatinSquarePuzzles.INPUTS)
                .add(TinyRecursiveModel.spec(16, 32, 2, 2, 2))
                .dense(LatinSquarePuzzles.OUTPUTS, Activation.IDENTITY)
                .seed(3).build();
        assertEquals(LatinSquarePuzzles.OUTPUTS, m.outputSize());
        Model h = Model.sequential().input(8).add(HierarchicalReasoningModel.spec(8, 3, 2, 2, 2)).seed(3).build();
        assertEquals(3, h.predict(new float[8]).length);
        Var out = m.forward(Var.of(LatinSquarePuzzles.inputs(List.of(LatinSquarePuzzles.split(4, 1, 1).get(0)
                .get(0)))), true);
        out.sum().backward();
        assertTrue(m.parameters().stream().allMatch(p -> p.grad() != null), "gradients reach every parameter");
    }

    @Test
    void anMlpBaselineForComparison() {
        Model mlp = Model.sequential().input(LatinSquarePuzzles.INPUTS).dense(128, Activation.TANH)
                .dense(LatinSquarePuzzles.OUTPUTS).seed(4).build();
        LatinSquarePuzzles.train(x -> List.of(mlp.forward(x, true)), mlp.parameters(), data.get(0), 300, 64, 4);
        LatinSquarePuzzles.Evaluation e = LatinSquarePuzzles.evaluate(x -> List.of(mlp.forward(x, false)),
                data.get(1));
        System.out.println("MLP " + Arrays.toString(e.blankAccuracy()) + " solved " + Arrays.toString(e.solved()));
        assertTrue(e.blankAccuracy()[0] > 0.25);
    }
}
