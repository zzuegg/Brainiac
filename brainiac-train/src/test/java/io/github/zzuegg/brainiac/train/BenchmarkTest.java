package io.github.zzuegg.brainiac.train;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.train.Benchmark.Report;
import io.github.zzuegg.brainiac.train.Benchmark.Task;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Supplier;

import org.junit.jupiter.api.Test;

class BenchmarkTest {

    /** Three clusters around different corners of a square. */
    private static Dataset clusters(int samples, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        float[][] centers = {{-1, -1}, {1, -1}, {0, 1}};
        float[][] x = new float[samples][];
        float[][] y = new float[samples][3];
        for (int i = 0; i < samples; i++) {
            int c = i % 3;
            x[i] = new float[] {centers[c][0] + (float) random.nextGaussian() * 0.2f,
                centers[c][1] + (float) random.nextGaussian() * 0.2f};
            y[i][c] = 1;
        }
        return Dataset.of(x, y);
    }

    private static Task task() {
        return Task.classification("clusters", clusters(300, 1), clusters(90, 2), Shape.of(2), Shape.of(3))
                .withClassNames(List.of("a", "b", "c"))
                .withDefaults(t -> t.optimizer(Optimizer.adam(0.05f)).batchSize(16).epochs(20));
    }

    private static Model model() {
        return Model.sequential().input(2).dense(16, Activation.RELU).dense(3, Activation.SOFTMAX).seed(3).build();
    }

    @Test
    void reportsAccuracyAndAConsistentConfusionMatrix() {
        Report r = Benchmark.run(task(), "mlp", model());
        assertTrue(r.accuracy() > 0.95f, "accuracy " + r.accuracy());
        assertEquals(20, r.epochs());
        assertEquals(model().parameterCount(), r.parameters());
        int total = 0;
        int correct = 0;
        for (int a = 0; a < 3; a++) {
            for (int p = 0; p < 3; p++) {
                total += r.confusion()[a][p];
                correct += a == p ? r.confusion()[a][p] : 0;
            }
        }
        assertEquals(90, total);
        assertEquals(r.accuracy(), (float) correct / total, 1e-6f);
        assertEquals(3, r.perClassAccuracy().length);
        assertTrue(r.toString().contains("mlp on clusters"));
        assertTrue(r.confusionTable().contains("actual"));
    }

    @Test
    void settingsOverrideTheTaskDefaults() {
        Report r = Benchmark.run(task(), "short", model(), t -> t.epochs(2));
        assertEquals(2, r.epochs());
    }

    @Test
    void evaluateTestsWithoutTraining() {
        Model m = model();
        Benchmark.run(task(), "trained", m);
        Report again = Benchmark.evaluate(task(), "again", m);
        assertNull(again.training());
        assertEquals(0, again.epochs());
        assertTrue(again.accuracy() > 0.95f);
    }

    @Test
    void comparesSeveralModelsInOrder() {
        Map<String, Supplier<Model>> models = new LinkedHashMap<>();
        models.put("one", BenchmarkTest::model);
        models.put("linear", () -> Model.sequential().input(2).dense(3, Activation.SOFTMAX).seed(1).build());
        List<Report> reports = Benchmark.compare(task(), models, t -> t.epochs(3));
        assertEquals(List.of("one", "linear"), reports.stream().map(Report::model).toList());
        String table = Benchmark.table(reports);
        assertTrue(table.contains("linear") && table.contains("% acc"), table);
    }

    @Test
    void regressionReportsMeanAbsoluteError() {
        float[][] x = new float[200][1];
        float[][] y = new float[200][1];
        for (int i = 0; i < 200; i++) {
            x[i][0] = i / 100f - 1;
            y[i][0] = 2 * x[i][0] + 0.5f;
        }
        Dataset data = Dataset.of(x, y);
        Task line = Task.regression("line", data, data, Shape.of(1), Shape.of(1))
                .withDefaults(t -> t.optimizer(Optimizer.adam(0.05f)).batchSize(20).epochs(100));
        Report r = Benchmark.run(line, "dense", Model.sequential().input(1).dense(1).seed(1).build());
        assertNull(r.confusion());
        assertTrue(r.meanAbsoluteError() < 0.05f, "MAE " + r.meanAbsoluteError());
        assertTrue(Float.isNaN(r.accuracy()));
    }

    @Test
    void rejectsAModelThatDoesNotFit() {
        Model wrong = Model.sequential().input(3).dense(3, Activation.SOFTMAX).build();
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class,
                () -> Benchmark.run(task(), "wrong", wrong));
        assertTrue(e.getMessage().contains("[2]"), e.getMessage());
    }

    @Test
    void limitShrinksTheData() {
        Task small = task().limit(30, 9);
        assertEquals(30, small.train().size());
        assertEquals(9, small.test().size());
        assertArrayEquals(new float[] {1, 0, 0}, small.test().targetsOf(0));
    }
}
