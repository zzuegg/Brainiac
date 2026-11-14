package io.github.zzuegg.brainiac.train;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;

import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;

class TrainerTest {

    private static final Dataset XOR = Dataset.of(
            new float[][] {{0, 0}, {0, 1}, {1, 0}, {1, 1}},
            new float[][] {{0}, {1}, {1}, {0}});

    private static Model xorModel() {
        return Model.sequential().input(2).dense(8, Activation.TANH).dense(1, Activation.SIGMOID).seed(42).build();
    }

    private static TrainingResult trainXor(Model model) {
        return Trainer.of(model)
                .loss(Loss.MSE)
                .optimizer(Optimizer.adam(0.05f))
                .batchSize(4)
                .epochs(2_000)
                .metric(Metric.ACCURACY)
                .seed(42)
                .fit(XOR);
    }

    @Test
    void learnsXor() {
        Model model = xorModel();
        TrainingResult result = trainXor(model);
        assertTrue(result.finalLoss() < 0.01, "loss " + result.finalLoss());
        assertEquals(1f, result.last().metrics().get("accuracy"));
        assertTrue(model.predict(0, 1)[0] > 0.9);
        assertTrue(model.predict(1, 1)[0] < 0.1);
        assertEquals(2_000, result.history().size());
    }

    @Test
    void sameSeedsGiveIdenticalRuns() {
        Model a = xorModel();
        Model b = xorModel();
        trainXor(a);
        trainXor(b);
        assertArrayEquals(a.predict(0.3f, 0.7f), b.predict(0.3f, 0.7f));
    }

    @Test
    void reportsValidationAndCallsListeners() {
        List<EpochResult> seen = new ArrayList<>();
        TrainingResult r = Trainer.of(xorModel())
                .epochs(5)
                .batchSize(2)
                .metric(Metric.ACCURACY)
                .validation(XOR)
                .listener(seen::add)
                .fit(XOR);
        assertEquals(5, seen.size());
        assertEquals(r.history(), seen);
        assertTrue(r.last().hasValidation());
        assertTrue(r.last().validationMetrics().containsKey("accuracy"));
        assertTrue(r.last().toString().startsWith("epoch 5  loss "));
    }

    @Test
    void stopsEarlyAndRestoresTheBestWeights() {
        Model model = xorModel();
        // a huge learning rate makes the loss jump around, so improvements stop quickly
        TrainingResult r = Trainer.of(model)
                .optimizer(Optimizer.sgd(50f))
                .batchSize(1)
                .epochs(500)
                .earlyStopping(5)
                .validation(XOR)
                .fit(XOR);
        assertTrue(r.stoppedEarly());
        assertTrue(r.history().size() < 500);
        Evaluation ev = Trainer.of(model).evaluate(XOR);
        assertEquals(r.best().validationLoss(), ev.loss(), 1e-6);
    }

    @Test
    void evaluatesWithoutTraining() {
        Model model = xorModel();
        float before = model.predict(1, 0)[0];
        Evaluation ev = Trainer.of(model).metric(Metric.ACCURACY).evaluate(XOR);
        assertEquals(before, model.predict(1, 0)[0]);
        assertTrue(ev.loss() > 0);
        assertTrue(ev.get(Metric.ACCURACY) >= 0);
        assertThrows(IllegalArgumentException.class, () -> ev.get(Metric.MEAN_ABSOLUTE_ERROR));
    }

    @Test
    void rejectsDataThatDoesNotFitTheModel() {
        Dataset wrong = Dataset.of(new float[][] {{1, 2, 3}}, new float[][] {{1}});
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> Trainer.of(xorModel()).fit(wrong));
        assertEquals("model maps 2 inputs to 1 outputs, but the data has 3 features and 1 targets", e.getMessage());
        assertFalse(e.getMessage().isEmpty());
    }

    @Test
    void accuracyHandlesBinaryAndMultiClassOutputs() {
        var t = io.github.zzuegg.brainiac.tensor.Tensor.of(new float[][] {{1}, {0}, {1}});
        var p = io.github.zzuegg.brainiac.tensor.Tensor.of(new float[][] {{0.9f}, {0.2f}, {0.4f}});
        assertEquals(2f / 3, Metric.ACCURACY.compute(p, t), 1e-6);
        var t3 = io.github.zzuegg.brainiac.tensor.Tensor.of(new float[][] {{0, 1, 0}, {1, 0, 0}});
        var p3 = io.github.zzuegg.brainiac.tensor.Tensor.of(new float[][] {{0.1f, 0.7f, 0.2f}, {0.2f, 0.5f, 0.3f}});
        assertEquals(0.5f, Metric.ACCURACY.compute(p3, t3), 1e-6);
    }
}
