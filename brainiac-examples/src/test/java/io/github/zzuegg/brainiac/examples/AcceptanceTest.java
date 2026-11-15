package io.github.zzuegg.brainiac.examples;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.train.Benchmark;
import io.github.zzuegg.brainiac.train.Benchmark.Report;
import io.github.zzuegg.brainiac.train.Benchmark.Task;
import io.github.zzuegg.brainiac.train.Metric;
import io.github.zzuegg.brainiac.train.TrainingResult;

import java.util.List;

import org.junit.jupiter.api.Test;

/** The milestone 1 acceptance checks, run through the public examples. */
class AcceptanceTest {

    @Test
    void xorReachesLossBelowOneHundredthWithin2000Epochs() {
        TrainingResult r = XorExample.train(XorExample.model(), false);
        assertTrue(r.finalLoss() < 0.01, "loss " + r.finalLoss());
    }

    @Test
    void irisReachesAtLeast95PercentTestAccuracy() {
        IrisExample.Result r = IrisExample.run(IrisExample.DATA_FILE, false);
        float accuracy = r.testEvaluation().get(Metric.ACCURACY);
        assertTrue(accuracy >= 0.95f, "test accuracy " + accuracy);
    }

    @Test
    void spiralIsLearned() {
        Dataset data = SpiralExample.data(100, 3, 1);
        TrainingResult r = SpiralExample.train(SpiralExample.model(3), data, false);
        float accuracy = r.last().metrics().get("accuracy");
        assertTrue(accuracy >= 0.9f, "accuracy " + accuracy);
    }

    @Test
    void sameSeedGivesBitIdenticalTraining() {
        Model a = XorExample.model();
        Model b = XorExample.model();
        XorExample.train(a, false);
        XorExample.train(b, false);
        for (int i = 0; i < a.parameters().size(); i++) {
            assertArrayEquals(a.parameters().get(i).value().toArray(), b.parameters().get(i).value().toArray());
        }
    }

    @Test
    void convolutionalNetworkLearnsImageShapes() {
        Dataset train = MnistExample.shapes(600, 12, 1);
        Dataset test = MnistExample.shapes(200, 12, 2);
        Model model = MnistExample.model(12, 3);
        MnistExample.train(model, train, null, 6, false);
        float accuracy = io.github.zzuegg.brainiac.train.Trainer.of(model)
                .metric(Metric.ACCURACY).evaluate(test).get(Metric.ACCURACY);
        assertTrue(accuracy >= 0.95f, "test accuracy " + accuracy);
    }

    @Test
    void recurrentNetworkForecastsAWave() {
        Model model = SequenceExample.forecaster();
        float mse = SequenceExample.trainForecaster(model, SequenceExample.waves(800, 1), 10, false)
                .evaluate(SequenceExample.waves(300, 2)).loss();
        assertTrue(mse < 0.01f, "test MSE " + mse);
    }

    @Test
    void transformerLearnsToReverseSequences() {
        Model model = SequenceExample.reverser();
        float accuracy = SequenceExample.trainReverser(model, SequenceExample.reversal(2000, 3), 6, false)
                .evaluate(SequenceExample.reversal(500, 4)).get(Metric.ACCURACY);
        assertTrue(accuracy >= 0.95f, "test accuracy " + accuracy);
    }

    @Test
    void everyOfflineTaskRunsItsDefaultModelAndAGenericOne() {
        for (Task task : Tasks.offline()) {
            Task quick = task.limit(200, 100);
            for (String name : List.of("default", "mlp")) {
                Report r = Benchmark.run(
                        quick, name, BenchmarkExample.model(name, quick), t -> t.epochs(1));
                assertTrue(Float.isFinite(r.test().loss()), task.name() + " " + name + ": " + r);
            }
        }
    }
}
