package io.github.zzuegg.brainiac.evolve;

import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.train.Optimizer;
import io.github.zzuegg.brainiac.train.Trainer;

import java.util.Optional;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class GpuEvolutionTest {

    private static Device gpu() {
        Optional<Device> d = Device.available().stream().filter(x -> x.name().startsWith("vulkan")).findFirst();
        assumeTrue(d.isPresent(), "no Vulkan device");
        return d.get();
    }

    private static Dataset data(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[][] x = new float[n][2];
        float[][] y = new float[n][2];
        for (int i = 0; i < n; i++) {
            x[i][0] = (float) r.nextGaussian();
            x[i][1] = (float) r.nextGaussian();
            y[i][x[i][0] * x[i][1] > 0 ? 1 : 0] = 1;
        }
        return Dataset.of(x, y);
    }

    @Test
    void mixedActivationsTrainOnTheGpu() {
        Device gpu = gpu();
        Genome g = Genome.dense(2, Activation.RELU, 8).with(d -> d.region("h1").neurons
                .replaceAll(n -> new Genome.Neuron(n.id(), n.id() % 2 == 0 ? Activation.TANH : Activation.SIGMOID)));
        Model m = g.build(Shape.of(2), null, 2, Activation.SOFTMAX, 1);
        Trainer.of(m).loss(Loss.CROSS_ENTROPY).optimizer(Optimizer.adam(0.05f)).epochs(3).batchSize(16).seed(1)
                .device(gpu).fit(data(64, 1));
    }

    @Test
    void evolutionOnTheGpuTrainsEveryCandidate() {
        Device gpu = gpu();
        io.github.zzuegg.brainiac.train.Benchmark.Task task = io.github.zzuegg.brainiac.train.Benchmark.Task
                .classification("quadrants", data(120, 2), data(40, 3), Shape.of(2), Shape.of(2))
                .withDefaults(t -> t.optimizer(Optimizer.adam(0.03f)).batchSize(16).epochs(3));
        for (boolean batched : new boolean[] {false, true}) {
            Evolution e = Evolution.of(task).template(Genome.dense(2, Activation.RELU, 6))
                    .mutations("connections,neurons,activations,learning-rate")
                    .population(6).generations(2).rescore(1).device(gpu).batched(batched).inheritWeights(true)
                    .seed(3);
            Evolution.Result r = e.run();
            assertNull(e.firstFailure(), "batched " + batched);
            long distinct = r.population().stream().mapToDouble(Evolution.Individual::fitness).distinct().count();
            assertTrue(distinct > 1, "batched " + batched + ": every candidate scored the same");
            assertTrue(r.population().stream().allMatch(i -> Double.isFinite(i.fitness())));
        }
    }
}
