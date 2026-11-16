package io.github.zzuegg.brainiac.evolve;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import io.github.zzuegg.brainiac.train.Trainer;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class PopulationTest {

    private static final Shape IN = Shape.of(3);

    private static Dataset data(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[][] x = new float[n][3];
        float[][] y = new float[n][2];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < 3; k++) {
                x[i][k] = (float) r.nextGaussian();
            }
            y[i][x[i][0] * x[i][1] > x[i][2] ? 1 : 0] = 1;
        }
        return Dataset.of(x, y);
    }

    /** Three different designs with the same region order: sizes, activations, wiring and recurrence differ. */
    private static List<Genome> designs() {
        Genome a = Genome.dense(3, Activation.RELU, 6, 5);
        Genome b = a.with(d -> {
            d.removeNeuron(d.region("h1"), 2);
            d.region("h2").neurons.replaceAll(n -> new Genome.Neuron(n.id(), n.id() % 2 == 0
                    ? Activation.TANH : Activation.SIGMOID));
            d.connectRandomly("input", "h2", 0.5, new SplittableRandom(3));
        });
        Genome c = a.with(d -> {
            for (int i = 0; i < 4; i++) {
                d.region("h2").neurons.add(new Genome.Neuron(d.newId(), Activation.GELU));
            }
            d.connectFully("h1", "h2");
            d.projection("h1", "h2").connections.removeIf(k -> Genome.source(k) % 3 == 0);
        });
        return List.of(a, b, c);
    }

    private static List<Model> build(List<Genome> genomes) {
        return genomes.stream().map(g -> g.build(IN, null, 2, Activation.SOFTMAX, 5)).toList();
    }

    @Test
    void stackedForwardMatchesEachModel() {
        List<Genome> genomes = designs();
        List<Model> models = build(genomes);
        Population pop = new Population(genomes, models, new float[] {1, 0.5f, 2}, Activation.SOFTMAX);
        Model stacked = Model.sequential().input(3).layer(pop).build();
        Tensor x = data(7, 1).features();
        float[] all = stacked.predict(x).toArray();
        for (int c = 0; c < genomes.size(); c++) {
            float[] own = models.get(c).predict(x).toArray();
            for (int i = 0; i < 7; i++) {
                for (int k = 0; k < 2; k++) {
                    assertEquals(own[i * 2 + k], all[i * 6 + c * 2 + k], 1e-5, "candidate " + c + " row " + i);
                }
            }
        }
    }

    @Test
    void stackedTrainingMatchesTrainingAlone() {
        List<Genome> genomes = designs();
        float[] rates = {0.01f, 0.03f, 0.005f};
        Dataset train = data(64, 2);
        Tensor x = data(16, 3).features();

        float[][] alone = new float[3][];
        for (int c = 0; c < 3; c++) {
            Model m = genomes.get(c).build(IN, null, 2, Activation.SOFTMAX, 5);
            Trainer.of(m).loss(Loss.CROSS_ENTROPY).optimizer(Optimizer.adam(rates[c])).batchSize(16).epochs(3)
                    .seed(9).fit(train);
            alone[c] = m.predict(x).toArray();
        }
        float[] scales = new float[3];
        for (int c = 0; c < 3; c++) {
            scales[c] = rates[c] / rates[0];
        }
        Population pop = new Population(genomes, build(genomes), scales, Activation.SOFTMAX);
        Model stacked = Model.sequential().input(3).layer(pop).build();
        Trainer.of(stacked).loss(pop.loss(Loss.CROSS_ENTROPY)).optimizer(Optimizer.adam(rates[0])).batchSize(16)
                .epochs(3).seed(9).fit(pop.tile(train));
        float[] together = stacked.predict(x).toArray();
        for (int c = 0; c < 3; c++) {
            for (int i = 0; i < 16; i++) {
                for (int k = 0; k < 2; k++) {
                    assertEquals(alone[c][i * 2 + k], together[i * 6 + c * 2 + k], 2e-3,
                            "candidate " + c + " row " + i);
                }
            }
        }
        // trained weights come back out by neuron id
        for (int c = 0; c < 3; c++) {
            Model fresh = genomes.get(c).build(IN, null, 2, Activation.SOFTMAX, 77);
            pop.weights(c).applyTo(genomes.get(c), fresh);
            assertArrayEquals(alone[c], fresh.predict(x).toArray(), 2e-3f);
        }
    }

    @Test
    void childrenInheritSharedWeights() {
        Genome parent = Genome.dense(3, Activation.TANH, 5);
        Model trained = parent.build(IN, null, 2, Activation.SOFTMAX, 1);
        Trainer.of(trained).loss(Loss.CROSS_ENTROPY).optimizer(Optimizer.adam(0.05f)).epochs(5).seed(1)
                .fit(data(64, 4));
        Weights w = Weights.of(parent, trained);
        assertEquals(15, w.connections());

        Model copy = parent.build(IN, null, 2, Activation.SOFTMAX, 99);
        assertEquals(15 + 5 + 10, w.applyTo(parent, copy));
        Tensor x = data(5, 5).features();
        assertArrayEquals(trained.predict(x).toArray(), copy.predict(x).toArray(), 1e-6f);

        // a child that lost a neuron and gained one keeps everything it shares
        Genome child = parent.with(d -> {
            d.removeNeuron(d.region("h1"), 0);
            d.region("h1").neurons.add(new Genome.Neuron(d.newId(), Activation.TANH));
            d.projection("input", "h1").connections.add(Genome.key(0, d.region("h1").neurons.getLast().id()));
        });
        Model kid = child.build(IN, null, 2, Activation.SOFTMAX, 7);
        int copied = w.applyTo(child, kid);
        assertEquals(12 + 4 + 8, copied);
        assertTrue(Weights.NONE.isEmpty());
    }
}
