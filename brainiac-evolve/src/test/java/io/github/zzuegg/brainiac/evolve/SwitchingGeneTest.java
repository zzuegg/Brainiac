package io.github.zzuegg.brainiac.evolve;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Brain;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Benchmark.Task;
import io.github.zzuegg.brainiac.train.Optimizer;
import io.github.zzuegg.brainiac.train.Trainer;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

/** Neurons that choose their activation by their input, as an evolvable gene. */
class SwitchingGeneTest {

    private static final Shape IN = Shape.of(3);
    private static final Brain.Switching TANH_RELU = new Brain.Switching(List.of(Activation.TANH, Activation.RELU),
            true);

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

    /**
     * Same region order; sizes, wiring, which neurons switch, the candidates and the mode
     * differ, and they still stack into one model.
     */
    private static List<Genome> designs() {
        Genome a = Genome.dense(3, Activation.RELU, 6, 5).with(d ->
                d.region("h2").neurons.replaceAll(n -> n.withSwitching(true)));
        Genome b = a.with(d -> {
            d.removeNeuron(d.region("h1"), 2);
            d.region("h2").neurons.replaceAll(n -> n.withSwitching(n.id() % 2 == 0));
            d.region("h2").switching = TANH_RELU;   // hard, two candidates
            d.region("h1").neurons.set(0, d.region("h1").neurons.get(0).withSwitching(true));
            d.connectRandomly("input", "h2", 0.5, new SplittableRandom(3));
        });
        Genome c = a.with(d -> {
            for (int i = 0; i < 3; i++) {
                d.region("h2").neurons.add(new Genome.Neuron(d.newId(), Activation.GELU, i != 1));
            }
            d.region("h2").switching = new Brain.Switching(List.of(Activation.SILU, Activation.TANH,
                    Activation.GELU), false);
            d.projection("h1", "h2").connections.removeIf(k -> Genome.source(k) % 3 == 0);
        });
        return List.of(a, b, c);
    }

    private static List<Model> build(List<Genome> genomes) {
        return genomes.stream().map(g -> g.build(IN, null, 2, Activation.SOFTMAX, 5)).toList();
    }

    @Test
    void switchingIsAPerNeuronGene() {
        Genome g = Genome.dense(3, Activation.RELU, 40);
        assertNull(g.region("h1").switching());
        Genome.Draft d = g.draft();
        Mutations.switching(0.25).apply(d, new SplittableRandom(1));
        Genome m = d.freeze();
        int on = m.region("h1").switchingNeurons();
        assertTrue(on > 0 && on < 40, "switched " + on);
        assertEquals(Genome.DEFAULT_SWITCHING, m.region("h1").switching());
        // each switching neuron adds control weights on its connections, a bias, and u and e per candidate
        assertEquals(g.effectiveParameters(2) + on * 3L + on * (1 + 2 * 4), m.effectiveParameters(2));
        assertTrue(m.describe().contains(on + " switching (soft: relu/tanh/sigmoid/identity)"), m.describe());

        Genome hard = m.with(x -> Mutations.switchMode(1).apply(x, new SplittableRandom(2)));
        assertTrue(hard.region("h1").switching().hard());
        Genome more = m;
        for (int i = 0; i < 5; i++) {
            Genome current = more;
            more = current.with(x -> Mutations.switchCandidates(1, Activation.GELU, Activation.SILU)
                    .apply(x, new SplittableRandom(current.hashCode())));
        }
        assertTrue(more.region("h1").switching().candidates().size() >= 2);
        // no switching neuron left: the settings go too
        Genome off = m.with(x -> x.region("h1").neurons.replaceAll(n -> n.withSwitching(false)));
        assertNull(off.region("h1").switching());
        assertEquals(g, off);
        // and the models build and run
        for (Genome x : List.of(m, hard, more)) {
            assertEquals(2, x.build(IN, null, 2, Activation.SOFTMAX, 1).predict(new float[] {1, 0, -1}).length);
        }
    }

    @Test
    void crossoverCarriesSwitchingByNeuron() {
        Genome a = Genome.dense(3, Activation.RELU, 8);
        Genome b = a.with(d -> {
            d.region("h1").neurons.replaceAll(n -> n.withSwitching(true));
            d.region("h1").switching = TANH_RELU;
        });
        Genome child = Crossover.UNIFORM.cross(a, b, new SplittableRandom(4));
        int on = child.region("h1").switchingNeurons();
        assertTrue(on > 0 && on < 8, "took " + on + " switching neurons");
        assertNotNull(child.region("h1").switching());
    }

    @Test
    void stackedSwitchingMatchesEachModelAndTrainsAlike() {
        List<Genome> genomes = designs();
        List<Model> models = build(genomes);
        Population pop = new Population(genomes, models, new float[] {1, 0.5f, 2}, Activation.SOFTMAX, List.of());
        Model stacked = Model.sequential().input(3).layer(pop).build();
        Tensor x = data(7, 1).features();
        float[] all = stacked.predict(x).toArray();
        for (int c = 0; c < 3; c++) {
            float[] own = models.get(c).predict(x).toArray();
            for (int i = 0; i < 7; i++) {
                for (int k = 0; k < 2; k++) {
                    assertEquals(own[i * 2 + k], all[i * 6 + c * 2 + k], 1e-5, "candidate " + c + " row " + i);
                }
            }
        }

        float[] rates = {0.01f, 0.03f, 0.005f};
        Dataset train = data(64, 2);
        float[][] alone = new float[3][];
        for (int c = 0; c < 3; c++) {
            Model m = genomes.get(c).build(IN, null, 2, Activation.SOFTMAX, 5);
            Trainer.of(m).loss(Loss.CROSS_ENTROPY).optimizer(Optimizer.adam(rates[c])).batchSize(16).epochs(3)
                    .seed(9).fit(train);
            alone[c] = m.predict(x).toArray();
        }
        float[] scales = {1, rates[1] / rates[0], rates[2] / rates[0]};
        Population trained = new Population(genomes, build(genomes), scales, Activation.SOFTMAX, List.of());
        Model together = Model.sequential().input(3).layer(trained).build();
        Trainer.of(together).loss(trained.loss(Loss.CROSS_ENTROPY)).optimizer(Optimizer.adam(rates[0]))
                .batchSize(16).epochs(3).seed(9).fit(trained.tile(train));
        float[] t = together.predict(x).toArray();
        for (int c = 0; c < 3; c++) {
            for (int i = 0; i < 7; i++) {
                for (int k = 0; k < 2; k++) {
                    assertEquals(alone[c][i * 2 + k], t[i * 6 + c * 2 + k], 2e-3, "candidate " + c + " row " + i);
                }
            }
            // and the trained control weights come back out by neuron id
            Model fresh = genomes.get(c).build(IN, null, 2, Activation.SOFTMAX, 77);
            trained.weights(c).applyTo(genomes.get(c), fresh);
            assertArrayEquals(alone[c], fresh.predict(x).toArray(), 2e-3f);
        }
    }

    @Test
    void childrenInheritControlWeights() {
        Genome parent = designs().getFirst();
        Model m = parent.build(IN, null, 2, Activation.SOFTMAX, 1);
        Trainer.of(m).loss(Loss.CROSS_ENTROPY).optimizer(Optimizer.adam(0.05f)).epochs(3).seed(1).fit(data(64, 4));
        Weights w = Weights.of(parent, m);
        Model copy = parent.build(IN, null, 2, Activation.SOFTMAX, 99);
        w.applyTo(parent, copy);
        Tensor x = data(5, 5).features();
        assertArrayEquals(m.predict(x).toArray(), copy.predict(x).toArray(), 1e-6f);
    }

    @Test
    void evolutionWithSwitchingRunsBothWays() {
        SplittableRandom random = new SplittableRandom(1);
        float[][] xs = new float[120][2];
        float[][] ys = new float[120][2];
        for (int i = 0; i < 120; i++) {
            xs[i] = new float[] {(float) random.nextGaussian(), (float) random.nextGaussian()};
            ys[i][xs[i][0] * xs[i][1] > 0 ? 1 : 0] = 1;
        }
        Dataset all = Dataset.of(xs, ys);
        Task task = Task.classification("quadrants", all, all, Shape.of(2), Shape.of(2))
                .withDefaults(t -> t.optimizer(Optimizer.adam(0.03f)).batchSize(16).epochs(5));
        for (boolean batched : new boolean[] {false, true}) {
            Evolution e = Evolution.of(task).template(Genome.dense(2, Activation.TANH, 8))
                    .mutations("switching,switch-mode,switch-candidates,activations")
                    .population(6).generations(3).rescore(1).batched(batched).inheritWeights(true).seed(3);
            Evolution.Result r = e.run();
            assertNull(e.firstFailure(), "batched " + batched);
            assertTrue(r.best().score().metric() > 0.6, "batched " + batched + ": " + r.best().score().metric());
        }
    }
}
