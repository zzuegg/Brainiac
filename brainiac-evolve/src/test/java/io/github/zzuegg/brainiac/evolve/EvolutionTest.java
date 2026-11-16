package io.github.zzuegg.brainiac.evolve;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.train.Benchmark.Task;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class EvolutionTest {

    /** Three noisy clusters around corners of a square. */
    private static Dataset clusters(int samples, long seed) {
        SplittableRandom random = new SplittableRandom(seed);
        float[][] centers = {{-1, -1}, {1, -1}, {0, 1}};
        float[][] x = new float[samples][];
        float[][] y = new float[samples][3];
        for (int i = 0; i < samples; i++) {
            int c = i % 3;
            x[i] = new float[] {centers[c][0] + (float) random.nextGaussian() * 0.4f,
                centers[c][1] + (float) random.nextGaussian() * 0.4f};
            y[i][c] = 1;
        }
        return Dataset.of(x, y);
    }

    private static Task task() {
        return Task.classification("clusters", clusters(150, 1), clusters(60, 2), Shape.of(2), Shape.of(3))
                .withDefaults(t -> t.optimizer(Optimizer.adam(0.03f)).batchSize(16).epochs(5));
    }

    @Test
    void denseTemplateMatchesAPlainMlp() {
        Genome g = Genome.dense(2, Activation.RELU, 8, 4);
        assertEquals(12, g.neurons());
        assertEquals(2 * 8 + 8 * 4, g.connections());
        assertEquals(2 * 8 + 8 + 8 * 4 + 4 + 4 * 3 + 3, g.effectiveParameters(3));
        assertTrue(g.isViable());
        Model m = g.build(Shape.of(2), null, 3, Activation.SOFTMAX, 1);
        Model mlp = Model.sequential().input(2).dense(8, Activation.RELU).dense(4, Activation.RELU)
                .dense(3, Activation.SOFTMAX).build();
        assertEquals(mlp.parameterCount() - 0, g.effectiveParameters(3));
        assertEquals(3, m.predict(new float[] {0.1f, 0.2f}).length);
    }

    @Test
    void removingANeuronRemovesItsConnections() {
        Genome g = Genome.dense(2, Activation.TANH, 4, 4);
        Genome smaller = g.with(d -> d.removeNeuron(d.region("h1"), 0));
        assertEquals(7, smaller.neurons());
        assertEquals(2 * 3 + 3 * 4, smaller.connections());
        assertEquals(g, Genome.dense(2, Activation.TANH, 4, 4));
        assertNotEquals(g, smaller);
    }

    @Test
    void everyBuiltInMutationKeepsGenomesBuildable() {
        Genome g = Genome.dense(2, Activation.RELU, 6, 6);
        SplittableRandom random = new SplittableRandom(3);
        for (String name : Mutations.names()) {
            Mutation m = Mutations.byName(name, 12);
            assertEquals(name, m.name());
            Genome current = g;
            for (int i = 0; i < 30; i++) {
                Genome.Draft d = current.draft();
                m.apply(d, random);
                Genome next = d.freeze();
                if (next.isViable()) {
                    current = next;
                }
            }
            for (Genome.Region r : current.regions()) {
                assertTrue(r.neurons().size() <= 12, name + " grew " + r.name() + " past the limit");
            }
            Model model = current.build(Shape.of(2), null, 3, Activation.SOFTMAX, 1);
            assertEquals(3, model.predict(new float[] {0.5f, -0.5f}).length, name);
        }
    }

    @Test
    void perNeuronActivationsChangeSingleNeurons() {
        Genome g = Genome.dense(2, Activation.RELU, 50);
        Genome.Draft d = g.draft();
        Mutations.activations(0.1, Mutations.Scope.NEURON, Activation.TANH).apply(d, new SplittableRandom(1));
        Genome m = d.freeze();
        int tanh = m.activationCounts().getOrDefault("tanh", 0);
        assertTrue(tanh > 0 && tanh < 50, "changed " + tanh);

        Genome.Draft r = g.draft();
        Mutations.activations(1, Mutations.Scope.REGION, Activation.TANH).apply(r, new SplittableRandom(1));
        assertEquals(50, r.freeze().activationCounts().get("tanh"));
    }

    @Test
    void crossoverLinesParentsUpById() {
        Genome a = Genome.dense(2, Activation.RELU, 4);
        Genome b = a.with(d -> d.region("h1").neurons.replaceAll(n -> new Genome.Neuron(n.id(), Activation.TANH)));
        Genome child = Crossover.UNIFORM.cross(a, b, new SplittableRandom(5));
        assertEquals(4, child.neurons());
        assertEquals(a.connections(), child.connections());
        int tanh = child.activationCounts().getOrDefault("tanh", 0);
        assertTrue(tanh > 0 && tanh < 4, "took " + tanh + " activations from the other parent");
    }

    @Test
    void sameSeedGivesTheSameEvolution() {
        Evolution.Result r1 = small(7, 1);
        Evolution.Result r2 = small(7, 2);   // threads must not matter
        assertEquals(r1.history().size(), r2.history().size());
        for (int i = 0; i < r1.history().size(); i++) {
            assertEquals(r1.history().get(i).best().fitness(), r2.history().get(i).best().fitness(), 0);
            assertEquals(r1.history().get(i).best().genome(), r2.history().get(i).best().genome());
        }
    }

    @Test
    void bestFitnessNeverDropsAndBeatsATinyStart() {
        Evolution.Result r = small(11, 1);
        List<Evolution.Generation> h = r.history();
        for (int i = 1; i < h.size(); i++) {
            assertTrue(h.get(i).best().fitness() >= h.get(i - 1).best().fitness() - 1e-9,
                    "generation " + i + " lost its best");
        }
        assertTrue(r.best().score().metric() > 0.8, "accuracy " + r.best().score().metric());
        assertFalse(r.table().isEmpty());
    }

    private static Evolution.Result small(long seed, int threads) {
        return Evolution.of(task())
                .template(Genome.dense(2, Activation.RELU, 4))
                .neuronScale(0.5)
                .mutations("connections,neurons,activations,learning-rate")
                .population(6).generations(4).rescore(1)
                .seed(seed).threads(threads)
                .run();
    }

    @Test
    void batchedAndInheritingRunsAreReproducibleAndLearn() {
        for (boolean batched : new boolean[] {false, true}) {
            Evolution.Result r1 = withOptions(batched, true);
            Evolution.Result r2 = withOptions(batched, true);
            for (int i = 0; i < r1.history().size(); i++) {
                assertEquals(r1.history().get(i).best().genome(), r2.history().get(i).best().genome());
                assertEquals(r1.history().get(i).best().fitness(), r2.history().get(i).best().fitness(), 0);
            }
            assertTrue(r1.best().score().metric() > 0.8, "batched " + batched + ": " + r1.best().score().metric());
        }
    }

    @Test
    void batchedScoresMatchOneByOne() {
        Evolution.Result alone = withOptions(false, false);
        Evolution.Result together = withOptions(true, false);
        // same designs, same starting weights and batches: the first generation scores agree
        assertEquals(alone.history().getFirst().best().score().metric(),
                together.history().getFirst().best().score().metric(), 0.02);
    }

    @Test
    void batchedRunsReallyStack() {
        Evolution e = setup(true, false);
        e.run();
        assertTrue(e.stackedRuns() > 0);
    }

    private static Evolution.Result withOptions(boolean batched, boolean inherit) {
        return setup(batched, inherit).run();
    }

    private static Evolution setup(boolean batched, boolean inherit) {
        return Evolution.of(task())
                .template(Genome.dense(2, Activation.RELU, 6))
                .mutations("connections,neurons,activations,learning-rate")
                .population(6).generations(3).rescore(1)
                .batched(batched).inheritWeights(inherit)
                .seed(5);
    }

    @Test
    void rejectsBadSetups() {
        assertThrows(IllegalStateException.class, () -> Evolution.of(task()).run());
        assertThrows(IllegalArgumentException.class, () -> Evolution.of(task()).mutations("telepathy"));
        assertThrows(IllegalArgumentException.class,
                () -> Evolution.of(task()).template(Genome.dense(5, Activation.RELU, 4)).run());
    }
}
