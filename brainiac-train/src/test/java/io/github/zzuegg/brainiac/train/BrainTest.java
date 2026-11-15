package io.github.zzuegg.brainiac.train;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Brain;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.nio.file.Path;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** The growing brain: regions, sparse projections, rewiring and neurogenesis. */
class BrainTest {

    /** Points on {@code arms} interleaved spiral arms. */
    static Dataset spiral(int perArm, int arms, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[][] x = new float[perArm * arms][2];
        float[][] y = new float[perArm * arms][arms];
        for (int a = 0; a < arms; a++) {
            for (int i = 0; i < perArm; i++) {
                double t = (double) i / perArm;
                double angle = 2 * Math.PI * a / arms + 4 * t + r.nextGaussian() * 0.15;
                x[a * perArm + i][0] = (float) (t * Math.cos(angle));
                x[a * perArm + i][1] = (float) (t * Math.sin(angle));
                y[a * perArm + i][a] = 1;
            }
        }
        return Dataset.of(x, y);
    }

    private static Brain.Spec small() {
        return Brain.builder()
                .region("sensory", 32).initialNeurons(8).activation(Activation.RELU)
                .region("association", 32).initialNeurons(8)
                .connect("input", "sensory", 1.0)
                .connect("sensory", "association", 0.5)
                .connect("association", "sensory", 0.25)
                .connect("association", "association", 0.25)
                .output("association")
                .ticks(2)
                .build();
    }

    private static Brain brainOf(Model m) {
        return (Brain) m.layers().stream().filter(Brain.class::isInstance).findFirst().orElseThrow();
    }

    @Test
    void startsSparseWithOnlyInitialNeuronsActive() {
        Model m = Model.sequential().input(2).add(small()).dense(3, Activation.SOFTMAX).seed(1).build();
        Brain b = brainOf(m);
        assertEquals(16, b.activeNeurons());
        Brain.ProjectionStats ff = b.projections().get(1);
        assertEquals(64, ff.possible());
        assertTrue(ff.density() > 0.25 && ff.density() < 0.75, "density " + ff.density());
        assertEquals(Shape.of(32), b.outputShape(Shape.of(2)));
        float[] out = b.forward(Var.of(Tensor.uniform(Shape.of(5, 2), -1, 1, new SplittableRandom(2))))
                .value().toArray();
        for (int n = 0; n < 5; n++) {
            for (int k = 8; k < 32; k++) {
                assertEquals(0, out[n * 32 + k], "dormant neurons are silent");
            }
        }
        assertTrue(b.describe().contains("association"));
    }

    @Test
    void missingConnectionsGetNoGradientAndStayMissing() {
        Model m = Model.sequential().input(2).add(small()).dense(3, Activation.SOFTMAX).seed(1).build();
        Brain b = brainOf(m);
        long before = b.connections();
        Dataset data = spiral(30, 3, 1);
        Trainer.of(m).loss(Loss.CROSS_ENTROPY).optimizer(Optimizer.adam(0.01f)).epochs(3).batchSize(30).fit(data);
        assertEquals(before, b.connections(), "plain training keeps the wiring");
    }

    @Test
    void pruneGrowAndNeurons() {
        Model m = Model.sequential().input(2).add(small()).dense(3).seed(1).build();
        Brain b = brainOf(m);
        long before = b.connections();
        int[] removed = b.prune(0.5);
        assertEquals(before - java.util.Arrays.stream(removed).sum(), b.connections());
        for (int j = 0; j < removed.length; j++) {
            assertEquals(removed[j], b.grow(j, removed[j], false, new SplittableRandom(3)));
        }
        assertEquals(before, b.connections());
        assertEquals(8, b.addNeurons("association", 8, new SplittableRandom(4)));
        assertEquals(16, b.regions().get(1).active());
        assertTrue(b.connections() > before);
        assertEquals(16, b.addNeurons("association", 100, new SplittableRandom(4)), "only up to capacity");
        assertThrows(IllegalArgumentException.class, () -> b.addNeurons("nowhere", 1, new SplittableRandom(1)));
    }

    @Test
    void plasticityRewiresAndGrowsWhileLearning() {
        Model m = Model.sequential().input(2)
                .add(Brain.builder()
                        .region("hidden", 64).initialNeurons(4).activation(Activation.RELU)
                        .connect("input", "hidden", 1.0)
                        .build())
                .dense(3, Activation.SOFTMAX).seed(1).build();
        Brain b = brainOf(m);
        Plasticity p = Plasticity.of(m).rewire(20, 0.3).until(2_000).neurogenesis(2, 0.02, 8)
                .pruneDeadNeurons(1e-4).seed(1);
        Dataset data = spiral(100, 3, 1);
        TrainingResult r = Trainer.of(m).loss(Loss.CROSS_ENTROPY).optimizer(Optimizer.adam(0.02f)).epochs(80)
                .batchSize(30).metric(Metric.ACCURACY).listener(p).seed(1).fit(data);
        assertTrue(b.activeNeurons() > 4, "the brain grew: " + b);
        assertTrue(p.events().stream().anyMatch(e -> e.kind().equals("grow neurons")));
        assertTrue(p.events().stream().anyMatch(e -> e.kind().equals("rewire")));
        assertEquals(80, p.history().size());
        float accuracy = r.last().metrics().get("accuracy");
        assertTrue(accuracy > 0.8f, "accuracy " + accuracy + " with " + b);
    }

    @Test
    void savesLoadsAndMovesTypes(@TempDir Path dir) throws Exception {
        Model m = Model.sequential().input(2).add(small()).dense(3, Activation.SOFTMAX).seed(1).build();
        Brain b = brainOf(m);
        b.addNeurons("sensory", 5, new SplittableRandom(1));
        b.prune(0.2);
        Tensor x = Tensor.uniform(Shape.of(4, 2), -1, 1, new SplittableRandom(5));
        float[] before = m.predict(x).toArray();
        Path file = dir.resolve("brain.bnm");
        m.save(file);
        Model loaded = Model.load(file);
        assertArrayEquals(before, loaded.predict(x).toArray(), 1e-6f);
        Brain lb = brainOf(loaded);
        assertEquals(b.activeNeurons(), lb.activeNeurons());
        assertEquals(b.connections(), lb.connections());
        Model wide = Model.load(file).to(DType.FLOAT64);
        assertArrayEquals(before, wide.predict(x).to(DType.FLOAT32).toArray(), 1e-5f);
    }

    @Test
    void badBrainsAreRejected() {
        assertThrows(IllegalArgumentException.class, () -> Brain.builder().region("input", 3));
        assertThrows(IllegalStateException.class, () -> Brain.builder().region("a", 3).build());
        assertThrows(IllegalArgumentException.class,
                () -> Brain.builder().region("a", 3).connect("input", "b", 0.5).build());
        assertThrows(IllegalArgumentException.class,
                () -> Brain.builder().region("a", 3).connect("input", "a", 0).build());
        assertThrows(IllegalArgumentException.class, () -> Brain.builder().region("a", 3).initialNeurons(4));
        assertThrows(IllegalArgumentException.class,
                () -> Brain.builder().region("a", 3).connect("input", "a", 1).build().outputShape(Shape.of(2, 2)));
    }
}
