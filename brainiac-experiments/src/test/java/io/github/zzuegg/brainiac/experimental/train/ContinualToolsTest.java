package io.github.zzuegg.brainiac.experimental.train;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ReservoirReplay;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

import org.junit.jupiter.api.Test;

class ContinualToolsTest {

    /** Task A only constrains w[0] (towards 1); task B wants w = (-1, 2). */
    private static Var taskA(Var w) {
        return w.mul(Tensor.of(1, 0)).sum().sub(1).square();
    }

    private static Var taskB(Var w) {
        return w.sub(Tensor.of(-1, 2)).square().sum();
    }

    private static void minimise(Var w, Optimizer opt, Supplier<Var> loss, int steps) {
        for (int i = 0; i < steps; i++) {
            loss.get().backward();
            opt.step(List.of(w));
        }
    }

    @Test
    void ewcAnchorsOnlyTheWeightsTheOldTaskNeeded() {
        Var w = Var.param(Tensor.of(0, 0));
        minimise(w, Optimizer.adam(0.05f), () -> taskA(w), 300);
        ElasticWeightConsolidation ewc = new ElasticWeightConsolidation(List.of(w), 50);
        // two noisy samples of task A: their gradients differ, so the empirical Fisher is not zero
        ewc.consolidate(List.of(() -> w.mul(Tensor.of(1, 0)).sum().sub(0.8f).square(),
                () -> w.mul(Tensor.of(1, 0)).sum().sub(1.2f).square()));
        float[] f = ewc.fisher(w).toArray();
        assertTrue(f[0] > 0 && f[1] == 0, "importance only on w0: " + f[0] + ", " + f[1]);
        assertEquals(0, ewc.penalty().value().item(), 1e-9);
        minimise(w, Optimizer.adam(0.05f), () -> taskB(w).add(ewc.penalty()), 400);
        float[] v = w.value().toArray();
        assertTrue(v[0] > 0.5f, "w0 stays near task A's value: " + v[0]);
        assertEquals(2, v[1], 0.05, "w1 is free to learn task B");
    }

    @Test
    void synapticIntelligenceMeasuresImportanceAlongThePath() {
        Var w = Var.param(Tensor.of(0, 0));
        SynapticIntelligence si = new SynapticIntelligence(List.of(w), 20, 0.1f);
        minimise(w, si.track(Optimizer.adam(0.05f)), () -> taskA(w), 300);
        si.consolidate();
        float[] omega = si.importance(w).toArray();
        assertTrue(omega[0] > 0.5f && omega[1] == 0, "w0 earned importance: " + omega[0] + ", " + omega[1]);
        minimise(w, si.track(Optimizer.adam(0.05f)), () -> taskB(w).add(si.penalty()), 400);
        float[] v = w.value().toArray();
        assertTrue(v[0] > 0f, "protected: " + v[0]);
        assertEquals(2, v[1], 0.05);
    }

    @Test
    void reservoirKeepsAUniformSampleOfTheStream() {
        ReservoirReplay replay = new ReservoirReplay(200, 1);
        for (int s = 0; s < 100; s++) {
            float[] x = new float[100];
            for (int i = 0; i < 100; i++) {
                x[i] = s * 100 + i;
            }
            replay.add(Tensor.of(Shape.of(100, 1), x), Tensor.of(Shape.of(100), x));
        }
        assertEquals(200, replay.size());
        assertEquals(10_000, replay.seen());
        float[] kept = replay.toDataset().features().toArray();
        double mean = 0;
        for (float k : kept) {
            mean += k / kept.length;
        }
        assertEquals(5000, mean, 700, "uniform over the stream, not just recent samples");
        Tensor[] mixed = replay.mix(Tensor.zeros(Shape.of(8, 1)), Tensor.zeros(Shape.of(8)), 4);
        assertEquals(Shape.of(12, 1), mixed[0].shape());
        assertEquals(Shape.of(12), mixed[1].shape());
    }

    @Test
    void continualBackpropReplacesUnitsWithoutChangingTheFunction() {
        Model model = Model.sequential().input(2).dense(16, Activation.RELU).dense(16, Activation.RELU).dense(3)
                .seed(4).build();
        List<Dense> dense = new ArrayList<>();
        for (Layer l : model.layers()) {
            dense.add((Dense) l);
        }
        ContinualBackprop cbp = new ContinualBackprop(dense, 0.05, 5, 1);
        Tensor[] data = Fixtures.spiral(20, 2);
        cbp.reinitialise(0, 3);
        Tensor before = model.predict(data[0]);
        cbp.reinitialise(0, 3);   // new incoming weights do not matter while outgoing ones are zero
        assertArrayEquals(before.toArray(), model.predict(data[0]).toArray(), 1e-5f,
                "a new unit starts with zero outgoing weights");
        Optimizer opt = Optimizer.adam(0.01f);
        for (int i = 0; i < 60; i++) {
            Functional.crossEntropy(model.forward(Var.of(data[0])), data[1]).backward();
            opt.step(model.parameters());
            cbp.update(data[0]);
        }
        assertTrue(cbp.replaced() > 10, "mature low-utility units were replaced: " + cbp.replaced());
    }
}
