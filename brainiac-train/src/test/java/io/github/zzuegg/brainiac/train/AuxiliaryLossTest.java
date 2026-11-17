package io.github.zzuegg.brainiac.train;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;

import org.junit.jupiter.api.Test;

/** Layers that add their own loss terms and carry state between forward passes. */
class AuxiliaryLossTest {

    /** Multiplies by one weight; while training it also pulls the weight towards a target. */
    static final class Pulled implements Layer {
        final Var weight = Var.param(Tensor.of(new float[] {1}));
        final float target;
        final float strength;
        Var aux;
        int calls;

        Pulled(float target, float strength) {
            this.target = target;
            this.strength = strength;
        }

        @Override
        public Var forward(Var input) {
            return forward(input, false);
        }

        @Override
        public Var forward(Var input, boolean training) {
            calls++;
            if (training && strength > 0) {
                Var d = weight.sub(Var.of(Tensor.of(new float[] {target})));
                aux = d.mul(d).sum().mul(strength);
            }
            return input.mul(weight);
        }

        @Override
        public List<Var> parameters() {
            return List.of(weight);
        }

        @Override
        public Var auxiliaryLoss() {
            return aux;
        }

        @Override
        public void resetState() {
            calls = 0;
        }
    }

    private static final Dataset IDENTITY = Dataset.of(
            new float[][] {{1}, {2}, {-1}, {0.5f}},
            new float[][] {{1}, {2}, {-1}, {0.5f}});

    private static float train(float strength) {
        Pulled layer = new Pulled(3, strength);
        Model model = Model.sequential().input(1).layer(layer).seed(1).build();
        Trainer.of(model).loss(Loss.MSE).optimizer(Optimizer.adam(0.05f)).batchSize(4).epochs(400).seed(1)
                .fit(IDENTITY);
        return layer.weight.value().toArray()[0];
    }

    @Test
    void trainerAddsTheLayersLoss() {
        assertEquals(1, train(0), 0.05f);   // only the data: the identity
        assertTrue(train(10) > 2.5f);       // the layer's own term dominates
    }

    @Test
    void modelSumsAndResets() {
        Pulled a = new Pulled(1, 1);
        Pulled b = new Pulled(2, 1);
        Model model = Model.sequential().input(1).layer(a).layer(b).seed(1).build();
        assertNull(model.auxiliaryLoss());
        model.forward(Var.of(Tensor.of(new float[] {1}).reshape(1, 1)), true);
        // (1 - 1)^2 + (1 - 2)^2
        assertEquals(1, model.auxiliaryLoss().value().item(), 1e-6f);
        model.resetState();
        assertEquals(0, a.calls);
        assertEquals(0, b.calls);
    }
}
