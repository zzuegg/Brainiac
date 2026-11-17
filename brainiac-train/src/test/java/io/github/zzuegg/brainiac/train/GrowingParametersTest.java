package io.github.zzuegg.brainiac.train;

import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

import org.junit.jupiter.api.Test;

/** Layers that add or replace parameters while the trainer runs. */
class GrowingParametersTest {

    /** y = x * w, and after a few training steps also + b; w is replaced once on the way. */
    static final class Growing implements Layer {
        Var weight = Var.param(Tensor.of(new float[] {0.5f}));
        Var bias;
        Var replaced;
        int steps;

        @Override
        public Var forward(Var input) {
            return forward(input, false);
        }

        @Override
        public Var forward(Var input, boolean training) {
            if (training) {
                steps++;
                if (steps == 5) {
                    replaced = weight;
                    weight = Var.param(weight.value().add(0f));   // same value, a new parameter
                }
                if (steps == 10) {
                    bias = Var.param(Tensor.of(new float[] {0}));
                }
            }
            Var y = input.mul(weight);
            return bias == null ? y : y.add(bias);
        }

        @Override
        public List<Var> parameters() {
            List<Var> out = new ArrayList<>(List.of(weight));
            if (bias != null) {
                out.add(bias);
            }
            return out;
        }
    }

    @Test
    void newParametersAreTrainedAndOldStateIsDropped() {
        Growing layer = new Growing();
        Model model = Model.sequential().input(1).layer(layer).seed(1).build();
        Dataset data = Dataset.of(new float[][] {{1}, {2}, {-1}, {0}}, new float[][] {{3}, {5}, {-1}, {1}});   // 2x + 1
        StatefulOptimizer adam = (StatefulOptimizer) Optimizer.adam(0.05f);
        Trainer.of(model).loss(Loss.MSE).optimizer(adam).batchSize(4).epochs(400).seed(1).fit(data);
        assertTrue(layer.replaced != null && layer.bias != null);
        for (Tensor slot : adam.slots(layer.replaced)) {
            assertNull(slot);
        }
        assertNotEquals(0f, layer.bias.value().item());
        assertTrue(Math.abs(layer.weight.value().item() - 2) < 0.05f, "w " + layer.weight.value().item());
        assertTrue(Math.abs(layer.bias.value().item() - 1) < 0.05f, "b " + layer.bias.value().item());
    }
}
