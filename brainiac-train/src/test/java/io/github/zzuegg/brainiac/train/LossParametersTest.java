package io.github.zzuegg.brainiac.train;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Tensor;
import java.util.List;
import org.junit.jupiter.api.Test;

/** A loss's own parameters are trained with the model's. */
class LossParametersTest {

    @Test
    void theTrainerOptimizesTheLossParameters() {
        // targets are 3x the inputs; the model is one weight and the loss has a trainable scale
        Dataset data = Dataset.of(new float[][] {{1}, {2}, {-1}, {0.5f}}, new float[][] {{3}, {6}, {-3}, {1.5f}});
        Model model = Model.sequential().input(1).dense(1).seed(1).build();
        Var scale = Var.param(Tensor.of(0.1f));
        Loss loss = Loss.withParameters((p, t) -> Loss.MSE.compute(p.mul(scale), t), List.of(scale));
        Trainer.of(model).loss(loss).optimizer(Optimizer.adam(0.05f)).batchSize(4).epochs(200).seed(1).fit(data);
        assertTrue(Math.abs(scale.value().item() - 0.1f) > 0.05f, "scale stayed at " + scale.value().item());
    }
}
