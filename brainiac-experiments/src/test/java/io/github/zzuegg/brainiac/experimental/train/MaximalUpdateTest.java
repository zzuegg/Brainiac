package io.github.zzuegg.brainiac.experimental.train;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import org.junit.jupiter.api.Test;

/** The μP "coordinate check": how much the output moves after a few steps, across widths. */
class MaximalUpdateTest {

    private static double outputChange(int width, boolean mup) {
        Model model = Model.sequential().input(2).dense(width, Activation.TANH).dense(width, Activation.TANH)
                .dense(3).seed(1).build();
        Optimizer opt = mup ? new MaximalUpdate(32, width).apply(model, 0.01f, 2) : Optimizer.adam(0.01f);
        Tensor[] data = Fixtures.spiral(30, 3);
        Tensor before = model.predict(data[0]);
        for (int i = 0; i < 5; i++) {
            Functional.crossEntropy(model.forward(Var.of(data[0])), data[1]).backward();
            opt.step(model.parameters());
        }
        return model.predict(data[0]).sub(before).abs().mean().item();
    }

    @Test
    void outputUpdatesStayTheSameSizeAcrossWidthsUnderMuP() {
        double spRatio = outputChange(512, false) / outputChange(32, false);
        double mupRatio = outputChange(512, true) / outputChange(32, true);
        System.out.println("output change 512/32: standard " + spRatio + ", muP " + mupRatio);
        assertTrue(spRatio > 2 * mupRatio, "standard parametrization blows up with width: " + spRatio);
        assertTrue(mupRatio < 2 && mupRatio > 0.5, "muP keeps updates width-independent: " + mupRatio);
        assertTrue(new MaximalUpdate(32, 512).adamLearningRate(MaximalUpdate.Role.HIDDEN, 0.01f) < 0.001f);
    }
}
