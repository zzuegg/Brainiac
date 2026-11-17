package io.github.zzuegg.brainiac.experimental.rl;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.NoisyDense;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.rl.GridWorld;
import io.github.zzuegg.brainiac.rl.Nets;
import io.github.zzuegg.brainiac.rl.Rl;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.SplittableRandom;

class RainbowDqnTest {

    private static GridWorld maze() {
        return GridWorld.of("S....", ".###.", "....G").maxSteps(30);
    }

    private static double solve(RainbowDqn dqn) {
        GridWorld env = maze();
        dqn.learningStarts(100).seed(4).fit(env, 2500);
        double score = Rl.evaluate(dqn, env, 1, 1);
        System.out.println(dqn + ": " + score + " loss " + dqn.lastLoss());
        return score;
    }

    @Test
    void categoricalC51SolvesTheMaze() {
        RainbowDqn dqn = RainbowDqn.of(() -> Nets.mlp(15, 4 * 21, Activation.RELU, 1, 64), 4).categorical(21, -2, 2);
        assertTrue(solve(dqn) > 0.9);
    }

    @Test
    void quantileRegressionSolvesTheMaze() {
        RainbowDqn dqn = RainbowDqn.of(() -> Nets.mlp(15, 4 * 8, Activation.RELU, 1, 64), 4).quantile(8);
        assertTrue(solve(dqn) > 0.9);
    }

    @Test
    void noisyDuelingNStepPrioritizedSolvesTheMaze() {
        RainbowDqn dqn = RainbowDqn.of(() -> Model.sequential().input(15).dense(64, Activation.RELU)
                .add(NoisyDense.spec(5 * 11, 0.5f, 3)).seed(2).build(), 4)
                .categorical(11, -2, 2).dueling(true).nSteps(3)
                .prioritized(0.5, io.github.zzuegg.brainiac.rl.Schedule.linear(0.4, 1, 2500));
        assertTrue(solve(dqn) > 0.9);
    }

    @Test
    void noisyDenseUsesNoiseOnlyInTraining() {
        NoisyDense layer = new NoisyDense(3, 2, 0.5f, new SplittableRandom(1));
        Var x = Var.of(Tensor.of(Shape.of(1, 3), 1, 2, 3));
        float[] eval1 = layer.forward(x, false).value().toArray();
        float[] train1 = layer.forward(x, true).value().toArray();
        float[] train2 = layer.forward(x, true).value().toArray();
        layer.resample();
        float[] train3 = layer.forward(x, true).value().toArray();
        assertEquals(train1[0], train2[0], 1e-6);
        assertTrue(Math.abs(train1[0] - eval1[0]) > 1e-4);
        assertTrue(Math.abs(train3[0] - train1[0]) > 1e-4);
        layer.forward(x, true).sum().backward();
        assertEquals(4, layer.parameters().stream().filter(p -> p.grad() != null).count());
    }
}
