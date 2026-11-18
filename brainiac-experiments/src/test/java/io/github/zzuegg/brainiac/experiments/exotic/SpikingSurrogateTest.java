package io.github.zzuegg.brainiac.experiments.exotic;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.autograd.AdaptiveSurrogateSpike;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experiments.exotic.SpikingClassifier.SpikeLoss;
import io.github.zzuegg.brainiac.experiments.exotic.SpikingClassifier.Surrogate;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import org.junit.jupiter.api.Test;

class SpikingSurrogateTest {

    @Test
    void rectangularSurrogateHasUnitMassInItsWindow() {
        AdaptiveSurrogateSpike op = AdaptiveSurrogateSpike.fixed(0.5f, 0.5f);
        Var v = Var.param(Tensor.of(Shape.of(4), -0.3f, -0.2f, 0.7f, 0.8f));
        Var s = op.spike(v);
        s.sum().backward();
        float[] g = v.grad().toArray();
        // window [-0.25, 0.75], height 1
        assertTrue(g[0] == 0 && g[1] == 1 && g[2] == 1 && g[3] == 0, java.util.Arrays.toString(g));
    }

    @Test
    void deeperNetworksAndGradientMismatch() {
        double base = SpikingClassifier.direct(4, SpikeLoss.MEAN_OUTPUT, Surrogate.FAST_SIGMOID, 3, 150, 1);
        double tet = SpikingClassifier.direct(4, SpikeLoss.TET, Surrogate.FAST_SIGMOID, 3, 150, 1);
        double grid = SpikingClassifier.direct(4, SpikeLoss.MEAN_OUTPUT, Surrogate.GRID, 3, 150, 1);
        double gridTet = SpikingClassifier.direct(4, SpikeLoss.TET, Surrogate.GRID, 3, 150, 1);
        System.out.printf("SNN direct T 4 (3 hidden): mean-output fast-sigmoid %.3f  TET %.3f  A2SG grid %.3f  "
                + "TET + grid %.3f%n", base, tet, grid, gridTet);
        float[] widths = {0.1f, 0.25f, 0.5f, 1f};
        double[] cos = SpikingClassifier.gradientCosine(4, 2, widths, 24, 0.5f, 1);
        System.out.printf("SNN gradient cosine vs finite differences (2 hidden, T 4): fast-sigmoid %.3f  "
                + "rect 0.1 %.3f  rect 0.25 %.3f  rect 0.5 %.3f  rect 1.0 %.3f%n", cos[0], cos[1], cos[2], cos[3],
                cos[4]);
        assertTrue(base > 0.5, "deeper net learns: " + base);
        assertTrue(cos[0] > 0, "the surrogate gradient points along the finite differences");
    }
}
