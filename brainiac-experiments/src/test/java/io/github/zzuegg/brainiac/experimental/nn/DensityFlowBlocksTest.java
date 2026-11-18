package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.SplittableRandom;

import static org.junit.jupiter.api.Assertions.assertEquals;

class DensityFlowBlocksTest {

    /** Perturbs every parameter so the flows are not the identity, then checks the round trip. */
    private static void roundTrip(DensityFlow flow, String name) {
        SplittableRandom r = new SplittableRandom(5);
        Tensor x = Tensor.normal(Shape.of(16, flow.dimension()), 0, 1, r);
        flow.logProb(Var.of(x));   // data-dependent initialisation, if any
        for (Var p : flow.parameters()) {
            p.assign(p.value().add(Tensor.normal(p.value().shape(), 0, 0.2f, r)));
        }
        Tensor z = flow.forward(Var.of(x)).output().value();
        float[] back = flow.inverse(z).toArray();
        float[] orig = x.toArray();
        for (int i = 0; i < orig.length; i++) {
            assertEquals(orig[i], back[i], 2e-3, name + " inverse at " + i);
        }
    }

    @Test
    void flowsInvertExactly() {
        roundTrip(GlowFlow.of(4, 3, 16, 1), "Glow");
        roundTrip(SplineFlow.of(4, 3, 6, 3f, 16, 1), "spline");
        roundTrip(MaskedAutoregressiveFlow.of(4, 3, 16, 1), "MAF");
    }

    @Test
    void splineLogDeterminantMatchesFiniteDifferences() {
        SplineFlow flow = SplineFlow.of(2, 1, 5, 3f, 8, 3);
        SplittableRandom r = new SplittableRandom(2);
        for (Var p : flow.parameters()) {
            p.assign(p.value().add(Tensor.normal(p.value().shape(), 0, 0.5f, r)));
        }
        float[] x = {0.3f, -0.7f};
        float ld = flow.forward(Var.of(Tensor.of(Shape.of(1, 2), x))).logDet().value().item();
        float h = 1e-3f;
        float[] up = flow.forward(Var.of(Tensor.of(Shape.of(1, 2), x[0], x[1] + h))).output().value().toArray();
        float[] dn = flow.forward(Var.of(Tensor.of(Shape.of(1, 2), x[0], x[1] - h))).output().value().toArray();
        // mask {1, 0}: feature 0 passes, feature 1 is splined, so the Jacobian is triangular
        double numeric = Math.log((up[1] - dn[1]) / (2 * h));
        assertEquals(numeric, ld, 1e-2);
    }

    @Test
    void continuousFlowLikelihoodOfALinearFieldIsExact() {
        // dz/dt = -z/2 maps N(0, e) to N(0, 1) at t = 1
        ContinuousFlow flow = ContinuousFlow.of((z, t) -> z.mul(-0.5f), List.of(), 2, OdeSolver.rk4(20), 1);
        float[] x = {0.4f, -1.2f};
        double lp = flow.logProb(Var.of(Tensor.of(Shape.of(1, 2), x))).value().item();
        double var = Math.E;
        double exact = -Math.log(2 * Math.PI * var) - (x[0] * x[0] + x[1] * x[1]) / (2 * var);
        assertEquals(exact, lp, 1e-3);
        flow.trace(ContinuousFlow.Trace.HUTCHINSON);
        double hutch = flow.logProb(Var.of(Tensor.of(Shape.of(1, 2), x))).value().item();
        assertEquals(exact, hutch, 1.5, "Hutchinson is unbiased but noisy");
        float[] back = flow.inverse(flow.forward(Var.of(Tensor.of(Shape.of(1, 2), x))).output().value()).toArray();
        assertEquals(x[0], back[0], 1e-3);
    }
}
