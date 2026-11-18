package io.github.zzuegg.brainiac.experimental.sciml;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

class SciMlBlocksTest {

    @Test
    void spectralHeatSolverMatchesExactDecay() {
        int n = 64;
        double[] u0 = new double[n];
        for (int j = 0; j < n; j++) {
            u0[j] = Math.sin(2 * Math.PI * j / n * 3);
        }
        PdeProblems.Solution s = PdeProblems.solve(PdeProblems.Equation.HEAT, u0, 0, 2 * Math.PI, 0.1, 1, 2, 0.01);
        double decay = Math.exp(-0.1 * 9);
        double[] last = s.last();
        for (int j = 0; j < n; j++) {
            assertEquals(u0[j] * decay, last[j], 1e-6);
        }
    }

    @Test
    void pinnDerivativesMatchFiniteDifferences() {
        Layer field = PhysicsInformedNetwork.tanhMlp(2, 8, 2, 1, 3);
        PhysicsInformedNetwork net = PhysicsInformedNetwork.of(field);
        double[] p = {0.3, -0.2};
        PhysicsInformedNetwork.Derivatives d = net.derivatives(Tensor.of(Shape.of(1, 2), new float[] {0.3f, -0.2f}));
        double h = 1e-2;
        double[] up = net.predict(new double[][] {{p[0] + h, p[1]}, {p[0] - h, p[1]}, {p[0], p[1]}});
        double fd1 = (up[0] - up[1]) / (2 * h);
        double fd2 = (up[0] - 2 * up[2] + up[1]) / (h * h);
        assertEquals(fd1, d.first(0).value().item(), 1e-3);
        assertEquals(fd2, d.second(0).value().item(), 2e-2);
    }

    @Test
    void leapfrogKeepsOscillatorEnergyBounded() {
        SymplecticIntegrator.Gradient h = (q, p) -> new double[][] {{q[0]}, {p[0]}};
        double[] q = {1};
        double[] p = {0};
        double worst = 0;
        for (int t = 0; t < 10_000; t++) {
            double[][] s = SymplecticIntegrator.leapfrog(h, q, p, 0.1);
            q = s[0];
            p = s[1];
            worst = Math.max(worst, Math.abs(0.5 * (q[0] * q[0] + p[0] * p[0]) - 0.5));
        }
        assertTrue(worst < 5e-3, "energy error " + worst);
    }

    @Test
    void sindyRecoversLorenzExactly() {
        MechanicalSystems.Ode lorenz = MechanicalSystems.lorenz63(10, 28, 8.0 / 3);
        double[][] x = MechanicalSystems.trajectory(lorenz, new double[] {1, 1, 20}, 0.01, 1000, 5);
        double[][] dx = new double[x.length][];
        for (int t = 0; t < x.length; t++) {
            dx[t] = lorenz.derivative(x[t]);
        }
        Sindy.Library lib = Sindy.polynomial(new String[] {"x", "y", "z"}, 2, false);
        Sindy model = Sindy.fit(x, dx, lib, 0.1);
        double[][] xi = model.coefficients();
        int nonZero = 0;
        for (double[] row : xi) {
            for (double v : row) {
                nonZero += v != 0 ? 1 : 0;
            }
        }
        assertEquals(7, nonZero, model.equations(new String[] {"x", "y", "z"}));
    }

    @Test
    void dmdFindsRotationEigenvalues() {
        double angle = 0.3;
        double[][] snaps = new double[20][];
        snaps[0] = new double[] {1, 0};
        for (int t = 1; t < snaps.length; t++) {
            double[] v = snaps[t - 1];
            snaps[t] = new double[] {0.95 * (Math.cos(angle) * v[0] - Math.sin(angle) * v[1]),
                0.95 * (Math.sin(angle) * v[0] + Math.cos(angle) * v[1])};
        }
        double[][] eig = DynamicModeDecomposition.dmd(snaps, 2).eigenvalues();
        for (double[] e : eig) {
            assertEquals(0.95, Math.hypot(e[0], e[1]), 1e-6);
            assertEquals(angle, Math.abs(Math.atan2(e[1], e[0])), 1e-6);
        }
    }

    @Test
    void equivariantBasisHasTheRightDimension() {
        EquivariantMlp.Representation so2 = EquivariantMlp.rotations2d();
        assertEquals(2, EquivariantMlp.basis(so2, so2, 1e-6).length);   // identity and the 90-degree turn
        EquivariantMlp.Representation so3 = EquivariantMlp.rotations3d(false);
        assertEquals(1, EquivariantMlp.basis(so3, so3, 1e-6).length);
        assertEquals(1, EquivariantMlp.basis(so3.times(so3), so3.scalar(), 1e-6).length);   // the dot product
        EquivariantMlp.Representation o3 = EquivariantMlp.rotations3d(true);
        assertEquals(0, EquivariantMlp.basis(o3, o3.scalar(), 1e-6).length);
    }

    @Test
    void frameAveragingIsInvariant() {
        SimpleMlp mlp = SimpleMlp.tanh(1, 12, 16, 1);
        Layer backbone = new Layer() {
            @Override
            public Var forward(Var input) {
                return mlp.forward(input);
            }
        };
        FrameAveraging fa = FrameAveraging.invariant(backbone, 4, false);
        SplittableRandom r = new SplittableRandom(2);
        float[] cloud = new float[12];
        for (int i = 0; i < 12; i++) {
            cloud[i] = (float) r.nextGaussian();
        }
        // rotate by 90 degrees about z and reflect x, then translate
        float[] moved = new float[12];
        for (int i = 0; i < 4; i++) {
            moved[3 * i] = -cloud[3 * i + 1] + 2;
            moved[3 * i + 1] = -cloud[3 * i];
            moved[3 * i + 2] = cloud[3 * i + 2] - 1;
        }
        float a = fa.forward(Var.of(Tensor.of(Shape.of(1, 12), cloud))).value().item();
        float b = fa.forward(Var.of(Tensor.of(Shape.of(1, 12), moved))).value().item();
        assertEquals(a, b, 1e-4);
    }

    @Test
    void schNetForcesAreEnergyGradients() {
        SchNet net = SchNet.of(8, 1, 8, 3.0, 4);
        float[] x = {0, 0, 0, 1.1f, 0.1f, 0, 0.2f, 1.0f, 0.3f};
        Var pos = Var.param(Tensor.of(Shape.of(1, 3, 3), x));
        float[] f = net.forces(pos, false).value().toArray();
        float h = 1e-2f;
        for (int k : new int[] {0, 4, 8}) {
            float[] xp = x.clone();
            float[] xm = x.clone();
            xp[k] += h;
            xm[k] -= h;
            float ep = net.energy(Var.of(Tensor.of(Shape.of(1, 3, 3), xp))).value().item();
            float em = net.energy(Var.of(Tensor.of(Shape.of(1, 3, 3), xm))).value().item();
            assertEquals(-(ep - em) / (2 * h), f[k], 2e-2);
        }
    }
}
