package io.github.zzuegg.brainiac.experimental.nn;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

import org.junit.jupiter.api.Test;

/** ODE solvers, neural ODE/CDE and liquid cells. */
class ContinuousTimeTest {

    /** dz/dt = A z with A a rotation generator: z(t) = R(t) z(0). */
    private static Var rotate(float t, Var z) {
        return z.matmul(Tensor.of(Shape.of(2, 2), 0, 1, -1, 0));
    }

    @Test
    void solversReachTheirOrderAndDopri5AdaptsItsSteps() {
        Var z0 = Var.of(Tensor.of(Shape.of(1, 2), 1, 0));
        double exactX = Math.cos(2);
        double exactY = Math.sin(2);
        double[] errors = new double[4];
        int i = 0;
        for (OdeSolver s : List.of(OdeSolver.euler(20), OdeSolver.euler(40), OdeSolver.rk4(10), OdeSolver.rk4(20))) {
            float[] z = s.integrate(ContinuousTimeTest::rotate, z0, 0, 2).value().toArray();
            errors[i++] = Math.hypot(z[0] - exactX, z[1] - exactY);
        }
        assertEquals(2, errors[0] / errors[1], 0.4);    // first order
        assertEquals(16, errors[2] / errors[3], 3);     // fourth order
        OdeSolver loose = OdeSolver.dopri5(1e-3f, 1e-5f);
        OdeSolver tight = OdeSolver.dopri5(1e-6f, 1e-8f);
        float[] a = loose.integrate(ContinuousTimeTest::rotate, z0, 0, 2).value().toArray();
        float[] b = tight.integrate(ContinuousTimeTest::rotate, z0, 0, 2).value().toArray();
        assertTrue(Math.hypot(b[0] - exactX, b[1] - exactY) < 1e-5);
        assertTrue(Math.hypot(a[0] - exactX, a[1] - exactY) < 1e-2);
        assertTrue(tight.acceptedSteps() > loose.acceptedSteps());
        // backwards in time returns to the start
        float[] back = tight.integrate(ContinuousTimeTest::rotate, Var.of(Tensor.of(Shape.of(1, 2), b)), 2, 0)
                .value().toArray();
        assertArrayEquals(new float[] {1, 0}, back, 1e-4f);
    }

    @Test
    void adjointGradientsMatchBackpropagationThroughTheSolver() {
        NeuralOde direct = new NeuralOde(3, 8, OdeSolver.rk4(20), 1f, NeuralOde.Gradient.DIRECT, new SplittableRandom(1));
        NeuralOde adjoint = new NeuralOde(3, 8, OdeSolver.rk4(20), 1f, NeuralOde.Gradient.ADJOINT,
                new SplittableRandom(1));
        Tensor x = Tensor.uniform(Shape.of(4, 3), -1, 1, new SplittableRandom(2));
        Tensor w = Tensor.uniform(Shape.of(4, 3), -1, 1, new SplittableRandom(3));
        Var xd = Var.param(x);
        Var xa = Var.param(x);
        Var yd = direct.forward(xd);
        Var ya = adjoint.forward(xa);
        assertArrayEquals(yd.value().toArray(), ya.value().toArray(), 1e-5f);
        yd.mul(w).sum().backward();
        ya.mul(w).sum().backward();
        assertArrayEquals(xd.grad().toArray(), xa.grad().toArray(), 1e-3f);
        for (int i = 0; i < 4; i++) {
            assertArrayEquals(direct.parameters().get(i).grad().toArray(), adjoint.parameters().get(i).grad().toArray(),
                    1e-3f, "parameter " + i);
        }
    }

    @Test
    void neuralOdeLearnsAFlowThatSeparatesNestedClasses() {
        // inner disk vs outer ring: a flow that moves them apart, then a linear readout
        SplittableRandom r = new SplittableRandom(4);
        NeuralOde ode = new NeuralOde(2, 16, OdeSolver.rk4(4), 1f, NeuralOde.Gradient.ADJOINT, r);
        Var readout = MixerOps.weight(r, 2, 2);
        Var bias = MixerOps.filled(0f, 2);
        List<Var> params = new ArrayList<>(ode.parameters());
        params.add(readout);
        params.add(bias);
        Optimizer opt = Optimizer.adam(0.03f);
        double acc = 0;
        for (int step = 0; step < 150; step++) {
            float[] pts = new float[64 * 2];
            float[] cls = new float[64];
            for (int i = 0; i < 64; i++) {
                boolean outer = i % 2 == 0;
                double radius = outer ? 1.5 + r.nextDouble() * 0.5 : r.nextDouble() * 0.7;
                double angle = r.nextDouble() * 2 * Math.PI;
                pts[2 * i] = (float) (radius * Math.cos(angle));
                pts[2 * i + 1] = (float) (radius * Math.sin(angle));
                cls[i] = outer ? 1 : 0;
            }
            Var logits = ode.forward(Var.of(Tensor.of(Shape.of(64, 2), pts))).matmul(readout).add(bias);
            Var loss = Functional.crossEntropy(logits, Tensor.of(Shape.of(64), cls));
            loss.backward();
            opt.step(params);
            float[] l = logits.value().toArray();
            int correct = 0;
            for (int i = 0; i < 64; i++) {
                correct += (l[2 * i + 1] > l[2 * i]) == (cls[i] == 1) ? 1 : 0;
            }
            acc = correct / 64.0;
        }
        assertTrue(acc > 0.9, "accuracy " + acc);
    }

    @Test
    void splinesInterpolateKnotsWithSmoothDerivatives() {
        Tensor obs = Tensor.of(Shape.of(1, 5, 1), 0, 1, 0, -1, 3);
        for (ControlPath.Interpolation kind : ControlPath.Interpolation.values()) {
            ControlPath path = new ControlPath(obs, kind);
            for (int i = 0; i < 5; i++) {
                assertEquals(obs.toArray()[i], path.value(i).item(), 1e-5f);
            }
            for (double s = 0.1; s < 3.9; s += 0.37) {
                double fd = (path.value(s + 1e-3).item() - path.value(s - 1e-3).item()) / 2e-3;
                assertEquals(fd, path.derivative(s).item(), 2e-2);
            }
        }
        ControlPath cubic = new ControlPath(obs, ControlPath.Interpolation.CUBIC);
        // continuous derivative at an inner knot
        assertEquals(cubic.derivative(2 - 1e-6).item(), cubic.derivative(2 + 1e-6).item(), 1e-3);
    }

    /** Irregularly sampled sines, slow (class 0) or fast (class 1); channels: time, value. */
    private static Tensor[] irregular(SplittableRandom r, int batch, int points) {
        float[] x = new float[batch * points * 2];
        float[] cls = new float[batch];
        for (int b = 0; b < batch; b++) {
            boolean fast = r.nextBoolean();
            cls[b] = fast ? 1 : 0;
            double freq = fast ? 2.5 : 0.8;
            double phase = r.nextDouble() * 2 * Math.PI;
            double t = 0;
            for (int i = 0; i < points; i++) {
                t += 0.1 + r.nextDouble() * 0.4;
                x[(b * points + i) * 2] = (float) (t / 3);
                x[(b * points + i) * 2 + 1] = (float) Math.sin(freq * t + phase);
            }
        }
        return new Tensor[] {Tensor.of(Shape.of(batch, points, 2), x), Tensor.of(Shape.of(batch), cls)};
    }

    @Test
    void neuralCdeClassifiesIrregularlySampledSeries() {
        SplittableRandom r = new SplittableRandom(5);
        NeuralCde cde = new NeuralCde(2, 8, 16, ControlPath.Interpolation.CUBIC, 2, r);
        Var readout = MixerOps.weight(r, 8, 2);
        List<Var> params = new ArrayList<>(cde.parameters());
        params.add(readout);
        Optimizer opt = Optimizer.adam(0.02f);
        for (int step = 0; step < 120; step++) {
            Tensor[] d = irregular(r, 16, 10);
            Var loss = Functional.crossEntropy(cde.forward(Var.of(d[0])).matmul(readout), d[1]);
            loss.backward();
            opt.step(params);
        }
        Tensor[] test = irregular(new SplittableRandom(6), 64, 10);
        float[] l = cde.forward(Var.of(test[0])).matmul(readout).value().toArray();
        float[] c = test[1].toArray();
        int correct = 0;
        for (int i = 0; i < 64; i++) {
            correct += (l[2 * i + 1] > l[2 * i]) == (c[i] == 1) ? 1 : 0;
        }
        assertTrue(correct / 64.0 > 0.85, "accuracy " + correct / 64.0);
    }

    @Test
    void liquidCellsForecastSines() {
        for (LiquidCell.Kind kind : LiquidCell.Kind.values()) {
            double mse = SineForecast.train(new LiquidCell(kind, 16, 16, new SplittableRandom(7)), 16, 16, 250,
                    0.01f, 8);
            assertTrue(mse < 0.05, kind + ": mse " + mse);
        }
    }

    @Test
    void ltcStaysBoundedAndCfcUsesTheElapsedTime() {
        LiquidCell ltc = new LiquidCell(LiquidCell.Kind.LTC, 2, 4, new SplittableRandom(9));
        float[] states = ltc.forward(Var.of(Tensor.uniform(Shape.of(2, 30, 2), -20, 20, new SplittableRandom(10))))
                .value().toArray();
        float bound = 0;
        for (float a : ltc.parameters().get(3).value().toArray()) {
            bound = Math.max(bound, Math.abs(a));
        }
        for (float v : states) {
            assertTrue(Math.abs(v) <= bound + 1e-4, "state " + v + " beyond " + bound);
        }
        LiquidCell cfc = new LiquidCell(LiquidCell.Kind.CFC, 2, 4, new SplittableRandom(11));
        Var x = Var.of(Tensor.uniform(Shape.of(1, 3, 2), -1, 1, new SplittableRandom(12)));
        float[] slow = cfc.forward(x, new float[][] {{1, 1, 1}}).value().toArray();
        float[] fast = cfc.forward(x, new float[][] {{1, 1, 0.1f}}).value().toArray();
        assertArrayEquals(java.util.Arrays.copyOf(slow, 8), java.util.Arrays.copyOf(fast, 8), 1e-6f);
        double diff = 0;
        for (int i = 8; i < 12; i++) {
            diff += Math.abs(slow[i] - fast[i]);
        }
        assertTrue(diff > 1e-3, "the gap changed nothing");
    }
}
