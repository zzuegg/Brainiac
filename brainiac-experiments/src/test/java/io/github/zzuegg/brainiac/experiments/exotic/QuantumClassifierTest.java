package io.github.zzuegg.brainiac.experiments.exotic;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.QuantumCircuitLayer;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import org.junit.jupiter.api.Test;

/** Wave 6, 6.3: a simulated variational quantum classifier against random Fourier features. */
class QuantumClassifierTest {

    @Test
    void parameterShiftMatchesFiniteDifferences() {
        QuantumCircuitLayer c = QuantumCircuitLayer.of(2, 3, 2, true, 3);
        Tensor x = Tensor.of(Shape.of(1, 2), new float[] {0.4f, -1.1f});
        Var out = c.forward(Var.of(x)).sum();
        out.backward();
        float[] g = c.weights().grad().toArray();
        float[] t = c.weights().value().toArray();
        double eps = 1e-3;
        for (int i = 0; i < 4; i++) {
            float[] p = t.clone();
            p[i] += (float) eps;
            c.weights().assign(Tensor.of(c.weights().shape(), p));
            double up = c.forward(Var.of(x)).sum().value().item();
            p[i] -= (float) (2 * eps);
            c.weights().assign(Tensor.of(c.weights().shape(), p));
            double down = c.forward(Var.of(x)).sum().value().item();
            c.weights().assign(Tensor.of(c.weights().shape(), t));
            assertEquals((up - down) / (2 * eps), g[i], 2e-2, "weight " + i);
        }
    }

    @Test
    void circuitClassifiesMoonsAndCircles() {
        ToyTasks.Data train = QuantumClassifier.moons(80, 0.1, 1);
        ToyTasks.Data test = QuantumClassifier.moons(200, 0.1, 2);
        QuantumClassifier.Result small = QuantumClassifier.quantum(train, test, 2, 1, 0, 40, 1);
        QuantumClassifier.Result deep = QuantumClassifier.quantum(train, test, 4, 3, 0, 40, 1);
        QuantumClassifier.Result noisy = QuantumClassifier.quantum(train, test, 4, 3, 50, 40, 1);
        QuantumClassifier.Result rff = QuantumClassifier.fourierFeatures(train, test, deep.parameters(), 0.7, 200, 1);
        ToyTasks.Data ctrain = QuantumClassifier.circles(80, 0.08, 3);
        ToyTasks.Data ctest = QuantumClassifier.circles(200, 0.08, 4);
        QuantumClassifier.Result circle = QuantumClassifier.quantum(ctrain, ctest, 4, 3, 0, 40, 1);
        QuantumClassifier.Result crff = QuantumClassifier.fourierFeatures(ctrain, ctest, circle.parameters(), 0.7,
                200, 1);
        double v1 = QuantumClassifier.gradientVariance(6, 1, 30, 1);
        double v4 = QuantumClassifier.gradientVariance(6, 4, 30, 1);
        double v16 = QuantumClassifier.gradientVariance(6, 16, 30, 1);
        double q2 = QuantumClassifier.gradientVariance(2, 12, 30, 1);
        double q8 = QuantumClassifier.gradientVariance(8, 12, 30, 1);
        System.out.printf("VQC moons acc: 2q1L %.2f, 4q3L %.2f, 4q3L 50 shots %.2f vs RFF(%d params) %.2f;"
                + " circles 4q3L %.2f vs RFF %.2f; barren plateau mean Var[dE/dθ] 6q depth 1/4/16:"
                + " %.4f/%.4f/%.4f, depth 12 at 2q/8q: %.4f/%.4f%n",
                small.accuracy(), deep.accuracy(), noisy.accuracy(), rff.parameters(), rff.accuracy(),
                circle.accuracy(), crff.accuracy(), v1, v4, v16, q2, q8);
        assertTrue(deep.accuracy() > 0.8, "moons " + deep.accuracy());
        assertTrue(circle.accuracy() > 0.75, "circles " + circle.accuracy());
        assertTrue(noisy.accuracy() > 0.7, "shot noise still trains " + noisy.accuracy());
        assertTrue(v16 < v1, "gradients shrink with depth");
        assertTrue(q8 < q2 / 2, "gradients vanish with width");
    }
}
