package io.github.zzuegg.brainiac.experiments.exotic;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.QuantumCircuitLayer;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;

/**
 * A simulated variational quantum classifier ({@link QuantumCircuitLayer}) on two moons and
 * two circles, against random Fourier features (Rahimi and Recht 2007) with a logistic
 * read-out at the same number of trained parameters, the classical kernel baseline that
 * "dequantization" results say a small quantum model should be compared with.
 *
 * <p>The circuit sees the standardized point {@code (x, y)} as rotation angles (re-uploaded
 * before every layer), its {@code <Z_i>} expectations go through a two-class linear head,
 * and everything trains with Adam on the parameter-shift gradients, optionally with a finite
 * number of shots per expectation. {@link #gradientVariance} reproduces the barren-plateau
 * measurement of McClean et al. (<a href="https://arxiv.org/abs/1803.11173">arXiv:1803.11173</a>):
 * the variance over random initializations of one gradient component of {@code <Z_0>},
 * which falls as the circuit deepens and widens.
 */
public final class QuantumClassifier {

    /**
     * One trained model.
     *
     * @param model "vqc" or "rff"
     * @param qubits qubits (0 for RFF)
     * @param layers ansatz layers (RFF: the number of random features)
     * @param shots shots per expectation (0 exact)
     * @param parameters trained parameters
     * @param accuracy held-out accuracy
     */
    public record Result(String model, int qubits, int layers, int shots, int parameters, double accuracy) {
    }

    private QuantumClassifier() {
    }

    /**
     * Two interleaving half circles with Gaussian noise, standardized.
     *
     * @param n points
     * @param noise noise standard deviation
     * @param seed the seed
     * @return the data, two classes
     */
    public static ToyTasks.Data moons(int n, double noise, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[][] x = new float[n][];
        int[] y = new int[n];
        for (int i = 0; i < n; i++) {
            int c = i % 2;
            double a = r.nextDouble() * Math.PI;
            double px = c == 0 ? Math.cos(a) : 1 - Math.cos(a);
            double py = c == 0 ? Math.sin(a) : 0.5 - Math.sin(a);
            x[i] = new float[] {(float) ((px - 0.5 + r.nextGaussian() * noise) / 0.87),
                (float) ((py - 0.25 + r.nextGaussian() * noise) / 0.5)};
            y[i] = c;
        }
        return new ToyTasks.Data(x, y, 2);
    }

    /**
     * Two concentric circles with Gaussian noise.
     *
     * @param n points
     * @param noise noise standard deviation
     * @param seed the seed
     * @return the data, two classes (inner circle radius 0.5, outer 1)
     */
    public static ToyTasks.Data circles(int n, double noise, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[][] x = new float[n][];
        int[] y = new int[n];
        for (int i = 0; i < n; i++) {
            int c = i % 2;
            double a = r.nextDouble() * 2 * Math.PI;
            double rad = c == 0 ? 1 : 0.5;
            x[i] = new float[] {(float) (rad * Math.cos(a) + r.nextGaussian() * noise) * 1.4f,
                (float) (rad * Math.sin(a) + r.nextGaussian() * noise) * 1.4f};
            y[i] = c;
        }
        return new ToyTasks.Data(x, y, 2);
    }

    /**
     * Trains the quantum classifier.
     *
     * @param train training points
     * @param test held-out points
     * @param qubits qubits
     * @param layers ansatz layers
     * @param shots shots per expectation, 0 for exact
     * @param steps full-batch Adam steps
     * @param seed the seed
     * @return the result
     */
    public static Result quantum(ToyTasks.Data train, ToyTasks.Data test, int qubits, int layers, int shots,
            int steps, long seed) {
        QuantumCircuitLayer circuit = QuantumCircuitLayer.of(2, qubits, layers, true, seed).shots(shots);
        SplittableRandom r = new SplittableRandom(seed + 1);
        Var head = Var.param(Tensor.of(Shape.of(qubits, 2), gaussian(qubits * 2, 0.5, r)));
        Var bias = Var.param(Tensor.zeros(Shape.of(2)));
        List<Var> params = List.of(circuit.weights(), head, bias);
        Optimizer opt = Optimizer.adam(0.1f);
        Tensor x = train.inputs();
        Tensor ids = train.ids();
        for (int s = 0; s < steps; s++) {
            Var loss = Functional.crossEntropy(circuit.forward(Var.of(x)).matmul(head).add(bias), ids);
            loss.backward();
            opt.step(params);
        }
        circuit.shots(0);
        Tensor scores = circuit.forward(Var.of(test.inputs())).matmul(head).add(bias).value();
        int p = qubits * layers * 2 + qubits * 2 + 2;
        return new Result("vqc", qubits, layers, shots, p, ToyTasks.accuracy(scores, test.y()));
    }

    /**
     * Random Fourier features {@code sqrt(2/D) cos(Wx + b)} of an RBF kernel with a trained
     * logistic (softmax) read-out.
     *
     * @param train training points
     * @param test held-out points
     * @param parameters the parameter budget; {@code D = parameters / 2 - 1} features
     * @param bandwidth the RBF length scale
     * @param steps full-batch Adam steps
     * @param seed the seed
     * @return the result
     */
    public static Result fourierFeatures(ToyTasks.Data train, ToyTasks.Data test, int parameters, double bandwidth,
            int steps, long seed) {
        int d = Math.max(1, parameters / 2 - 1);
        SplittableRandom r = new SplittableRandom(seed);
        Tensor w = Tensor.of(Shape.of(2, d), gaussian(2 * d, 1 / bandwidth, r));
        float[] phase = new float[d];
        for (int i = 0; i < d; i++) {
            phase[i] = (float) (r.nextDouble() * 2 * Math.PI);
        }
        Tensor b = Tensor.of(Shape.of(d), phase);
        float scale = (float) Math.sqrt(2.0 / d);
        Var head = Var.param(Tensor.zeros(Shape.of(d, 2)));
        Var bias = Var.param(Tensor.zeros(Shape.of(2)));
        Optimizer opt = Optimizer.adam(0.1f);
        Tensor features = Var.of(train.inputs()).matmul(Var.of(w)).add(b).cos().mul(scale).value();
        for (int s = 0; s < steps; s++) {
            Var loss = Functional.crossEntropy(Var.of(features).matmul(head).add(bias), train.ids());
            loss.backward();
            opt.step(List.of(head, bias));
        }
        Tensor tf = Var.of(test.inputs()).matmul(Var.of(w)).add(b).cos().mul(scale).value();
        Tensor scores = Var.of(tf).matmul(head).add(bias).value();
        return new Result("rff", 0, d, 0, 2 * d + 2, ToyTasks.accuracy(scores, test.y()));
    }

    /**
     * The barren-plateau measurement: the variance, over random initial angles, of the
     * gradient of {@code <Z_0>} with respect to each circuit weight at a fixed input, averaged
     * over the weights.
     *
     * @param qubits qubits
     * @param layers ansatz layers
     * @param inits random initializations
     * @param seed the seed
     * @return the mean gradient variance
     */
    public static double gradientVariance(int qubits, int layers, int inits, long seed) {
        int n = qubits * layers * 2;
        double[] sum = new double[n];
        double[] sq = new double[n];
        Tensor x = Tensor.of(Shape.of(1, 2), new float[] {0.3f, -0.7f});
        for (int i = 0; i < inits; i++) {
            QuantumCircuitLayer c = QuantumCircuitLayer.of(2, qubits, layers, false, seed + 7919L * i);
            Var out = c.forward(Var.of(x)).slice(1, 0, 1).sum();
            out.backward();
            float[] g = c.weights().grad().toArray();
            for (int k = 0; k < n; k++) {
                sum[k] += g[k];
                sq[k] += g[k] * g[k];
            }
        }
        double v = 0;
        for (int k = 0; k < n; k++) {
            double mean = sum[k] / inits;
            v += sq[k] / inits - mean * mean;
        }
        return v / n;
    }

    private static float[] gaussian(int n, double sd, SplittableRandom r) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
            v[i] = (float) (r.nextGaussian() * sd);
        }
        return v;
    }
}
