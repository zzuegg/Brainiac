package io.github.zzuegg.brainiac.experiments.exotic;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.HyperDense;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A hypernetwork (Ha et al., <a href="https://arxiv.org/abs/1609.09106">1609.09106</a>) that
 * writes a whole regression network per task: given a task's code
 * {@code (A cos phi, A sin phi)}, three {@link HyperDense} layers generate a 1-16-16-1 MLP
 * fitting {@code y = A sin(x + phi)}. Trained on random tasks, it produces working networks
 * for amplitudes and phases it never saw.
 */
@SuppressWarnings("exports")   // experimental.* packages are exported by the lead at integration
public final class HyperSines {
    private final HyperDense first;
    private final HyperDense second;
    private final HyperDense last;
    private final Optimizer optimizer;
    private final SplittableRandom random;

    private HyperSines(float learningRate, long seed) {
        this.first = HyperDense.of(2, 1, 16, 32, seed);
        this.second = HyperDense.of(2, 16, 16, 32, seed + 1);
        this.last = HyperDense.of(2, 16, 1, 32, seed + 2);
        this.optimizer = Optimizer.adam(learningRate);
        this.random = new SplittableRandom(seed + 3);
    }

    /**
     * Creates the experiment.
     *
     * @param learningRate the Adam learning rate
     * @param seed the seed
     * @return the experiment
     */
    public static HyperSines of(float learningRate, long seed) {
        return new HyperSines(learningRate, seed);
    }

    /**
     * Runs the generated networks.
     *
     * @param code task codes {@code [batch, 2]}
     * @param x inputs {@code [batch, 1]}
     * @return predictions {@code [batch, 1]}
     */
    public Var predict(Tensor code, Tensor x) {
        Var z = Var.of(code);
        Var h = first.forward(z, Var.of(x)).relu();
        h = second.forward(z, h).relu();
        return last.forward(z, h);
    }

    /**
     * Builds a batch.
     *
     * @param tasks {@code {amplitude, phase}} per task
     * @param points points per task
     * @param r the random source for the x positions
     * @return code, x and y tensors
     */
    public static Tensor[] batch(List<double[]> tasks, int points, SplittableRandom r) {
        int n = tasks.size() * points;
        float[] code = new float[2 * n];
        float[] x = new float[n];
        float[] y = new float[n];
        int i = 0;
        for (double[] t : tasks) {
            for (int p = 0; p < points; p++, i++) {
                code[2 * i] = (float) (t[0] * Math.cos(t[1]));
                code[2 * i + 1] = (float) (t[0] * Math.sin(t[1]));
                x[i] = (float) r.nextDouble(-Math.PI, Math.PI);
                y[i] = (float) (t[0] * Math.sin(x[i] + t[1]));
            }
        }
        return new Tensor[] {Tensor.of(Shape.of(n, 2), code), Tensor.of(Shape.of(n, 1), x), Tensor.of(Shape.of(n, 1), y)};
    }

    /**
     * Draws a random task.
     *
     * @param r the random source
     * @return {@code {amplitude in [0.5, 2], phase in [0, 2 pi)}}
     */
    public static double[] task(SplittableRandom r) {
        return new double[] {r.nextDouble(0.5, 2), r.nextDouble(0, 2 * Math.PI)};
    }

    /**
     * Trains on fresh random tasks.
     *
     * @param steps optimizer steps
     * @return the last loss
     */
    public float train(int steps) {
        float last = Float.NaN;
        for (int s = 0; s < steps; s++) {
            List<double[]> tasks = new ArrayList<>();
            for (int t = 0; t < 16; t++) {
                tasks.add(task(random));
            }
            Tensor[] b = batch(tasks, 16, random);
            Var loss = predict(b[0], b[1]).sub(Var.of(b[2])).square().mean();
            last = loss.value().item();
            loss.backward();
            List<Var> params = new ArrayList<>(first.parameters());
            params.addAll(second.parameters());
            params.addAll(this.last.parameters());
            optimizer.step(params);
        }
        return last;
    }

    /**
     * Mean squared error on given tasks.
     *
     * @param tasks the tasks
     * @param seed the seed for x positions
     * @return the MSE
     */
    public float error(List<double[]> tasks, long seed) {
        Tensor[] b = batch(tasks, 64, new SplittableRandom(seed));
        return predict(b[0], b[1]).value().sub(b[2]).square().mean().item();
    }
}
