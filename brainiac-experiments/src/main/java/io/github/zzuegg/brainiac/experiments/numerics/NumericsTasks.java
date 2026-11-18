package io.github.zzuegg.brainiac.experiments.numerics;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experimental.train.KroneckerCapture;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

/**
 * Shared glue for the numerics experiments: a {@link SimpleMlp} forward pass routed through
 * {@link KroneckerCapture} hooks, minibatch sampling, and the small evaluation helpers every
 * shootout uses.
 */
public final class NumericsTasks {

    private NumericsTasks() {
    }

    /**
     * The network's logits, with every dense layer recorded by the capture.
     *
     * @param net the network
     * @param x the inputs
     * @param capture the hooks, or {@code null} for a plain forward pass
     * @return the logits
     */
    public static Var forward(SimpleMlp net, Var x, KroneckerCapture capture) {
        Var h = x;
        for (int l = 0; l < net.depth(); l++) {
            Var z = capture == null ? net.linear(l, h) : capture.dense(net.weight(l), net.bias(l), h);
            h = l < net.depth() - 1 ? net.activation().apply(z) : z;
        }
        return h;
    }

    /**
     * A random minibatch, drawn with replacement.
     *
     * @param data the data
     * @param size the batch size
     * @param random the random source
     * @return {@code {inputs, ids}}
     */
    public static Tensor[] batch(ToyTasks.Data data, int size, SplittableRandom random) {
        int f = data.features();
        float[] x = new float[size * f];
        float[] y = new float[size];
        for (int i = 0; i < size; i++) {
            int k = random.nextInt(data.size());
            for (int j = 0; j < f; j++) {
                x[i * f + j] = data.x()[k][j];
            }
            y[i] = data.y()[k];
        }
        return new Tensor[] {Tensor.of(Shape.of(size, f), x), Tensor.of(Shape.of(size), y)};
    }

    /**
     * The mean cross-entropy of a network on data, without a graph.
     *
     * @param net the network
     * @param data the data
     * @return the loss
     */
    public static double loss(SimpleMlp net, ToyTasks.Data data) {
        return Var.noGrad(() -> Functional.crossEntropy(net.forward(Var.of(data.inputs())), data.ids())).value().item();
    }

    /**
     * @param net the network
     * @param data the data
     * @return the accuracy of the network on the data
     */
    public static double accuracy(SimpleMlp net, ToyTasks.Data data) {
        return ToyTasks.accuracy(Var.noGrad(() -> net.forward(Var.of(data.inputs()))).value(), data.y());
    }

    /**
     * @param values some numbers
     * @return their mean
     */
    public static double mean(double[] values) {
        double s = 0;
        for (double v : values) {
            s += v;
        }
        return s / values.length;
    }
}
