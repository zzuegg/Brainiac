package io.github.zzuegg.brainiac.experiments.exotic;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.BitLinear;
import io.github.zzuegg.brainiac.experimental.tensor.TernaryMatrix;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.SplittableRandom;

/**
 * BitNet b1.58 at toy scale: the same small classifier trained three ways on a nonlinear
 * 2-D task (interleaved spirals): in full precision, in full precision and then
 * ternarized after training (post-training quantization), and with quantization-aware
 * training through {@code BitLinear} layers. QAT recovers most of the float accuracy that
 * naive ternarization loses, and the QAT model runs on packed {@code TernaryMatrix}
 * weights with additions only, giving the same predictions.
 */
public final class BitNetExperiment {
    private BitNetExperiment() {
    }

    /**
     * Two interleaved spirals.
     *
     * @param n points
     * @param random the random source
     * @param labels receives the class of each point
     * @return {@code [n, 2]}
     */
    public static Tensor spirals(int n, SplittableRandom random, float[] labels) {
        float[] x = new float[n * 2];
        for (int i = 0; i < n; i++) {
            int c = i % 2;
            double t = 0.5 + 2.5 * random.nextDouble();
            double a = t * 2.2 + c * Math.PI;
            x[2 * i] = (float) (t * Math.cos(a) / 3 + 0.03 * random.nextGaussian());
            x[2 * i + 1] = (float) (t * Math.sin(a) / 3 + 0.03 * random.nextGaussian());
            labels[i] = c;
        }
        return Tensor.of(Shape.of(n, 2), x);
    }

    /**
     * A classifier with two hidden layers, either {@code BitLinear} or float {@code Dense}.
     *
     * @param ternary whether the hidden layers are BitLinear
     * @param seed the seed
     * @return the model
     */
    public static Model model(boolean ternary, long seed) {
        Model.Builder b = Model.sequential().input(2).dense(64, Activation.TANH);   // full-precision input layer
        if (ternary) {
            // separate activation instances: Model.layers() lists each distinct layer once
            b.add(BitLinear.spec(64)).activation(Activation.named("relu", Var::relu))
                    .add(BitLinear.spec(64)).activation(Activation.named("relu", Var::relu));
        } else {
            b.dense(64, Activation.RELU).dense(64, Activation.RELU);
        }
        return b.dense(2).seed(seed).build();
    }

    /**
     * Trains a model on the spirals.
     *
     * @param model the model
     * @param steps Adam steps on batches of 128
     * @param seed the data seed
     */
    public static void train(Model model, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        Optimizer opt = Optimizer.adam(5e-3f);
        for (int s = 0; s < steps; s++) {
            float[] y = new float[128];
            Tensor x = spirals(128, r, y);
            Var loss = Functional.crossEntropy(model.forward(Var.of(x), true), Tensor.of(Shape.of(128), y));
            loss.backward();
            opt.step(model.parameters());
        }
    }

    /**
     * Classification accuracy.
     *
     * @param model the model
     * @param x inputs
     * @param y labels
     * @return the share correct
     */
    public static double accuracy(Model model, Tensor x, float[] y) {
        float[] pred = model.forward(Var.of(x)).value().argmax(1).toArray();
        int right = 0;
        for (int i = 0; i < y.length; i++) {
            right += Math.round(pred[i]) == Math.round(y[i]) ? 1 : 0;
        }
        return (double) right / y.length;
    }

    /**
     * Post-training ternarization: replaces every hidden Dense weight matrix by its
     * absmean ternary version.
     *
     * @param model a trained float model
     */
    public static void ternarizeHidden(Model model) {
        int dense = 0;
        for (Layer l : model.layers()) {
            if (l instanceof Dense d) {
                dense++;
                if (dense == 2 || dense == 3) {
                    d.weight().assign(BitLinear.quantizeWeights(d.weight().value()));
                }
            }
        }
    }

    /**
     * Runs a trained BitLinear model with its hidden layers on packed ternary weights.
     *
     * @param model a model built by {@code model(true, ...)}
     * @param x inputs {@code [n, 2]}
     * @return the logits {@code [n, 2]} as a flat array
     */
    public static float[] packedForward(Model model, Tensor x) {
        int n = x.shape().dim(0);
        Var h = Var.of(x);
        for (Layer l : model.layers()) {
            if (l instanceof BitLinear b) {
                TernaryMatrix m = b.pack();
                h = Var.of(Tensor.of(Shape.of(n, m.cols()), m.matmul(h.value().toArray(), n)));
            } else {
                h = l.forward(h);
            }
        }
        return h.value().toArray();
    }
}
