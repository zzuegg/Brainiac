package io.github.zzuegg.brainiac.experiments.exotic;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.FractalBlock;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

/**
 * FractalNet's drop-path claim (Larsson et al., <a href="https://arxiv.org/abs/1605.07648">1605.07648</a>)
 * on the two-spirals task: a dense {@link FractalBlock} trained with drop-path yields columns
 * that each classify well on their own, from the one-layer column to the deepest; trained
 * without it, the columns co-adapt and individually fall apart.
 */
public final class FractalDropPath {
    private FractalDropPath() {
    }

    /**
     * A spirals classifier around one fractal block of dense ReLU units.
     *
     * @param columns the block's columns
     * @param localDrop the local drop-path probability
     * @param globalShare the share of global drop-path batches
     * @param seed the seed
     * @return the model
     */
    public static Model model(int columns, double localDrop, double globalShare, long seed) {
        return Model.sequential().input(2).dense(32, Activation.TANH)
                .add(FractalBlock.spec(columns, Dense.spec(32, Activation.RELU, Init.HE), localDrop, globalShare))
                .dense(2).seed(seed).build();
    }

    /**
     * Finds the fractal block of a model.
     *
     * @param model a model from {@link #model}
     * @return its block
     */
    public static FractalBlock block(Model model) {
        for (Layer l : model.layers()) {
            if (l instanceof FractalBlock f) {
                return f;
            }
        }
        throw new IllegalArgumentException("no fractal block");
    }

    /**
     * Accuracy of each column alone, then of the full joined block.
     *
     * @param model the trained model
     * @param n test points
     * @param seed the test seed
     * @return {@code columns + 1} accuracies; the last is the full block
     */
    public static double[] columnAccuracies(Model model, int n, long seed) {
        float[] y = new float[n];
        Tensor x = BitNetExperiment.spirals(n, new SplittableRandom(seed), y);
        FractalBlock f = block(model);
        double[] out = new double[f.columns() + 1];
        for (int c = 1; c <= f.columns(); c++) {
            f.useColumn(c);
            out[c - 1] = BitNetExperiment.accuracy(model, x, y);
        }
        f.useColumn(0);
        out[f.columns()] = BitNetExperiment.accuracy(model, x, y);
        return out;
    }

    /**
     * Trains on spirals (delegates to the BitNet experiment's training loop).
     *
     * @param model the model
     * @param steps Adam steps
     * @param seed the data seed
     */
    public static void train(Model model, int steps, long seed) {
        BitNetExperiment.train(model, steps, seed);
    }

    /**
     * Runs a forward pass, for direct use.
     *
     * @param model the model
     * @param x inputs
     * @return logits
     */
    public static Tensor logits(Model model, Tensor x) {
        return model.forward(Var.of(x)).value();
    }
}
