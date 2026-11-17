package io.github.zzuegg.brainiac.experiments.exotic;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.AnalogDense;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.SplittableRandom;

/**
 * Hardware-aware training for analog in-memory inference (Joshi et al.,
 * <a href="https://arxiv.org/abs/1906.03138">1906.03138</a>): the same spirals classifier
 * built from {@link AnalogDense} layers is trained once with weight-noise injection and once
 * without, then both are programmed onto many simulated noisy chips. The noise-aware network
 * keeps most of its accuracy; the conventionally trained one, with its sharper and larger
 * weights, degrades much more.
 */
public final class AnalogNoiseAwareTraining {
    private AnalogNoiseAwareTraining() {
    }

    /**
     * A spirals classifier of three analog layers with 6-bit ADCs.
     *
     * @param trainingNoise the weight noise injected while training, relative to the weight limit
     * @param seed the seed
     * @return the model
     */
    public static Model model(float trainingNoise, long seed) {
        return Model.sequential().input(2)
                .add(AnalogDense.spec(64, 1, trainingNoise, 6, 8)).activation(Activation.named("tanh", Var::tanh))
                .add(AnalogDense.spec(64, 1, trainingNoise, 6, 8)).activation(Activation.named("relu", Var::relu))
                .add(AnalogDense.spec(2, 1, trainingNoise, 6, 8))
                .seed(seed).build();
    }

    /**
     * Trains on spirals.
     *
     * @param model the model
     * @param steps Adam steps
     * @param seed the data seed
     */
    public static void train(Model model, int steps, long seed) {
        BitNetExperiment.train(model, steps, seed);
    }

    /**
     * Mean accuracy over simulated chips.
     *
     * @param model the trained model
     * @param noise the programming noise relative to the weight limit
     * @param chips how many chip instances to average
     * @param seed the seed for test data and chips
     * @return the mean accuracy
     */
    public static double chipAccuracy(Model model, float noise, int chips, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[] y = new float[500];
        Tensor x = BitNetExperiment.spirals(500, r, y);
        double sum = 0;
        for (int c = 0; c < chips; c++) {
            SplittableRandom chip = r.split();
            for (Layer l : model.layers()) {
                if (l instanceof AnalogDense a) {
                    a.program(noise, chip);
                }
            }
            sum += BitNetExperiment.accuracy(model, x, y);
        }
        for (Layer l : model.layers()) {
            if (l instanceof AnalogDense a) {
                a.unprogram();
            }
        }
        return sum / chips;
    }
}
