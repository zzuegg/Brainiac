package io.github.zzuegg.brainiac.examples;

import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.train.Optimizer;
import io.github.zzuegg.brainiac.train.Progress;
import io.github.zzuegg.brainiac.train.Trainer;
import io.github.zzuegg.brainiac.train.TrainingResult;

/**
 * The classic first network: learn XOR, which no single straight line can separate.
 * Two inputs, one hidden layer of 8 tanh units, one sigmoid output.
 */
public final class XorExample {

    private XorExample() {
    }

    /** The four XOR cases. */
    public static final Dataset DATA = Dataset.of(
            new float[][] {{0, 0}, {0, 1}, {1, 0}, {1, 1}},
            new float[][] {{0}, {1}, {1}, {0}});

    /**
     * Builds the model.
     *
     * @return an untrained model
     */
    public static Model model() {
        return Model.sequential()
                .input(2)
                .dense(8, Activation.TANH)
                .dense(1, Activation.SIGMOID)
                .seed(42)
                .build();
    }

    /**
     * Trains a model on XOR.
     *
     * @param model the model to train
     * @param verbose whether to print progress
     * @return the training result
     */
    public static TrainingResult train(Model model, boolean verbose) {
        Trainer trainer = Trainer.of(model)
                .loss(Loss.MSE)
                .optimizer(Optimizer.adam(0.05f))
                .batchSize(4)
                .epochs(2_000)
                .seed(42);
        if (verbose) {
            trainer.listener(Progress.every(250));
        }
        return trainer.fit(DATA);
    }

    /**
     * Runs the example.
     *
     * @param args ignored
     */
    public static void main(String[] args) {
        Model model = model();
        System.out.println(model.summary());
        TrainingResult result = train(model, true);
        System.out.printf("%nfinal loss %.6f%n", result.finalLoss());
        for (int i = 0; i < DATA.size(); i++) {
            float[] in = DATA.featuresOf(i);
            System.out.printf("%.0f XOR %.0f = %.3f%n", in[0], in[1], model.predict(in)[0]);
        }
    }
}
