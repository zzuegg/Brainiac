package io.github.zzuegg.brainiac.examples;

import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.data.Standardizer;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.train.Evaluation;
import io.github.zzuegg.brainiac.train.Metric;
import io.github.zzuegg.brainiac.train.Optimizer;
import io.github.zzuegg.brainiac.train.Progress;
import io.github.zzuegg.brainiac.train.Trainer;

import java.nio.file.Path;

/**
 * Classify iris flowers into three species from four measurements, loaded from CSV.
 * Shows a train/test split, standardization fitted on training data, validation and
 * early stopping.
 */
public final class IrisExample {

    private IrisExample() {
    }

    /** Where the data file lives, relative to the examples module. */
    public static final Path DATA_FILE = Path.of("data", "iris.csv");

    /**
     * The result of a run.
     *
     * @param model the trained model
     * @param standardizer the feature scaling fitted on the training data
     * @param test the held-out test data, standardized
     * @param testEvaluation loss and accuracy on the test data
     */
    public record Result(Model model, Standardizer standardizer, Dataset test, Evaluation testEvaluation) {
    }

    /**
     * Loads the data, trains and evaluates on held-out samples.
     *
     * @param file the CSV file
     * @param verbose whether to print progress
     * @return the trained model and its test evaluation
     */
    public static Result run(Path file, boolean verbose) {
        Dataset iris = Dataset.csv(file)
                .features("sepal_length", "sepal_width", "petal_length", "petal_width")
                .label("species").oneHot()
                .load();
        Dataset.Split split = iris.shuffle(42).split(0.8);
        Standardizer standardizer = Standardizer.fit(split.train());
        Dataset train = split.train().standardize(standardizer);
        Dataset test = split.test().standardize(standardizer);

        Model model = Model.sequential()
                .input(4)
                .dense(16, Activation.RELU)
                .dense(3, Activation.SOFTMAX)
                .seed(42)
                .build();

        Trainer trainer = Trainer.of(model)
                .loss(Loss.CROSS_ENTROPY)
                .optimizer(Optimizer.adam(0.01f))
                .batchSize(16)
                .epochs(300)
                .metric(Metric.ACCURACY)
                .validation(test)
                .earlyStopping(30)
                .seed(42);
        if (verbose) {
            trainer.listener(Progress.every(25));
        }
        trainer.fit(train);
        return new Result(model, standardizer, test, trainer.evaluate(test));
    }

    /**
     * Runs the example.
     *
     * @param args optional path to iris.csv
     */
    public static void main(String[] args) {
        Path file = args.length > 0 ? Path.of(args[0]) : DATA_FILE;
        Result r = run(file, true);
        System.out.printf("%ntest accuracy %.1f%% on %d unseen flowers%n",
                100 * r.testEvaluation().get(Metric.ACCURACY), r.test().size());
        float[] flower = r.standardizer().apply(new float[] {6.1f, 2.8f, 4.7f, 1.2f});
        float[] p = r.model().predict(flower);
        System.out.printf("a 6.1/2.8/4.7/1.2 flower is: setosa %.2f, versicolor %.2f, virginica %.2f%n", p[0], p[1], p[2]);
    }
}
