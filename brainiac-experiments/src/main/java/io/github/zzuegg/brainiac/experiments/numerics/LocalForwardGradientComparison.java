package io.github.zzuegg.brainiac.experiments.numerics;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experimental.train.CurvatureProducts;
import io.github.zzuegg.brainiac.experimental.train.FeedbackAlignment;
import io.github.zzuegg.brainiac.experimental.train.ForwardGradient;
import io.github.zzuegg.brainiac.experimental.train.ForwardGradientLearner;
import io.github.zzuegg.brainiac.experimental.train.LocalForwardGradient;
import io.github.zzuegg.brainiac.experiments.learning.LearnerComparison;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The fair test of forward-mode learning (wave 7, item 7.3.18, the follow-up of the
 * {@link LearnerComparison} result where weight-perturbed forward gradients reached 0.79 against
 * 0.987 for backprop on spirals): {@link LocalForwardGradient} (local losses, activity
 * perturbation, unit groups) against weight-perturbed {@link ForwardGradientLearner}, feedback
 * alignment and backprop, on the same spirals and budget, plus the cosine of each gradient
 * estimate to the gradient it estimates.
 */
public final class LocalForwardGradientComparison {

    private LocalForwardGradientComparison() {
    }

    /**
     * One learner's outcome.
     *
     * @param result the train and test accuracy
     * @param cosine the mean cosine of its gradient estimate to the true gradient (1 for backprop,
     *     NaN where not measured)
     */
    public record Row(LearnerComparison.Result result, double cosine) {
    }

    /**
     * Runs every learner on spirals.
     *
     * @param epochs passes over the data
     * @param seed the seed
     * @return one row per learner
     */
    public static Map<String, Row> spirals(int epochs, long seed) {
        ToyTasks.Data train = ToyTasks.spiral(100, 3, 0.1, seed);
        ToyTasks.Data test = ToyTasks.spiral(50, 3, 0.1, seed + 1);
        Map<String, Row> out = new LinkedHashMap<>();
        out.put("backprop", new Row(LearnerComparison.backprop(train, test, epochs, 50, 0.02f, seed, 32, 32), 1));
        SimpleMlp weightNet = SimpleMlp.tanh(seed, 2, 32, 32, 3);
        double weightCos = weightPerturbedCosine(weightNet, train, 32, seed);
        out.put("forward gradient (weights, 32 dirs)", new Row(LearnerComparison.run("forward gradient",
                new ForwardGradientLearner(weightNet, Optimizer.adam(0.01f), 32, seed), train, test, epochs, 50),
                weightCos));
        LocalForwardGradient local = new LocalForwardGradient(2, 3, new int[] {32, 32}, 4, 2, Optimizer.adam(0.02f),
                seed);
        out.put("local activity FG (4 groups x 2 dirs)", new Row(LearnerComparison.run("local activity FG", local,
                train, test, epochs, 50), local.meanCosine()));
        LocalForwardGradient ungrouped = new LocalForwardGradient(2, 3, new int[] {32, 32}, 1, 8,
                Optimizer.adam(0.02f), seed);
        out.put("local activity FG (1 group x 8 dirs)", new Row(LearnerComparison.run("local activity FG ungrouped",
                ungrouped, train, test, epochs, 50), ungrouped.meanCosine()));
        out.put("feedback alignment", new Row(LearnerComparison.run("feedback alignment",
                new FeedbackAlignment(SimpleMlp.tanh(seed, 2, 32, 32, 3), Optimizer.adam(0.02f),
                        FeedbackAlignment.Mode.values()[0], seed), train, test, epochs, 50), Double.NaN));
        return out;
    }

    /** The cosine of a weight-perturbed forward-gradient estimate to the true gradient at init. */
    private static double weightPerturbedCosine(SimpleMlp net, ToyTasks.Data data, int directions, long seed) {
        List<Var> params = net.parameters();
        double[] exact = CurvatureProducts.gradient(
                Functional.crossEntropy(net.forward(Var.of(data.inputs())), data.ids()), params);
        double sum = 0;
        int trials = 5;
        ForwardGradient fg = new ForwardGradient(seed + 9, directions);
        for (int k = 0; k < trials; k++) {
            fg.estimate(params, p -> Functional.crossEntropy(net.forward(Var.of(data.inputs())), data.ids()));
            sum += CurvatureProducts.cosine(CurvatureProducts.grads(params), exact);
            for (Var p : params) {
                p.zeroGrad();
            }
        }
        return sum / trials;
    }
}
