package io.github.zzuegg.brainiac.experiments.agents;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.SplittableRandom;

/**
 * A toy AI Scientist (Lu et al., <a href="https://arxiv.org/abs/2408.06292">2408.06292</a>),
 * item 5.7: an automated idea, experiment, write-up and review loop over a real Brainiac
 * experiment, training a small MLP on two interleaved moons. Each cycle:
 * <ol>
 *   <li><b>idea</b>: 24 candidate ideas, each changing one or two settings of a promising
 *       earlier configuration (learning rate, width, optimizer, activation, steps), stated
 *       as a hypothesis;</li>
 *   <li><b>review before running</b>: a reviewer model, a logistic regression over one-hot
 *       settings refit on every result so far, predicts whether each idea will beat the
 *       baseline; the most promising novel idea is run;</li>
 *   <li><b>experiment</b>: train and test the network;</li>
 *   <li><b>write-up and review after running</b>: the result is written up as a claim; a
 *       claim of beating the baseline is accepted only if a rerun with a new seed reproduces
 *       it, the reviewer's check of the paper against the evidence.</li>
 * </ol>
 * The baseline search is random configuration search with the same number of training runs
 * (the reproductions count against the researcher's budget).
 */
public final class AutoResearcher {

    static final double[] LRS = {0.001, 0.003, 0.01, 0.03, 0.1, 0.3};
    static final int[] WIDTHS = {2, 4, 8, 16, 32};
    static final String[] OPTIMIZERS = {"sgd", "momentum", "adam"};
    static final String[] ACTIVATIONS = {"tanh", "relu"};
    static final int[] STEPS = {50, 100, 200};

    /**
     * An experiment configuration.
     *
     * @param lr the index into the learning rates
     * @param width the index into the hidden widths
     * @param optimizer the index into the optimizers
     * @param activation the index into the activations
     * @param steps the index into the step counts
     */
    public record Config(int lr, int width, int optimizer, int activation, int steps) {
        @Override
        public String toString() {
            return String.format(Locale.ROOT, "lr %s, width %d, %s, %s, %d steps", LRS[lr], WIDTHS[width],
                    OPTIMIZERS[optimizer], ACTIVATIONS[activation], STEPS[steps]);
        }

        /** One-hot of every setting plus a bias: the reviewer is additive over settings. */
        double[] features() {
            double[] f = new double[1 + LRS.length + WIDTHS.length + OPTIMIZERS.length + ACTIVATIONS.length
                    + STEPS.length];
            f[0] = 1;
            int at = 1;
            f[at + lr] = 1;
            at += LRS.length;
            f[at + width] = 1;
            at += WIDTHS.length;
            f[at + optimizer] = 1;
            at += OPTIMIZERS.length;
            f[at + activation] = 1;
            at += ACTIVATIONS.length;
            f[at + steps] = 1;
            return f;
        }
    }

    /**
     * The outcome.
     *
     * @param baselineAccuracy the baseline experiment's test accuracy
     * @param researcherBeat share of the researcher's idea runs that beat the baseline
     * @param randomBeat share of random-search runs that beat the baseline
     * @param researcherBest the researcher's best test accuracy
     * @param randomBest random search's best test accuracy
     * @param accepted findings that survived the reproduction review
     * @param writeUps the write-ups, in order
     */
    public record Result(double baselineAccuracy, double researcherBeat, double randomBeat, double researcherBest,
            double randomBest, int accepted, List<String> writeUps) {
    }

    private AutoResearcher() {
    }

    private record Data(Tensor trainX, Tensor trainY, float[][] testX, float[] testY) {
    }

    private static Data moons(long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = 200;
        float[] x = new float[2 * n * 2];
        float[] y = new float[2 * n];
        for (int i = 0; i < 2 * n; i++) {
            int c = i % 2;
            double t = Math.PI * r.nextDouble();
            double px = c == 0 ? Math.cos(t) : 1 - Math.cos(t);
            double py = c == 0 ? Math.sin(t) : 0.5 - Math.sin(t);
            x[2 * i] = (float) (px + 0.15 * r.nextGaussian());
            x[2 * i + 1] = (float) (py + 0.15 * r.nextGaussian());
            y[i] = c;
        }
        float[][] testX = new float[n][];
        float[] testY = new float[n];
        for (int i = 0; i < n; i++) {
            testX[i] = new float[] {x[2 * (n + i)], x[2 * (n + i) + 1]};
            testY[i] = y[n + i];
        }
        return new Data(Tensor.of(Shape.of(n, 2), java.util.Arrays.copyOf(x, 2 * n)),
                Tensor.of(Shape.of(n, 1), java.util.Arrays.copyOf(y, n)), testX, testY);
    }

    /**
     * Runs the experiment under study.
     *
     * @param c the configuration
     * @param seed the initialization seed
     * @return the test accuracy
     */
    static double experiment(Config c, Data d, long seed) {
        Activation act = c.activation() == 0 ? Activation.TANH : Activation.RELU;
        Model net = Model.sequential().input(2).dense(WIDTHS[c.width()], act).dense(WIDTHS[c.width()], act).dense(1)
                .seed(seed).build();
        float lr = (float) LRS[c.lr()];
        Optimizer opt = switch (c.optimizer()) {
            case 0 -> Optimizer.sgd(lr);
            case 1 -> Optimizer.momentum(lr, 0.9f);
            default -> Optimizer.adam(lr);
        };
        Var x = Var.of(d.trainX());
        for (int s = 0; s < STEPS[c.steps()]; s++) {
            Var z = net.forward(x, true);
            Var loss = z.softplus().sub(z.mul(d.trainY())).mean();
            loss.backward();
            opt.step(net.parameters());
        }
        float[] flat = new float[d.testX().length * 2];
        for (int i = 0; i < d.testX().length; i++) {
            flat[2 * i] = d.testX()[i][0];
            flat[2 * i + 1] = d.testX()[i][1];
        }
        float[] z = net.forward(Var.of(Tensor.of(Shape.of(d.testX().length, 2), flat)), false).value()
                .to(DType.FLOAT32).toArray();
        int right = 0;
        for (int i = 0; i < z.length; i++) {
            right += (z[i] > 0) == (d.testY()[i] > 0.5f) ? 1 : 0;
        }
        return right / (double) z.length;
    }

    /** The reviewer: logistic regression on configuration features, refit on every result. */
    static double[] fitReviewer(List<Config> configs, List<Boolean> beat) {
        double[] w = new double[configs.getFirst().features().length];
        for (int it = 0; it < 300; it++) {
            double[] g = new double[w.length];
            for (int i = 0; i < configs.size(); i++) {
                double[] f = configs.get(i).features();
                double p = predict(w, f);
                double y = beat.get(i) ? 1 : 0;
                for (int j = 0; j < w.length; j++) {
                    g[j] += (p - y) * f[j];
                }
            }
            for (int j = 0; j < w.length; j++) {
                w[j] -= 0.5 * (g[j] / configs.size() + 0.01 * w[j]);
            }
        }
        return w;
    }

    static double predict(double[] w, double[] f) {
        double z = 0;
        for (int j = 0; j < w.length; j++) {
            z += w[j] * f[j];
        }
        return 1 / (1 + Math.exp(-z));
    }

    static Config idea(Config parent, SplittableRandom r) {
        int axis = r.nextInt(5);
        int step = r.nextBoolean() ? 1 : -1;
        return switch (axis) {
            case 0 -> new Config(Math.clamp(parent.lr() + step, 0, LRS.length - 1), parent.width(),
                    parent.optimizer(), parent.activation(), parent.steps());
            case 1 -> new Config(parent.lr(), Math.clamp(parent.width() + step, 0, WIDTHS.length - 1),
                    parent.optimizer(), parent.activation(), parent.steps());
            case 2 -> new Config(parent.lr(), parent.width(), r.nextInt(OPTIMIZERS.length), parent.activation(),
                    parent.steps());
            case 3 -> new Config(parent.lr(), parent.width(), parent.optimizer(), 1 - parent.activation(),
                    parent.steps());
            default -> new Config(parent.lr(), parent.width(), parent.optimizer(), parent.activation(),
                    Math.clamp(parent.steps() + step, 0, STEPS.length - 1));
        };
    }

    static Config randomConfig(SplittableRandom r) {
        return new Config(r.nextInt(LRS.length), r.nextInt(WIDTHS.length), r.nextInt(OPTIMIZERS.length),
                r.nextInt(ACTIVATIONS.length), r.nextInt(STEPS.length));
    }

    /**
     * @param budget training runs per searcher
     * @param seed the seed
     * @return the comparison
     */
    public static Result run(int budget, long seed) {
        Data d = moons(seed);
        Config baseline = new Config(2, 2, 2, 0, 1);   // a sensible default: Adam 0.01, width 8, tanh
        double base = experiment(baseline, d, seed);
        SplittableRandom r = new SplittableRandom(seed);

        // the researcher
        List<Config> tried = new ArrayList<>(List.of(baseline));
        List<Double> scores = new ArrayList<>(List.of(base));
        List<Boolean> beat = new ArrayList<>(List.of(false));
        Set<Config> seen = new HashSet<>(tried);
        List<String> writeUps = new ArrayList<>();
        int ideaRuns = 0;
        int ideaBeat = 0;
        int accepted = 0;
        double best = base;
        int runs = 0;
        while (runs < budget) {
            int top = 0;
            for (int i = 1; i < tried.size(); i++) {
                top = scores.get(i) > scores.get(top) ? i : top;
            }
            double[] reviewer = tried.size() >= 4 ? fitReviewer(tried, beat) : null;
            Config chosen = null;
            double chosenScore = Double.NEGATIVE_INFINITY;
            for (int k = 0; k < 24; k++) {
                Config parent = r.nextDouble() < 0.7 ? tried.get(top) : tried.get(r.nextInt(tried.size()));
                Config c = idea(parent, r);
                if (r.nextBoolean()) {
                    c = idea(c, r);   // some ideas change two settings
                }
                if (seen.contains(c)) {
                    continue;   // the reviewer rejects ideas that are not novel
                }
                double s = reviewer == null ? r.nextDouble() : predict(reviewer, c.features());
                if (s > chosenScore) {
                    chosenScore = s;
                    chosen = c;
                }
            }
            if (chosen == null) {
                chosen = randomConfig(r);
                if (seen.contains(chosen)) {
                    continue;
                }
            }
            seen.add(chosen);
            double acc = experiment(chosen, d, seed + 10 + runs);
            runs++;
            ideaRuns++;
            boolean claim = acc > base;
            ideaBeat += claim ? 1 : 0;
            String verdict = claim ? "not reviewed (budget spent)" : "refuted";
            if (claim && runs < budget) {
                double again = experiment(chosen, d, seed + 1000 + runs);
                runs++;
                verdict = again > base ? "accepted (reproduced: " + fmt(again) + ")" : "rejected (did not reproduce)";
                accepted += again > base ? 1 : 0;
            }
            writeUps.add("Hypothesis: " + chosen + " beats the baseline. Result: " + fmt(acc) + " vs " + fmt(base)
                    + ". Review: " + verdict + ".");
            tried.add(chosen);
            scores.add(acc);
            beat.add(claim);
            best = Math.max(best, acc);
        }

        // random configuration search
        SplittableRandom rr = new SplittableRandom(seed + 7);
        int randomBeat = 0;
        double randomBest = base;
        for (int i = 0; i < budget; i++) {
            double acc = experiment(randomConfig(rr), d, seed + 10 + i);
            randomBeat += acc > base ? 1 : 0;
            randomBest = Math.max(randomBest, acc);
        }
        return new Result(base, ideaBeat / (double) ideaRuns, randomBeat / (double) budget, best, randomBest,
                accepted, List.copyOf(writeUps));
    }

    private static String fmt(double v) {
        return String.format(Locale.ROOT, "%.3f", v);
    }
}
