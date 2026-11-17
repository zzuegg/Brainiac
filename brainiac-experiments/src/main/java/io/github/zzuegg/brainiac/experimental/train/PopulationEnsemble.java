package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.evolve.Evolution;
import io.github.zzuegg.brainiac.evolve.Genome;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.train.Benchmark;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;
import java.util.function.Function;

/**
 * Several classifiers evaluated together, with their disagreement as a measure of
 * uncertainty and the option to <em>abstain</em> when unsure (selective prediction).
 * Members can be any models: independently trained copies (a deep ensemble), MC-dropout
 * samples, or, cheapest of all, the best distinct designs of an evolutionary search,
 * which a search has produced anyway ({@link #fromEvolution}).
 *
 * <p>For each input the ensemble reports the mean probabilities and three uncertainty
 * scores: predictive entropy (total uncertainty), mutual information (BALD: the part due
 * to disagreement, high off the training data) and the vote disagreement rate. Sorting
 * inputs by a score and answering only the most certain ones gives the risk-coverage
 * curve ({@link #selective}); its area (AURC) summarises how well the score ranks errors.
 *
 * <p>Lakshminarayanan et al. 2017, "Simple and Scalable Predictive Uncertainty Estimation
 * using Deep Ensembles", <a href="https://arxiv.org/abs/1612.01474">arXiv:1612.01474</a>;
 * Geifman and El-Yaniv 2017, "Selective Classification for Deep Neural Networks",
 * <a href="https://arxiv.org/abs/1705.08500">arXiv:1705.08500</a>.
 */
public final class PopulationEnsemble {

    /** An uncertainty score for selective prediction; higher means less sure. */
    public enum Uncertainty {
        /** {@code 1 - max mean probability} */
        CONFIDENCE,
        /** entropy of the mean prediction */
        ENTROPY,
        /** mutual information between prediction and member (BALD) */
        MUTUAL_INFORMATION,
        /** share of members whose vote differs from the majority */
        DISAGREEMENT
    }

    /**
     * The ensemble's view of a batch.
     *
     * @param members probabilities {@code [member][sample][class]}
     * @param mean the mean probabilities {@code [sample][class]}
     */
    public record Prediction(double[][][] members, double[][] mean) {

        /**
         * The predicted class of a sample.
         *
         * @param i the sample
         * @return the class with the largest mean probability
         */
        public int label(int i) {
            return Calibration.argmax(mean[i]);
        }

        /**
         * A sample's uncertainty.
         *
         * @param i the sample
         * @param kind which score
         * @return the score, higher is less certain
         */
        public double uncertainty(int i, Uncertainty kind) {
            return switch (kind) {
                case CONFIDENCE -> 1 - mean[i][label(i)];
                case ENTROPY -> Calibration.entropy(mean[i]);
                case MUTUAL_INFORMATION -> {
                    double[][] m = new double[members.length][];
                    for (int k = 0; k < members.length; k++) {
                        m[k] = members[k][i];
                    }
                    yield Calibration.mutualInformation(m);
                }
                case DISAGREEMENT -> {
                    int majority = label(i);
                    int differ = 0;
                    for (double[][] member : members) {
                        differ += Calibration.argmax(member[i]) == majority ? 0 : 1;
                    }
                    yield differ / (double) members.length;
                }
            };
        }

        /**
         * Every sample's uncertainty.
         *
         * @param kind which score
         * @return one score per sample
         */
        public double[] uncertainties(Uncertainty kind) {
            double[] out = new double[mean.length];
            for (int i = 0; i < out.length; i++) {
                out[i] = uncertainty(i, kind);
            }
            return out;
        }
    }

    /**
     * A risk-coverage curve: answering the {@code k} most certain samples, for every k.
     *
     * @param coverage the share answered
     * @param risk the error rate among the answered
     * @param thresholds the uncertainty of the last answered sample
     */
    public record SelectiveCurve(double[] coverage, double[] risk, double[] thresholds) {

        /** @return the area under the risk-coverage curve (lower is better) */
        public double aurc() {
            return Arrays.stream(risk).average().orElse(0);
        }

        /**
         * The risk when answering a share of the samples.
         *
         * @param share the coverage wanted
         * @return the error rate on the most certain {@code share} of samples
         */
        public double riskAt(double share) {
            int k = Math.max(1, (int) Math.round(share * coverage.length)) - 1;
            return risk[Math.min(k, risk.length - 1)];
        }

        /**
         * The largest uncertainty threshold whose risk stays at or under a target.
         *
         * @param target the acceptable error rate
         * @return the threshold to abstain above, or negative infinity if none qualifies
         */
        public double thresholdForRisk(double target) {
            double best = Double.NEGATIVE_INFINITY;
            for (int k = 0; k < risk.length; k++) {
                if (risk[k] <= target) {
                    best = thresholds[k];
                }
            }
            return best;
        }
    }

    private final List<Function<float[][], double[][]>> members;

    private PopulationEnsemble(List<Function<float[][], double[][]>> members) {
        if (members.isEmpty()) {
            throw new IllegalArgumentException("an ensemble needs members");
        }
        this.members = List.copyOf(members);
    }

    /**
     * An ensemble of trained models whose outputs are class probabilities.
     *
     * @param models the members
     * @return the ensemble
     */
    public static PopulationEnsemble of(List<Model> models) {
        List<Function<float[][], double[][]>> f = new ArrayList<>();
        for (Model m : models) {
            Objects.requireNonNull(m);
            f.add(x -> toDouble(m.predict(x)));
        }
        return new PopulationEnsemble(f);
    }

    /**
     * An ensemble of arbitrary probability functions.
     *
     * @param members each maps a batch of samples to class probabilities
     * @return the ensemble
     */
    public static PopulationEnsemble ofFunctions(List<Function<float[][], double[][]>> members) {
        return new PopulationEnsemble(members);
    }

    /**
     * An ensemble of the best distinct designs of an evolutionary search, each trained
     * from scratch with the search's settings on the task's full training data.
     *
     * @param evolution the search that produced the result
     * @param result its result
     * @param task the task it ran on
     * @param count how many designs to take
     * @param seed the seed for the members' starting weights
     * @return the ensemble
     */
    public static PopulationEnsemble fromEvolution(Evolution evolution, Evolution.Result result, Benchmark.Task task,
            int count, long seed) {
        List<Model> models = new ArrayList<>();
        Set<Genome> seen = new HashSet<>();
        List<Evolution.Individual> ranked = new ArrayList<>(result.population());
        ranked.sort(Comparator.comparingDouble(i -> -i.fitness()));
        for (Evolution.Individual ind : ranked) {
            if (models.size() == count) {
                break;
            }
            if (!seen.add(ind.genome())) {
                continue;
            }
            Model m = evolution.build(ind.genome(), seed + models.size());
            Benchmark.run(task, "member " + models.size(), m, evolution.settings(ind.genome()));
            models.add(m);
        }
        return of(models);
    }

    /** @return the number of members */
    public int size() {
        return members.size();
    }

    /**
     * Runs every member.
     *
     * @param samples the inputs
     * @return the members' and the mean probabilities
     */
    public Prediction predict(float[][] samples) {
        double[][][] all = new double[members.size()][][];
        for (int k = 0; k < all.length; k++) {
            all[k] = members.get(k).apply(samples);
        }
        int n = samples.length;
        int c = all[0][0].length;
        double[][] mean = new double[n][c];
        for (double[][] m : all) {
            for (int i = 0; i < n; i++) {
                for (int j = 0; j < c; j++) {
                    mean[i][j] += m[i][j] / all.length;
                }
            }
        }
        return new Prediction(all, mean);
    }

    /**
     * Predicts, abstaining (label -1) where the uncertainty is above a threshold.
     *
     * @param samples the inputs
     * @param kind the uncertainty score
     * @param threshold the most uncertainty still answered
     * @return one label per sample, -1 for abstentions
     */
    public int[] predictOrAbstain(float[][] samples, Uncertainty kind, double threshold) {
        Prediction p = predict(samples);
        int[] out = new int[samples.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = p.uncertainty(i, kind) > threshold ? -1 : p.label(i);
        }
        return out;
    }

    /**
     * The risk-coverage curve of a scoring rule on labelled data.
     *
     * @param samples the inputs
     * @param labels the true classes
     * @param kind the uncertainty score
     * @return the curve
     */
    public SelectiveCurve selective(float[][] samples, int[] labels, Uncertainty kind) {
        Prediction p = predict(samples);
        return curve(p.uncertainties(kind), i -> p.label(i) == labels[i]);
    }

    /**
     * A risk-coverage curve from uncertainties and correctness.
     *
     * @param uncertainty one score per sample
     * @param correct whether the prediction for a sample is right
     * @return the curve
     */
    public static SelectiveCurve curve(double[] uncertainty, java.util.function.IntPredicate correct) {
        int n = uncertainty.length;
        Integer[] order = new Integer[n];
        for (int i = 0; i < n; i++) {
            order[i] = i;
        }
        Arrays.sort(order, Comparator.comparingDouble(i -> uncertainty[i]));
        double[] coverage = new double[n];
        double[] risk = new double[n];
        double[] thresholds = new double[n];
        int errors = 0;
        for (int k = 0; k < n; k++) {
            errors += correct.test(order[k]) ? 0 : 1;
            coverage[k] = (k + 1) / (double) n;
            risk[k] = errors / (double) (k + 1);
            thresholds[k] = uncertainty[order[k]];
        }
        return new SelectiveCurve(coverage, risk, thresholds);
    }

    private static double[][] toDouble(float[][] p) {
        double[][] out = new double[p.length][];
        for (int i = 0; i < p.length; i++) {
            out[i] = new double[p[i].length];
            for (int j = 0; j < p[i].length; j++) {
                out[i][j] = p[i][j];
            }
        }
        return out;
    }
}
