package io.github.zzuegg.brainiac.experimental.ensemble;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.DistanceAwareHead;
import io.github.zzuegg.brainiac.experimental.nn.EvidentialHead;
import io.github.zzuegg.brainiac.experimental.nn.TabM;
import io.github.zzuegg.brainiac.experimental.ood.FeatureClassifier;
import io.github.zzuegg.brainiac.experimental.ood.OodMetrics;
import io.github.zzuegg.brainiac.experimental.train.Calibration;
import io.github.zzuegg.brainiac.experimental.train.PopulationEnsemble;
import io.github.zzuegg.brainiac.experimental.train.WeightPosterior;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.function.UnaryOperator;

/**
 * One protocol for every ensemble and uncertainty method: a common {@link Member} interface
 * (probabilities of a batch of inputs), adapters for the repository's existing blocks, and the
 * diversity measures of an ensemble (Lakshminarayanan et al. 2017,
 * <a href="https://arxiv.org/abs/1612.01474">arXiv:1612.01474</a>; Ovadia et al. 2019,
 * <a href="https://arxiv.org/abs/1906.02530">arXiv:1906.02530</a>):
 *
 * <ul>
 *   <li>{@link #disagreement}: the share of member pairs and inputs whose top classes differ;</li>
 *   <li>{@link #ambiguity}: the ambiguity decomposition of the log loss of the averaged
 *       prediction, {@code NLL(p̄) = mean_m NLL(p_m) - ambiguity}, where the ambiguity is the
 *       mean of {@code log p̄_y - log p_{m,y}} and never negative (Jensen);</li>
 *   <li>{@link #functionalDiversity}: the mean mutual information {@code H(p̄) - mean H(p_m)},
 *       which is the mean KL divergence of the members from the ensemble mean;</li>
 *   <li>{@link #evaluate}: accuracy, NLL, Brier, ECE, disagreement, ambiguity and diversity;
 *       {@link #oodAuroc}: OOD detection by mutual information or predictive entropy.</li>
 * </ul>
 *
 * A single model is an ensemble of one (its diversity is zero), so single-model baselines run
 * through the same code.
 */
public final class EnsembleAudit {

    /** One member of an ensemble. */
    @FunctionalInterface
    public interface Member {

        /**
         * @param x inputs {@code [n, features]}
         * @return probabilities {@code [n][classes]}
         */
        double[][] probabilities(Tensor x);
    }

    /** The uncertainty score used for OOD detection. */
    public enum Score {
        /** the mutual information between prediction and member */
        MUTUAL_INFORMATION,
        /** the entropy of the mean prediction */
        ENTROPY
    }

    /**
     * Ensemble metrics.
     *
     * @param accuracy accuracy of the mean prediction
     * @param nll its NLL
     * @param brier its Brier score
     * @param ece its 15-bin ECE
     * @param disagreement pairwise top-class disagreement
     * @param ambiguity the ambiguity term of the NLL decomposition
     * @param diversity the mean mutual information
     * @param memberNll the mean member NLL
     */
    public record Report(double accuracy, double nll, double brier, double ece, double disagreement, double ambiguity,
            double diversity, double memberNll) {
    }

    private EnsembleAudit() {
    }

    // ------------------------------------------------------------------ adapters

    /**
     * @param model a classifier
     * @return its softmax probabilities in inference mode
     */
    public static Member of(FeatureClassifier model) {
        return x -> softmaxRows(model.logits(x));
    }

    /**
     * @param model a model whose output is logits
     * @return its softmax probabilities in inference mode
     */
    public static Member of(Model model) {
        return x -> softmaxRows(Var.noGrad(() -> model.forward(Var.of(x), false).value()));
    }

    /**
     * @param logits a function from inputs to logits
     * @return its softmax probabilities
     */
    public static Member ofLogits(UnaryOperator<Tensor> logits) {
        return x -> softmaxRows(logits.apply(x));
    }

    /**
     * MC dropout: one member per stochastic pass of a model run in training mode.
     *
     * @param model a model with dropout, output logits
     * @param passes the number of passes
     * @return the members (each call of a member is a fresh dropout mask)
     */
    public static List<Member> mcDropout(Model model, int passes) {
        List<Member> out = new ArrayList<>();
        for (int i = 0; i < passes; i++) {
            out.add(x -> softmaxRows(Var.noGrad(() -> model.forward(Var.of(x), true).value())));
        }
        return out;
    }

    /**
     * The members of a {@link PopulationEnsemble}.
     *
     * @param ensemble the ensemble
     * @return one member per population model
     */
    public static List<Member> of(PopulationEnsemble ensemble) {
        List<Member> out = new ArrayList<>();
        for (int i = 0; i < ensemble.size(); i++) {
            int m = i;
            out.add(x -> ensemble.predict(rows(x)).members()[m]);
        }
        return out;
    }

    /**
     * Members from weight samples or snapshots of one network (SWAG samples, snapshot and FGE
     * ensembles), predicted through {@link WeightPosterior#average}.
     *
     * @param parameters the network's parameters
     * @param samples the weight samples
     * @param logits the network's logits for the current weights
     * @return one member per sample
     */
    public static List<Member> ofWeights(List<Var> parameters, List<List<Tensor>> samples,
            UnaryOperator<Tensor> logits) {
        List<Member> out = new ArrayList<>();
        for (List<Tensor> s : samples) {
            out.add(x -> softmaxRows(WeightPosterior.average(parameters, List.of(s), () -> logits.apply(x)).get(1)));
        }
        return out;
    }

    /**
     * The members of TabM (or any model with a {@code [n, k, classes]} member output).
     *
     * @param model the model
     * @return one member per BatchEnsemble member
     */
    public static List<Member> of(TabM model) {
        List<Member> out = new ArrayList<>();
        for (int i = 0; i < model.size(); i++) {
            int m = i;
            out.add(x -> {
                Tensor z = Var.noGrad(() -> model.members(Var.of(x), false).value());
                int n = z.shape().dim(0);
                int c = z.shape().dim(2);
                return softmaxRows(z.slice(1, m, m + 1).reshape(n, c));
            });
        }
        return out;
    }

    /**
     * An evidential (Dirichlet) head: its mean probabilities.
     *
     * @param alpha maps inputs to the Dirichlet parameters
     * @return the member
     */
    public static Member evidential(UnaryOperator<Tensor> alpha) {
        return x -> rows2(EvidentialHead.probabilities(alpha.apply(x)));
    }

    /**
     * A distance-aware head (SNGP or DUQ) on a feature extractor.
     *
     * @param head the head
     * @param features maps inputs to the head's input features
     * @return the member
     */
    public static Member distanceAware(DistanceAwareHead head, UnaryOperator<Tensor> features) {
        return x -> rows2(head.probabilities(features.apply(x)));
    }

    // ------------------------------------------------------------------ measures

    /**
     * Every member's probabilities.
     *
     * @param members the ensemble
     * @param x the inputs
     * @return {@code [member][sample][class]}
     */
    public static double[][][] predict(List<Member> members, Tensor x) {
        double[][][] out = new double[members.size()][][];
        for (int m = 0; m < out.length; m++) {
            out[m] = members.get(m).probabilities(x);
        }
        return out;
    }

    /**
     * @param p {@code [member][sample][class]}
     * @return the mean prediction {@code [sample][class]}
     */
    public static double[][] mean(double[][][] p) {
        int n = p[0].length;
        int k = p[0][0].length;
        double[][] out = new double[n][k];
        for (double[][] m : p) {
            for (int i = 0; i < n; i++) {
                for (int c = 0; c < k; c++) {
                    out[i][c] += m[i][c] / p.length;
                }
            }
        }
        return out;
    }

    /**
     * @param p {@code [member][sample][class]}
     * @return the mean pairwise top-class disagreement (0 for one member)
     */
    public static double disagreement(double[][][] p) {
        if (p.length < 2) {
            return 0;
        }
        double d = 0;
        int pairs = 0;
        for (int a = 0; a < p.length; a++) {
            for (int b = a + 1; b < p.length; b++) {
                int diff = 0;
                for (int i = 0; i < p[a].length; i++) {
                    diff += Calibration.argmax(p[a][i]) != Calibration.argmax(p[b][i]) ? 1 : 0;
                }
                d += diff / (double) p[a].length;
                pairs++;
            }
        }
        return d / pairs;
    }

    /**
     * The ambiguity term of the log-loss decomposition.
     *
     * @param p {@code [member][sample][class]}
     * @param labels the labels
     * @return {@code mean_m NLL(p_m) - NLL(p̄)}, at least 0
     */
    public static double ambiguity(double[][][] p, int[] labels) {
        double members = 0;
        for (double[][] m : p) {
            members += Calibration.negativeLogLikelihood(m, labels) / p.length;
        }
        return members - Calibration.negativeLogLikelihood(mean(p), labels);
    }

    /**
     * @param p {@code [member][sample][class]}
     * @return the mean mutual information (the members' mean KL from the ensemble)
     */
    public static double functionalDiversity(double[][][] p) {
        double s = 0;
        for (double v : mutualInformation(p)) {
            s += v;
        }
        return s / p[0].length;
    }

    /**
     * @param p {@code [member][sample][class]}
     * @return the mutual information of every sample
     */
    public static double[] mutualInformation(double[][][] p) {
        int n = p[0].length;
        double[] out = new double[n];
        double[][] members = new double[p.length][];
        for (int i = 0; i < n; i++) {
            for (int m = 0; m < p.length; m++) {
                members[m] = p[m][i];
            }
            out[i] = Calibration.mutualInformation(members);
        }
        return out;
    }

    /**
     * Ensemble metrics on labelled inputs.
     *
     * @param members the ensemble
     * @param x the inputs
     * @param labels the labels
     * @return the report
     */
    public static Report evaluate(List<Member> members, Tensor x, int[] labels) {
        return evaluate(predict(members, x), labels);
    }

    /**
     * Ensemble metrics from member predictions.
     *
     * @param p {@code [member][sample][class]}
     * @param labels the labels
     * @return the report
     */
    public static Report evaluate(double[][][] p, int[] labels) {
        double[][] mean = mean(p);
        double memberNll = 0;
        for (double[][] m : p) {
            memberNll += Calibration.negativeLogLikelihood(m, labels) / p.length;
        }
        return new Report(Calibration.accuracy(mean, labels), Calibration.negativeLogLikelihood(mean, labels),
                Calibration.brierScore(mean, labels), Calibration.expectedCalibrationError(mean, labels, 15),
                disagreement(p), ambiguity(p, labels), functionalDiversity(p), memberNll);
    }

    /**
     * OOD detection AUROC (OOD inputs as positives).
     *
     * @param members the ensemble
     * @param in in-distribution inputs
     * @param out OOD inputs
     * @param score the uncertainty score
     * @return the AUROC
     */
    public static double oodAuroc(List<Member> members, Tensor in, Tensor out, Score score) {
        return OodMetrics.auroc(uncertainty(predict(members, out), score), uncertainty(predict(members, in), score));
    }

    /**
     * Per-sample uncertainty.
     *
     * @param p {@code [member][sample][class]}
     * @param score the score
     * @return the uncertainty of every sample
     */
    public static double[] uncertainty(double[][][] p, Score score) {
        if (score == Score.MUTUAL_INFORMATION) {
            return mutualInformation(p);
        }
        double[][] mean = mean(p);
        double[] out = new double[mean.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = Calibration.entropy(mean[i]);
        }
        return out;
    }

    // ------------------------------------------------------------------ conversions

    /**
     * @param logits {@code [n, classes]}
     * @return softmax rows
     */
    public static double[][] softmaxRows(Tensor logits) {
        return rows2(logits.sub(logits.logsumexp(1, true)).exp());
    }

    static double[][] rows2(Tensor t) {
        int c = t.shape().dim(1);
        double[] v = t.toDoubleArray();
        double[][] out = new double[v.length / c][c];
        for (int i = 0; i < out.length; i++) {
            System.arraycopy(v, i * c, out[i], 0, c);
        }
        return out;
    }

    static float[][] rows(Tensor x) {
        int d = x.shape().dim(1);
        float[] v = x.toArray();
        float[][] out = new float[v.length / d][d];
        for (int i = 0; i < out.length; i++) {
            System.arraycopy(v, i * d, out[i], 0, d);
        }
        return out;
    }
}
