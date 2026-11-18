package io.github.zzuegg.brainiac.experimental.ensemble;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.ml.DirichletProcessMixture;
import io.github.zzuegg.brainiac.experimental.nn.DistillationLoss;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Distilling an ensemble into one network without erasing its disagreement:
 *
 * <ul>
 *   <li>{@link #meanKd}: plain distillation of the ensemble's mean prediction (reusing
 *       {@link DistillationLoss}), which keeps accuracy but loses the members' disagreement;</li>
 *   <li>{@link #dirichletNll}: ensemble distribution distillation, EnD² (Malinin, Mlodozeniec
 *       &amp; Gales, ICLR 2020, <a href="https://arxiv.org/abs/1905.00076">arXiv:1905.00076</a>):
 *       the student outputs Dirichlet concentrations {@code α = exp(z)} and maximises the
 *       likelihood of every member's (temperature-softened, centrally smoothed) predictive
 *       distribution under that Dirichlet, so total and knowledge uncertainty stay separable;
 *       {@link #annealedTemperature} anneals the temperature to 1;</li>
 *   <li>{@link Hydra}: a shared trunk with one light head per member, each head distilled from
 *       its member (Tran et al. 2020, <a href="https://arxiv.org/abs/2001.04694">arXiv:2001.04694</a>),
 *       keeping member diversity at almost one model's cost.</li>
 * </ul>
 * {@link #dirichletProbabilities} and {@link #dirichletMutualInformation} read a Dirichlet
 * student's mean prediction and knowledge uncertainty.
 */
public final class EnsembleDistillation {

    private EnsembleDistillation() {
    }

    /**
     * Distillation of the ensemble mean.
     *
     * @param studentLogits the student's logits {@code [n, k]}
     * @param meanProbs the ensemble's mean probabilities {@code [n, k]}
     * @param temperature the distillation temperature
     * @return the loss
     */
    public static Var meanKd(Var studentLogits, Tensor meanProbs, float temperature) {
        return new DistillationLoss(temperature).compute(studentLogits, meanProbs.add(1e-8f).log());
    }

    /**
     * The EnD² loss: the mean negative Dirichlet log-likelihood of the members' predictions.
     *
     * @param studentLogits the student's log-concentrations {@code [n, k]}
     * @param memberLogits each member's logits {@code [n, k]}
     * @param temperature the softening temperature of the members' predictions (and of the
     *     student's concentrations, {@code α = exp(z / T)}), annealed to 1
     * @param smoothing central smoothing {@code ε} of the targets, such as {@code 1e-4}
     * @return the loss
     */
    public static Var dirichletNll(Var studentLogits, List<Tensor> memberLogits, float temperature, float smoothing) {
        int k = studentLogits.shape().dim(1);
        Var alpha = studentLogits.div(temperature).exp().add(1e-4f);
        Var alpha0 = alpha.sum(1, true);
        Var norm = alpha0.lgamma().sum().sub(alpha.lgamma().sum());
        Var loss = null;
        int n = studentLogits.shape().dim(0);
        for (Tensor z : memberLogits) {
            Tensor scaled = z.div(temperature);
            Tensor p = scaled.sub(scaled.logsumexp(1, true)).exp().mul(1 - smoothing).add(smoothing / k);
            Var ll = norm.add(alpha.sub(1f).mul(p.log()).sum());
            loss = loss == null ? ll.neg() : loss.sub(ll);
        }
        return loss.div((float) n * memberLogits.size());
    }

    /**
     * A temperature annealed linearly from {@code start} to 1 over the first half of training.
     *
     * @param step the step
     * @param steps the total steps
     * @param start the initial temperature
     * @return the temperature
     */
    public static float annealedTemperature(int step, int steps, float start) {
        double f = Math.min(1, step / (0.5 * steps));
        return (float) (start + (1 - start) * f);
    }

    /**
     * @param logits a Dirichlet student's log-concentrations {@code [n, k]}
     * @return the mean prediction {@code α / α0} per row
     */
    public static double[][] dirichletProbabilities(Tensor logits) {
        double[][] a = alphas(logits);
        for (double[] row : a) {
            double s = 0;
            for (double v : row) {
                s += v;
            }
            for (int c = 0; c < row.length; c++) {
                row[c] /= s;
            }
        }
        return a;
    }

    /**
     * The Dirichlet's knowledge uncertainty: the mutual information
     * {@code H(E[π]) - E[H(π)]}, with {@code E[H(π)] = -Σ_c (α_c/α0)(ψ(α_c + 1) - ψ(α0 + 1))}.
     *
     * @param logits log-concentrations {@code [n, k]}
     * @return the mutual information of every row
     */
    public static double[] dirichletMutualInformation(Tensor logits) {
        double[][] a = alphas(logits);
        double[] out = new double[a.length];
        for (int i = 0; i < a.length; i++) {
            double a0 = 0;
            for (double v : a[i]) {
                a0 += v;
            }
            double total = 0;
            double expected = 0;
            for (double v : a[i]) {
                double p = v / a0;
                total -= p * Math.log(Math.max(1e-12, p));
                expected -= p * (DirichletProcessMixture.digamma(v + 1) - DirichletProcessMixture.digamma(a0 + 1));
            }
            out[i] = Math.max(0, total - expected);
        }
        return out;
    }

    private static double[][] alphas(Tensor logits) {
        int k = logits.shape().dim(1);
        double[] v = logits.toDoubleArray();
        double[][] out = new double[v.length / k][k];
        for (int i = 0; i < out.length; i++) {
            for (int c = 0; c < k; c++) {
                out[i][c] = Math.exp(Math.min(30, v[i * k + c])) + 1e-4;
            }
        }
        return out;
    }

    /**
     * Hydra: a shared trunk and one linear head per ensemble member, each head distilled from
     * its member.
     */
    public static final class Hydra {

        private final Model trunk;
        private final List<Dense> heads = new ArrayList<>();

        /**
         * Creates the student.
         *
         * @param inputs input features
         * @param hidden the trunk's hidden widths
         * @param classes the number of classes
         * @param members the number of heads
         * @param seed the seed
         */
        public Hydra(int inputs, int[] hidden, int classes, int members, long seed) {
            Model.Builder b = Model.sequential().input(inputs);
            for (int h : hidden) {
                b.dense(h, Activation.RELU);
            }
            this.trunk = b.seed(seed).build();
            SplittableRandom r = new SplittableRandom(seed + 77);
            for (int m = 0; m < members; m++) {
                heads.add(new Dense(hidden[hidden.length - 1], classes, Activation.IDENTITY, Init.XAVIER, r));
            }
        }

        /**
         * Every head's logits.
         *
         * @param x the inputs
         * @param training the training flag
         * @return one {@code [n, k]} logit tensor per head
         */
        public List<Var> heads(Var x, boolean training) {
            Var h = trunk.forward(x, training);
            List<Var> out = new ArrayList<>();
            for (Dense d : heads) {
                out.add(d.forward(h, training));
            }
            return out;
        }

        /**
         * The distillation loss: the sum over heads of the KD loss to the matching member.
         *
         * @param x the inputs
         * @param memberLogits each member's logits on the inputs
         * @param temperature the distillation temperature
         * @return the loss
         */
        public Var loss(Var x, List<Tensor> memberLogits, float temperature) {
            List<Var> z = heads(x, true);
            DistillationLoss kd = new DistillationLoss(temperature);
            Var loss = null;
            for (int m = 0; m < z.size(); m++) {
                Var l = kd.compute(z.get(m), memberLogits.get(m));
                loss = loss == null ? l : loss.add(l);
            }
            return loss;
        }

        /** @return all parameters */
        public List<Var> parameters() {
            List<Var> p = new ArrayList<>(trunk.parameters());
            for (Dense d : heads) {
                p.addAll(d.parameters());
            }
            return p;
        }

        /** @return one audit member per head */
        public List<EnsembleAudit.Member> auditMembers() {
            List<EnsembleAudit.Member> out = new ArrayList<>();
            for (int i = 0; i < heads.size(); i++) {
                int m = i;
                out.add(x -> EnsembleAudit.softmaxRows(Var.noGrad(() -> heads(Var.of(x), false).get(m).value())));
            }
            return out;
        }

        /** @return the number of heads */
        public int size() {
            return heads.size();
        }
    }
}
