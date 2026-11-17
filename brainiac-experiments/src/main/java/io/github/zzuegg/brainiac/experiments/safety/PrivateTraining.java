package io.github.zzuegg.brainiac.experiments.safety;

import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.experimental.train.DpSgd;
import io.github.zzuegg.brainiac.experimental.train.MachineUnlearning;
import io.github.zzuegg.brainiac.experimental.train.OneRunAudit;
import io.github.zzuegg.brainiac.experimental.train.RdpAccountant;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Differentially private training and its audit, on the {@link ToyTasks#digits} task (8x8
 * noisy glyphs, 10 classes, an MLP with one hidden layer).
 *
 * <p>{@link #budgets}: {@link DpSgd} at several privacy budgets, with the noise multiplier
 * set by the {@link RdpAccountant} so the run ends exactly at the budget, against the same
 * loop without noise or clipping. All runs use the same sampling rate, steps and step
 * size.
 *
 * <p>{@link #audit}: a one-run audit ({@link OneRunAudit}). Canaries are random binary
 * images (no glyph behind them) with random labels, each put into the training data with
 * probability one half. The attacker's score is the model's log-probability of the canary's label; it guesses "in" for the
 * highest and "out" for the lowest scores. The empirical {@code ε} lower bound is compared
 * with the accountant's {@code ε} for non-private SGD, DP-SGD, and the non-private model
 * after NPO unlearning of the canaries it was trained on. With {@code 2k} guesses the
 * bound cannot exceed what {@code 2k} right answers refute (about 2.1 for 40 guesses at
 * {@code β = 0.05}), so it shows a gap, not the true {@code ε}.
 *
 * <p>Abadi et al. 2016, <a href="https://arxiv.org/abs/1607.00133">arXiv:1607.00133</a>;
 * Steinke, Nasr and Jagielski 2023, <a href="https://arxiv.org/abs/2305.08846">arXiv:2305.08846</a>.
 */
public final class PrivateTraining {

    /**
     * One privacy budget.
     *
     * @param targetEpsilon the budget, infinite for the non-private run
     * @param noiseMultiplier the noise multiplier used
     * @param epsilon the accountant's {@code ε} at the end of the run
     * @param accuracy test accuracy
     */
    public record Budget(double targetEpsilon, double noiseMultiplier, double epsilon, double accuracy) {
    }

    /**
     * One audited model.
     *
     * @param method the training
     * @param theoreticalEpsilon the accountant's {@code ε}, infinite without a guarantee
     * @param audit the audit's outcome
     * @param membershipAuc the loss-threshold AUC between included and excluded canaries
     * @param accuracy test accuracy
     */
    public record Audit(String method, double theoreticalEpsilon, OneRunAudit.Result audit, double membershipAuc,
            double accuracy) {
    }

    static final double DELTA = 1e-5;
    private static final double LR = 2;
    private static final int SIDE = 8;
    private static final int CLASSES = 10;

    private PrivateTraining() {
    }

    /**
     * Trains at each budget.
     *
     * @param budgets the target {@code ε} values
     * @param perClass training glyphs per class
     * @param steps DP-SGD steps
     * @param samplingRate the sampling rate
     * @param seed the seed
     * @return the non-private run, then one entry per budget
     */
    public static List<Budget> budgets(double[] budgets, int perClass, int steps, double samplingRate, long seed) {
        Dataset train = dataset(ToyTasks.digits(CLASSES, perClass, SIDE, 0.05, seed, seed + 1));
        Dataset test = dataset(ToyTasks.digits(CLASSES, 50, SIDE, 0.05, seed, seed + 2));
        List<Budget> out = new ArrayList<>();
        Model plain = model(seed);
        DpSgd.of(plain, Loss.CROSS_ENTROPY).clipNorm(1e9).noiseMultiplier(0).learningRate(LR)
                .samplingRate(samplingRate).seed(seed).train(train, steps);
        out.add(new Budget(Double.POSITIVE_INFINITY, 0, Double.POSITIVE_INFINITY,
                MachineUnlearning.accuracy(plain, test)));
        for (double eps : budgets) {
            double sigma = RdpAccountant.noiseMultiplierFor(eps, samplingRate, steps, DELTA);
            Model m = model(seed);
            DpSgd dp = DpSgd.of(m, Loss.CROSS_ENTROPY).clipNorm(1).noiseMultiplier(sigma).learningRate(LR)
                    .samplingRate(samplingRate).seed(seed).train(train, steps);
            out.add(new Budget(eps, sigma, dp.epsilon(DELTA), MachineUnlearning.accuracy(m, test)));
        }
        return out;
    }

    /**
     * Audits non-private training, DP-SGD and unlearning with canaries.
     *
     * @param perClass clean training glyphs per class
     * @param canaries canaries, each included with probability one half
     * @param guesses guesses of each kind ("in" and "out")
     * @param dpEpsilon the DP-SGD budget
     * @param steps training steps
     * @param seed the seed
     * @return non-private, DP-SGD, then non-private plus NPO unlearning
     */
    public static List<Audit> audit(int perClass, int canaries, int guesses, double dpEpsilon, int steps,
            long seed) {
        double q = 0.1;
        Dataset clean = dataset(ToyTasks.digits(CLASSES, perClass, SIDE, 0.05, seed, seed + 1));
        Dataset test = dataset(ToyTasks.digits(CLASSES, 50, SIDE, 0.05, seed, seed + 2));
        SplittableRandom r = new SplittableRandom(seed + 4);
        float[][] cx = new float[canaries][];
        float[][] cy = new float[canaries][CLASSES];
        boolean[] included = new boolean[canaries];
        List<float[]> tx = new ArrayList<>();
        List<float[]> ty = new ArrayList<>();
        List<float[]> inX = new ArrayList<>();
        List<float[]> inY = new ArrayList<>();
        List<float[]> outX = new ArrayList<>();
        List<float[]> outY = new ArrayList<>();
        for (int i = 0; i < clean.size(); i++) {
            tx.add(clean.featuresOf(i));
            ty.add(clean.targetsOf(i));
        }
        for (int c = 0; c < canaries; c++) {
            cx[c] = new float[SIDE * SIDE];
            for (int p = 0; p < cx[c].length; p++) {
                cx[c][p] = r.nextDouble() < 0.35 ? 1 : 0;
            }
            cy[c][r.nextInt(CLASSES)] = 1;
            included[c] = r.nextBoolean();
            (included[c] ? inX : outX).add(cx[c]);
            (included[c] ? inY : outY).add(cy[c]);
            if (included[c]) {
                tx.add(cx[c]);
                ty.add(cy[c]);
            }
        }
        Dataset train = Dataset.of(tx.toArray(new float[0][]), ty.toArray(new float[0][]));
        Dataset canarySet = Dataset.of(cx, cy);
        Dataset in = Dataset.of(inX.toArray(new float[0][]), inY.toArray(new float[0][]));
        Dataset outSet = Dataset.of(outX.toArray(new float[0][]), outY.toArray(new float[0][]));

        List<Audit> out = new ArrayList<>();
        Model plain = model(seed);
        DpSgd.of(plain, Loss.CROSS_ENTROPY).clipNorm(1e9).noiseMultiplier(0).learningRate(LR).samplingRate(q)
                .seed(seed).train(train, steps);
        out.add(audited("non-private", Double.POSITIVE_INFINITY, plain, canarySet, included, in, outSet, test,
                guesses));
        double sigma = RdpAccountant.noiseMultiplierFor(dpEpsilon, q, steps, DELTA);
        Model dp = model(seed);
        DpSgd run = DpSgd.of(dp, Loss.CROSS_ENTROPY).clipNorm(1).noiseMultiplier(sigma).learningRate(LR)
                .samplingRate(q).seed(seed).train(train, steps);
        out.add(audited("DP-SGD", run.epsilon(DELTA), dp, canarySet, included, in, outSet, test, guesses));
        Model unlearned = plain.copy(plain.parameters().get(0).value().device());
        MachineUnlearning.of(unlearned, Loss.CROSS_ENTROPY).method(MachineUnlearning.Method.NPO)
                .optimizer(() -> Optimizer.adam(0.003f)).steps(150).batchSize(16).seed(seed).run(in, clean);
        out.add(audited("NPO unlearned", Double.POSITIVE_INFINITY, unlearned, canarySet, included, in, outSet, test,
                guesses));
        return out;
    }

    private static Audit audited(String method, double eps, Model m, Dataset canaries, boolean[] included,
            Dataset in, Dataset out, Dataset test, int guesses) {
        double[] losses = MachineUnlearning.sampleLosses(m, canaries);
        double[] scores = new double[losses.length];
        for (int i = 0; i < scores.length; i++) {
            scores[i] = -losses[i];
        }
        return new Audit(method, eps, OneRunAudit.audit(scores, included, guesses, guesses, 0.05),
                MachineUnlearning.membershipAuc(m, in, out), MachineUnlearning.accuracy(m, test));
    }

    static Model model(long seed) {
        return Model.sequential().input(SIDE * SIDE).dense(32, Activation.TANH).dense(CLASSES, Activation.SOFTMAX)
                .seed(seed).build();
    }

    static Dataset dataset(ToyTasks.Data d) {
        float[][] y = new float[d.size()][d.classes()];
        for (int i = 0; i < y.length; i++) {
            y[i][d.y()[i]] = 1;
        }
        return Dataset.of(d.x(), y);
    }
}
