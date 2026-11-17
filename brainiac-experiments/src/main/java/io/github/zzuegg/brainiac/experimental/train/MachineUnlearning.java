package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.data.Batch;
import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.Supplier;

/**
 * Approximate machine unlearning: makes a trained model behave as if a <em>forget set</em>
 * had not been in its training data, without retraining from scratch, while a
 * <em>retain set</em> keeps the rest of its behaviour in place. Each step adds a forgetting
 * term on a forget batch to the ordinary loss on a retain batch (weighted by
 * {@code retainWeight}):
 *
 * <ul>
 *   <li>{@link Method#GRADIENT_ASCENT}: the negated loss, so the forget samples are
 *       actively un-learned ("NegGrad+"); stops once the forget loss passes a ceiling,
 *       since ascent is unbounded;</li>
 *   <li>{@link Method#RANDOM_LABELS}: the loss towards a random wrong class, a bounded
 *       alternative (Graves et al. 2021, "Amnesiac Machine Learning",
 *       <a href="https://arxiv.org/abs/2010.10981">arXiv:2010.10981</a>).</li>
 * </ul>
 *
 * <p>{@link #membershipAuc} is a loss-threshold membership-inference attack (Yeom et al.
 * 2018, <a href="https://arxiv.org/abs/1709.01604">arXiv:1709.01604</a>): an AUC near 0.5
 * between the forget set and unseen data means an attacker cannot tell that the forget set
 * was ever trained on. Kurmanji et al. 2023, "Towards Unbounded Machine Unlearning",
 * <a href="https://arxiv.org/abs/2302.09880">arXiv:2302.09880</a>, compare these baselines.
 */
public final class MachineUnlearning {

    /** How the forget set is unlearned. */
    public enum Method {
        /** Gradient ascent on the forget loss. */
        GRADIENT_ASCENT,
        /** Training the forget samples towards random wrong classes. */
        RANDOM_LABELS
    }

    /**
     * One unlearning step.
     *
     * @param step the step number
     * @param forgetLoss the loss on the forget batch (with its true targets)
     * @param retainLoss the loss on the retain batch
     */
    public record Step(int step, double forgetLoss, double retainLoss) {
    }

    private final Model model;
    private final Loss loss;
    private Method method = Method.GRADIENT_ASCENT;
    private Supplier<Optimizer> optimizer = () -> Optimizer.adam(1e-3f);
    private int steps = 100;
    private int batchSize = 32;
    private double retainWeight = 1;
    private double ceiling = 5;
    private long seed = 1;

    private MachineUnlearning(Model model, Loss loss) {
        this.model = Objects.requireNonNull(model);
        this.loss = Objects.requireNonNull(loss);
    }

    /**
     * Sets up unlearning for a trained model.
     *
     * @param model the model, changed in place
     * @param loss the training loss (targets are one-hot for {@link Method#RANDOM_LABELS})
     * @return the settings
     */
    public static MachineUnlearning of(Model model, Loss loss) {
        return new MachineUnlearning(model, loss);
    }

    /**
     * @param m the method
     * @return this
     */
    public MachineUnlearning method(Method m) {
        this.method = Objects.requireNonNull(m);
        return this;
    }

    /**
     * @param o makes the optimizer
     * @return this
     */
    public MachineUnlearning optimizer(Supplier<Optimizer> o) {
        this.optimizer = Objects.requireNonNull(o);
        return this;
    }

    /**
     * @param n the maximum number of steps
     * @return this
     */
    public MachineUnlearning steps(int n) {
        this.steps = n;
        return this;
    }

    /**
     * @param n the forget and retain batch size
     * @return this
     */
    public MachineUnlearning batchSize(int n) {
        this.batchSize = n;
        return this;
    }

    /**
     * @param w the weight of the retain loss
     * @return this
     */
    public MachineUnlearning retainWeight(double w) {
        this.retainWeight = w;
        return this;
    }

    /**
     * @param c stop gradient ascent once the mean forget loss passes this
     * @return this
     */
    public MachineUnlearning forgetLossCeiling(double c) {
        this.ceiling = c;
        return this;
    }

    /**
     * @param s the seed for batch order and random labels
     * @return this
     */
    public MachineUnlearning seed(long s) {
        this.seed = s;
        return this;
    }

    /**
     * Unlearns the forget set.
     *
     * @param forget the samples to forget
     * @param retain samples whose behaviour should be kept (the rest of the training data)
     * @return the steps taken
     */
    public List<Step> run(Dataset forget, Dataset retain) {
        Optimizer opt = optimizer.get();
        SplittableRandom r = new SplittableRandom(seed);
        List<Var> params = model.parameters();
        List<Step> history = new ArrayList<>();
        List<Batch> fb = new ArrayList<>();
        List<Batch> rb = new ArrayList<>();
        for (int s = 0; s < steps; s++) {
            if (fb.isEmpty()) {
                fb.addAll(forget.shuffle(r.nextLong()).batches(batchSize));
            }
            if (rb.isEmpty()) {
                rb.addAll(retain.shuffle(r.nextLong()).batches(batchSize));
            }
            Batch f = fb.removeLast();
            Batch k = rb.removeLast();
            Var pf = model.forward(Var.of(f.features()), true);
            Var lf = loss.compute(pf, f.targets());
            double forgetLoss = lf.value().item();
            if (method == Method.GRADIENT_ASCENT && forgetLoss > ceiling) {
                break;
            }
            Var term = method == Method.GRADIENT_ASCENT ? lf.neg()
                    : loss.compute(pf, randomWrong(f.targets(), r));
            Var lr = loss.compute(model.forward(Var.of(k.features()), true), k.targets());
            term.add(lr.mul((float) retainWeight)).backward();
            opt.step(params);
            history.add(new Step(s, forgetLoss, lr.value().item()));
        }
        return history;
    }

    private static Tensor randomWrong(Tensor oneHot, SplittableRandom r) {
        float[] t = oneHot.toArray();
        int c = oneHot.shape().dim(-1);
        float[] out = new float[t.length];
        for (int i = 0; i < t.length / c; i++) {
            int label = 0;
            for (int j = 1; j < c; j++) {
                if (t[i * c + j] > t[i * c + label]) {
                    label = j;
                }
            }
            int wrong = (label + 1 + r.nextInt(c - 1)) % c;
            out[i * c + wrong] = 1;
        }
        return Tensor.of(oneHot.shape(), out);
    }

    /**
     * The share of samples whose most probable output is the target's largest entry.
     *
     * @param model a classifier
     * @param data labelled data with one-hot targets
     * @return the accuracy, 0 for empty data
     */
    public static double accuracy(Model model, Dataset data) {
        if (data.size() == 0) {
            return 0;
        }
        int ok = 0;
        float[][] p = model.predict(features(data));
        for (int i = 0; i < data.size(); i++) {
            ok += argmax(p[i]) == argmax(data.targetsOf(i)) ? 1 : 0;
        }
        return ok / (double) data.size();
    }

    /**
     * Per-sample cross-entropy, {@code -log p(target)}, of a classifier.
     *
     * @param model a classifier whose outputs are probabilities
     * @param data labelled data with one-hot targets
     * @return one loss per sample
     */
    public static double[] sampleLosses(Model model, Dataset data) {
        float[][] p = model.predict(features(data));
        double[] out = new double[data.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = -Math.log(Math.max(p[i][argmax(data.targetsOf(i))], 1e-12));
        }
        return out;
    }

    /**
     * The loss-threshold membership-inference AUC: the probability that a random member
     * has a lower loss than a random non-member (ties count half). 0.5 means members are
     * indistinguishable; near 1 means the model visibly remembers them.
     *
     * @param model a classifier
     * @param members samples that may have been trained on
     * @param nonMembers samples from the same distribution that were not
     * @return the AUC
     */
    public static double membershipAuc(Model model, Dataset members, Dataset nonMembers) {
        return auc(sampleLosses(model, members), sampleLosses(model, nonMembers));
    }

    /**
     * The AUC of telling members from non-members by a low score.
     *
     * @param memberScores scores (losses) of members
     * @param nonMemberScores scores of non-members
     * @return the probability that a member scores lower, ties counting half
     */
    public static double auc(double[] memberScores, double[] nonMemberScores) {
        double s = 0;
        for (double m : memberScores) {
            for (double n : nonMemberScores) {
                s += m < n ? 1 : m == n ? 0.5 : 0;
            }
        }
        return s / ((double) memberScores.length * nonMemberScores.length);
    }

    private static float[][] features(Dataset data) {
        float[][] x = new float[data.size()][];
        for (int i = 0; i < x.length; i++) {
            x[i] = data.featuresOf(i);
        }
        return x;
    }

    private static int argmax(float[] p) {
        int b = 0;
        for (int i = 1; i < p.length; i++) {
            if (p[i] > p[b]) {
                b = i;
            }
        }
        return b;
    }
}
