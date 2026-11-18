package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.data.Dataset;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * SCRUB, approximate unlearning by a teacher-student scheme (Kurmanji et al., NeurIPS 2023,
 * <a href="https://arxiv.org/abs/2302.09880">arXiv:2302.09880</a>). The student starts as a
 * copy of the original model (the frozen teacher) and alternates "max-steps" that push its
 * output away from the teacher on the forget set, maximising {@code KL(teacher ‖ student)},
 * with "min-steps" that keep it close to the teacher and to the labels on the retain set,
 * minimising {@code KL(teacher ‖ student) + α CE}. Max-steps stop after a few epochs while
 * min-steps continue, which repairs retain performance. Models output class probabilities
 * (softmax), as with {@link Loss#CROSS_ENTROPY}.
 */
public final class Scrub {

    private final Model student;
    private final Model teacher;
    private int epochs = 10;
    private int maxEpochs = 4;
    private double alpha = 1;
    private float learningRate = 0.003f;
    private int batchSize = 32;
    private long seed = 1;

    private Scrub(Model student, Model teacher) {
        this.student = Objects.requireNonNull(student);
        this.teacher = Objects.requireNonNull(teacher);
    }

    /**
     * SCRUB.
     *
     * @param student the model to unlearn, changed in place (a copy of the teacher)
     * @param teacher the original model, not changed
     * @return the settings
     */
    public static Scrub of(Model student, Model teacher) {
        return new Scrub(student, teacher);
    }

    /**
     * @param total epochs of min-steps
     * @param max epochs that also run max-steps
     * @return this
     */
    public Scrub epochs(int total, int max) {
        this.epochs = total;
        this.maxEpochs = max;
        return this;
    }

    /**
     * @param a the weight of the retain cross-entropy
     * @return this
     */
    public Scrub alpha(double a) {
        this.alpha = a;
        return this;
    }

    /**
     * @param lr Adam's step size
     * @return this
     */
    public Scrub learningRate(float lr) {
        this.learningRate = lr;
        return this;
    }

    /**
     * @param b the batch size
     * @return this
     */
    public Scrub batchSize(int b) {
        this.batchSize = b;
        return this;
    }

    /**
     * @param s the seed
     * @return this
     */
    public Scrub seed(long s) {
        this.seed = s;
        return this;
    }

    /**
     * Unlearns.
     *
     * @param forget the forget set
     * @param retain the retain set
     */
    public void run(Dataset forget, Dataset retain) {
        Optimizer opt = Optimizer.adam(learningRate);
        SplittableRandom r = new SplittableRandom(seed);
        for (int e = 0; e < epochs; e++) {
            if (e < maxEpochs) {
                for (Dataset b : batches(forget, r)) {
                    kl(b).neg().backward();
                    opt.step(student.parameters());
                }
            }
            for (Dataset b : batches(retain, r)) {
                Var p = student.forward(Var.of(b.features()), true);
                kl(p, b).add(Loss.CROSS_ENTROPY.compute(p, b.targets()).mul((float) alpha)).backward();
                opt.step(student.parameters());
            }
        }
    }

    private Var kl(Dataset b) {
        return kl(student.forward(Var.of(b.features()), true), b);
    }

    private Var kl(Var studentProbabilities, Dataset b) {
        Tensor t = teacher.predict(b.features());
        return Loss.KL_DIVERGENCE.compute(studentProbabilities, t);
    }

    private List<Dataset> batches(Dataset d, SplittableRandom r) {
        int[] idx = ShardedUnlearning.permutation(d.size(), r);
        List<Dataset> out = new ArrayList<>();
        for (int b = 0; b < d.size(); b += batchSize) {
            List<Integer> rows = new ArrayList<>();
            for (int i = b; i < Math.min(d.size(), b + batchSize); i++) {
                rows.add(idx[i]);
            }
            out.add(ShardedUnlearning.subset(d, rows));
        }
        return out;
    }
}
