package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Hidden-state distillation: the student's intermediate features, mapped through a learned
 * linear projection, regress onto the teacher's features at the matching layer
 * (FitNets, Romero et al. 2015, <a href="https://arxiv.org/abs/1412.6550">arXiv:1412.6550</a>;
 * TinyBERT, Jiao et al. 2020, <a href="https://arxiv.org/abs/1909.10351">arXiv:1909.10351</a>):
 * {@code mean((h_S W - h_T)^2)}. The projection {@code W [studentDim, teacherDim]} lets a
 * narrow student match a wide teacher, and is trained together with the student
 * ({@link #parameters()}). Matching features gives the student a dense, per-layer target
 * instead of only the teacher's output distribution.
 *
 * <p>{@link #attentionTransfer} matches attention maps, TinyBERT's other layer-wise loss
 * (and Zagoruyko and Komodakis 2017, <a href="https://arxiv.org/abs/1612.03928">arXiv:1612.03928</a>).
 *
 * <p>As a core {@link Loss} the prediction is the student's hidden state and the target the
 * teacher's, so it adds to a logit distillation loss in any training loop.
 */
public final class FeatureDistillationLoss implements Loss {

    private final Var projection;

    /**
     * @param studentDim the width of the student's features
     * @param teacherDim the width of the teacher's features
     * @param seed the initialization seed of the projection
     */
    public FeatureDistillationLoss(int studentDim, int teacherDim, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[] w = new float[studentDim * teacherDim];
        double scale = 1 / Math.sqrt(studentDim);
        for (int i = 0; i < w.length; i++) {
            w[i] = (float) (r.nextGaussian() * scale);
        }
        this.projection = Var.param(Tensor.of(Shape.of(studentDim, teacherDim), w));
    }

    /** @return the learned projection {@code [studentDim, teacherDim]} */
    public Var projection() {
        return projection;
    }

    /** @return the trainable values: the projection */
    public List<Var> parameters() {
        return List.of(projection);
    }

    @Override
    public Var compute(Var studentHidden, Tensor teacherHidden) {
        return studentHidden.matmul(projection).sub(teacherHidden).square().mean();
    }

    /**
     * Attention transfer: the mean squared difference of row-normalized attention maps (or
     * any non-negative maps of the same shape).
     *
     * @param student the student's attention weights {@code [..., queries, keys]}
     * @param teacher the teacher's attention weights, same shape
     * @return the loss
     */
    public static Var attentionTransfer(Var student, Tensor teacher) {
        Var s = student.div(student.sum(-1, true).add(1e-8f));
        Tensor t = teacher.div(teacher.sum(-1, true).add(1e-8f));
        return s.sub(t).square().mean();
    }

    @Override
    public String toString() {
        return "FeatureDistillation(" + projection.shape() + ")";
    }
}
