package io.github.zzuegg.brainiac.experimental.compress;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Decoupled knowledge distillation, DKD (Zhao et al. 2022, <a href="https://arxiv.org/abs/2203.08679">arXiv:2203.08679</a>).
 * The classic KD loss splits exactly into
 * <ul>
 * <li>target-class KD, {@code TCKD = KL(b_T || b_S)} over the binary split
 *     {@code b = [p_target, 1 - p_target]}: how confident the teacher is in the true class;</li>
 * <li>non-target-class KD, {@code NCKD = KL(p^_T || p^_S)} over the non-target classes
 *     renormalised among themselves: the "dark knowledge" of which wrong classes look alike;</li>
 * </ul>
 * with {@code KD = TCKD + (1 - p^T_target) NCKD}, so the more confident the teacher, the more
 * the dark knowledge is suppressed. DKD weights the two independently,
 * {@code T^2 (alpha TCKD + beta NCKD)}, typically {@code alpha = 1, beta = 8}.
 *
 * <p>As a core {@link Loss}, the prediction is the student's logits and the target the
 * teacher's logits; the target class is then the teacher's top class. {@link #compute(Var, Tensor, Tensor)}
 * takes the labels instead.
 */
public final class DecoupledKdLoss implements Loss {

    private static final float MASK = 1000f;

    private final float temperature;
    private final float alpha;
    private final float beta;

    /**
     * @param temperature the softening temperature, for example 4
     * @param alpha the weight of target-class KD, for example 1
     * @param beta the weight of non-target-class KD, for example 8
     */
    public DecoupledKdLoss(float temperature, float alpha, float beta) {
        if (!(temperature > 0)) {
            throw new IllegalArgumentException("temperature must be positive");
        }
        this.temperature = temperature;
        this.alpha = alpha;
        this.beta = beta;
    }

    @Override
    public Var compute(Var studentLogits, Tensor teacherLogits) {
        return compute(studentLogits, teacherLogits, teacherLogits.argmax(1));
    }

    /**
     * DKD with given labels.
     *
     * @param studentLogits {@code [rows, classes]}
     * @param teacherLogits {@code [rows, classes]}
     * @param labels {@code [rows]} class ids
     * @return the loss, averaged over rows
     */
    public Var compute(Var studentLogits, Tensor teacherLogits, Tensor labels) {
        int rows = studentLogits.shape().dim(0);
        int classes = studentLogits.shape().dim(1);
        Tensor gt = oneHot(labels, rows, classes);
        Tensor notGt = gt.neg().add(1f);
        // teacher, as constants
        Tensor logT = logSoftmax(teacherLogits.div(temperature));
        Tensor pT = logT.exp().mul(gt).sum(1, true);
        Tensor logPt = pT.add(1e-12f).log();
        Tensor log1mPt = pT.neg().add(1f + 1e-12f).log();
        Tensor logTn = logSoftmax(teacherLogits.div(temperature).sub(gt.mul(MASK)));
        // student
        Var logS = Activation.LOG_SOFTMAX.forward(studentLogits.div(temperature));
        Var pS = logS.exp().mul(gt).sum(1, true);
        Var logPs = pS.add(1e-12f).log();
        Var log1mPs = pS.neg().add(1f + 1e-12f).log();
        Var tckd = logPs.neg().add(Var.of(logPt)).mul(pT)
                .add(log1mPs.neg().add(Var.of(log1mPt)).mul(pT.neg().add(1f))).sum();
        Var logSn = Activation.LOG_SOFTMAX.forward(studentLogits.div(temperature).sub(gt.mul(MASK)));
        Var nckd = logSn.neg().add(Var.of(logTn)).mul(logTn.exp().mul(notGt)).sum();
        return tckd.mul(alpha).add(nckd.mul(beta)).mul(temperature * temperature / rows);
    }

    private static Tensor logSoftmax(Tensor x) {
        Tensor shifted = x.sub(x.max(1, true));
        return shifted.sub(shifted.exp().sum(1, true).log());
    }

    private static Tensor oneHot(Tensor labels, int rows, int classes) {
        float[] ids = labels.toArray();
        float[] v = new float[rows * classes];
        for (int i = 0; i < rows; i++) {
            v[i * classes + (int) ids[i]] = 1;
        }
        return Tensor.of(Shape.of(rows, classes), v);
    }

    @Override
    public String toString() {
        return "DecoupledKd(T=" + temperature + ", alpha=" + alpha + ", beta=" + beta + ")";
    }
}
