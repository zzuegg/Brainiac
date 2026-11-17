package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Knowledge distillation (Hinton, Vinyals and Dean 2015, <a href="https://arxiv.org/abs/1503.02531">arXiv:1503.02531</a>):
 * a student learns from a teacher's <em>soft</em> outputs at temperature {@code T},
 * {@code T^2 KL(softmax(teacher/T) || softmax(student/T))}, which carry the "dark
 * knowledge" of how similar the wrong classes are. The {@code T^2} factor keeps gradient
 * sizes independent of {@code T}.
 *
 * <p>As a core {@link Loss} the prediction is the student's logits and the target the
 * teacher's logits, so a {@code Trainer} can distil from a data set whose targets are a
 * teacher's outputs. {@link #withLabels} mixes in the ordinary hard-label cross-entropy.
 */
public final class DistillationLoss implements Loss {

    private final float temperature;

    /**
     * @param temperature the softening temperature, for example 4
     */
    public DistillationLoss(float temperature) {
        if (!(temperature > 0)) {
            throw new IllegalArgumentException("temperature must be positive");
        }
        this.temperature = temperature;
    }

    @Override
    public Var compute(Var studentLogits, Tensor teacherLogits) {
        Tensor scaled = teacherLogits.div(temperature);
        Tensor shifted = scaled.sub(scaled.max(1, true));
        Tensor logTeacher = shifted.sub(shifted.exp().sum(1, true).log());
        Tensor teacher = logTeacher.exp();
        Var logStudent = Activation.LOG_SOFTMAX.forward(studentLogits.div(temperature));
        int rows = studentLogits.shape().dim(0);
        // KL = sum p_t (log p_t - log p_s)
        return logStudent.neg().add(logTeacher).mul(teacher).sum().mul(temperature * temperature / rows);
    }

    /**
     * {@code alpha * distillation + (1 - alpha) * crossEntropy(student, labels)}.
     *
     * @param studentLogits the student's logits
     * @param teacherLogits the teacher's logits
     * @param labels {@code [rows]} class ids
     * @param alpha the weight of the soft targets, for example 0.9
     * @return the combined loss
     */
    public Var withLabels(Var studentLogits, Tensor teacherLogits, Tensor labels, float alpha) {
        return compute(studentLogits, teacherLogits).mul(alpha)
                .add(Functional.crossEntropy(studentLogits, labels).mul(1 - alpha));
    }

    @Override
    public String toString() {
        return "Distillation(T=" + temperature + ")";
    }
}
