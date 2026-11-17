package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;

import java.util.List;

/**
 * A momentum ("mean") teacher: a second copy of a network whose weights follow the trained
 * student's as an exponential moving average, {@code teacher <- m teacher + (1 - m) student},
 * and are never trained by gradients. The slowly moving teacher gives stable targets; it is
 * the target network of BYOL, DINO, I-JEPA, MoCo and Mean Teacher. Unlike the core {@code Ema},
 * which shadows the same parameters, this keeps two separate networks that run side by side.
 *
 * <p>Paper: Tarvainen and Valpola, "Mean teachers are better role models" (2017),
 * <a href="https://arxiv.org/abs/1703.01780">arXiv:1703.01780</a>.
 */
public final class MomentumTeacher {

    private final List<Var> teacher;
    private final List<Var> student;
    private float momentum;

    /**
     * Creates the teacher and copies the student's weights into it.
     *
     * @param teacher the teacher's parameters (same structure as the student's)
     * @param student the student's parameters
     * @param momentum the averaging momentum, such as 0.99 or 0.996
     */
    public MomentumTeacher(List<Var> teacher, List<Var> student, float momentum) {
        if (teacher.size() != student.size()) {
            throw new IllegalArgumentException("teacher and student need the same parameters");
        }
        this.teacher = List.copyOf(teacher);
        this.student = List.copyOf(student);
        this.momentum = momentum;
        for (int i = 0; i < teacher.size(); i++) {
            this.teacher.get(i).assign(this.student.get(i).value());
        }
    }

    /** Moves the teacher towards the student. */
    public void update() {
        for (int i = 0; i < teacher.size(); i++) {
            Var t = teacher.get(i);
            t.assign(t.value().mul(momentum).add(student.get(i).value().mul(1 - momentum)));
        }
    }

    /**
     * Changes the momentum, for the cosine increase towards 1 the papers use.
     *
     * @param m the new momentum
     */
    public void momentum(float m) {
        this.momentum = m;
    }

    /** @return the current momentum */
    public float momentum() {
        return momentum;
    }
}
