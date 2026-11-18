package io.github.zzuegg.brainiac.experimental.semisup;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.train.MomentumTeacher;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;

/**
 * The three classic semi-supervised consistency losses, added to the supervised loss on
 * unlabelled inputs with a weight ramped up from zero:
 *
 * <ul>
 *   <li>{@link Mode#PSEUDO_LABEL} (Lee 2013): cross-entropy to the argmax of the model's own
 *       prediction, optionally only where its confidence exceeds a threshold.</li>
 *   <li>{@link Mode#PI_MODEL} (Laine and Aila, ICLR 2017,
 *       <a href="https://arxiv.org/abs/1610.02242">arXiv:1610.02242</a>): the squared difference
 *       between the softmax outputs for two stochastic augmentations of the same input.</li>
 *   <li>{@link Mode#MEAN_TEACHER} (Tarvainen and Valpola, NeurIPS 2017,
 *       <a href="https://arxiv.org/abs/1703.01780">arXiv:1703.01780</a>): the same squared
 *       difference, but the target comes from an exponential moving average of the student's
 *       weights ({@link MomentumTeacher}), a smoother and more accurate target.</li>
 * </ul>
 *
 * <p>The caller runs the forward passes (so any model works) and passes the student's logits and
 * the target logits: a second augmented pass of the student for the Π-model, the teacher's pass
 * for Mean Teacher, or the student's own (detached) pass for pseudo-labelling.
 */
public final class ConsistencyTraining {

    /** Which consistency loss to use. */
    public enum Mode {
        /** Hard self-training on the argmax of the model's own prediction. */
        PSEUDO_LABEL,
        /** Squared difference between two augmented passes of the student. */
        PI_MODEL,
        /** Squared difference to an EMA teacher's prediction. */
        MEAN_TEACHER
    }

    private final Mode mode;
    private final float weight;
    private final int rampSteps;
    private final float threshold;

    /**
     * Creates the loss.
     *
     * @param mode the consistency loss
     * @param weight the unlabelled weight after the ramp-up
     * @param rampSteps the length of the sigmoid ramp-up, in steps
     * @param threshold for pseudo-labelling, the confidence below which rows are skipped (0 for
     *     plain pseudo-labelling); ignored by the other modes
     */
    public ConsistencyTraining(Mode mode, float weight, int rampSteps, float threshold) {
        this.mode = mode;
        this.weight = weight;
        this.rampSteps = rampSteps;
        this.threshold = threshold;
    }

    /** @return the mode */
    public Mode mode() {
        return mode;
    }

    /**
     * The sigmoid ramp-up of Laine and Aila, {@code exp(-5 (1 - t)^2)} for {@code t = step /
     * rampSteps} clamped to {@code [0, 1]}.
     *
     * @param step the training step
     * @param rampSteps the ramp length
     * @return the ramp factor in {@code [0, 1]}
     */
    public static float rampUp(int step, int rampSteps) {
        if (rampSteps <= 0) {
            return 1;
        }
        double t = Math.clamp(step / (double) rampSteps, 0, 1);
        return (float) Math.exp(-5 * (1 - t) * (1 - t));
    }

    /**
     * The weighted unlabelled loss at this step.
     *
     * @param studentLogits the student's logits on (augmented) unlabelled inputs, with gradient
     * @param targetLogits the target logits: a second student pass, the teacher's, or the
     *     student's own, used without gradient
     * @param step the training step, for the ramp-up
     * @return {@code ramp(step) * weight * consistency}
     */
    public Var loss(Var studentLogits, Tensor targetLogits, int step) {
        float w = weight * rampUp(step, rampSteps);
        float[][] p = SoftTargets.probabilities(targetLogits, 1f);
        int n = p.length;
        if (mode == Mode.PSEUDO_LABEL) {
            int[] hard = SoftTargets.argmax(p);
            float[] keep = new float[n];
            for (int i = 0; i < n; i++) {
                keep[i] = p[i][hard[i]] >= threshold ? 1 : 0;
            }
            return SoftTargets.softCrossEntropy(studentLogits, SoftTargets.tensor(SoftTargets.oneHot(hard,
                    p[0].length)), keep, n).mul(w);
        }
        Var q = SoftTargets.softmax(studentLogits);
        return q.sub(SoftTargets.tensor(p)).square().sum(-1, false).mean().mul(w);
    }

    /**
     * Creates the Mean Teacher's EMA of the student's weights.
     *
     * @param teacher the teacher's parameters (a second copy of the network)
     * @param student the student's parameters
     * @param momentum the EMA momentum, such as 0.99
     * @return the teacher; call {@link MomentumTeacher#update()} after every optimizer step
     */
    public static MomentumTeacher teacher(List<Var> teacher, List<Var> student, float momentum) {
        return new MomentumTeacher(teacher, student, momentum);
    }
}
