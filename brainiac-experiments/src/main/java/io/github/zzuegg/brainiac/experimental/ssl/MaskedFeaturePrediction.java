package io.github.zzuegg.brainiac.experimental.ssl;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.SelfSupervisedLosses;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;

/**
 * Masked prediction of teacher features, the two recipes that sit between MAE (predict pixels)
 * and I-JEPA (predict the target encoder's final features of hidden blocks).
 *
 * <ul>
 *   <li><b>data2vec</b> (Baevski et al., ICML 2022): an EMA teacher sees the whole input; the
 *       student sees it masked and regresses, at the masked positions, the average of the
 *       teacher's top {@code K} layer outputs, each instance-normalised over the positions
 *       ({@link #data2vecTargets}). Contextual, layer-averaged targets are richer than pixels
 *       and the same recipe serves images, speech and text.</li>
 *   <li><b>iBOT</b> (Zhou et al., ICLR 2022): DINO self-distillation applied to masked patch
 *       tokens ({@link #ibotPatchLoss}): the student's head output at a masked patch must match
 *       the centred, sharpened distribution of the online teacher's head at the same,
 *       unmasked patch (the teacher is an "online tokenizer"), added to DINO's image-level
 *       loss.</li>
 * </ul>
 *
 * <p>Papers: <a href="https://arxiv.org/abs/2202.03555">arXiv:2202.03555</a>,
 * <a href="https://arxiv.org/abs/2111.07832">arXiv:2111.07832</a>.
 */
public final class MaskedFeaturePrediction {

    private MaskedFeaturePrediction() {
    }

    /**
     * data2vec's regression targets: the mean of the last {@code topK} layer outputs, each
     * instance-normalised (zero mean, unit variance over the positions, per channel and sample).
     *
     * @param layers the teacher's layer outputs, first to last, each {@code [n, tokens, d]}
     * @param topK how many of the last layers to average, at most {@code layers.size()}
     * @return {@code [n, tokens, d]}
     */
    public static Tensor data2vecTargets(List<Tensor> layers, int topK) {
        if (topK < 1 || topK > layers.size()) {
            throw new IllegalArgumentException("topK must be in [1, " + layers.size() + "]: " + topK);
        }
        Tensor sum = null;
        for (int i = layers.size() - topK; i < layers.size(); i++) {
            Tensor x = layers.get(i);
            Tensor c = x.sub(x.mean(1, true));
            Tensor normed = c.div(c.square().mean(1, true).add(1e-5f).sqrt());
            sum = sum == null ? normed : sum.add(normed);
        }
        return sum.div(topK);
    }

    /**
     * The regression loss on the masked positions (mean squared error; data2vec uses a smooth
     * L1 that is quadratic at these scales).
     *
     * @param predicted the student's predictions at the masked positions {@code [n, m, d]}
     * @param targets the targets at the same positions
     * @return the mean squared error
     */
    public static Var regression(Var predicted, Tensor targets) {
        return predicted.sub(targets).square().mean();
    }

    /**
     * iBOT's masked-patch self-distillation: DINO's cross-entropy between the teacher's
     * centred, sharpened patch distributions and the student's, over all masked patches.
     *
     * @param student the student's head outputs at the masked patches {@code [n, m, k]}
     * @param teacher the teacher's head outputs at the same patches {@code [n, m, k]}
     * @param center the running center of the teacher's patch outputs {@code [k]}
     * @param studentTemperature 0.1 in the paper
     * @param teacherTemperature 0.04 to 0.07 in the paper
     * @return the loss
     */
    public static Var ibotPatchLoss(Var student, Tensor teacher, Tensor center, float studentTemperature,
            float teacherTemperature) {
        int k = student.shape().dim(-1);
        int rows = Math.toIntExact(student.shape().size() / k);
        return SelfSupervisedLosses.dino(student.reshape(rows, k), Var.of(teacher.reshape(rows, k)), center,
                studentTemperature, teacherTemperature);
    }
}
