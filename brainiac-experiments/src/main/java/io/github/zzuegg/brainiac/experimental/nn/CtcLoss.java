package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Ctc;
import io.github.zzuegg.brainiac.nn.Loss;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;

/**
 * Connectionist Temporal Classification (Graves et al., 2006,
 * <a href="https://www.cs.toronto.edu/~graves/icml_2006.pdf">ICML</a>) as a {@link Loss}
 * over logits, with class 0 as the blank. The forward-backward recursion and the decoders
 * are the core {@link Ctc}; this class only adds the log-softmax in front and the
 * {@link Loss} adapter, so CTC drops into {@code Trainer}s and {@code Model.loss} like any
 * other loss.
 *
 * <p>As a {@link Loss}, predictions are logits {@code [batch, frames, classes]} and targets
 * are {@code [batch, maxLength]} label ids {@code 1..classes-1}, padded with -1.
 */
public final class CtcLoss implements Loss {
    /** The shared instance; CTC has no settings besides the blank, which is class 0. */
    public static final CtcLoss INSTANCE = new CtcLoss();

    private CtcLoss() {
    }

    @Override
    public Var compute(Var prediction, Tensor target) {
        Shape s = prediction.shape();
        if (s.rank() != 3 || target.shape().rank() != 2 || target.shape().dim(0) != s.dim(0)) {
            throw new IllegalArgumentException("CTC takes logits [batch, frames, classes] and labels [batch, length],"
                    + " got " + s + " and " + target.shape());
        }
        int b = s.dim(0);
        int len = target.shape().dim(1);
        float[] t = target.toArray();
        int[][] labels = new int[b][];
        for (int i = 0; i < b; i++) {
            List<Integer> l = new ArrayList<>();
            for (int j = 0; j < len; j++) {
                int v = Math.round(t[i * len + j]);
                if (v >= 0) {
                    l.add(v);
                }
            }
            labels[i] = l.stream().mapToInt(Integer::intValue).toArray();
        }
        return loss(prediction, labels);
    }

    /**
     * The mean CTC loss over a batch: {@code Ctc.loss(logSoftmax(logits), labels, null, 0)}.
     *
     * @param logits {@code [batch, frames, classes]}
     * @param labels one label sequence per batch element, ids {@code 1..classes-1}
     * @return the scalar loss
     */
    public static Var loss(Var logits, int[][] labels) {
        return Ctc.loss(Activation.LOG_SOFTMAX.forward(logits), labels, null, 0);
    }

    /**
     * Best-path decoding: the most likely class per frame, repeats merged, blanks dropped
     * ({@link Ctc#greedy} with blank 0; the argmax of logits is that of log-probabilities).
     *
     * @param logits {@code [batch, frames, classes]}
     * @return the decoded label sequence per batch element
     */
    public static int[][] greedyDecode(Tensor logits) {
        return Ctc.greedy(logits, 0);
    }
}
