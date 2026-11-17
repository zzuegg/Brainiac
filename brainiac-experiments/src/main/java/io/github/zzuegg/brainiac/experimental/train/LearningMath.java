package io.github.zzuegg.brainiac.experimental.train;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;

/**
 * Small tensor helpers shared by the learning rules and optimizers: softmax, one-hot
 * targets, accuracy, and norms and dot products over lists of parameters.
 */
public final class LearningMath {

    private LearningMath() {
    }

    /**
     * Softmax over the last axis of a {@code [rows, classes]} tensor.
     *
     * @param logits the scores
     * @return probabilities, rows summing to 1
     */
    public static Tensor softmax(Tensor logits) {
        Tensor e = logits.sub(logits.max(1, true)).exp();
        return e.div(e.sum(1, true));
    }

    /**
     * One-hot rows for class ids.
     *
     * @param ids {@code [rows]} class ids
     * @param classes the number of classes
     * @return {@code [rows, classes]}, 1 at each id
     */
    public static Tensor oneHot(Tensor ids, int classes) {
        int rows = ids.shape().dim(0);
        return ids.reshape(rows, 1).equalTo(Tensor.arange(classes).to(ids.device()));
    }

    /**
     * The share of rows whose largest score is at the label.
     *
     * @param scores {@code [rows, classes]}
     * @param labels {@code [rows]} class ids
     * @return the accuracy in {@code [0, 1]}
     */
    public static double accuracy(Tensor scores, Tensor labels) {
        float[] s = scores.toArray();
        float[] y = labels.toArray();
        int classes = scores.shape().dim(-1);
        int correct = 0;
        for (int i = 0; i < y.length; i++) {
            int best = 0;
            for (int c = 1; c < classes; c++) {
                if (s[i * classes + c] > s[i * classes + best]) {
                    best = c;
                }
            }
            if (best == Math.round(y[i])) {
                correct++;
            }
        }
        return (double) correct / y.length;
    }

    /**
     * The derivative of tanh from its output, {@code 1 - y^2}.
     *
     * @param y {@code tanh(x)}
     * @return the derivative at {@code x}
     */
    public static Tensor tanhDerivative(Tensor y) {
        return y.square().neg().add(1);
    }

    /**
     * Copies every value of a list of tensors into one array.
     *
     * @param tensors the tensors
     * @return their values, one after another
     */
    public static float[] flatten(List<Tensor> tensors) {
        long total = 0;
        for (Tensor t : tensors) {
            total += t.size();
        }
        float[] out = new float[Math.toIntExact(total)];
        int at = 0;
        for (Tensor t : tensors) {
            float[] v = t.toArray();
            System.arraycopy(v, 0, out, at, v.length);
            at += v.length;
        }
        return out;
    }

    /**
     * @param params parameters
     * @return their current values, copied
     */
    public static List<Tensor> values(List<Var> params) {
        return params.stream().map(p -> Tensor.of(p.shape(), p.value().toArray())).toList();
    }

    /**
     * Writes values into parameters.
     *
     * @param params the parameters
     * @param values one tensor per parameter, same shapes
     */
    public static void assign(List<Var> params, List<Tensor> values) {
        for (int i = 0; i < params.size(); i++) {
            params.get(i).assign(values.get(i));
        }
    }

    /**
     * The cosine of the angle between two vectors.
     *
     * @param a a vector
     * @param b another, same length
     * @return {@code a.b / (|a| |b|)}
     */
    public static double cosine(float[] a, float[] b) {
        double dot = 0;
        double na = 0;
        double nb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += (double) a[i] * b[i];
            na += (double) a[i] * a[i];
            nb += (double) b[i] * b[i];
        }
        return dot / Math.sqrt(na * nb + 1e-30);
    }

    /**
     * @param t a tensor
     * @return its Euclidean norm
     */
    public static float norm(Tensor t) {
        return (float) Math.sqrt(t.square().sum().item());
    }

    /**
     * A scalar tensor broadcastable to any shape.
     *
     * @param value the value
     * @return a {@code []}-shaped tensor
     */
    public static Tensor scalar(float value) {
        return Tensor.full(Shape.of(), value);
    }
}
