package io.github.zzuegg.brainiac.experimental.graph;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Transductive label propagation and Correct and Smooth: parameter-free post-processing
 * that spreads information over the graph after (or instead of) training a model.
 * <ul>
 *   <li>{@link #labelPropagation} (Zhu and Ghahramani, 2002): iterate
 *       {@code Y <- alpha S Y + (1 - alpha) Y0} from the one-hot training labels
 *       {@code Y0} (each class column scaled to unit mass, so a class with fewer labelled
 *       nodes is not drowned out), with {@code S = D^-1/2 A D^-1/2}; uses no features at
 *       all;</li>
 *   <li>{@link #correctAndSmooth} (Huang et al., 2021, <a href="https://arxiv.org/abs/2010.13993">arXiv:2010.13993</a>):
 *       start from the class probabilities {@code Z} of a cheap base predictor (an MLP on
 *       features, trained without the graph). <em>Correct</em>: the residuals
 *       {@code Y - Z} on training nodes are propagated (errors are correlated along edges),
 *       rescaled to the typical training error ("autoscale") and added to {@code Z}.
 *       <em>Smooth</em>: training rows are reset to their true labels and the corrected
 *       predictions are propagated once more, like label propagation with a prior.</li>
 * </ul>
 * Neither step has parameters, so the graph is only used at inference; it often matches
 * or beats message-passing networks at a fraction of the training cost.
 */
public final class CorrectAndSmooth {

    private CorrectAndSmooth() {
    }

    private static Tensor propagate(Tensor s, Tensor start, float alpha, int iterations) {
        Tensor y = start;
        for (int i = 0; i < iterations; i++) {
            y = s.matmul(y).mul(alpha).add(start.mul(1 - alpha));
        }
        return y;
    }

    private static Tensor oneHot(int[] labels, boolean[] train, int classes) {
        float[] y = new float[labels.length * classes];
        for (int i = 0; i < labels.length; i++) {
            if (train[i]) {
                y[i * classes + labels[i]] = 1;
            }
        }
        return Tensor.of(Shape.of(labels.length, classes), y);
    }

    private static Tensor trainMask(boolean[] train) {
        float[] m = new float[train.length];
        for (int i = 0; i < train.length; i++) {
            m[i] = train[i] ? 1 : 0;
        }
        return Tensor.of(Shape.of(train.length, 1), m);
    }

    /**
     * Label propagation.
     *
     * @param adjacency {@code [n, n]}
     * @param labels node labels (read on training nodes only)
     * @param train which labels are known
     * @param classes the class count
     * @param alpha the propagation weight
     * @param iterations propagation steps
     * @return soft labels {@code [n, classes]}
     */
    public static Tensor labelPropagation(Tensor adjacency, int[] labels, boolean[] train, int classes, float alpha,
            int iterations) {
        float[] count = new float[classes];
        for (int i = 0; i < labels.length; i++) {
            if (train[i]) {
                count[labels[i]]++;
            }
        }
        for (int c = 0; c < classes; c++) {
            count[c] = 1 / Math.max(1, count[c]);
        }
        // class-mass normalisation: every class starts with the same total mass
        Tensor start = oneHot(labels, train, classes).mul(Tensor.of(Shape.of(1, classes), count));
        return propagate(GraphMath.symmetricNormalize(adjacency), start, alpha, iterations);
    }

    /**
     * Correct and Smooth.
     *
     * @param adjacency {@code [n, n]}
     * @param base the base predictor's class probabilities {@code [n, classes]}
     * @param labels node labels (read on training nodes only)
     * @param train which labels are known
     * @param correctAlpha the propagation weight of the correct step
     * @param smoothAlpha the propagation weight of the smooth step
     * @param iterations propagation steps per stage
     * @return the final scores {@code [n, classes]}
     */
    public static Tensor correctAndSmooth(Tensor adjacency, Tensor base, int[] labels, boolean[] train,
            float correctAlpha, float smoothAlpha, int iterations) {
        int n = labels.length;
        int classes = base.shape().dim(1);
        Tensor s = GraphMath.symmetricNormalize(adjacency);
        Tensor y = oneHot(labels, train, classes);
        Tensor mask = trainMask(train);
        Tensor error = y.sub(base).mul(mask);
        Tensor spread = propagate(s, error, correctAlpha, iterations);
        float[] e = error.toArray();
        double sigma = 0;
        int count = 0;
        for (int i = 0; i < n; i++) {
            if (train[i]) {
                for (int c = 0; c < classes; c++) {
                    sigma += Math.abs(e[i * classes + c]);
                }
                count++;
            }
        }
        sigma /= Math.max(1, count);
        float[] sp = spread.toArray();
        float[] scale = new float[n];
        for (int i = 0; i < n; i++) {
            double l1 = 0;
            for (int c = 0; c < classes; c++) {
                l1 += Math.abs(sp[i * classes + c]);
            }
            scale[i] = (float) (l1 > 1e-12 ? sigma / l1 : 0);
        }
        Tensor corrected = base.add(spread.mul(Tensor.of(Shape.of(n, 1), scale)));
        Tensor start = corrected.mul(mask.neg().add(1f)).add(y);
        return propagate(s, start, smoothAlpha, iterations);
    }
}
