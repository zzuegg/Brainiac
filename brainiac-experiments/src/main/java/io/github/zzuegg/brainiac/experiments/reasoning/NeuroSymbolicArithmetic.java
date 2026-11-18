package io.github.zzuegg.brainiac.experiments.reasoning;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.NeuralPredicateProgram;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Digit addition with only the sum supervised, the DeepProbLog benchmark, on the
 * MNIST-like {@link ToyTasks#digits} images. A digit classifier is the neural predicate
 * {@code digit(img, d)} of a {@link NeuralPredicateProgram} whose query is
 * {@code sum(img1, img2, s)}; the program's exact probability of the observed sum trains
 * the classifier without any digit label. Because the program, not the network, does the
 * arithmetic, the same classifier then adds numbers of 2, 3 and 4 digits.
 *
 * <p>Baselines: an end-to-end MLP that reads both images and classifies the sum (19
 * classes), which has no route to multi-digit numbers, and the same classifier trained
 * with the true digit labels (the oracle).
 */
public final class NeuroSymbolicArithmetic {

    /** The compared learners. */
    public enum Method {
        /** The neural predicate inside the sum program, all proofs. */
        NEURAL_PREDICATE,
        /** The same, keeping the 3 most probable proofs per example. */
        NEURAL_PREDICATE_TOP3,
        /** An MLP from both images to the sum. */
        END_TO_END,
        /** The digit classifier trained on digit labels. */
        ORACLE
    }

    /**
     * Scores.
     *
     * @param sumAccuracy one-digit sums right on test pairs
     * @param digitAccuracy the implied single-digit accuracy (NaN for end-to-end)
     * @param multiDigit exact accuracy of 2-, 3- and 4-digit sums (NaN for end-to-end)
     */
    public record Scores(double sumAccuracy, double digitAccuracy, double[] multiDigit) {
    }

    private static final int SIDE = 8;

    private NeuroSymbolicArithmetic() {
    }

    private static Tensor images(ToyTasks.Data d, int[] idx) {
        float[] x = new float[idx.length * SIDE * SIDE];
        for (int i = 0; i < idx.length; i++) {
            System.arraycopy(d.x()[idx[i]], 0, x, i * SIDE * SIDE, SIDE * SIDE);
        }
        return Tensor.of(Shape.of(idx.length, SIDE * SIDE), x);
    }

    /**
     * Trains and scores one method.
     *
     * @param method the method
     * @param steps training steps of 32 pairs
     * @param seed the seed
     * @return the scores
     */
    public static Scores run(Method method, int steps, long seed) {
        ToyTasks.Data train = ToyTasks.digits(10, 60, SIDE, 0.1, 3, seed);
        ToyTasks.Data test = ToyTasks.digits(10, 40, SIDE, 0.1, 3, seed + 99);
        SplittableRandom r = new SplittableRandom(seed);
        Model classifier = Model.sequential().input(SIDE * SIDE).dense(64, Activation.RELU).dense(10).seed(seed).build();
        Model e2e = Model.sequential().input(2 * SIDE * SIDE).dense(128, Activation.RELU).dense(19).seed(seed).build();
        Optimizer opt = Optimizer.adam(3e-3f);
        NeuralPredicateProgram program = new NeuralPredicateProgram(2, 10);
        int n = train.y().length;
        for (int s = 0; s < steps; s++) {
            int[] a = new int[32];
            int[] b = new int[32];
            int[] sum = new int[32];
            for (int i = 0; i < 32; i++) {
                a[i] = r.nextInt(n);
                b[i] = r.nextInt(n);
                sum[i] = train.y()[a[i]] + train.y()[b[i]];
            }
            Var loss = switch (method) {
                case NEURAL_PREDICATE, NEURAL_PREDICATE_TOP3 -> {
                    Var pa = softmax(classifier.forward(Var.of(images(train, a)), true));
                    Var pb = softmax(classifier.forward(Var.of(images(train, b)), true));
                    yield program.loss(List.of(pa, pb), program.sumProofs(sum),
                            method == Method.NEURAL_PREDICATE_TOP3 ? 3 : 0);
                }
                case END_TO_END -> Functional.crossEntropy(e2e.forward(Var.of(
                        Tensor.concat(1, images(train, a), images(train, b))), true), ids(sum));
                case ORACLE -> {
                    int[] da = new int[32];
                    for (int i = 0; i < 32; i++) {
                        da[i] = train.y()[a[i]];
                    }
                    yield Functional.crossEntropy(classifier.forward(Var.of(images(train, a)), true), ids(da));
                }
            };
            loss.backward();
            opt.step(method == Method.END_TO_END ? e2e.parameters() : classifier.parameters());
        }
        int m = test.y().length;
        int[] all = new int[m];
        for (int i = 0; i < m; i++) {
            all[i] = i;
        }
        SplittableRandom tr = new SplittableRandom(seed + 5);
        int pairs = 400;
        int[] a = new int[pairs];
        int[] b = new int[pairs];
        for (int i = 0; i < pairs; i++) {
            a[i] = tr.nextInt(m);
            b[i] = tr.nextInt(m);
        }
        if (method == Method.END_TO_END) {
            float[] pred = e2e.forward(Var.of(Tensor.concat(1, images(test, a), images(test, b)))).value().argmax(1)
                    .toArray();
            int right = 0;
            for (int i = 0; i < pairs; i++) {
                right += (int) pred[i] == test.y()[a[i]] + test.y()[b[i]] ? 1 : 0;
            }
            return new Scores(right / (double) pairs, Double.NaN, new double[] {Double.NaN, Double.NaN, Double.NaN});
        }
        float[] digits = classifier.forward(Var.of(images(test, all))).value().argmax(1).toArray();
        int digitRight = 0;
        for (int i = 0; i < m; i++) {
            digitRight += (int) digits[i] == test.y()[i] ? 1 : 0;
        }
        int sumRight = 0;
        for (int i = 0; i < pairs; i++) {
            sumRight += (int) digits[a[i]] + (int) digits[b[i]] == test.y()[a[i]] + test.y()[b[i]] ? 1 : 0;
        }
        double[] multi = new double[3];
        for (int len = 2; len <= 4; len++) {
            int right = 0;
            for (int q = 0; q < 200; q++) {
                long x = 0;
                long y = 0;
                long px = 0;
                long py = 0;
                for (int d = 0; d < len; d++) {
                    int i1 = tr.nextInt(m);
                    int i2 = tr.nextInt(m);
                    x = 10 * x + test.y()[i1];
                    y = 10 * y + test.y()[i2];
                    px = 10 * px + (long) digits[i1];
                    py = 10 * py + (long) digits[i2];
                }
                right += px + py == x + y ? 1 : 0;
            }
            multi[len - 2] = right / 200.0;
        }
        return new Scores(sumRight / (double) pairs, digitRight / (double) m, multi);
    }

    private static Var softmax(Var logits) {
        return logits.sub(logits.logsumexp(1, true)).exp();
    }

    private static Tensor ids(int[] v) {
        float[] f = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            f[i] = v[i];
        }
        return Tensor.of(Shape.of(v.length), f);
    }

    /** @return the methods in report order */
    public static List<Method> methods() {
        return new ArrayList<>(List.of(Method.values()));
    }
}
