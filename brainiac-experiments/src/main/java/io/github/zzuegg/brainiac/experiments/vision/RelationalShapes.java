package io.github.zzuegg.brainiac.experiments.vision;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ShapeImages;
import io.github.zzuegg.brainiac.experimental.nn.RelationNetwork;
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
 * Wave 6 item 2.11: a Sort-of-CLEVR-lite visual question answering task on the two-shape scenes
 * of {@link ShapeSegmentation}. Questions are a type flag plus a class one-hot:
 *
 * <ul>
 *   <li>non-relational "is a shape of class {@code c} in the scene?";</li>
 *   <li>relational "are the two shapes the same class?" (asked on a pool balanced between
 *       same and different pairs).</li>
 * </ul>
 *
 * Both models share a small CNN ({@code conv3, conv2, 3x3 maxpool -> 3x3x16}). The
 * {@link RelationNetwork} treats the 9 feature cells plus their coordinates as objects and the
 * question as its conditioning input; the baseline flattens the feature map, concatenates the
 * question and applies an MLP (CNN+MLP of the paper). Reported: accuracy per question type.
 */
public final class RelationalShapes {

    /** The compared models. */
    public enum Architecture {
        /** CNN cells as objects into a relation network */
        RELATION_NETWORK,
        /** CNN, flatten, concatenate question, MLP */
        CNN_MLP
    }

    private static final int QUESTION = 2 + ShapeImages.CLASSES;
    private static final int CELLS = 3;
    private static final int CHANNELS = 16;

    private RelationalShapes() {
    }

    /** A pool of scenes with the set of classes visible in each. */
    private record Pool(float[] images, boolean[][] present, List<Integer> same, List<Integer> different) {
    }

    private static Pool pool(int n, long seed) {
        Tensor[] s = ShapeSegmentation.scenes(n, seed);
        float[] labels = s[1].toArray();
        int p = ShapeSegmentation.SIDE * ShapeSegmentation.SIDE;
        boolean[][] present = new boolean[n][ShapeImages.CLASSES];
        List<Integer> same = new ArrayList<>();
        List<Integer> different = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            int count = 0;
            for (int j = 0; j < p; j++) {
                int c = (int) labels[i * p + j] - 1;
                if (c >= 0 && !present[i][c]) {
                    present[i][c] = true;
                    count++;
                }
            }
            (count == 1 ? same : different).add(i);
        }
        return new Pool(s[0].toArray(), present, same, different);
    }

    /** {@code {images, questions, answers, relational flags}}. */
    private static Object[] batch(Pool pool, int b, SplittableRandom r) {
        int p = ShapeSegmentation.SIDE * ShapeSegmentation.SIDE;
        float[] img = new float[b * p];
        float[] q = new float[b * QUESTION];
        float[] a = new float[b];
        boolean[] relational = new boolean[b];
        for (int i = 0; i < b; i++) {
            relational[i] = r.nextBoolean();
            int scene;
            if (relational[i]) {
                boolean same = r.nextBoolean();
                List<Integer> from = same ? pool.same() : pool.different();
                scene = from.get(r.nextInt(from.size()));
                q[i * QUESTION] = 1;
                a[i] = same ? 1 : 0;
            } else {
                scene = r.nextInt(pool.present().length);
                boolean ask = r.nextBoolean();
                int c;
                do {
                    c = r.nextInt(ShapeImages.CLASSES);
                } while (pool.present()[scene][c] != ask);
                q[i * QUESTION + 1] = 1;
                q[i * QUESTION + 2 + c] = 1;
                a[i] = ask ? 1 : 0;
            }
            System.arraycopy(pool.images(), scene * p, img, i * p, p);
        }
        int side = ShapeSegmentation.SIDE;
        return new Object[] {Tensor.of(Shape.of(b, 1, side, side), img), Tensor.of(Shape.of(b, QUESTION), q),
            Tensor.of(Shape.of(b), a), relational};
    }

    private static final class Net {
        private final Model cnn;
        private final RelationNetwork rn;
        private final Model head;
        private final Tensor coordinates;

        Net(Architecture a, long seed) {
            int side = ShapeSegmentation.SIDE;
            cnn = Model.sequential().input(1, side, side).seed(seed).conv2d(CHANNELS, 3, Activation.RELU)
                    .conv2d(CHANNELS, 2, Activation.RELU).maxPool(3, 3).build();
            float[] xy = new float[CELLS * CELLS * 2];
            for (int i = 0; i < CELLS * CELLS; i++) {
                xy[2 * i] = (i / CELLS) / (CELLS - 1f) * 2 - 1;
                xy[2 * i + 1] = (i % CELLS) / (CELLS - 1f) * 2 - 1;
            }
            coordinates = Tensor.of(Shape.of(1, CELLS * CELLS, 2), xy);
            if (a == Architecture.RELATION_NETWORK) {
                rn = new RelationNetwork(CHANNELS + 2, QUESTION, 64, 2, new SplittableRandom(seed + 1));
                head = null;
            } else {
                rn = null;
                head = Model.sequential().input(CELLS * CELLS * CHANNELS + QUESTION).seed(seed + 1)
                        .dense(96, Activation.RELU).dense(96, Activation.RELU).dense(2, Activation.IDENTITY).build();
            }
        }

        Var logits(Tensor images, Tensor questions, boolean training) {
            int b = images.shape().dim(0);
            Var f = cnn.forward(Var.of(images), training);
            if (rn != null) {
                Var cells = f.reshape(b, CHANNELS, CELLS * CELLS).permute(0, 2, 1);
                Var xy = Var.of(coordinates).expand(Shape.of(b, CELLS * CELLS, 2));
                return rn.forward(List.of(Var.concat(2, cells, xy), Var.of(questions)), training);
            }
            return head.forward(Var.concat(1, f.reshape(b, CELLS * CELLS * CHANNELS), Var.of(questions)), training);
        }

        List<Var> parameters() {
            List<Var> p = new ArrayList<>(cnn.parameters());
            p.addAll(rn != null ? rn.parameters() : head.parameters());
            return p;
        }
    }

    /**
     * Trains a question-answering model.
     *
     * @param a the model
     * @param steps Adam steps of 16 questions
     * @param seed the seed
     * @return {@code {non-relational accuracy, relational accuracy}} on 1000 fresh questions
     */
    public static double[] run(Architecture a, int steps, long seed) {
        Net net = new Net(a, seed);
        Pool train = pool(3000, seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 2);
        for (int s = 0; s < steps; s++) {
            Object[] d = batch(train, 16, r);
            Functional.crossEntropy(net.logits((Tensor) d[0], (Tensor) d[1], true), (Tensor) d[2]).backward();
            opt.step(net.parameters());
        }
        Pool test = pool(1500, seed + 50);
        SplittableRandom t = new SplittableRandom(seed + 3);
        int[] correct = new int[2];
        int[] total = new int[2];
        for (int chunk = 0; chunk < 10; chunk++) {
            Object[] d = batch(test, 100, t);
            float[] pred = net.logits((Tensor) d[0], (Tensor) d[1], false).value().argmax(1).toArray();
            float[] y = ((Tensor) d[2]).toArray();
            boolean[] rel = (boolean[]) d[3];
            for (int i = 0; i < y.length; i++) {
                int k = rel[i] ? 1 : 0;
                total[k]++;
                correct[k] += pred[i] == y[i] ? 1 : 0;
            }
        }
        return new double[] {(double) correct[0] / total[0], (double) correct[1] / total[1]};
    }
}
