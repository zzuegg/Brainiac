package io.github.zzuegg.brainiac.experiments.timeseries;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.SeriesClassificationTasks;
import io.github.zzuegg.brainiac.experimental.data.ShapeImages;
import io.github.zzuegg.brainiac.experimental.ml.DynamicTimeWarping;
import io.github.zzuegg.brainiac.experimental.nn.SelfSupervisedLosses;
import io.github.zzuegg.brainiac.experimental.nn.TemporalConvNet;
import io.github.zzuegg.brainiac.experimental.timeseries.RandomConvolutionFeatures;
import io.github.zzuegg.brainiac.experimental.timeseries.RidgeClassifier;
import io.github.zzuegg.brainiac.experimental.timeseries.ScatteringTransform;
import io.github.zzuegg.brainiac.experimental.timeseries.TimesBlock;
import io.github.zzuegg.brainiac.experimental.timeseries.Ts2Vec;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.tensor.spi.ActivationOp;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * Series classification (8.2.9, 8.2.10, 8.2.17, and TimesNet as a classifier for 8.2.4) on
 * {@link SeriesClassificationTasks}: cylinder-bell-funnel, shapelets and frequency bands.
 * <ul>
 *   <li>{@link #compare}: {@link RandomConvolutionFeatures} (ROCKET and MiniRocket) with a
 *       {@link RidgeClassifier}, the wavelet {@link ScatteringTransform} with the same read-out,
 *       against 1-NN {@link DynamicTimeWarping}, an MLP, a TCN and a TimesNet classifier trained
 *       end to end; accuracy and fit-plus-predict time;</li>
 *   <li>{@link #lowLabels}: self-supervised {@link Ts2Vec} on all training series, then a
 *       linear probe on 10% of the labels, against SimCLR on series crops with the same encoder
 *       ({@link SelfSupervisedLosses#ntXent}), the raw window with the same probe, ROCKET with the
 *       same labels, and a TCN trained on them;</li>
 *   <li>{@link #scatteringCurve}: accuracy against training-set size for scattering, ROCKET and
 *       the raw window;</li>
 *   <li>{@link #imagesLowData}: 2-D scattering with a linear read-out against a small CNN on
 *       {@link ShapeImages} with 64 training images (the shared low-data split of WP1's G-CNN
 *       row: images 0-63 of seed 7 for training).</li>
 * </ul>
 */
public final class SeriesClassification {

    private SeriesClassification() {
    }

    /** The three tasks by name. */
    static SeriesClassificationTasks.Task task(String name, int n, long seed) {
        SeriesClassificationTasks.Task t = switch (name) {
            case "CBF" -> SeriesClassificationTasks.cylinderBellFunnel(n, 128, seed);
            case "shapelets" -> SeriesClassificationTasks.shapelets(n, 96, seed);
            case "frequencies" -> SeriesClassificationTasks.frequencies(n, 96, 4, seed);
            default -> throw new IllegalArgumentException(name);
        };
        return t;
    }

    private static double[][] raw(float[][] x) {
        double[][] out = new double[x.length][];
        for (int i = 0; i < x.length; i++) {
            out[i] = new double[x[i].length];
            for (int j = 0; j < x[i].length; j++) {
                out[i][j] = x[i][j];
            }
        }
        return out;
    }

    private static double[] timed(Function<Void, int[]> f, int[] truth) {
        long t = System.nanoTime();
        int[] p = f.apply(null);
        return new double[] {RidgeClassifier.accuracy(p, truth), (System.nanoTime() - t) / 1e6};
    }

    /**
     * Accuracy and time of every classifier on one task.
     *
     * @param name "CBF", "shapelets" or "frequencies"
     * @param trainSize training series
     * @param testSize test series
     * @param steps training steps of the neural baselines
     * @param seed the seed
     * @return method to {accuracy, milliseconds}
     */
    public static Map<String, double[]> compare(String name, int trainSize, int testSize, int steps, long seed) {
        SeriesClassificationTasks.Task train = task(name, trainSize, seed);
        SeriesClassificationTasks.Task test = task(name, testSize, seed + 1000);
        Map<String, double[]> out = new LinkedHashMap<>();
        int length = train.length();
        out.put("ROCKET", timed(v -> {
            RandomConvolutionFeatures rocket = RandomConvolutionFeatures.rocket(500, length, 1, seed);
            return RidgeClassifier.fit(rocket.transform(RandomConvolutionFeatures.univariate(train.series())),
                    train.labels(), train.classes(), 10).predict(rocket.transform(RandomConvolutionFeatures
                    .univariate(test.series())));
        }, test.labels()));
        out.put("MiniRocket", timed(v -> {
            float[][][] x = RandomConvolutionFeatures.univariate(train.series());
            RandomConvolutionFeatures mini = RandomConvolutionFeatures.miniRocket(x, 840, seed);
            return RidgeClassifier.fit(mini.transform(x), train.labels(), train.classes(), 10).predict(
                    mini.transform(RandomConvolutionFeatures.univariate(test.series())));
        }, test.labels()));
        out.put("scattering + ridge", timed(v -> scattering(train, test), test.labels()));
        out.put("DTW 1-NN", timed(v -> dtw(train, test), test.labels()));
        out.put("raw + ridge", timed(v -> RidgeClassifier.fit(raw(train.series()), train.labels(), train.classes(), 10)
                .predict(raw(test.series())), test.labels()));
        out.put("MLP", timed(v -> mlp(train, test, steps, seed), test.labels()));
        out.put("TCN", timed(v -> tcn(train, test, steps, seed), test.labels()));
        out.put("TimesNet", timed(v -> timesNet(train, test, steps, seed), test.labels()));
        return out;
    }

    static int[] scattering(SeriesClassificationTasks.Task train, SeriesClassificationTasks.Task test) {
        int size = Integer.highestOneBit(train.length() - 1) * 2;
        // J = 4: coefficients every 16 steps keep the coarse order of events (ramp up or down)
        ScatteringTransform st = ScatteringTransform.oneDimensional(size, 4, 4);
        double[][] a = ScatteringTransform.logCompress(st.transform(train.series(), 0, false), 10);
        double[][] b = ScatteringTransform.logCompress(st.transform(test.series(), 0, false), 10);
        return RidgeClassifier.fit(a, train.labels(), train.classes(), 1).predict(b);
    }

    static int[] dtw(SeriesClassificationTasks.Task train, SeriesClassificationTasks.Task test) {
        int band = Math.max(1, train.length() / 10);
        DynamicTimeWarping.TemplateClassifier c = new DynamicTimeWarping.TemplateClassifier(band);
        for (int i = 0; i < train.size(); i++) {
            c.add(train.labels()[i], column(train.series()[i]));
        }
        int[] out = new int[test.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = c.classify(column(test.series()[i]));
        }
        return out;
    }

    private static float[][] column(float[] x) {
        float[][] c = new float[x.length][1];
        for (int t = 0; t < x.length; t++) {
            c[t][0] = x[t];
        }
        return c;
    }

    private static Tensor stack(float[][] x) {
        int n = x.length;
        int l = x[0].length;
        float[] v = new float[n * l];
        for (int i = 0; i < n; i++) {
            System.arraycopy(x[i], 0, v, i * l, l);
        }
        return Tensor.of(Shape.of(n, l), v);
    }

    private static Tensor labels(int[] y) {
        float[] v = new float[y.length];
        for (int i = 0; i < y.length; i++) {
            v[i] = y[i];
        }
        return Tensor.of(Shape.of(y.length), v);
    }

    /** Trains {@code logits = f([b, length])} with cross-entropy on minibatches of 32 and predicts the test set. */
    private static int[] trainClassifier(Function<Var, Var> f, List<Var> params, SeriesClassificationTasks.Task train,
            SeriesClassificationTasks.Task test, int steps, float lr, long seed) {
        Tensor x = stack(train.series());
        Tensor y = labels(train.labels());
        Optimizer opt = Optimizer.adam(lr);
        SplittableRandom r = new SplittableRandom(seed);
        int b = Math.min(32, train.size());
        for (int s = 0; s < steps; s++) {
            float[] idx = new float[b];
            for (int i = 0; i < b; i++) {
                idx[i] = r.nextInt(train.size());
            }
            Tensor sel = Tensor.of(Shape.of(b), idx);
            Functional.crossEntropy(f.apply(Var.of(x.gather(0, sel))), y.gather(0, sel)).backward();
            opt.step(params);
        }
        float[] logits = Var.noGrad(() -> f.apply(Var.of(stack(test.series())))).value().toArray();
        int c = train.classes();
        int[] out = new int[test.size()];
        for (int i = 0; i < out.length; i++) {
            int best = 0;
            for (int k = 1; k < c; k++) {
                if (logits[i * c + k] > logits[i * c + best]) {
                    best = k;
                }
            }
            out[i] = best;
        }
        return out;
    }

    static int[] mlp(SeriesClassificationTasks.Task train, SeriesClassificationTasks.Task test, int steps, long seed) {
        Model m = Model.sequential().input(train.length()).dense(64, Activation.RELU).dense(train.classes()).seed(seed)
                .build();
        return trainClassifier(x -> m.forward(x, true), m.parameters(), train, test, steps, 3e-3f, seed);
    }

    static int[] tcn(SeriesClassificationTasks.Task train, SeriesClassificationTasks.Task test, int steps, long seed) {
        int l = train.length();
        TemporalConvNet net = new TemporalConvNet(1, 16, 3, 4, new SplittableRandom(seed));
        Var head = Var.param(Tensor.normal(Shape.of(16, train.classes()), 0, 0.25f, new SplittableRandom(seed + 1)));
        List<Var> params = new ArrayList<>(net.parameters());
        params.add(head);
        return trainClassifier(x -> net.forward(x.reshape(x.shape().dim(0), l, 1), true).mean(1, false).matmul(head),
                params, train, test, steps, 5e-3f, seed);
    }

    static int[] timesNet(SeriesClassificationTasks.Task train, SeriesClassificationTasks.Task test, int steps,
            long seed) {
        int l = train.length();
        SplittableRandom r = new SplittableRandom(seed);
        Var embed = Var.param(Tensor.normal(Shape.of(1, 8), 0, 1f, r));
        TimesBlock block = new TimesBlock(8, 8, 2, new int[] {1, 3}, r);
        Var head = Var.param(Tensor.normal(Shape.of(8, train.classes()), 0, 0.3f, r));
        List<Var> params = new ArrayList<>(block.parameters());
        params.add(embed);
        params.add(head);
        return trainClassifier(x -> block.forward(x.reshape(x.shape().dim(0), l, 1).matmul(embed))
                .activation(ActivationOp.GELU, 0).mean(1, false).matmul(head),
                params, train, test, steps, 3e-3f, seed);
    }

    /**
     * Linear-probe accuracy with a tenth of the labels.
     *
     * @param name the task
     * @param pool training series (all used without labels)
     * @param labelled labelled series among them
     * @param pretrainSteps self-supervised steps (TS2Vec and SimCLR)
     * @param seed the seed
     * @return method to accuracy
     */
    public static Map<String, Double> lowLabels(String name, int pool, int labelled, int pretrainSteps, long seed) {
        SeriesClassificationTasks.Task all = task(name, pool, seed);
        SeriesClassificationTasks.Task few = all.head(labelled);
        SeriesClassificationTasks.Task test = task(name, 300, seed + 1000);
        Map<String, Double> out = new LinkedHashMap<>();
        Ts2Vec ts2vec = new Ts2Vec(16, 32, 3, new SplittableRandom(seed));
        ts2vec.fit(all.series(), pretrainSteps, 16, seed);
        out.put("TS2Vec probe", RidgeClassifier.accuracy(RidgeClassifier.fit(ts2vec.instanceRepresentations(
                few.series()), few.labels(), few.classes(), 1).predict(ts2vec.instanceRepresentations(test.series())),
                test.labels()));
        Ts2Vec simclr = new Ts2Vec(16, 32, 3, new SplittableRandom(seed));
        simclr(simclr, all.series(), pretrainSteps, seed);
        out.put("SimCLR probe", RidgeClassifier.accuracy(RidgeClassifier.fit(simclr.instanceRepresentations(
                few.series()), few.labels(), few.classes(), 1).predict(simclr.instanceRepresentations(test.series())),
                test.labels()));
        Ts2Vec untrained = new Ts2Vec(16, 32, 3, new SplittableRandom(seed));
        out.put("random encoder probe", RidgeClassifier.accuracy(RidgeClassifier.fit(untrained
                .instanceRepresentations(few.series()), few.labels(), few.classes(), 1).predict(untrained
                .instanceRepresentations(test.series())), test.labels()));
        out.put("raw + ridge", RidgeClassifier.accuracy(RidgeClassifier.fit(raw(few.series()), few.labels(),
                few.classes(), 10).predict(raw(test.series())), test.labels()));
        RandomConvolutionFeatures rocket = RandomConvolutionFeatures.rocket(500, all.length(), 1, seed);
        out.put("ROCKET", RidgeClassifier.accuracy(RidgeClassifier.fit(rocket.transform(RandomConvolutionFeatures
                .univariate(few.series())), few.labels(), few.classes(), 10).predict(rocket.transform(
                RandomConvolutionFeatures.univariate(test.series()))), test.labels()));
        out.put("supervised TCN", RidgeClassifier.accuracy(tcn(few, test, pretrainSteps, seed), test.labels()));
        return out;
    }

    /** SimCLR with the TS2Vec encoder: two random crops (resized by cropping to equal length) plus jitter, NT-Xent on max-pooled codes. */
    private static void simclr(Ts2Vec encoder, float[][] series, int steps, long seed) {
        SplittableRandom r = new SplittableRandom(seed + 3);
        Optimizer opt = Optimizer.adamW(1e-3f, 1e-4f);
        int length = series[0].length;
        int crop = length * 3 / 4;
        int b = 16;
        for (int s = 0; s < steps; s++) {
            float[] a = new float[b * crop];
            float[] c = new float[b * crop];
            for (int i = 0; i < b; i++) {
                float[] row = series[r.nextInt(series.length)];
                int o1 = r.nextInt(length - crop + 1);
                int o2 = r.nextInt(length - crop + 1);
                for (int t = 0; t < crop; t++) {
                    a[i * crop + t] = row[o1 + t] + (float) (0.1 * r.nextGaussian());
                    c[i * crop + t] = row[o2 + t] + (float) (0.1 * r.nextGaussian());
                }
            }
            Var z1 = encoder.encode(Var.of(Tensor.of(Shape.of(b, crop), a)), null).max(1, false);
            Var z2 = encoder.encode(Var.of(Tensor.of(Shape.of(b, crop), c)), null).max(1, false);
            SelfSupervisedLosses.ntXent(z1, z2, 0.2f).backward();
            opt.step(encoder.parameters());
        }
    }

    /**
     * Accuracy against training-set size.
     *
     * @param name the task
     * @param sizes training sizes
     * @param seed the seed
     * @return size to method to accuracy
     */
    public static Map<Integer, Map<String, Double>> scatteringCurve(String name, int[] sizes, long seed) {
        SeriesClassificationTasks.Task test = task(name, 300, seed + 1000);
        Map<Integer, Map<String, Double>> out = new LinkedHashMap<>();
        for (int n : sizes) {
            SeriesClassificationTasks.Task train = task(name, n, seed);
            Map<String, Double> row = new LinkedHashMap<>();
            row.put("scattering + ridge", RidgeClassifier.accuracy(scattering(train, test), test.labels()));
            RandomConvolutionFeatures rocket = RandomConvolutionFeatures.rocket(500, train.length(), 1, seed);
            row.put("ROCKET", RidgeClassifier.accuracy(RidgeClassifier.fit(rocket.transform(RandomConvolutionFeatures
                    .univariate(train.series())), train.labels(), train.classes(), 10).predict(rocket.transform(
                    RandomConvolutionFeatures.univariate(test.series()))), test.labels()));
            row.put("raw + ridge", RidgeClassifier.accuracy(RidgeClassifier.fit(raw(train.series()), train.labels(),
                    train.classes(), 10).predict(raw(test.series())), test.labels()));
            out.put(n, row);
        }
        return out;
    }

    /**
     * {@link ShapeImages} (12 x 12, six shapes) with 64 training images: 2-D scattering with a
     * ridge read-out against a small CNN (two 3x3 convolutions, global pooling) and raw pixels
     * with the same read-out, scored on 300 test images.
     *
     * @param trainImages training images (64 for the shared split)
     * @param cnnSteps CNN training steps
     * @return method to accuracy
     */
    public static Map<String, Double> imagesLowData(int trainImages, int cnnSteps) {
        ShapeImages gen = new ShapeImages(12, 0.1f);
        Tensor[] train = gen.generate(trainImages, 7);
        Tensor[] test = gen.generate(300, 8);
        float[][] trainPx = rows(train[0]);
        float[][] testPx = rows(test[0]);
        int[] trainY = ids(train[1]);
        int[] testY = ids(test[1]);
        Map<String, Double> out = new LinkedHashMap<>();
        ScatteringTransform st = ScatteringTransform.twoDimensional(16, 2, 4);
        double[][] a = ScatteringTransform.logCompress(st.transform(trainPx, 12, false), 10);
        double[][] b = ScatteringTransform.logCompress(st.transform(testPx, 12, false), 10);
        out.put("2-D scattering + ridge", RidgeClassifier.accuracy(RidgeClassifier.fit(a, trainY,
                ShapeImages.CLASSES, 10).predict(b), testY));
        out.put("pixels + ridge", RidgeClassifier.accuracy(RidgeClassifier.fit(raw(trainPx), trainY,
                ShapeImages.CLASSES, 10).predict(raw(testPx)), testY));
        Model cnn = Model.sequential().input(1, 12, 12).conv2d(16, 3, Activation.RELU).conv2d(16, 3, Activation.RELU)
                .globalAvgPool().dense(ShapeImages.CLASSES).seed(7).build();
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(7);
        for (int s = 0; s < cnnSteps; s++) {
            float[] idx = new float[32];
            for (int i = 0; i < 32; i++) {
                idx[i] = r.nextInt(trainImages);
            }
            Tensor sel = Tensor.of(Shape.of(32), idx);
            Functional.crossEntropy(cnn.forward(Var.of(train[0].gather(0, sel)), true), train[1].gather(0, sel))
                    .backward();
            opt.step(cnn.parameters());
        }
        float[] logits = cnn.predict(test[0]).toArray();
        int right = 0;
        for (int i = 0; i < testY.length; i++) {
            int best = 0;
            for (int k = 1; k < ShapeImages.CLASSES; k++) {
                if (logits[i * ShapeImages.CLASSES + k] > logits[i * ShapeImages.CLASSES + best]) {
                    best = k;
                }
            }
            right += best == testY[i] ? 1 : 0;
        }
        out.put("CNN", right / (double) testY.length);
        return out;
    }

    private static float[][] rows(Tensor images) {
        int n = images.shape().dim(0);
        float[] v = images.toArray();
        int d = v.length / n;
        float[][] out = new float[n][d];
        for (int i = 0; i < n; i++) {
            System.arraycopy(v, i * d, out[i], 0, d);
        }
        return out;
    }

    private static int[] ids(Tensor t) {
        float[] v = t.toArray();
        int[] out = new int[v.length];
        for (int i = 0; i < v.length; i++) {
            out[i] = (int) v[i];
        }
        return out;
    }
}
