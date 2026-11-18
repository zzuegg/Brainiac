package io.github.zzuegg.brainiac.experiments.techniques;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ShapeImages;
import io.github.zzuegg.brainiac.experimental.nn.BayesianDense;
import io.github.zzuegg.brainiac.experimental.nn.DistanceAwareHead;
import io.github.zzuegg.brainiac.experimental.nn.EvidentialHead;
import io.github.zzuegg.brainiac.experimental.nn.LastLayerLaplace;
import io.github.zzuegg.brainiac.experimental.nn.SpectralNormDense;
import io.github.zzuegg.brainiac.experimental.train.Calibration;
import io.github.zzuegg.brainiac.experimental.train.WeightPosterior;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Linalg;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Function;

/**
 * One uncertainty benchmark for weight posteriors and single-pass uncertainty heads.
 *
 * <p><b>Image OOD</b> ({@link #classification}): a classifier of {@link ShapeImages} shapes
 * 0-3; shapes 4 and 5 are the out-of-distribution inputs. Scores: in-distribution accuracy,
 * NLL and ECE ({@link Calibration}), the AUROC of telling OOD inputs from test inputs by the
 * method's uncertainty (predictive entropy unless the method has its own score), and the
 * forward passes per prediction.
 *
 * <p><b>Regression gap</b> ({@link #regression}): 1-D regression of {@code sin(1.5 x)} with
 * noise 0.1 from 40 points in {@code [-3, -1] u [1, 3]}; a good posterior widens its error
 * bars in the gap {@code (-1, 1)}. Scores: Gaussian NLL over {@code [-3, 3]} and in the gap,
 * and the mean predictive standard deviation in the gap and in the data. A GP with a fitted
 * RBF kernel is the gold standard.
 */
public final class UncertaintyBenchmark {

    /** The classification methods. */
    public enum Method {
        /** One network, maximum a posteriori. */
        MAP,
        /** Five networks from different seeds. */
        DEEP_ENSEMBLE,
        /** Dropout kept on at test time, 10 samples. */
        MC_DROPOUT,
        /** Last-layer Laplace approximation. */
        LAPLACE,
        /** SWAG over SGD iterates (momentum 0.9, lr 0.2, weight decay) after 20 Adam epochs, 10 samples. */
        SWAG,
        /** Bayes by backprop, 10 samples. */
        BAYES_BY_BACKPROP,
        /** SGLD, 10 samples. */
        SGLD,
        /** Evidential head, one pass. */
        EVIDENTIAL,
        /**
         * Spectral-normalized body plus GP head, one pass; scored by the entropy of the
         * variance-adjusted softmax (the raw variance separates OOD inputs worse here).
         */
        SNGP,
        /** RBF-centroid head with gradient penalty, one pass. */
        DUQ
    }

    /**
     * Classification scores.
     *
     * @param accuracy in-distribution test accuracy
     * @param nll in-distribution negative log-likelihood
     * @param ece expected calibration error (10 bins)
     * @param oodAuroc AUROC of the uncertainty separating OOD from test inputs
     * @param forwardPasses network evaluations per prediction
     */
    public record Scores(double accuracy, double nll, double ece, double oodAuroc, int forwardPasses) {
    }

    /** The regression methods. */
    public enum RegressionMethod {
        /** Exact GP, RBF kernel fitted by marginal likelihood. */
        GP,
        /** Five networks. */
        DEEP_ENSEMBLE,
        /** MC dropout, 20 samples. */
        MC_DROPOUT,
        /** SWAG, 20 samples. */
        SWAG,
        /** Bayes by backprop, 20 samples. */
        BAYES_BY_BACKPROP,
        /** SGLD, 20 samples. */
        SGLD
    }

    /**
     * Regression scores.
     *
     * @param nll mean Gaussian NLL over the test grid
     * @param gapNll mean NLL inside the gap
     * @param gapStd mean predictive standard deviation inside the gap
     * @param dataStd mean predictive standard deviation where there is data
     */
    public record RegressionScores(double nll, double gapNll, double gapStd, double dataStd) {
    }

    private static final int CLASSES = 4;
    private static final int EPOCHS = 40;
    private static final int BATCH = 100;
    private static final float NOISE_PIXELS = 0.2f;

    private UncertaintyBenchmark() {
    }

    private record Images(Tensor trainX, Tensor trainY, int[] trainLabels, Tensor testX, int[] testLabels,
            Tensor oodX) {
    }

    private static Images images(long seed) {
        Tensor[] pool = new ShapeImages(8, NOISE_PIXELS).generate(4000, seed);
        float[] x = pool[0].toArray();
        float[] y = pool[1].toArray();
        List<float[]> in = new ArrayList<>();
        List<Integer> labels = new ArrayList<>();
        List<float[]> ood = new ArrayList<>();
        for (int i = 0; i < y.length; i++) {
            float[] img = Arrays.copyOfRange(x, i * 64, (i + 1) * 64);
            if (y[i] < CLASSES) {
                in.add(img);
                labels.add((int) y[i]);
            } else if (ood.size() < 400) {
                ood.add(img);
            }
        }
        int train = 1200;
        int[] tl = new int[train];
        int[] sl = new int[400];
        for (int i = 0; i < train; i++) {
            tl[i] = labels.get(i);
        }
        for (int i = 0; i < 400; i++) {
            sl[i] = labels.get(train + i);
        }
        return new Images(Tensor.of(in.subList(0, train).toArray(new float[0][])), oneHot(tl),
                tl, Tensor.of(in.subList(train, train + 400).toArray(new float[0][])), sl,
                Tensor.of(ood.toArray(new float[0][])));
    }

    private static Tensor oneHot(int[] labels) {
        float[] v = new float[labels.length * CLASSES];
        for (int i = 0; i < labels.length; i++) {
            v[i * CLASSES + labels[i]] = 1;
        }
        return Tensor.of(Shape.of(labels.length, CLASSES), v);
    }

    private static Var nll(Var probs, Tensor oneHot) {
        return probs.add(1e-7f).log().mul(oneHot).sum(1, false).mean().neg();
    }

    private static Model mlp(long seed, float dropout) {
        Model.Builder b = Model.sequential().input(64).dense(64, Activation.RELU);
        if (dropout > 0) {
            b.dropout(dropout);
        }
        b.dense(64, Activation.RELU);
        if (dropout > 0) {
            b.dropout(dropout);
        }
        return b.dense(CLASSES, Activation.SOFTMAX).seed(seed).build();
    }

    /** Minibatch epochs; {@code step} gets each batch's inputs and one-hot targets and the epoch. */
    private interface BatchStep {
        void step(Tensor x, Tensor y, int epoch);
    }

    private static void epochs(Images d, int epochs, long seed, BatchStep step) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = d.trainLabels().length;
        for (int e = 0; e < epochs; e++) {
            int[] order = r.ints(0, n).distinct().limit(n).toArray();
            for (int from = 0; from < n; from += BATCH) {
                int to = Math.min(n, from + BATCH);
                Tensor[] xs = new Tensor[to - from];
                Tensor[] ys = new Tensor[to - from];
                for (int i = from; i < to; i++) {
                    xs[i - from] = d.trainX().slice(0, order[i], order[i] + 1);
                    ys[i - from] = d.trainY().slice(0, order[i], order[i] + 1);
                }
                step.step(Tensor.concat(0, xs), Tensor.concat(0, ys), e);
            }
        }
    }

    private static Model trainMap(Images d, long seed, float dropout) {
        Model m = mlp(seed, dropout);
        Optimizer opt = Optimizer.adam(3e-3f);
        epochs(d, EPOCHS, seed, (x, y, e) -> {
            nll(m.forward(Var.of(x), true), y).backward();
            opt.step(m.parameters());
        });
        return m;
    }

    /**
     * Trains and scores one method on the image OOD task.
     *
     * @param method the method
     * @param seed the seed
     * @return the scores
     */
    public static Scores classification(Method method, long seed) {
        Images d = images(seed);
        return switch (method) {
            case MAP -> {
                Model m = trainMap(d, seed, 0);
                yield score(d, x -> toRows(m.forward(Var.of(x)).value()), null, 1);
            }
            case DEEP_ENSEMBLE -> {
                List<Model> ms = new ArrayList<>();
                for (int k = 0; k < 5; k++) {
                    ms.add(trainMap(d, seed + 31L * k, 0));
                }
                yield score(d, x -> {
                    Tensor sum = null;
                    for (Model m : ms) {
                        Tensor p = m.forward(Var.of(x)).value();
                        sum = sum == null ? p : sum.add(p);
                    }
                    return toRows(sum.div(ms.size()));
                }, null, 5);
            }
            case MC_DROPOUT -> {
                Model m = trainMap(d, seed, 0.2f);
                yield score(d, x -> {
                    Tensor sum = null;
                    for (int s = 0; s < 10; s++) {
                        Tensor p = m.forward(Var.of(x), true).value();
                        sum = sum == null ? p : sum.add(p);
                    }
                    return toRows(sum.div(10));
                }, null, 10);
            }
            case LAPLACE -> {
                Model m = trainMap(d, seed, 0);
                LastLayerLaplace la = LastLayerLaplace.fit(m, d.trainX(), 1.0);
                yield score(d, la::predict, null, 1);
            }
            case SWAG -> {
                Model m = mlp(seed, 0);
                Optimizer adam = Optimizer.adam(3e-3f);
                epochs(d, EPOCHS / 2, seed, (x, y, e) -> {
                    nll(m.forward(Var.of(x), true), y).backward();
                    adam.step(m.parameters());
                });
                WeightPosterior.Swag swag = new WeightPosterior.Swag(m.parameters(), 10, 1);
                Optimizer sgd = Optimizer.momentum(0.2f, 0.9f);
                int[] step = {0};
                epochs(d, EPOCHS / 2, seed + 1, (x, y, e) -> {
                    nll(m.forward(Var.of(x), true), y).add(l2(m.parameters()).mul(5e-4f)).backward();
                    sgd.step(m.parameters());
                    if (++step[0] % 12 == 0) {
                        swag.collect();
                    }
                });
                List<List<Tensor>> samples = swag.sample(10, new SplittableRandom(seed));
                yield score(d, x -> toRows(WeightPosterior.average(m.parameters(), samples,
                        () -> m.forward(Var.of(x)).value()).getFirst()), null, 10);
            }
            case BAYES_BY_BACKPROP -> {
                int n = d.trainLabels().length;
                Model m = Model.sequential().input(64).add(BayesianDense.spec(64, Activation.RELU, 1f, n))
                        .add(BayesianDense.spec(64, Activation.RELU, 1f, n))
                        .add(BayesianDense.spec(CLASSES, Activation.SOFTMAX, 1f, n)).seed(seed).build();
                Optimizer opt = Optimizer.adam(3e-3f);
                epochs(d, EPOCHS, seed, (x, y, e) -> {
                    Var loss = nll(m.forward(Var.of(x), true), y);
                    loss.add(m.auxiliaryLoss()).backward();
                    opt.step(m.parameters());
                });
                yield score(d, x -> {
                    Tensor sum = null;
                    for (int s = 0; s < 10; s++) {
                        Tensor p = m.forward(Var.of(x)).value();
                        sum = sum == null ? p : sum.add(p);
                    }
                    return toRows(sum.div(10));
                }, null, 10);
            }
            case SGLD -> {
                Model m = trainMap(d, seed, 0);
                WeightPosterior.Sgld sgld = new WeightPosterior.Sgld(5e-3f, d.trainLabels().length, 1f, 1f, seed);
                int[] step = {0};
                epochs(d, 20, seed + 2, (x, y, e) -> {
                    nll(m.forward(Var.of(x), true), y).backward();
                    sgld.step(m.parameters());
                    if (++step[0] % 24 == 0) {
                        sgld.snapshot(m.parameters());
                    }
                });
                List<List<Tensor>> samples = sgld.samples();
                yield score(d, x -> toRows(WeightPosterior.average(m.parameters(), samples,
                        () -> m.forward(Var.of(x)).value()).getFirst()), null, samples.size());
            }
            case EVIDENTIAL -> {
                Model body = Model.sequential().input(64).dense(64, Activation.RELU).dense(64, Activation.RELU)
                        .seed(seed).build();
                EvidentialHead head = new EvidentialHead(64, CLASSES, new SplittableRandom(seed));
                Optimizer opt = Optimizer.adam(3e-3f);
                List<Var> ps = new ArrayList<>(body.parameters());
                ps.addAll(head.parameters());
                epochs(d, EPOCHS, seed, (x, y, e) -> {
                    EvidentialHead.loss(head.forward(body.forward(Var.of(x), true)), y,
                            Math.min(1f, e / 10f)).backward();
                    opt.step(ps);
                });
                yield score(d, x -> toRows(EvidentialHead.probabilities(head.forward(body.forward(Var.of(x))).value())),
                        x -> EvidentialHead.uncertainty(head.forward(body.forward(Var.of(x))).value()).toDoubleArray(),
                        1);
            }
            case SNGP -> {
                SplittableRandom r = new SplittableRandom(seed);
                Model body = Model.sequential().input(64).layer(new SpectralNormDense(64, 64, Activation.RELU, r))
                        .layer(new SpectralNormDense(64, 64, Activation.RELU, r)).seed(seed).build();
                DistanceAwareHead head = DistanceAwareHead.sngp(64, CLASSES, 512, 1f, r);
                Optimizer opt = Optimizer.adam(3e-3f);
                List<Var> ps = new ArrayList<>(body.parameters());
                ps.addAll(head.parameters());
                epochs(d, EPOCHS, seed, (x, y, e) -> {
                    head.loss(head.forward(body.forward(Var.of(x), true)), ids(y)).backward();
                    opt.step(ps);
                });
                head.fitCovariance(body.forward(Var.of(d.trainX())).value(), 1f, false);
                yield score(d, x -> toRows(head.probabilities(body.forward(Var.of(x)).value())),
                        null, 1);
            }
            case DUQ -> {
                SplittableRandom r = new SplittableRandom(seed);
                Model body = Model.sequential().input(64).dense(64, Activation.RELU).dense(64, Activation.RELU)
                        .seed(seed).build();
                DistanceAwareHead head = DistanceAwareHead.duq(64, CLASSES, 32, 0.3f, r);
                Optimizer opt = Optimizer.adam(3e-3f);
                List<Var> ps = new ArrayList<>(body.parameters());
                ps.addAll(head.parameters());
                epochs(d, EPOCHS, seed, (x, y, e) -> {
                    Var in = Var.param(x);
                    Var k = head.forward(body.forward(in, true));
                    head.loss(k, ids(y)).add(DistanceAwareHead.gradientPenalty(in, k, 0.05f)).backward();
                    opt.step(ps);
                    head.updateCentroids(body.forward(Var.of(x)).value(), ids(y));
                });
                yield score(d, x -> toRows(head.probabilities(body.forward(Var.of(x)).value())),
                        x -> head.uncertainty(body.forward(Var.of(x)).value()).toDoubleArray(), 1);
            }
        };
    }

    private static Var l2(List<Var> parameters) {
        Var sum = null;
        for (Var p : parameters) {
            sum = sum == null ? p.square().sum() : sum.add(p.square().sum());
        }
        return sum;
    }

    private static Tensor ids(Tensor oneHot) {
        return oneHot.argmax(1);
    }

    private static Scores score(Images d, Function<Tensor, double[][]> probs, Function<Tensor, double[]> uncertainty,
            int passes) {
        double[][] p = probs.apply(d.testX());
        double[][] q = probs.apply(d.oodX());
        double[] uIn = uncertainty == null ? entropy(p) : uncertainty.apply(d.testX());
        double[] uOut = uncertainty == null ? entropy(q) : uncertainty.apply(d.oodX());
        return new Scores(Calibration.accuracy(p, d.testLabels()), Calibration.negativeLogLikelihood(p, d.testLabels()),
                Calibration.expectedCalibrationError(p, d.testLabels(), 10), auroc(uIn, uOut), passes);
    }

    private static double[] entropy(double[][] p) {
        double[] out = new double[p.length];
        for (int i = 0; i < p.length; i++) {
            out[i] = Calibration.entropy(p[i]);
        }
        return out;
    }

    /**
     * The area under the ROC curve of a score separating positives from negatives (the
     * Mann-Whitney statistic; ties count half).
     *
     * @param negatives scores of the negatives (in-distribution)
     * @param positives scores of the positives (out-of-distribution)
     * @return the AUROC in {@code [0, 1]}
     */
    public static double auroc(double[] negatives, double[] positives) {
        double wins = 0;
        for (double pos : positives) {
            for (double neg : negatives) {
                wins += pos > neg ? 1 : pos == neg ? 0.5 : 0;
            }
        }
        return wins / ((double) positives.length * negatives.length);
    }

    private static double[][] toRows(Tensor t) {
        int n = t.shape().dim(0);
        int k = t.shape().dim(1);
        double[] v = t.toDoubleArray();
        double[][] out = new double[n][k];
        for (int i = 0; i < n; i++) {
            System.arraycopy(v, i * k, out[i], 0, k);
        }
        return out;
    }

    // ---------------------------------------------------------------- regression gap

    private static final double NOISE = 0.1;

    private record Curve(Tensor x, Tensor y, Tensor grid, double[] truth) {
    }

    private static Curve curve(long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        float[] x = new float[40];
        float[] y = new float[40];
        for (int i = 0; i < 40; i++) {
            double v = 1 + 2 * r.nextDouble();
            x[i] = (float) (r.nextBoolean() ? v : -v);
            y[i] = (float) (Math.sin(1.5 * x[i]) + NOISE * r.nextGaussian());
        }
        float[] g = new float[121];
        double[] truth = new double[121];
        for (int i = 0; i < g.length; i++) {
            g[i] = -3f + 6f * i / 120;
            truth[i] = Math.sin(1.5 * g[i]);
        }
        return new Curve(Tensor.of(Shape.of(40, 1), x), Tensor.of(Shape.of(40, 1), y), Tensor.of(Shape.of(121, 1), g),
                truth);
    }

    private static Model regressor(long seed, float dropout) {
        Model.Builder b = Model.sequential().input(1).dense(50, Activation.TANH);
        if (dropout > 0) {
            b.dropout(dropout);
        }
        b.dense(50, Activation.TANH);
        if (dropout > 0) {
            b.dropout(dropout);
        }
        return b.dense(1, Activation.IDENTITY).seed(seed).build();
    }

    private static Var gaussianLoss(Var pred, Tensor y) {
        return pred.sub(y).square().mean().mul((float) (0.5 / (NOISE * NOISE)));
    }

    private static Model fit(Curve c, long seed, float dropout, int steps) {
        Model m = regressor(seed, dropout);
        Optimizer opt = Optimizer.adam(1e-2f);
        for (int s = 0; s < steps; s++) {
            gaussianLoss(m.forward(Var.of(c.x()), true), c.y()).backward();
            opt.step(m.parameters());
        }
        return m;
    }

    /**
     * Trains and scores one method on the regression gap.
     *
     * @param method the method
     * @param seed the seed
     * @return the scores
     */
    public static RegressionScores regression(RegressionMethod method, long seed) {
        Curve c = curve(seed);
        List<double[]> samples = new ArrayList<>();
        switch (method) {
            case GP -> {
                return regressionScores(c, gp(c));
            }
            case DEEP_ENSEMBLE -> {
                for (int k = 0; k < 5; k++) {
                    samples.add(fit(c, seed + 17L * k, 0, 1500).forward(Var.of(c.grid())).value().toDoubleArray());
                }
            }
            case MC_DROPOUT -> {
                Model m = fit(c, seed, 0.1f, 1500);
                for (int s = 0; s < 20; s++) {
                    samples.add(m.forward(Var.of(c.grid()), true).value().toDoubleArray());
                }
            }
            case SWAG -> {
                Model m = fit(c, seed, 0, 1500);
                WeightPosterior.Swag swag = new WeightPosterior.Swag(m.parameters(), 20, 1);
                Optimizer sgd = Optimizer.sgd(2e-4f);
                SplittableRandom r = new SplittableRandom(seed);
                for (int s = 0; s < 400; s++) {
                    int from = r.nextInt(30);
                    gaussianLoss(m.forward(Var.of(c.x().slice(0, from, from + 10))), c.y().slice(0, from, from + 10))
                            .backward();
                    sgd.step(m.parameters());
                    if (s % 10 == 0) {
                        swag.collect();
                    }
                }
                for (List<Tensor> w : swag.sample(20, r)) {
                    samples.add(WeightPosterior.average(m.parameters(), List.of(w),
                            () -> m.forward(Var.of(c.grid())).value()).getFirst().toDoubleArray());
                }
            }
            case BAYES_BY_BACKPROP -> {
                Model m = Model.sequential().input(1).add(BayesianDense.spec(50, Activation.TANH, 1f, 40))
                        .add(BayesianDense.spec(50, Activation.TANH, 1f, 40))
                        .add(BayesianDense.spec(1, Activation.IDENTITY, 1f, 40)).seed(seed).build();
                Optimizer opt = Optimizer.adam(1e-2f);
                for (int s = 0; s < 2000; s++) {
                    gaussianLoss(m.forward(Var.of(c.x()), true), c.y()).add(m.auxiliaryLoss()).backward();
                    opt.step(m.parameters());
                }
                for (int s = 0; s < 20; s++) {
                    samples.add(m.forward(Var.of(c.grid())).value().toDoubleArray());
                }
            }
            case SGLD -> {
                Model m = fit(c, seed, 0, 1500);
                WeightPosterior.Sgld sgld = new WeightPosterior.Sgld(1e-4f, 40, 1f, 1f, seed);
                for (int s = 0; s < 4000; s++) {
                    gaussianLoss(m.forward(Var.of(c.x())), c.y()).backward();
                    sgld.step(m.parameters());
                    if (s % 200 == 199) {
                        sgld.snapshot(m.parameters());
                    }
                }
                for (List<Tensor> w : sgld.samples()) {
                    samples.add(WeightPosterior.average(m.parameters(), List.of(w),
                            () -> m.forward(Var.of(c.grid())).value()).getFirst().toDoubleArray());
                }
            }
        }
        int g = c.truth().length;
        double[][] pred = new double[2][g];
        for (int i = 0; i < g; i++) {
            double mean = 0;
            for (double[] s : samples) {
                mean += s[i] / samples.size();
            }
            double var = 0;
            for (double[] s : samples) {
                var += (s[i] - mean) * (s[i] - mean) / samples.size();
            }
            pred[0][i] = mean;
            pred[1][i] = Math.sqrt(var + NOISE * NOISE);
        }
        return regressionScores(c, pred);
    }

    private static RegressionScores regressionScores(Curve c, double[][] pred) {
        double[] grid = c.grid().toDoubleArray();
        double nll = 0;
        double gapNll = 0;
        double gapStd = 0;
        double dataStd = 0;
        int gap = 0;
        int data = 0;
        for (int i = 0; i < grid.length; i++) {
            double d = c.truth()[i] - pred[0][i];
            double s = pred[1][i];
            double l = Math.log(s) + 0.5 * d * d / (s * s) + 0.5 * Math.log(2 * Math.PI);
            nll += l / grid.length;
            if (Math.abs(grid[i]) < 1) {
                gapNll += l;
                gapStd += s;
                gap++;
            } else {
                dataStd += s;
                data++;
            }
        }
        return new RegressionScores(nll, gapNll / gap, gapStd / gap, dataStd / data);
    }

    /** An exact GP with an RBF kernel; amplitude and length scale by grid search on the marginal likelihood. */
    private static double[][] gp(Curve c) {
        double[] xs = c.x().toDoubleArray();
        double[] ys = c.y().toDoubleArray();
        double[] xq = c.grid().toDoubleArray();
        double bestLml = Double.NEGATIVE_INFINITY;
        double[] best = null;
        for (double amp : new double[] {0.5, 1, 2}) {
            for (double len : new double[] {0.3, 0.5, 0.75, 1, 1.5}) {
                Tensor l = Linalg.cholesky(kernel(xs, xs, amp, len, NOISE * NOISE));
                double[] a = solve(l, column(ys)).toDoubleArray();
                double[] ld = l.toDoubleArray();
                double lml = 0;
                for (int i = 0; i < ys.length; i++) {
                    lml += -0.5 * ys[i] * a[i] - Math.log(ld[i * ys.length + i]);
                }
                if (lml > bestLml) {
                    bestLml = lml;
                    best = new double[] {amp, len};
                }
            }
        }
        Tensor l = Linalg.cholesky(kernel(xs, xs, best[0], best[1], NOISE * NOISE));
        Tensor kq = kernel(xs, xq, best[0], best[1], 0);
        double[] mean = kq.transpose().matmul(solve(l, column(ys))).toDoubleArray();
        double[] v = Linalg.solveTriangular(l, kq, true).toDoubleArray();
        double[][] out = new double[2][xq.length];
        for (int j = 0; j < xq.length; j++) {
            double var = best[0] * best[0] + NOISE * NOISE;
            for (int i = 0; i < xs.length; i++) {
                var -= v[i * xq.length + j] * v[i * xq.length + j];
            }
            out[0][j] = mean[j];
            out[1][j] = Math.sqrt(Math.max(var, 1e-9));
        }
        return out;
    }

    private static Tensor solve(Tensor l, Tensor b) {
        return Linalg.solveTriangular(l.transpose(), Linalg.solveTriangular(l, b, true), false);
    }

    private static Tensor kernel(double[] a, double[] b, double amp, double len, double noise) {
        double[] k = new double[a.length * b.length];
        for (int i = 0; i < a.length; i++) {
            for (int j = 0; j < b.length; j++) {
                double d = a[i] - b[j];
                k[i * b.length + j] = amp * amp * Math.exp(-d * d / (2 * len * len)) + (i == j && noise > 0 ? noise : 0);
            }
        }
        return Tensor.of(Shape.of(a.length, b.length), k, DType.FLOAT64, Device.cpu());
    }

    private static Tensor column(double[] v) {
        return Tensor.of(Shape.of(v.length, 1), v, DType.FLOAT64, Device.cpu());
    }
}
