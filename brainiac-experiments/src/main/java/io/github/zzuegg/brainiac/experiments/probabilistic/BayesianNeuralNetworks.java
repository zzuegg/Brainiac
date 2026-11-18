package io.github.zzuegg.brainiac.experiments.probabilistic;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ShapeImages;
import io.github.zzuegg.brainiac.experimental.ml.GaussianProcess;
import io.github.zzuegg.brainiac.experimental.ml.LogDensity;
import io.github.zzuegg.brainiac.experimental.ml.StationaryKernel;
import io.github.zzuegg.brainiac.experimental.nn.Epinet;
import io.github.zzuegg.brainiac.experimental.nn.FeatureDensityUncertainty;
import io.github.zzuegg.brainiac.experimental.nn.SpectralNormDense;
import io.github.zzuegg.brainiac.experimental.train.Calibration;
import io.github.zzuegg.brainiac.experimental.train.HamiltonianMonteCarlo;
import io.github.zzuegg.brainiac.experimental.train.Ivon;
import io.github.zzuegg.brainiac.experimental.train.SteinVariationalGradientDescent;
import io.github.zzuegg.brainiac.experimental.train.WeightPosterior;
import io.github.zzuegg.brainiac.experiments.techniques.UncertaintyBenchmark;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Weight-space inference and cheap epistemic uncertainty on the two tasks of
 * {@link UncertaintyBenchmark}, scaled down for speed.
 *
 * <p><b>Regression gap</b> ({@link #regressionGap}): {@code sin(1.5 x)} with noise 0.1 from 40
 * points in {@code [-3, -1] u [1, 3]}, a 1-20-20-1 tanh network with an {@code N(0, 1)} weight
 * prior. Full-batch HMC is the reference posterior (Izmailov et al. 2021); every other method
 * (SGLD and SWAG from {@link WeightPosterior}, a deep ensemble, MC dropout, SVGD, IVON, and the
 * exact RBF GP) is scored by Gaussian NLL over {@code [-3, 3]} and inside the gap, 95% coverage
 * in the data and in the gap, and the mean total-variation distance of its predictive from
 * HMC's. {@link #coldPosterior} sweeps the HMC temperature.
 *
 * <p><b>Image OOD</b> ({@link #imageOod}): {@link ShapeImages} shapes 0-3 in distribution,
 * shapes 4-5 out of distribution, a 64-64-64-4 MLP. Methods: MAP, a five-member deep ensemble,
 * IVON, SVGD with five particles, an {@link Epinet} on the MAP features and DDU
 * ({@link FeatureDensityUncertainty} on a dense layer followed by two residual
 * spectral-normalised blocks). Scores: accuracy,
 * NLL, ECE, OOD AUROC, forward passes per prediction, and Osband's dyadic joint NLL
 * ({@code tau = 10}, per point).
 */
public final class BayesianNeuralNetworks {

    private static final double NOISE = 0.1;

    private BayesianNeuralNetworks() {
    }

    // ------------------------------------------------------------ regression gap

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
        float[] g = new float[61];
        double[] truth = new double[61];
        for (int i = 0; i < g.length; i++) {
            g[i] = -3f + 6f * i / 60;
            truth[i] = Math.sin(1.5 * g[i]);
        }
        return new Curve(Tensor.of(Shape.of(40, 1), x), Tensor.of(Shape.of(40, 1), y), Tensor.of(Shape.of(61, 1), g),
                truth);
    }

    private static Model regressor(long seed, float dropout) {
        Model.Builder b = Model.sequential().input(1).dense(20, Activation.TANH);
        if (dropout > 0) {
            b.dropout(dropout);
        }
        b.dense(20, Activation.TANH);
        if (dropout > 0) {
            b.dropout(dropout);
        }
        return b.dense(1, Activation.IDENTITY).seed(seed).build();
    }

    /** Negative log joint: summed Gaussian NLL plus the N(0, 1) prior. */
    private static Var energy(Model m, Curve c) {
        Var e = m.forward(Var.of(c.x())).sub(c.y()).square().sum().mul((float) (0.5 / (NOISE * NOISE)));
        for (Var p : m.parameters()) {
            e = e.add(p.square().sum().mul(0.5f));
        }
        return e;
    }

    private static Model map(Curve c, long seed, float dropout, int steps) {
        Model m = regressor(seed, dropout);
        Optimizer opt = Optimizer.adam(1e-2f);
        for (int s = 0; s < steps; s++) {
            Var loss = m.forward(Var.of(c.x()), true).sub(c.y()).square().mean().mul((float) (0.5 / (NOISE * NOISE)));
            for (Var p : m.parameters()) {
                loss = loss.add(p.square().sum().mul(0.5f / 40));
            }
            loss.backward();
            opt.step(m.parameters());
        }
        return m;
    }

    private static double[] predictGrid(Model m, Curve c, boolean training) {
        return m.forward(Var.of(c.grid()), training).value().toDoubleArray();
    }

    private static List<double[]> hmcSamples(Curve c, long seed, double temperature, int samples) {
        Model m = map(c, seed, 0, 800);
        List<Var> ps = m.parameters();
        LogDensity post = HamiltonianMonteCarlo.parameterDensity(ps, () -> energy(m, c), temperature);
        HamiltonianMonteCarlo.Result res = HamiltonianMonteCarlo.hmc(post, HamiltonianMonteCarlo.flatten(ps), 0, 20,
                100, samples, 0.75, new SplittableRandom(seed));
        List<double[]> out = new ArrayList<>();
        for (int i = 0; i < samples; i += 4) {
            HamiltonianMonteCarlo.assign(ps, res.samples()[i]);
            out.add(predictGrid(m, c, false));
        }
        return out;
    }

    /** Predictive mean and standard deviation (with the observation noise) from function samples. */
    private static double[][] moments(List<double[]> samples) {
        int g = samples.getFirst().length;
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
        return pred;
    }

    private static double gaussianTv(double m1, double s1, double m2, double s2) {
        double lo = Math.min(m1 - 6 * s1, m2 - 6 * s2);
        double hi = Math.max(m1 + 6 * s1, m2 + 6 * s2);
        int n = 400;
        double h = (hi - lo) / n;
        double tv = 0;
        for (int i = 0; i <= n; i++) {
            double x = lo + i * h;
            double p = Math.exp(-0.5 * Math.pow((x - m1) / s1, 2)) / (s1 * Math.sqrt(2 * Math.PI));
            double q = Math.exp(-0.5 * Math.pow((x - m2) / s2, 2)) / (s2 * Math.sqrt(2 * Math.PI));
            tv += Math.abs(p - q) * h * (i == 0 || i == n ? 0.5 : 1);
        }
        return 0.5 * tv;
    }

    private static double[] scores(Curve c, double[][] pred, double[][] reference) {
        double[] grid = c.grid().toDoubleArray();
        double nll = 0;
        double gapNll = 0;
        double covIn = 0;
        double covGap = 0;
        double tv = 0;
        int gap = 0;
        int in = 0;
        for (int i = 0; i < grid.length; i++) {
            double d = c.truth()[i] - pred[0][i];
            double s = pred[1][i];
            double l = Math.log(s) + 0.5 * d * d / (s * s) + 0.5 * Math.log(2 * Math.PI);
            nll += l / grid.length;
            // coverage of the noisy target's 95% interval by the predictive
            boolean covered = Math.abs(d) < 1.96 * s;
            if (Math.abs(grid[i]) < 1) {
                gapNll += l;
                covGap += covered ? 1 : 0;
                gap++;
            } else {
                covIn += covered ? 1 : 0;
                in++;
            }
            if (reference != null) {
                tv += gaussianTv(pred[0][i], pred[1][i], reference[0][i], reference[1][i]) / grid.length;
            }
        }
        return new double[] {nll, gapNll / gap, covIn / in, covGap / gap, reference == null ? 0 : tv};
    }

    /**
     * Scores every method on the regression gap.
     *
     * @param seed the seed
     * @return method to {@code {NLL, gap NLL, coverage in data, coverage in gap, TV to HMC}}
     */
    public static Map<String, double[]> regressionGap(long seed) {
        Curve c = curve(seed);
        Map<String, double[]> out = new LinkedHashMap<>();
        double[][] hmc = moments(hmcSamples(c, seed, 1, 240));
        out.put("HMC", scores(c, hmc, hmc));
        // exact GP with a fitted RBF kernel
        double[][] xs = column(c.x().toDoubleArray());
        GaussianProcess gp = GaussianProcess.optimize(xs, c.y().toDoubleArray(), StationaryKernel.rbf(1, 1), 0.01, 60);
        GaussianProcess.Prediction p = gp.predict(column(c.grid().toDoubleArray()));
        double[][] gpPred = new double[2][p.mean().length];
        for (int i = 0; i < gpPred[0].length; i++) {
            gpPred[0][i] = p.mean()[i];
            gpPred[1][i] = Math.sqrt(p.latentVariance()[i] + NOISE * NOISE);
        }
        out.put("GP (RBF)", scores(c, gpPred, hmc));
        List<double[]> ens = new ArrayList<>();
        for (int k = 0; k < 5; k++) {
            ens.add(predictGrid(map(c, seed + 17L * k, 0, 800), c, false));
        }
        out.put("deep ensemble (5)", scores(c, moments(ens), hmc));
        Model drop = map(c, seed, 0.1f, 800);
        List<double[]> mc = new ArrayList<>();
        for (int s = 0; s < 20; s++) {
            mc.add(predictGrid(drop, c, true));
        }
        out.put("MC dropout", scores(c, moments(mc), hmc));
        // SGLD from the MAP
        Model sg = map(c, seed, 0, 800);
        WeightPosterior.Sgld sgld = new WeightPosterior.Sgld(2e-4f, 40, 1f, 1f, seed);
        for (int s = 0; s < 2000; s++) {
            sg.forward(Var.of(c.x())).sub(c.y()).square().mean().mul((float) (0.5 / (NOISE * NOISE))).backward();
            sgld.step(sg.parameters());
            if (s % 100 == 99) {
                sgld.snapshot(sg.parameters());
            }
        }
        out.put("SGLD", scores(c, moments(gridSamples(sg, c, sgld.samples())), hmc));
        // SWAG
        Model sw = map(c, seed, 0, 800);
        WeightPosterior.Swag swag = new WeightPosterior.Swag(sw.parameters(), 20, 1);
        Optimizer sgd = Optimizer.sgd(2e-4f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < 400; s++) {
            int from = r.nextInt(30);
            sw.forward(Var.of(c.x().slice(0, from, from + 10))).sub(c.y().slice(0, from, from + 10)).square().mean()
                    .mul((float) (0.5 / (NOISE * NOISE))).backward();
            sgd.step(sw.parameters());
            if (s % 10 == 0) {
                swag.collect();
            }
        }
        out.put("SWAG", scores(c, moments(gridSamples(sw, c, swag.sample(20, r))), hmc));
        // SVGD: eight networks as particles of the weight posterior
        Model sv = regressor(seed, 0);
        List<Var> ps = sv.parameters();
        double[][] init = new double[8][];
        for (int k = 0; k < 8; k++) {
            init[k] = HamiltonianMonteCarlo.flatten(map(c, seed + 31L * k, 0, 200).parameters());
        }
        LogDensity post = HamiltonianMonteCarlo.parameterDensity(ps, () -> energy(sv, c), 1);
        double[][] particles = SteinVariationalGradientDescent.create(post, init, 0.01).run(400);
        List<double[]> svgd = new ArrayList<>();
        for (double[] w : particles) {
            HamiltonianMonteCarlo.assign(ps, w);
            svgd.add(predictGrid(sv, c, false));
        }
        out.put("SVGD (8)", scores(c, moments(svgd), hmc));
        // IVON
        Model iv = regressor(seed, 0);
        Ivon ivon = new Ivon(0.05f, 40, 1f / 40, 10f, 0.9f, 0.99999f, 1f, seed);
        for (int s = 0; s < 2000; s++) {
            iv.forward(Var.of(c.x())).sub(c.y()).square().mean().mul((float) (0.5 / (NOISE * NOISE))).backward();
            ivon.step(iv.parameters());
        }
        out.put("IVON", scores(c, moments(gridSamples(iv, c, ivon.samples(iv.parameters(), 20))), hmc));
        return out;
    }

    private static double[][] column(double[] v) {
        double[][] x = new double[v.length][1];
        for (int i = 0; i < v.length; i++) {
            x[i][0] = v[i];
        }
        return x;
    }

    private static List<double[]> gridSamples(Model m, Curve c, List<List<Tensor>> weights) {
        List<double[]> out = new ArrayList<>();
        for (List<Tensor> w : weights) {
            out.add(WeightPosterior.average(m.parameters(), List.of(w), () -> m.forward(Var.of(c.grid())).value())
                    .getFirst().toDoubleArray());
        }
        return out;
    }

    /**
     * The cold-posterior sweep: HMC at temperatures 1, 0.3 and 0.1.
     *
     * @param seed the seed
     * @return {@code "T=..."} to {@code {NLL, gap NLL, coverage in data, coverage in gap}}
     */
    public static Map<String, double[]> coldPosterior(long seed) {
        Curve c = curve(seed);
        Map<String, double[]> out = new LinkedHashMap<>();
        for (double t : new double[] {1, 0.3, 0.1}) {
            double[] s = scores(c, moments(hmcSamples(c, seed, t, 160)), null);
            out.put("T=" + t, Arrays.copyOf(s, 4));
        }
        return out;
    }

    // ------------------------------------------------------------ image OOD

    private static final int CLASSES = 4;

    private record Images(Tensor trainX, Tensor trainY, int[] trainLabels, Tensor testX, int[] testLabels,
            Tensor oodX) {
    }

    private static Images images(long seed) {
        Tensor[] pool = new ShapeImages(8, 0.2f).generate(2400, seed);
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
            } else if (ood.size() < 300) {
                ood.add(img);
            }
        }
        int train = 600;
        int test = 300;
        int[] tl = new int[train];
        int[] sl = new int[test];
        float[] oh = new float[train * CLASSES];
        for (int i = 0; i < train; i++) {
            tl[i] = labels.get(i);
            oh[i * CLASSES + tl[i]] = 1;
        }
        for (int i = 0; i < test; i++) {
            sl[i] = labels.get(train + i);
        }
        return new Images(Tensor.of(in.subList(0, train).toArray(new float[0][])), Tensor.of(Shape.of(train, CLASSES), oh),
                tl, Tensor.of(in.subList(train, train + test).toArray(new float[0][])), sl,
                Tensor.of(ood.toArray(new float[0][])));
    }

    private static Model classifier(long seed) {
        return Model.sequential().input(64).dense(64, Activation.RELU).dense(64, Activation.RELU)
                .dense(CLASSES, Activation.IDENTITY).seed(seed).build();
    }

    private static Var crossEntropy(Var logits, Tensor oneHot) {
        return logits.logsumexp(1, false).sub(logits.mul(oneHot).sum(1, false)).mean();
    }

    private interface Step {
        void run(Tensor x, Tensor y);
    }

    private static void epochs(Images d, int epochs, long seed, Step step) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = d.trainLabels().length;
        int batch = 50;
        for (int e = 0; e < epochs; e++) {
            int[] order = r.ints(0, n).distinct().limit(n).toArray();
            for (int from = 0; from < n; from += batch) {
                Tensor[] xs = new Tensor[batch];
                Tensor[] ys = new Tensor[batch];
                for (int i = 0; i < batch; i++) {
                    xs[i] = d.trainX().slice(0, order[from + i], order[from + i] + 1);
                    ys[i] = d.trainY().slice(0, order[from + i], order[from + i] + 1);
                }
                step.run(Tensor.concat(0, xs), Tensor.concat(0, ys));
            }
        }
    }

    private static Model trainMap(Images d, long seed) {
        Model m = classifier(seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        epochs(d, 15, seed, (x, y) -> {
            crossEntropy(m.forward(Var.of(x), true), y).backward();
            opt.step(m.parameters());
        });
        return m;
    }

    private static double[][] softmaxRows(Tensor logits) {
        int n = logits.shape().dim(0);
        double[] v = logits.toDoubleArray();
        double[][] p = new double[n][];
        for (int i = 0; i < n; i++) {
            p[i] = Calibration.softmax(Arrays.copyOfRange(v, i * CLASSES, (i + 1) * CLASSES), 1);
        }
        return p;
    }

    /** Member probabilities {@code [K][n][C]} on test and OOD inputs. */
    private record Members(double[][][] test, double[][][] ood) {
    }

    private static double[][] meanOf(double[][][] members) {
        int n = members[0].length;
        double[][] m = new double[n][CLASSES];
        for (double[][] k : members) {
            for (int i = 0; i < n; i++) {
                for (int c = 0; c < CLASSES; c++) {
                    m[i][c] += k[i][c] / members.length;
                }
            }
        }
        return m;
    }

    private static double[] entropies(double[][] p) {
        double[] e = new double[p.length];
        for (int i = 0; i < p.length; i++) {
            e[i] = Calibration.entropy(p[i]);
        }
        return e;
    }

    /** Osband et al.'s dyadic joint log-loss: tau points drawn from two anchors, per point. */
    private static double jointNll(double[][][] members, int[] labels, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int tau = 10;
        double total = 0;
        int trials = 300;
        for (int t = 0; t < trials; t++) {
            int a = r.nextInt(labels.length);
            int b = r.nextInt(labels.length);
            int[] pts = new int[tau];
            for (int i = 0; i < tau; i++) {
                pts[i] = r.nextBoolean() ? a : b;
            }
            double[] lk = new double[members.length];
            double max = Double.NEGATIVE_INFINITY;
            for (int k = 0; k < members.length; k++) {
                for (int i : pts) {
                    lk[k] += Math.log(Math.max(members[k][i][labels[i]], 1e-12));
                }
                max = Math.max(max, lk[k]);
            }
            double s = 0;
            for (double v : lk) {
                s += Math.exp(v - max);
            }
            total -= (max + Math.log(s / members.length)) / tau;
        }
        return total / trials;
    }

    private static double[] classificationScores(Images d, Members m, double[] uIn, double[] uOut, int passes,
            long seed) {
        double[][] p = meanOf(m.test());
        double[] in = uIn == null ? entropies(p) : uIn;
        double[] out = uOut == null ? entropies(meanOf(m.ood())) : uOut;
        return new double[] {Calibration.accuracy(p, d.testLabels()), Calibration.negativeLogLikelihood(p, d.testLabels()),
            Calibration.expectedCalibrationError(p, d.testLabels(), 10), UncertaintyBenchmark.auroc(in, out), passes,
            jointNll(m.test(), d.testLabels(), seed)};
    }

    private static Members members(Images d, java.util.function.Function<Tensor, Tensor> logits, int k,
            Runnable between) {
        double[][][] test = new double[k][][];
        double[][][] ood = new double[k][][];
        for (int i = 0; i < k; i++) {
            between.run();
            test[i] = softmaxRows(logits.apply(d.testX()));
            ood[i] = softmaxRows(logits.apply(d.oodX()));
        }
        return new Members(test, ood);
    }

    /** A residual block {@code h + SN(h)} over spectrally normalised layers (DDU's body). */
    private static final class ResidualSn implements Layer {
        private final SpectralNormDense inner;

        ResidualSn(int width, SplittableRandom r) {
            inner = new SpectralNormDense(width, width, Activation.RELU, r);
        }

        @Override
        public Var forward(Var input) {
            return input.add(inner.forward(input));
        }

        @Override
        public List<Var> parameters() {
            return inner.parameters();
        }
    }

    /**
     * Scores every method on the image OOD task.
     *
     * @param seed the seed
     * @return method to {@code {accuracy, NLL, ECE, OOD AUROC, forward passes, joint NLL}}
     */
    public static Map<String, double[]> imageOod(long seed) {
        Images d = images(seed);
        Map<String, double[]> out = new LinkedHashMap<>();
        Model map = trainMap(d, seed);
        out.put("MAP", classificationScores(d, members(d, x -> map.forward(Var.of(x)).value(), 1, () -> { }), null,
                null, 1, seed));
        Model[] ens = new Model[5];
        ens[0] = map;
        for (int k = 1; k < 5; k++) {
            ens[k] = trainMap(d, seed + 101L * k);
        }
        double[][][] et = new double[5][][];
        double[][][] eo = new double[5][][];
        for (int k = 0; k < 5; k++) {
            et[k] = softmaxRows(ens[k].forward(Var.of(d.testX())).value());
            eo[k] = softmaxRows(ens[k].forward(Var.of(d.oodX())).value());
        }
        Members em = new Members(et, eo);
        out.put("deep ensemble (5)", classificationScores(d, em, null, null, 5, seed));
        // IVON
        Model iv = classifier(seed);
        Ivon ivon = new Ivon(0.3f, 600, 1e-3f, 1f, 0.9f, 0.99999f, 1f, seed);
        epochs(d, 15, seed, (x, y) -> {
            crossEntropy(iv.forward(Var.of(x), true), y).backward();
            ivon.step(iv.parameters());
        });
        Members im = members(d, x -> iv.forward(Var.of(x)).value(), 10, () -> ivon.sample(iv.parameters()));
        out.put("IVON (10 samples)", classificationScores(d, im, null, null, 10, seed));
        ivon.useMean(iv.parameters());
        out.put("IVON (mean)", classificationScores(d, members(d, x -> iv.forward(Var.of(x)).value(), 1, () -> { }),
                null, null, 1, seed));
        // SVGD: five networks, full-batch posterior gradient
        Model sv = classifier(seed);
        List<Var> ps = sv.parameters();
        double[][] init = new double[5][];
        for (int k = 0; k < 5; k++) {
            init[k] = HamiltonianMonteCarlo.flatten(ens[k].parameters());
        }
        LogDensity post = HamiltonianMonteCarlo.parameterDensity(ps, () -> {
            Var e = crossEntropy(sv.forward(Var.of(d.trainX())), d.trainY()).mul(600f);
            for (Var p : ps) {
                e = e.add(p.square().sum().mul(0.5f));
            }
            return e;
        }, 1);
        SteinVariationalGradientDescent svgd = SteinVariationalGradientDescent.create(post, init, 0.002);
        double[][] particles = svgd.run(40);
        int[] idx = new int[1];
        Members sm = members(d, x -> sv.forward(Var.of(x)).value(), 5, () -> HamiltonianMonteCarlo.assign(ps,
                particles[idx[0]++]));
        out.put("SVGD (5, from ensemble)", classificationScores(d, sm, null, null, 5, seed));
        // epinet on the MAP network's last hidden features, trained jointly with a fresh base
        SplittableRandom r = new SplittableRandom(seed);
        Model body = Model.sequential().input(64).dense(64, Activation.RELU).dense(64, Activation.RELU).seed(seed)
                .build();
        Var headW = Var.param(Tensor.normal(Shape.of(64, CLASSES), 0, 0.1f, r));
        Var headB = Var.param(Tensor.zeros(Shape.of(1, CLASSES)));
        Epinet epi = new Epinet(64 + 64, CLASSES, 8, 32, 1f, r);
        List<Var> eps = new ArrayList<>(body.parameters());
        eps.add(headW);
        eps.add(headB);
        eps.addAll(epi.parameters());
        Optimizer opt = Optimizer.adam(3e-3f);
        epochs(d, 15, seed, (x, y) -> {
            Var xin = Var.of(x);
            Var f = body.forward(xin, true);
            Var base = f.matmul(headW).add(headB);
            Var feats = Var.concat(1, f, xin);
            Var loss = null;
            for (int k = 0; k < 4; k++) {
                Var l = crossEntropy(base.add(epi.forward(feats)), y);
                loss = loss == null ? l : loss.add(l);
            }
            loss.mul(0.25f).backward();
            opt.step(eps);
        });
        double[][][] ept = new double[10][][];
        double[][][] epo = new double[10][][];
        for (int k = 0; k < 10; k++) {
            Tensor z = epi.index();
            ept[k] = softmaxRows(epiLogits(body, headW, headB, epi, d.testX(), z));
            epo[k] = softmaxRows(epiLogits(body, headW, headB, epi, d.oodX(), z));
        }
        out.put("epinet (10 indices)", classificationScores(d, new Members(ept, epo), null, null, 1, seed));
        // DDU: residual spectral-normalised body, softmax head, GDA on features
        SplittableRandom rr = new SplittableRandom(seed);
        Model sn = Model.sequential().input(64).dense(64, Activation.RELU).layer(new ResidualSn(64, rr))
                .layer(new ResidualSn(64, rr)).seed(seed).build();
        Var w = Var.param(Tensor.normal(Shape.of(64, CLASSES), 0, 0.1f, rr));
        Var b = Var.param(Tensor.zeros(Shape.of(1, CLASSES)));
        List<Var> dps = new ArrayList<>(sn.parameters());
        dps.add(w);
        dps.add(b);
        Optimizer dopt = Optimizer.adam(3e-3f);
        epochs(d, 15, seed, (x, y) -> {
            crossEntropy(sn.forward(Var.of(x), true).matmul(w).add(b), y).backward();
            dopt.step(dps);
        });
        double[][] trainF = rows(sn.forward(Var.of(d.trainX())).value(), 64);
        FeatureDensityUncertainty ddu = FeatureDensityUncertainty.fit(trainF, d.trainLabels(), CLASSES, 1e-3);
        Members dm = members(d, x -> sn.forward(Var.of(x)).matmul(w).add(b).value(), 1, () -> { });
        out.put("DDU", classificationScores(d, dm, ddu.uncertainty(rows(sn.forward(Var.of(d.testX())).value(), 64)),
                ddu.uncertainty(rows(sn.forward(Var.of(d.oodX())).value(), 64)), 1, seed));
        return out;
    }

    private static Tensor epiLogits(Model body, Var headW, Var headB, Epinet epi, Tensor x, Tensor z) {
        Var xin = Var.of(x);
        Var f = body.forward(xin);
        return f.matmul(headW).add(headB).add(epi.forward(Var.concat(1, f, xin), z)).value();
    }

    private static double[][] rows(Tensor t, int width) {
        double[] v = t.toDoubleArray();
        int n = v.length / width;
        double[][] out = new double[n][];
        for (int i = 0; i < n; i++) {
            out[i] = Arrays.copyOfRange(v, i * width, (i + 1) * width);
        }
        return out;
    }
}
