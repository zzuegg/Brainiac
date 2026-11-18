package io.github.zzuegg.brainiac.experiments.shift;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.data.Augmentation;
import io.github.zzuegg.brainiac.experimental.data.ShapeImages;
import io.github.zzuegg.brainiac.experimental.data.ShiftSuites;
import io.github.zzuegg.brainiac.experimental.ood.FeatureClassifier;
import io.github.zzuegg.brainiac.experimental.shift.DensityRatio;
import io.github.zzuegg.brainiac.experimental.shift.LastLayerRetraining;
import io.github.zzuegg.brainiac.experimental.shift.MixStyle;
import io.github.zzuegg.brainiac.experimental.shift.RobustFineTuning;
import io.github.zzuegg.brainiac.experimental.shift.Swad;
import io.github.zzuegg.brainiac.experimental.train.GroupRobustTraining;
import io.github.zzuegg.brainiac.experimental.train.ModelMerging;
import io.github.zzuegg.brainiac.experimental.train.StochasticWeightAveraging;
import io.github.zzuegg.brainiac.experiments.safety.CausalAndFairness;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.GlobalPool;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Wave 10 items 10.2.12-10.2.14 and 10.2.16: generalisation to unseen domains, spurious
 * correlations, covariate-shift weighting and robust fine-tuning, on {@link ShiftSuites}.
 *
 * <ul>
 *   <li>{@link #leaveOneOut}: four shape styles (contrast-reversed "colour", textured
 *       background, thick strokes, heavy noise); a small CNN is trained on three and tested on
 *       the fourth. ERM, ERM with RandAugment (core {@link Augmentation}), IRM, V-REx and
 *       GroupDRO over the training domains ({@link GroupRobustTraining}), plain SWA
 *       ({@link StochasticWeightAveraging}), SWAD ({@link Swad}) and MixStyle ({@link MixStyle}
 *       after the first convolution, domain-aware partners).</li>
 *   <li>{@link #spurious}: last-layer retraining (DFR, {@link LastLayerRetraining}) of an ERM
 *       model on a small group-balanced held-out set, on {@link CausalAndFairness}' coloured
 *       bars (where GroupDRO, JTT and IRM are that class's own rows) and on a Waterbirds-like
 *       shape-on-background split; worst-group and average accuracy.</li>
 *   <li>{@link #covariate}: Shimodaira's covariate-shift regression in 1 and 5 dimensions,
 *       a misspecified linear model fitted by importance-weighted least squares with KLIEP,
 *       uLSIF and classifier-odds weights ({@link DensityRatio}), against no weights and the
 *       true ratio; and {@link #conformal}, weighted against plain split conformal coverage on
 *       a shifted target.</li>
 *   <li>{@link #fineTuning}: an MLP encoder pretrained contrastively on unlabelled shapes of
 *       every style, then fine-tuned on plain shapes: linear probe, full fine-tuning, LP-FT,
 *       WiSE-FT along {@code alpha}, and a uniform soup of fine-tuned seeds
 *       ({@link ModelMerging#uniformSoup}); in-distribution and shifted accuracy.</li>
 * </ul>
 */
public final class DomainGeneralization {

    /** The domain-generalisation methods. */
    public enum Method {
        /** pooled empirical risk minimisation */
        ERM,
        /** ERM with RandAugment on half of each batch */
        ERM_AUGMENT,
        /** IRMv1 over the training domains */
        IRM,
        /** risk extrapolation */
        VREX,
        /** GroupDRO over the training domains */
        GROUP_DRO,
        /** plain SWA over the second half of training */
        SWA,
        /** SWAD's dense, validation-windowed averaging */
        SWAD,
        /** MixStyle after the first convolution */
        MIXSTYLE
    }

    /** The appearance shifts the fine-tuning comparison evaluates on, all seen in pretraining. */
    public static final ShiftSuites.Style[] FINE_TUNING_SHIFTS = {ShiftSuites.Style.THICK,
        ShiftSuites.Style.NOISY, ShiftSuites.Style.BLURRED, ShiftSuites.Style.LOW_CONTRAST};

    /** The four domains of {@link #leaveOneOut}. */
    public static final ShiftSuites.Style[] DOMAINS = {ShiftSuites.Style.INVERTED, ShiftSuites.Style.TEXTURED,
        ShiftSuites.Style.THICK, ShiftSuites.Style.NOISY};

    private static final int[] SHAPES = {0, 1, 2, 3, 4, 5};

    private DomainGeneralization() {
    }

    private static Model cnn(MixStyle mix, long seed) {
        Model.Builder b = Model.sequential().input(1, 8, 8).seed(seed).conv2d(16, 3, Activation.IDENTITY);
        if (mix != null) {
            b.layer(mix);
        }
        return b.activation(Activation.RELU).conv2d(32, 3, Activation.RELU).layer(GlobalPool.avg())
                .dense(6, Activation.IDENTITY).build();
    }

    /**
     * Trains on three domains and tests on the held-out one.
     *
     * @param method the method
     * @param heldOut the index into {@link #DOMAINS} of the test domain
     * @param steps Adam steps
     * @param seed the seed
     * @return the held-out domain accuracy
     */
    public static double leaveOneOut(Method method, int heldOut, int steps, long seed) {
        List<ShiftSuites.Split> train = new ArrayList<>();
        List<ShiftSuites.Split> val = new ArrayList<>();
        for (int d = 0; d < DOMAINS.length; d++) {
            if (d != heldOut) {
                train.add(ShiftSuites.shapes(600, SHAPES, DOMAINS[d], 0.15, seed + 10L * d));
                val.add(ShiftSuites.shapes(100, SHAPES, DOMAINS[d], 0.15, seed + 10L * d + 5));
            }
        }
        ShiftSuites.Split test = ShiftSuites.shapes(600, SHAPES, DOMAINS[heldOut], 0.15, seed + 99);
        MixStyle mix = method == Method.MIXSTYLE ? new MixStyle(0.5, 0.1, seed) : null;
        Model m = cnn(mix, seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        Augmentation rand = Augmentation.randAugment(1, 8, 8, 2, 0.5);
        GroupRobustTraining.GroupDro dro = new GroupRobustTraining.GroupDro(train.size(), 0.01);
        StochasticWeightAveraging swa = method == Method.SWA ? new StochasticWeightAveraging(m.parameters(),
                steps / 2, 5) : null;
        Swad swad = method == Method.SWAD ? new Swad(m.parameters()) : null;
        Tensor valX = Tensor.concat(0, val.stream().map(ShiftSuites.Split::images).toArray(Tensor[]::new));
        Tensor valY = Tensor.concat(0, val.stream().map(ShiftSuites.Split::ids).toArray(Tensor[]::new));
        int per = 32;
        for (int s = 0; s < steps; s++) {
            List<Var> risks = new ArrayList<>();
            List<Var> logitsPer = new ArrayList<>();
            List<Tensor> ys = new ArrayList<>();
            Tensor[] xs = new Tensor[train.size()];
            int[] domainOfRow = new int[per * train.size()];
            for (int d = 0; d < train.size(); d++) {
                ShiftSuites.Split b = ShiftHarness.batch(train.get(d), per, r);
                Tensor x = b.images();
                if (method == Method.ERM_AUGMENT && r.nextBoolean()) {
                    float[] v = x.toArray();
                    for (int i = 0; i < per; i++) {
                        rand.apply(v, i * 64, r);
                    }
                    x = Tensor.of(x.shape(), v);
                }
                xs[d] = x;
                ys.add(b.ids());
                java.util.Arrays.fill(domainOfRow, d * per, (d + 1) * per, d);
            }
            if (mix != null) {
                mix.domains(domainOfRow);
            }
            Var logits = m.forward(Var.of(Tensor.concat(0, xs)), true);
            for (int d = 0; d < train.size(); d++) {
                Var l = logits.slice(0, d * per, (d + 1) * per);
                logitsPer.add(l);
                risks.add(Functional.crossEntropy(l, ys.get(d)));
            }
            Var loss = switch (method) {
                case IRM -> {
                    Var mean = mean(risks);
                    Var pen = null;
                    for (int d = 0; d < risks.size(); d++) {
                        Tensor y = ys.get(d);
                        Var p = GroupRobustTraining.irmPenalty(logitsPer.get(d), z -> Functional.crossEntropy(z, y));
                        pen = pen == null ? p : pen.add(p);
                    }
                    float w = s < steps / 3 ? 1f : 10f;
                    yield mean.add(pen.mul(w)).mul(1f / w);
                }
                case VREX -> {
                    float w = s < steps / 3 ? 1f : 10f;
                    yield mean(risks).add(GroupRobustTraining.vrexPenalty(risks).mul(w)).mul(1f / w);
                }
                case GROUP_DRO -> dro.loss(risks);
                default -> mean(risks);
            };
            loss.backward();
            opt.step(m.parameters());
            if (swa != null && s >= steps / 2 && s % 5 == 0) {
                swa.sample();
            }
            if (swad != null) {
                swad.sample();
                if ((s + 1) % 20 == 0) {
                    swad.evaluate(Var.noGrad(() -> Functional.crossEntropy(m.forward(Var.of(valX), false), valY))
                            .value().item());
                }
            }
        }
        if (swa != null) {
            swa.swap();
        }
        if (swad != null) {
            swad.window(3, 1.3, 3);
            swad.apply();
        }
        int[] pred = Var.noGrad(() -> m.forward(Var.of(test.images()), false).value()).argmax(1).toIntArray();
        return ShiftHarness.accuracy(pred, test.y());
    }

    private static Var mean(List<Var> v) {
        Var s = v.getFirst();
        for (int i = 1; i < v.size(); i++) {
            s = s.add(v.get(i));
        }
        return s.mul(1f / v.size());
    }

    // ------------------------------------------------------------------ spurious correlations

    /** The spurious-correlation data sets. */
    public enum Spurious {
        /** {@link CausalAndFairness}' coloured bars */
        COLOURED_BARS,
        /** squares and plus signs on correlated stripe backgrounds */
        BACKGROUNDS
    }

    /**
     * Group-robustness scores.
     *
     * @param average mean test accuracy
     * @param worstGroup the lowest group accuracy
     * @param groupLabels group labels used (training set plus held-out set)
     */
    public record GroupScores(double average, double worstGroup, int groupLabels) {
    }

    /**
     * ERM, DFR and GroupDRO on a spurious split.
     *
     * @param data the data set
     * @param steps Adam steps
     * @param seed the seed
     * @return method name to scores
     */
    public static Map<String, GroupScores> spurious(Spurious data, int steps, long seed) {
        ShiftSuites.Grouped train = generate(data, 2000, 0.9, seed);
        ShiftSuites.Grouped held = generate(data, 400, 0.9, seed + 10);
        ShiftSuites.Grouped test = generate(data, 2000, 0.5, seed + 20);
        int inputs = train.split().x().shape().dim(1);
        Map<String, GroupScores> out = new LinkedHashMap<>();
        FeatureClassifier erm = ShiftHarness.mlp(inputs, 2, seed);
        ShiftHarness.train(erm, train.split(), steps, 1e-3f, seed);
        out.put("ERM", groupScores(erm.predict(test.split().x()), test, 0));
        Dense head = LastLayerRetraining.retrain(erm.features(held.split().x()), held.split().y(), held.group(), 4, 2,
                0.01, 10, seed);
        int[] dfr = Var.noGrad(() -> head.forward(Var.of(erm.features(test.split().x())))).value().argmax(1)
                .toIntArray();
        out.put("DFR", groupScores(dfr, test, held.group().length));
        // GroupDRO with full group labels, same network and steps
        FeatureClassifier gm = ShiftHarness.mlp(inputs, 2, seed);
        GroupRobustTraining.GroupDro dro = new GroupRobustTraining.GroupDro(4, 0.01);
        Optimizer opt = Optimizer.adam(1e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            int[] rows = new int[100];
            for (int i = 0; i < rows.length; i++) {
                rows[i] = r.nextInt(train.group().length);
            }
            ShiftSuites.Split b = train.split().rows(rows);
            Var logits = gm.logits(Var.of(b.x()), true);
            List<Var> losses = new ArrayList<>();
            for (int g = 0; g < 4; g++) {
                List<Float> sel = new ArrayList<>();
                for (int i = 0; i < rows.length; i++) {
                    if (train.group()[rows[i]] == g) {
                        sel.add((float) i);
                    }
                }
                if (sel.isEmpty()) {
                    losses.add(null);
                    continue;
                }
                float[] idx = new float[sel.size()];
                for (int i = 0; i < idx.length; i++) {
                    idx[i] = sel.get(i);
                }
                Tensor it = Tensor.of(Shape.of(idx.length), idx);
                losses.add(Functional.crossEntropy(logits.gather(0, it), b.ids().gather(0, it)));
            }
            dro.loss(losses).backward();
            opt.step(gm.parameters());
        }
        out.put("GroupDRO", groupScores(gm.predict(test.split().x()), test, train.group().length));
        return out;
    }

    private static ShiftSuites.Grouped generate(Spurious data, int n, double agreement, long seed) {
        return data == Spurious.COLOURED_BARS ? CausalAndFairness.colouredShapes(n, agreement, seed)
                : ShiftSuites.backgrounds(n, agreement, seed);
    }

    private static GroupScores groupScores(int[] pred, ShiftSuites.Grouped test, int groupLabels) {
        double[] hit = new double[4];
        double[] cnt = new double[4];
        int ok = 0;
        for (int i = 0; i < pred.length; i++) {
            boolean right = pred[i] == test.split().y()[i];
            ok += right ? 1 : 0;
            hit[test.group()[i]] += right ? 1 : 0;
            cnt[test.group()[i]]++;
        }
        double worst = 1;
        for (int g = 0; g < 4; g++) {
            if (cnt[g] > 0) {
                worst = Math.min(worst, hit[g] / cnt[g]);
            }
        }
        return new GroupScores(ok / (double) pred.length, worst, groupLabels);
    }

    // ------------------------------------------------------------------ covariate shift

    /** The importance weights of {@link #covariate}. */
    public enum Weighting {
        /** unweighted least squares */
        NONE,
        /** the true density ratio */
        ORACLE,
        /** KLIEP */
        KLIEP,
        /** uLSIF */
        ULSIF,
        /** logistic domain-classifier odds */
        CLASSIFIER,
        /** uLSIF clipped at 10x the mean weight */
        ULSIF_CLIPPED
    }

    /**
     * Covariate-shift regression results.
     *
     * @param targetMse the weighted linear fit's MSE on target data
     * @param ratioError mean squared error of the mean-normalised weights against the true ones
     * @param effectiveSampleSize {@code (sum w)^2 / sum w^2} over the training points
     */
    public record CovariateScores(double targetMse, double ratioError, double effectiveSampleSize) {
    }

    /**
     * Importance-weighted linear regression under Shimodaira's covariate shift.
     *
     * @param dim the input dimension (1 or 5)
     * @param n training points
     * @param seed the seed
     * @return scores per weighting
     */
    public static Map<Weighting, CovariateScores> covariate(int dim, int n, long seed) {
        Tensor[] src = ShiftSuites.shimodaira(n, dim, false, seed);
        Tensor[] tgtUnlabelled = ShiftSuites.shimodaira(n, dim, true, seed + 1);
        Tensor[] tgtTest = ShiftSuites.shimodaira(2000, dim, true, seed + 2);
        double[][] xs = ShiftHarness.rows(src[0]);
        double[][] xt = ShiftHarness.rows(tgtUnlabelled[0]);
        double[] ys = src[1].toDoubleArray();
        double[] oracle = new double[n];
        float[] xf = src[0].toArray();
        for (int i = 0; i < n; i++) {
            oracle[i] = Math.exp(ShiftSuites.shimodairaLogRatio(java.util.Arrays.copyOfRange(xf, i * dim, (i + 1) * dim)));
        }
        Map<Weighting, CovariateScores> out = new LinkedHashMap<>();
        DensityRatio ulsif = DensityRatio.ulsif(xs, xt, 100, seed);
        for (Weighting w : Weighting.values()) {
            double[] weights = switch (w) {
                case NONE -> ones(n);
                case ORACLE -> oracle;
                case KLIEP -> DensityRatio.kliep(xs, xt, 100, seed).weights(xs);
                case ULSIF -> ulsif.weights(xs);
                case CLASSIFIER -> DensityRatio.classifierOdds(xs, xt, 1e-3).weights(xs);
                case ULSIF_CLIPPED -> DensityRatio.clip(ulsif.weights(xs), 10 * mean(ulsif.weights(xs)));
            };
            double[] beta = weightedLeastSquares(xs, ys, weights);
            double[][] xtest = ShiftHarness.rows(tgtTest[0]);
            double[] ytest = tgtTest[1].toDoubleArray();
            double mse = 0;
            for (int i = 0; i < xtest.length; i++) {
                double p = beta[0];
                for (int j = 0; j < dim; j++) {
                    p += beta[j + 1] * xtest[i][j];
                }
                mse += (p - ytest[i]) * (p - ytest[i]) / xtest.length;
            }
            double mw = mean(weights);
            double mo = mean(oracle);
            double err = 0;
            for (int i = 0; i < n; i++) {
                double d = weights[i] / mw - oracle[i] / mo;
                err += d * d / n;
            }
            out.put(w, new CovariateScores(mse, err, DensityRatio.effectiveSampleSize(weights)));
        }
        return out;
    }

    private static double[] weightedLeastSquares(double[][] x, double[] y, double[] w) {
        int d = x[0].length + 1;
        float[] a = new float[d * d];
        float[] b = new float[d];
        for (int i = 0; i < x.length; i++) {
            double[] f = new double[d];
            f[0] = 1;
            System.arraycopy(x[i], 0, f, 1, d - 1);
            for (int p = 0; p < d; p++) {
                b[p] += (float) (w[i] * f[p] * y[i]);
                for (int q = 0; q < d; q++) {
                    a[p * d + q] += (float) (w[i] * f[p] * f[q]);
                }
            }
        }
        for (int p = 0; p < d; p++) {
            a[p * d + p] += 1e-6f;
        }
        return io.github.zzuegg.brainiac.tensor.Linalg.solve(Tensor.of(Shape.of(d, d), a), Tensor.of(Shape.of(d, 1), b))
                .toDoubleArray();
    }

    /**
     * Conformal coverage on the shifted target of {@link ShiftSuites#covariateClassification}.
     *
     * @param plain split conformal coverage on the target
     * @param weighted weighted conformal with uLSIF weights
     * @param oracle weighted conformal with the true weights
     * @param plainSize mean set size, plain
     * @param weightedSize mean set size, uLSIF-weighted
     */
    public record ConformalScores(double plain, double weighted, double oracle, double plainSize,
            double weightedSize) {
    }

    /**
     * Plain against weighted split conformal at miscoverage {@code alpha}.
     *
     * @param alpha the miscoverage, such as 0.1
     * @param seed the seed
     * @return the target coverages and set sizes
     */
    public static ConformalScores conformal(double alpha, long seed) {
        ShiftSuites.Split train = ShiftSuites.covariateClassification(1000, false, seed);
        ShiftSuites.Split cal = ShiftSuites.covariateClassification(1000, false, seed + 1);
        ShiftSuites.Split unlabelled = ShiftSuites.covariateClassification(1000, true, seed + 2);
        ShiftSuites.Split test = ShiftSuites.covariateClassification(1000, true, seed + 3);
        FeatureClassifier m = FeatureClassifier.mlp(2, new int[] {32, 32}, 2, seed);
        ShiftHarness.train(m, train, 600, 3e-3f, seed);
        double[][] pc = ShiftHarness.probabilities(m.logits(cal.x()));
        double[][] pt = ShiftHarness.probabilities(m.logits(test.x()));
        DensityRatio dr = DensityRatio.ulsif(ShiftHarness.rows(cal.x()), ShiftHarness.rows(unlabelled.x()), 100, seed);
        double[] wCal = dr.weights(ShiftHarness.rows(cal.x()));
        double[] wTest = dr.weights(ShiftHarness.rows(test.x()));
        double[][] calRows = ShiftHarness.rows(cal.x());
        double[][] testRows = ShiftHarness.rows(test.x());
        double[] oCal = new double[calRows.length];
        double[] oTest = new double[testRows.length];
        for (int i = 0; i < oCal.length; i++) {
            oCal[i] = Math.exp(ShiftSuites.covariateLogRatio(new float[] {(float) calRows[i][0], (float) calRows[i][1]}));
        }
        for (int i = 0; i < oTest.length; i++) {
            oTest[i] = Math.exp(ShiftSuites.covariateLogRatio(new float[] {(float) testRows[i][0],
                (float) testRows[i][1]}));
        }
        DensityRatio.WeightedConformal plain = new DensityRatio.WeightedConformal(pc, cal.y(), ones(pc.length), alpha);
        DensityRatio.WeightedConformal weighted = new DensityRatio.WeightedConformal(pc, cal.y(), wCal, alpha);
        DensityRatio.WeightedConformal oracle = new DensityRatio.WeightedConformal(pc, cal.y(), oCal, alpha);
        double[] cp = coverage(plain, pt, test.y(), ones(pt.length));
        double[] cw = coverage(weighted, pt, test.y(), wTest);
        double[] co = coverage(oracle, pt, test.y(), oTest);
        return new ConformalScores(cp[0], cw[0], co[0], cp[1], cw[1]);
    }

    private static double[] coverage(DensityRatio.WeightedConformal c, double[][] p, int[] y, double[] w) {
        int hit = 0;
        double size = 0;
        for (int i = 0; i < p.length; i++) {
            int[] set = c.predictSet(p[i], w[i]);
            size += set.length;
            for (int s : set) {
                hit += s == y[i] ? 1 : 0;
            }
        }
        return new double[] {hit / (double) p.length, size / p.length};
    }

    private static double[] ones(int n) {
        double[] o = new double[n];
        java.util.Arrays.fill(o, 1);
        return o;
    }

    private static double mean(double[] v) {
        double s = 0;
        for (double x : v) {
            s += x / v.length;
        }
        return s;
    }

    // ------------------------------------------------------------------ robust fine-tuning

    /**
     * Fine-tuning results.
     *
     * @param inDistribution accuracy on plain shapes
     * @param shifted mean accuracy over {@link #FINE_TUNING_SHIFTS}
     */
    public record FineTuneScores(double inDistribution, double shifted) {
    }

    /**
     * Pretrains an MLP encoder contrastively (InfoNCE between two augmented views of one shape,
     * each rendered in a random style) on unlabelled shapes: the local stand-in for {@code SelfSupervisedBenchmark}'s SimCLR
     * encoder.
     *
     * @param steps pretraining steps
     * @param seed the seed
     * @return the encoder with a fresh head
     */
    public static FeatureClassifier pretrain(int steps, long seed) {
        FeatureClassifier m = ShiftHarness.mlp(ShiftSuites.PIXELS, 6, seed);
        ShiftSuites.Split pool = ShiftSuites.shapes(2000, SHAPES, ShiftSuites.Style.PLAIN, 0.15, seed);
        ShiftSuites.Style[] styles = ShiftSuites.Style.values();
        ShapeImages aug = new ShapeImages(ShiftSuites.SIDE, 0.15f);
        Optimizer opt = Optimizer.adam(2e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            Tensor img = ShiftHarness.batch(pool, 64, r).images();
            // the two views are the same shape in two random styles: the diverse "web data" that
            // makes the pretrained features robust to the appearance shifts
            Tensor v1 = ShiftSuites.apply(aug.augment(img, 1, r).reshape(64, ShiftSuites.PIXELS),
                    styles[r.nextInt(styles.length)], r.nextLong());
            Tensor v2 = ShiftSuites.apply(aug.augment(img, 1, r).reshape(64, ShiftSuites.PIXELS),
                    styles[r.nextInt(styles.length)], r.nextLong());
            Functional.infoNce(m.features(Var.of(v1), true), m.features(Var.of(v2), true), 0.2f).backward();
            opt.step(m.body().parameters());
        }
        return m;
    }

    /**
     * Linear probe, fine-tuning, LP-FT, WiSE-FT and a soup, from one pretrained encoder.
     *
     * @param pretrainSteps contrastive pretraining steps
     * @param fineTuneSteps fine-tuning steps
     * @param alphas the WiSE-FT mixing weights to report
     * @param seed the seed
     * @return method name to scores
     */
    public static Map<String, FineTuneScores> fineTuning(int pretrainSteps, int fineTuneSteps, double[] alphas,
            long seed) {
        FeatureClassifier pre = pretrain(pretrainSteps, seed);
        List<Tensor> start = pre.snapshot();
        ShiftSuites.Split train = ShiftSuites.shapes(1200, SHAPES, ShiftSuites.Style.PLAIN, 0.15, seed + 50);
        Map<String, FineTuneScores> out = new LinkedHashMap<>();
        // linear probe
        RobustFineTuning.linearProbe(pre, train.x(), train.y(), 300, 1e-2f, seed);
        List<Tensor> probed = pre.snapshot();
        out.put("linear probe", evaluate(pre, seed));
        // LP-FT
        RobustFineTuning.fineTune(pre, train.x(), train.y(), fineTuneSteps, 1e-3f, seed);
        out.put("LP-FT", evaluate(pre, seed));
        List<Tensor> lpft = pre.snapshot();
        // full fine-tuning from the random head, three seeds for the soup
        List<List<Tensor>> tuned = new ArrayList<>();
        for (int k = 0; k < 3; k++) {
            pre.restore(start);
            FeatureClassifier fresh = ShiftHarness.mlp(ShiftSuites.PIXELS, 6, seed + 1000L * (k + 1));
            List<Var> headParams = pre.head().parameters();
            for (int i = 0; i < headParams.size(); i++) {
                headParams.get(i).assign(fresh.head().parameters().get(i).value().copy());
            }
            RobustFineTuning.fineTune(pre, train.x(), train.y(), fineTuneSteps, 1e-3f, seed + k);
            tuned.add(pre.snapshot());
            if (k == 0) {
                out.put("full fine-tuning", evaluate(pre, seed));
            }
        }
        pre.restore(ModelMerging.uniformSoup(tuned));
        out.put("soup of 3 fine-tuned", evaluate(pre, seed));
        for (double a : alphas) {
            pre.restore(RobustFineTuning.interpolate(probed, tuned.getFirst(), a));
            out.put(String.format("WiSE-FT a=%.2f", a), evaluate(pre, seed));
            pre.restore(RobustFineTuning.interpolate(probed, lpft, a));
            out.put(String.format("WiSE-(LP-FT) a=%.2f", a), evaluate(pre, seed));
        }
        return out;
    }

    private static FineTuneScores evaluate(FeatureClassifier m, long seed) {
        ShiftSuites.Split id = ShiftSuites.shapes(600, SHAPES, ShiftSuites.Style.PLAIN, 0.15, seed + 77);
        double shifted = 0;
        for (ShiftSuites.Style s : FINE_TUNING_SHIFTS) {
            ShiftSuites.Split t = ShiftSuites.shapes(600, SHAPES, s, 0.15, seed + 78 + s.ordinal());
            shifted += m.accuracy(t.x(), t.y()) / FINE_TUNING_SHIFTS.length;
        }
        return new FineTuneScores(m.accuracy(id.x(), id.y()), shifted);
    }

    /**
     * The long-budget run: every method on all four held-out domains over three seeds, the
     * spurious splits, covariate shift over ten draws, and fine-tuning.
     *
     * @param args ignored
     */
    public static void main(String[] args) {
        for (Method m : Method.values()) {
            double sum = 0;
            for (long seed = 1; seed <= 3; seed++) {
                for (int d = 0; d < DOMAINS.length; d++) {
                    sum += leaveOneOut(m, d, 1500, seed) / (3 * DOMAINS.length);
                }
            }
            System.out.printf("DG %-12s leave-one-domain-out accuracy %.3f%n", m, sum);
        }
        for (Spurious s : Spurious.values()) {
            System.out.println(s + " " + spurious(s, 2000, 1));
        }
        for (CausalAndFairness.RobustMethod m : CausalAndFairness.RobustMethod.values()) {
            System.out.println("coloured bars " + m + " " + CausalAndFairness.spurious(m, 1));
        }
        for (int dim : new int[] {1, 5}) {
            System.out.println("covariate dim " + dim + " " + covariate(dim, 200, 1));
        }
        System.out.println(conformal(0.1, 1));
        System.out.println(fineTuning(1500, 600, new double[] {0.25, 0.5, 0.75}, 1));
    }
}
