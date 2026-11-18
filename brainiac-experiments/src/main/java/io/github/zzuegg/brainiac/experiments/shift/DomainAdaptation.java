package io.github.zzuegg.brainiac.experiments.shift;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ShapeImages;
import io.github.zzuegg.brainiac.experimental.data.ShiftSuites;
import io.github.zzuegg.brainiac.experimental.ood.FeatureClassifier;
import io.github.zzuegg.brainiac.experimental.robust.TestTimeAdaptation;
import io.github.zzuegg.brainiac.experimental.shift.AdversarialAdaptation;
import io.github.zzuegg.brainiac.experimental.shift.DiscrepancyAdaptation;
import io.github.zzuegg.brainiac.experimental.shift.GradientReversal;
import io.github.zzuegg.brainiac.experimental.shift.RobustTestTimeAdaptation;
import io.github.zzuegg.brainiac.experimental.shift.SourceFreeAdaptation;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Wave 10 items 10.2.9-10.2.11 and 10.2.15: unsupervised domain adaptation from labelled source
 * data to unlabelled target data on {@link ShiftSuites} pairs (shapes plain to textured
 * background, digits in one glyph font to another, two-moons rotated by 35 degrees).
 *
 * <ul>
 *   <li>{@link #adapt}: source-only, a target-supervised oracle, DANN, CDAN and CDAN+E
 *       ({@link AdversarialAdaptation}), Deep CORAL and multi-kernel MMD
 *       ({@link DiscrepancyAdaptation}), all at equal steps on the same batch-norm MLP; and on the
 *       trained source model, with no source data, AdaBN (test-batch BN statistics), TENT
 *       ({@link TestTimeAdaptation}) and SHOT ({@link SourceFreeAdaptation}). Scores: target
 *       accuracy, the share of the source-only-to-oracle gap closed, and the number of classes
 *       predicted (collapse).</li>
 *   <li>{@link #continual}: a stream cycling through six corruptions for several rounds,
 *       optionally with small or single-class batches: no adaptation, BN statistics, TENT,
 *       EATA, SAR and CoTTA ({@link RobustTestTimeAdaptation}); accuracy overall, in the first
 *       and last round, and SAR's resets.</li>
 * </ul>
 */
public final class DomainAdaptation {

    /** A source/target pair. */
    public enum Task {
        /** shapes, plain to a textured background */
        SHAPES_TEXTURE,
        /** digits, block font to round font */
        DIGITS_FONT,
        /** two-moons rotated by 35 degrees */
        MOONS
    }

    /** An adaptation method. */
    public enum Method {
        /** source data only */
        SOURCE_ONLY,
        /** target labels (upper bound) */
        ORACLE,
        /** gradient-reversed domain discriminator */
        DANN,
        /** conditional discriminator */
        CDAN,
        /** conditional discriminator with entropy weights */
        CDAN_E,
        /** covariance alignment */
        CORAL,
        /** multi-kernel MMD */
        MMD,
        /** test-batch BN statistics on the source model (AdaBN) */
        ADABN,
        /** TENT on the source model */
        TENT,
        /** source-free SHOT on the source model */
        SHOT
    }

    /**
     * One method on one task.
     *
     * @param targetAccuracy accuracy on held-out target data
     * @param sourceAccuracy accuracy on held-out source data
     * @param classesPredicted distinct classes predicted on the target
     */
    public record Result(double targetAccuracy, double sourceAccuracy, int classesPredicted) {
    }

    private DomainAdaptation() {
    }

    /**
     * The data of a task.
     *
     * @param source labelled source rows
     * @param target unlabelled target rows (labels kept for the oracle only)
     * @param sourceTest held-out source rows
     * @param targetTest held-out target rows
     * @param classes the number of classes
     */
    public record Pair(ShiftSuites.Split source, ShiftSuites.Split target, ShiftSuites.Split sourceTest,
            ShiftSuites.Split targetTest, int classes) {
    }

    /**
     * Generates a task.
     *
     * @param task the pair
     * @param seed the seed
     * @return the data
     */
    public static Pair pair(Task task, long seed) {
        return switch (task) {
            case SHAPES_TEXTURE -> {
                int[] all = {0, 1, 2, 3, 4, 5};
                yield new Pair(ShiftSuites.shapes(1500, all, ShiftSuites.Style.PLAIN, 0.2, seed),
                        ShiftSuites.shapes(1500, all, ShiftSuites.Style.TEXTURED, 0.2, seed + 1),
                        ShiftSuites.shapes(500, all, ShiftSuites.Style.PLAIN, 0.2, seed + 2),
                        ShiftSuites.shapes(500, all, ShiftSuites.Style.TEXTURED, 0.2, seed + 3), 6);
            }
            case DIGITS_FONT -> {
                int[] all = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
                yield new Pair(digits(1500, all, ShiftSuites.Font.BLOCK, seed),
                        digits(1500, all, ShiftSuites.Font.ROUND, seed + 1),
                        digits(500, all, ShiftSuites.Font.BLOCK, seed + 2),
                        digits(500, all, ShiftSuites.Font.ROUND, seed + 3), 10);
            }
            case MOONS -> new Pair(ShiftSuites.twoMoons(1000, 0, 0.1, seed), ShiftSuites.twoMoons(1000, 35, 0.1, seed + 1),
                    ShiftSuites.twoMoons(500, 0, 0.1, seed + 2), ShiftSuites.twoMoons(500, 35, 0.1, seed + 3), 2);
        };
    }

    private static ShiftSuites.Split digits(int n, int[] all, ShiftSuites.Font font, long seed) {
        return ShiftSuites.digits(n, all, font, ShiftSuites.Style.PLAIN, 0.2, seed);
    }

    /**
     * The batch-norm MLP used for every method.
     *
     * @param inputs input features
     * @param classes the number of classes
     * @param seed the seed
     * @return the classifier (32 features)
     */
    public static FeatureClassifier network(int inputs, int classes, long seed) {
        Model body = Model.sequential().input(inputs).dense(64).batchNorm().activation(Activation.RELU).dense(32)
                .batchNorm().activation(Activation.RELU).seed(seed).build();
        return new FeatureClassifier(body, new Dense(32, classes, Activation.IDENTITY, Init.XAVIER,
                new SplittableRandom(seed + 5)));
    }

    /**
     * Trains and scores every method on a task.
     *
     * @param task the pair
     * @param methods the methods to run
     * @param steps training steps
     * @param seed the seed
     * @return the results in order
     */
    public static Map<Method, Result> adapt(Task task, List<Method> methods, int steps, long seed) {
        Pair d = pair(task, seed);
        int inputs = d.source().x().shape().dim(1);
        Map<Method, Result> out = new LinkedHashMap<>();
        FeatureClassifier sourceOnly = null;
        for (Method m : methods) {
            if (m == Method.ADABN || m == Method.TENT || m == Method.SHOT) {
                if (sourceOnly == null) {
                    sourceOnly = train(d, Method.SOURCE_ONLY, inputs, steps, seed);
                }
                out.put(m, sourceFree(sourceOnly, d, m, seed));
            } else {
                FeatureClassifier model = train(d, m, inputs, steps, seed);
                if (m == Method.SOURCE_ONLY) {
                    sourceOnly = model;
                }
                out.put(m, score(model, d));
            }
        }
        return out;
    }

    private static FeatureClassifier train(Pair d, Method m, int inputs, int steps, long seed) {
        FeatureClassifier model = network(inputs, d.classes(), seed);
        AdversarialAdaptation adv = switch (m) {
            case DANN -> new AdversarialAdaptation(AdversarialAdaptation.Mode.DANN, 32, d.classes(), 64, seed + 1);
            case CDAN -> new AdversarialAdaptation(AdversarialAdaptation.Mode.CDAN, 32, d.classes(), 64, seed + 1);
            case CDAN_E -> new AdversarialAdaptation(AdversarialAdaptation.Mode.CDAN_E, 32, d.classes(), 64, seed + 1);
            default -> null;
        };
        List<Var> params = new ArrayList<>(model.parameters());
        if (adv != null) {
            params.addAll(adv.parameters());
        }
        Optimizer opt = Optimizer.adam(2e-3f);
        SplittableRandom r = new SplittableRandom(seed);
        for (int s = 0; s < steps; s++) {
            ShiftSuites.Split sb = ShiftHarness.batch(m == Method.ORACLE ? d.target() : d.source(), 64, r);
            if (m == Method.SOURCE_ONLY || m == Method.ORACLE) {
                Functional.crossEntropy(model.logits(Var.of(sb.x()), true), sb.ids()).backward();
                opt.step(params);
                continue;
            }
            ShiftSuites.Split tb = ShiftHarness.batch(d.target(), 64, r);
            // one forward over both domains, so batch norm sees the mixed batch as in DANN's practice
            Var feats = model.features(Var.of(Tensor.concat(0, sb.x(), tb.x())), true);
            Var fs = feats.slice(0, 0, 64);
            Var ft = feats.slice(0, 64, 128);
            Var ls = model.head().forward(fs);
            Var lt = model.head().forward(ft);
            Var loss = Functional.crossEntropy(ls, sb.ids());
            float lambda = GradientReversal.schedule(s / (double) steps);
            loss = switch (m) {
                case CORAL -> loss.add(DiscrepancyAdaptation.coral(fs, ft).mul(10f * lambda));
                case MMD -> loss.add(DiscrepancyAdaptation.mmd(fs, ft, DiscrepancyAdaptation.medianBandwidth(fs.value(),
                        ft.value())).mul(lambda));
                default -> loss.add(adv.loss(fs, ls, ft, lt, lambda));
            };
            loss.backward();
            opt.step(params);
        }
        return model;
    }

    private static Result sourceFree(FeatureClassifier trained, Pair d, Method m, long seed) {
        java.util.List<Tensor> saved = trained.snapshot();
        Result res;
        switch (m) {
            case SHOT -> {
                SourceFreeAdaptation shot = new SourceFreeAdaptation(trained, 0.3f, 1e-3f);
                SplittableRandom r = new SplittableRandom(seed);
                for (int e = 0; e < 5; e++) {
                    shot.epoch(d.target().x(), 64, r);
                }
                res = score(trained, d);
            }
            default -> {
                TestTimeAdaptation tta = TestTimeAdaptation.of(trained.model(), m == Method.TENT
                        ? TestTimeAdaptation.Method.TENT : TestTimeAdaptation.Method.BN_STATS);
                int n = d.targetTest().size();
                int[] pred = new int[n];
                // TENT adapts online over one pass of the unlabelled target set, then predicts the test set
                if (m == Method.TENT) {
                    for (int from = 0; from + 64 <= d.target().size(); from += 64) {
                        tta.predict(d.target().range(from, from + 64).x());
                    }
                }
                for (int from = 0; from < n; from += 64) {
                    int to = Math.min(n, from + 64);
                    int[] p = tta.predict(d.targetTest().range(from, to).x()).argmax(1).toIntArray();
                    System.arraycopy(p, 0, pred, from, p.length);
                }
                tta.reset();
                res = new Result(ShiftHarness.accuracy(pred, d.targetTest().y()),
                        trained.accuracy(d.sourceTest().x(), d.sourceTest().y()), distinct(pred));
            }
        }
        trained.restore(saved);
        return res;
    }

    private static Result score(FeatureClassifier m, Pair d) {
        int[] p = m.predict(d.targetTest().x());
        return new Result(ShiftHarness.accuracy(p, d.targetTest().y()), m.accuracy(d.sourceTest().x(),
                d.sourceTest().y()), distinct(p));
    }

    private static int distinct(int[] p) {
        return (int) java.util.Arrays.stream(p).distinct().count();
    }

    /**
     * The share of the source-only-to-oracle gap a method closes.
     *
     * @param r the results
     * @param m the method
     * @return {@code (acc - source) / (oracle - source)}
     */
    public static double gapClosed(Map<Method, Result> r, Method m) {
        double s = r.get(Method.SOURCE_ONLY).targetAccuracy();
        double o = r.get(Method.ORACLE).targetAccuracy();
        return (r.get(m).targetAccuracy() - s) / Math.max(1e-9, o - s);
    }

    // ------------------------------------------------------------------ continual test-time adaptation

    /** A test-time method on the continual stream. */
    public enum StreamMethod {
        /** no adaptation */
        NONE,
        /** test-batch BN statistics */
        BN_STATS,
        /** TENT */
        TENT,
        /** EATA */
        EATA,
        /** SAR */
        SAR,
        /** CoTTA */
        COTTA
    }

    /**
     * Continual-stream results.
     *
     * @param accuracy over the whole stream
     * @param firstRound accuracy in the first round
     * @param lastRound accuracy in the last round
     * @param resets SAR's model resets (0 for the others)
     * @param adaptedShare the share of stream samples used in an update (EATA, SAR and CoTTA; 1 for TENT)
     */
    public record StreamResult(double accuracy, double firstRound, double lastRound, int resets, double adaptedShare) {
    }

    /** The six corruptions the stream cycles through. */
    private static final ShiftSuites.Style[] CORRUPTIONS = {ShiftSuites.Style.NOISY, ShiftSuites.Style.BLURRED,
        ShiftSuites.Style.INVERTED, ShiftSuites.Style.TEXTURED, ShiftSuites.Style.LOW_CONTRAST, ShiftSuites.Style.THICK};

    /**
     * Trains the clean source model of the continual stream.
     *
     * @param steps Adam steps
     * @param seed the seed
     * @return the classifier
     */
    public static FeatureClassifier streamSource(int steps, long seed) {
        int[] all = {0, 1, 2, 3, 4, 5};
        FeatureClassifier m = network(ShiftSuites.PIXELS, 6, seed);
        ShiftHarness.train(m, ShiftSuites.shapes(2000, all, ShiftSuites.Style.PLAIN, 0.2, seed), steps, 3e-3f, seed);
        return m;
    }

    /**
     * Runs a test-time method over the continual stream.
     *
     * @param source the trained source model (restored afterwards)
     * @param method the method
     * @param rounds passes over the six corruptions
     * @param batchesPerDomain batches per corruption per round
     * @param batchSize rows per batch
     * @param singleClass every batch holds one class (label shift)
     * @param seed the seed
     * @return the scores
     */
    public static StreamResult continual(FeatureClassifier source, StreamMethod method, int rounds,
            int batchesPerDomain, int batchSize, boolean singleClass, long seed) {
        List<Tensor> saved = source.snapshot();
        Model model = source.model();
        ShapeImages aug = new ShapeImages(ShiftSuites.SIDE, 0.1f);
        SplittableRandom ar = new SplittableRandom(seed + 99);
        TestTimeAdaptation tta = null;
        RobustTestTimeAdaptation rtta = null;
        switch (method) {
            case BN_STATS -> tta = TestTimeAdaptation.of(model, TestTimeAdaptation.Method.BN_STATS);
            case TENT -> tta = TestTimeAdaptation.of(model, TestTimeAdaptation.Method.TENT);
            case EATA -> rtta = RobustTestTimeAdaptation.of(model, RobustTestTimeAdaptation.Method.EATA)
                    .fisher(ShiftSuites.shapes(256, new int[] {0, 1, 2, 3, 4, 5}, ShiftSuites.Style.PLAIN, 0.2,
                            seed + 7).x());
            case SAR -> rtta = RobustTestTimeAdaptation.of(model, RobustTestTimeAdaptation.Method.SAR).learningRate(5e-3f);
            case COTTA -> rtta = RobustTestTimeAdaptation.of(model, RobustTestTimeAdaptation.Method.COTTA)
                    .cotta(0.99f, 0.01, 0.9, 4, x -> {
                        int n = x.shape().dim(0);
                        return aug.augment(x.reshape(n, 1, 8, 8), 1, ar).reshape(n, ShiftSuites.PIXELS);
                    });
            default -> {
            }
        }
        int[] all = {0, 1, 2, 3, 4, 5};
        long ok = 0;
        long total = 0;
        long okFirst = 0;
        long totalFirst = 0;
        long okLast = 0;
        long totalLast = 0;
        long s = seed;
        for (int round = 0; round < rounds; round++) {
            for (ShiftSuites.Style c : CORRUPTIONS) {
                for (int b = 0; b < batchesPerDomain; b++) {
                    s++;
                    ShiftSuites.Split batch;
                    if (singleClass) {
                        int cls = Math.floorMod(s, 6);
                        batch = ShiftSuites.shapes(batchSize, new int[] {cls}, c, 0.2, s);
                        batch = new ShiftSuites.Split(batch.x(), java.util.Collections.nCopies(batchSize, cls).stream()
                                .mapToInt(Integer::intValue).toArray());
                    } else {
                        batch = ShiftSuites.shapes(batchSize, all, c, 0.2, s);
                    }
                    Tensor logits;
                    if (tta != null) {
                        logits = tta.predict(batch.x());
                    } else if (rtta != null) {
                        logits = rtta.predict(batch.x());
                    } else {
                        logits = source.logits(batch.x());
                    }
                    int[] p = logits.argmax(1).toIntArray();
                    int hit = 0;
                    for (int i = 0; i < p.length; i++) {
                        hit += p[i] == batch.y()[i] ? 1 : 0;
                    }
                    ok += hit;
                    total += p.length;
                    if (round == 0) {
                        okFirst += hit;
                        totalFirst += p.length;
                    }
                    if (round == rounds - 1) {
                        okLast += hit;
                        totalLast += p.length;
                    }
                }
            }
        }
        int resets = rtta == null ? 0 : rtta.resets();
        double share = rtta != null ? rtta.adaptedSamples() / (double) total : method == StreamMethod.TENT ? 1 : 0;
        source.restore(saved);
        return new StreamResult(ok / (double) total, okFirst / (double) totalFirst, okLast / (double) totalLast, resets,
                share);
    }

    /**
     * The long-budget run: every task and method over three seeds, and the continual stream at
     * batch sizes 64, 8 and 1 and with single-class batches.
     *
     * @param args ignored
     */
    public static void main(String[] args) {
        for (Task t : Task.values()) {
            for (long sd = 1; sd <= 3; sd++) {
                long seed = sd;
                Map<Method, Result> r = adapt(t, List.of(Method.values()), 1500, seed);
                r.forEach((m, v) -> System.out.printf("%s seed %d %-12s target %.3f source %.3f classes %d gap %.2f%n",
                        t, seed, m, v.targetAccuracy(), v.sourceAccuracy(), v.classesPredicted(), gapClosed(r, m)));
            }
        }
        FeatureClassifier src = streamSource(1500, 1);
        for (int bs : new int[] {64, 8, 1}) {
            for (StreamMethod m : StreamMethod.values()) {
                System.out.printf("stream bs %d %-8s %s%n", bs, m, continual(src, m, 10, 4, bs, false, 1));
            }
        }
        for (StreamMethod m : StreamMethod.values()) {
            System.out.printf("stream single-class %-8s %s%n", m, continual(src, m, 10, 4, 32, true, 1));
        }
    }
}
