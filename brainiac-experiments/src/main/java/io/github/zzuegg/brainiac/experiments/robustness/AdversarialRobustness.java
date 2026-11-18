package io.github.zzuegg.brainiac.experiments.robustness;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.data.Augmentation;
import io.github.zzuegg.brainiac.experimental.data.BatchMixing;
import io.github.zzuegg.brainiac.experimental.data.ClientSplits;
import io.github.zzuegg.brainiac.experimental.data.ShapeImages;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experimental.nn.SpectralNormDense;
import io.github.zzuegg.brainiac.experimental.robust.AdversarialAttacks;
import io.github.zzuegg.brainiac.experimental.robust.AdversarialTraining;
import io.github.zzuegg.brainiac.experimental.robust.AugMix;
import io.github.zzuegg.brainiac.experimental.robust.BoundPropagation;
import io.github.zzuegg.brainiac.experimental.robust.CertifiedTraining;
import io.github.zzuegg.brainiac.experimental.robust.LipschitzLayers;
import io.github.zzuegg.brainiac.experimental.robust.RandomizedSmoothing;
import io.github.zzuegg.brainiac.experimental.robust.TrivialAugment;
import io.github.zzuegg.brainiac.experimental.security.AutoAttack;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.GlobalPool;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.DoubleFunction;
import java.util.function.UnaryOperator;

/**
 * Wave 8 items 8.6.8 to 8.6.11 on {@link ToyTasks#digits} (8x8, 10 classes, pixel values in
 * about {@code [0, 1]}, an MLP 64-64-10 with ReLU) and {@link ShapeImages} (a two-convolution
 * CNN):
 *
 * <ul>
 *   <li>{@link #adversarial}: standard training, random-noise training, fast FGSM, PGD-AT and
 *       TRADES, each evaluated clean and under PGD-20 at several {@code ℓ∞} budgets, with the
 *       training time.</li>
 *   <li>{@link #smoothing}: {@link RandomizedSmoothing} of a model trained with or without
 *       Gaussian noise: certified {@code ℓ2} accuracy at several radii and the abstention rate,
 *       next to PGD-AT's empirical {@code ℓ2} robust accuracy (an upper bound on what any
 *       certificate can show).</li>
 *   <li>{@link #certified}: {@link BoundPropagation} verified {@code ℓ∞} accuracy by IBP and
 *       CROWN of models trained normally, by PGD-AT, and by {@link CertifiedTraining} (IBP and
 *       CROWN-IBP).</li>
 *   <li>{@link #lipschitz}: 1-Lipschitz networks ({@link LipschitzLayers}: GroupSort with
 *       Björck or AOL weights) against {@link SpectralNormDense} with ReLU: clean and
 *       certified {@code ℓ2} accuracy at several radii (margin over {@code √2}).</li>
 *   <li>{@link #autoAttack} (wave 11, 11.1.10): the {@link #adversarial} defences re-evaluated
 *       with PGD-10, PGD-50 with restarts and {@link AutoAttack}.</li>
 *   <li>{@link #corruptions} (wave 11, 11.1.17): standard augmentation, RandAugment,
 *       {@link TrivialAugment}, {@link AugMix} with the Jensen-Shannon loss, mixup and PGD-AT
 *       under five held-out corruptions at five severities (mean corruption error).</li>
 * </ul>
 */
public final class AdversarialRobustness {

    /** The data set and architecture. */
    public enum Task {
        /** digits with an MLP */
        DIGITS_MLP,
        /** shapes with a CNN */
        SHAPES_CNN
    }

    /**
     * One adversarially trained model.
     *
     * @param method the training
     * @param clean clean test accuracy
     * @param robust PGD-20 accuracy at each evaluation budget
     * @param millis training time
     */
    public record Adversarial(AdversarialTraining.Method method, double clean, double[] robust, double millis) {
    }

    /**
     * Smoothing results.
     *
     * @param model the base model's training
     * @param certified certified accuracy at each radius
     * @param abstain the abstention rate
     * @param samples noisy samples per certificate
     */
    public record Smoothed(String model, double[] certified, double abstain, int samples) {
    }

    /**
     * Verified accuracy.
     *
     * @param model the training
     * @param clean clean accuracy
     * @param pgd PGD-20 accuracy at the radius
     * @param ibp IBP-verified accuracy
     * @param crown CROWN-verified accuracy
     * @param millis training time
     */
    public record Verified(String model, double clean, double pgd, double ibp, double crown, double millis) {
    }

    /**
     * A Lipschitz network.
     *
     * @param model the architecture
     * @param clean clean accuracy
     * @param certified certified {@code ℓ2} accuracy at each radius
     * @param pgd empirical PGD {@code ℓ2} accuracy at each radius
     * @param millis training time
     */
    public record Lipschitz(String model, double clean, double[] certified, double[] pgd, double millis) {
    }

    private static final int BATCH = 64;

    private AdversarialRobustness() {
    }

    /** Train and test data. */
    static ClientSplits.Labelled[] data(Task task, long seed) {
        if (task == Task.DIGITS_MLP) {
            ToyTasks.Data tr = ToyTasks.digits(10, 100, 8, 0.05, seed, seed + 1);
            ToyTasks.Data te = ToyTasks.digits(10, 30, 8, 0.05, seed, seed + 2);
            return new ClientSplits.Labelled[] {new ClientSplits.Labelled(tr.x(), tr.y(), 10),
                    new ClientSplits.Labelled(te.x(), te.y(), 10)};
        }
        ShapeImages images = new ShapeImages(8, 0.1f);
        return new ClientSplits.Labelled[] {ClientSplits.fromImages(images.generate(1200, seed + 1), 6),
                ClientSplits.fromImages(images.generate(300, seed + 2), 6)};
    }

    /** A model: logits and parameters. */
    record Net(UnaryOperator<Var> logits, List<Var> parameters, SimpleMlp mlp) {
    }

    static Net net(Task task, int classes, long seed) {
        if (task == Task.DIGITS_MLP) {
            SimpleMlp m = SimpleMlp.relu(seed, 64, 64, classes);
            return new Net(m::forward, m.parameters(), m);
        }
        Model cnn = Model.sequential().input(1, 8, 8).seed(seed).conv2d(8, 3, Activation.RELU)
                .conv2d(16, 3, Activation.RELU).layer(GlobalPool.avg()).dense(classes, Activation.IDENTITY).build();
        return new Net(x -> cnn.forward(x.reshape(x.shape().dim(0), 1, 8, 8), true), cnn.parameters(), null);
    }

    static Tensor[] batch(ClientSplits.Labelled train, int size, SplittableRandom r) {
        float[][] x = new float[size][];
        float[] y = new float[size];
        for (int i = 0; i < size; i++) {
            int j = r.nextInt(train.size());
            x[i] = train.x()[j];
            y[i] = train.y()[j];
        }
        return new Tensor[] {Tensor.of(x), Tensor.of(Shape.of(size), y)};
    }

    static double pgdAccuracy(Net net, ClientSplits.Labelled test, AdversarialAttacks.Norm norm, float eps,
            long seed) {
        if (eps == 0) {
            return AdversarialAttacks.accuracy(net.logits(), test.inputs(), test.ids());
        }
        Tensor adv = AdversarialAttacks.pgd(net.logits(), test.inputs(), test.ids(), norm, eps, 2.5f * eps / 20, 20, 1,
                new SplittableRandom(seed));
        return AdversarialAttacks.accuracy(net.logits(), adv, test.ids());
    }

    /**
     * Adversarial training against its baselines.
     *
     * @param task data and architecture
     * @param method the training
     * @param epsilon the training budget ({@code ℓ∞})
     * @param evalEpsilons the PGD-20 evaluation budgets
     * @param steps training steps
     * @param seed the seed
     * @return the row
     */
    public static Adversarial adversarial(Task task, AdversarialTraining.Method method, float epsilon,
            float[] evalEpsilons, int steps, long seed) {
        ClientSplits.Labelled[] d = data(task, seed);
        Net net = trainAdversarial(task, d, method, epsilon, steps, seed);
        double[] robust = new double[evalEpsilons.length];
        for (int i = 0; i < robust.length; i++) {
            robust[i] = pgdAccuracy(net, d[1], AdversarialAttacks.Norm.LINF, evalEpsilons[i], seed + 3);
        }
        return new Adversarial(method, AdversarialAttacks.accuracy(net.logits(), d[1].inputs(), d[1].ids()), robust,
                lastMillis);
    }

    private static double lastMillis;

    static Net trainAdversarial(Task task, ClientSplits.Labelled[] d, AdversarialTraining.Method method,
            float epsilon, int steps, long seed) {
        return trainAdversarial(task, d, method, AdversarialAttacks.Norm.LINF, epsilon, steps, seed);
    }

    static Net trainAdversarial(Task task, ClientSplits.Labelled[] d, AdversarialTraining.Method method,
            AdversarialAttacks.Norm norm, float epsilon, int steps, long seed) {
        Net net = net(task, d[0].classes(), seed);
        AdversarialTraining at = AdversarialTraining.of(net.logits(), net.parameters(), Optimizer.adam(3e-3f))
                .method(method).norm(norm).epsilon(epsilon).steps(5).beta(3).seed(seed);
        SplittableRandom r = new SplittableRandom(seed + 1);
        long start = System.nanoTime();
        for (int s = 0; s < steps; s++) {
            Tensor[] b = batch(d[0], BATCH, r);
            at.step(b[0], b[1]);
        }
        lastMillis = (System.nanoTime() - start) / 1e6;
        return net;
    }

    /**
     * Randomized smoothing on digits.
     *
     * @param sigma the smoothing noise
     * @param noiseTrained train the base model with Gaussian noise of the same {@code σ}
     * @param radii the {@code ℓ2} radii reported
     * @param examples test inputs certified
     * @param samples noisy samples per certificate ({@code n}; {@code n0 = 32})
     * @param seed the seed
     * @return the row
     */
    public static Smoothed smoothing(float sigma, boolean noiseTrained, double[] radii, int examples, int samples,
            long seed) {
        ClientSplits.Labelled[] d = data(Task.DIGITS_MLP, seed);
        SimpleMlp m = SimpleMlp.relu(seed, 64, 64, 10);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 1);
        for (int s = 0; s < 400; s++) {
            Tensor[] b = batch(d[0], BATCH, r);
            Tensor x = noiseTrained ? b[0].add(Tensor.normal(b[0].shape(), 0, sigma, r)) : b[0];
            Functional.crossEntropy(m.forward(Var.of(x)), b[1]).backward();
            opt.step(m.parameters());
        }
        RandomizedSmoothing smooth = RandomizedSmoothing.of(batch -> m.predict(batch).argmax(1).toIntArray(), sigma,
                samples);
        double[] certified = new double[radii.length];
        int abstain = 0;
        SplittableRandom nr = new SplittableRandom(seed + 2);
        for (int i = 0; i < examples; i++) {
            RandomizedSmoothing.Certificate c = smooth.certify(Tensor.of(Shape.of(64), d[1].x()[i]), 10, 32, samples,
                    0.001, nr);
            if (c.label() < 0) {
                abstain++;
                continue;
            }
            for (int k = 0; k < radii.length; k++) {
                if (c.label() == d[1].y()[i] && c.radius() >= radii[k]) {
                    certified[k] += 1.0 / examples;
                }
            }
        }
        return new Smoothed((noiseTrained ? "noise-trained" : "standard") + " sigma " + sigma, certified,
                (double) abstain / examples, samples + 32);
    }

    /**
     * Empirical {@code ℓ2} robustness of an {@code ℓ2} PGD-trained model, the reference for
     * smoothing's certificates.
     *
     * @param trainEpsilon the {@code ℓ2} training radius
     * @param radii the evaluation radii
     * @param seed the seed
     * @return PGD-20 accuracy at each radius
     */
    public static double[] pgdL2Reference(float trainEpsilon, double[] radii, long seed) {
        ClientSplits.Labelled[] d = data(Task.DIGITS_MLP, seed);
        Net net = trainAdversarial(Task.DIGITS_MLP, d, AdversarialTraining.Method.PGD_AT, AdversarialAttacks.Norm.L2,
                trainEpsilon, 400, seed);
        double[] out = new double[radii.length];
        for (int k = 0; k < radii.length; k++) {
            out[k] = pgdAccuracy(net, d[1], AdversarialAttacks.Norm.L2, (float) radii[k], seed + 3);
        }
        return out;
    }

    /** How a model for verification is trained. */
    public enum Certified {
        /** clean training */
        STANDARD,
        /** PGD adversarial training */
        PGD_AT,
        /** IBP certified training */
        IBP,
        /** CROWN-IBP certified training */
        CROWN_IBP
    }

    /**
     * Verified {@code ℓ∞} accuracy by IBP and CROWN.
     *
     * @param training the training
     * @param epsilon the radius (training and verification)
     * @param steps training steps
     * @param seed the seed
     * @return the row
     */
    public static Verified certified(Certified training, float epsilon, int steps, long seed) {
        ClientSplits.Labelled[] d = data(Task.DIGITS_MLP, seed);
        Net net;
        long start = System.nanoTime();
        switch (training) {
            case STANDARD -> net = trainAdversarial(Task.DIGITS_MLP, d, AdversarialTraining.Method.STANDARD, epsilon,
                    steps, seed);
            case PGD_AT -> net = trainAdversarial(Task.DIGITS_MLP, d, AdversarialTraining.Method.PGD_AT, epsilon,
                    steps, seed);
            default -> {
                net = net(Task.DIGITS_MLP, 10, seed);
                CertifiedTraining ct = CertifiedTraining.of(BoundPropagation.of(net.mlp()), net.parameters(),
                        Optimizer.adam(3e-3f), training == Certified.IBP ? CertifiedTraining.Method.IBP
                                : CertifiedTraining.Method.CROWN_IBP)
                        .schedule(epsilon, steps / 10, steps / 2);
                SplittableRandom r = new SplittableRandom(seed + 1);
                for (int s = 0; s < steps; s++) {
                    int[] rows = new int[BATCH];
                    for (int i = 0; i < BATCH; i++) {
                        rows[i] = r.nextInt(d[0].size());
                    }
                    ClientSplits.Labelled b = d[0].subset(rows);
                    ct.step(b.inputs(), b.y());
                }
            }
        }
        double millis = (System.nanoTime() - start) / 1e6;
        List<BoundPropagation.Affine> layers = BoundPropagation.of(net.mlp());
        Var x = Var.of(d[1].inputs());
        double ibp = BoundPropagation.verified(BoundPropagation.ibpMargins(layers, x, epsilon, d[1].y()).value(),
                d[1].y());
        double crown = BoundPropagation.verified(BoundPropagation.crown(layers, x, epsilon, d[1].y()).value(),
                d[1].y());
        return new Verified(training.toString(), AdversarialAttacks.accuracy(net.logits(), d[1].inputs(),
                d[1].ids()), pgdAccuracy(net, d[1], AdversarialAttacks.Norm.LINF, epsilon, seed + 3), ibp, crown,
                millis);
    }

    /** A 1-Lipschitz architecture. */
    public enum LipschitzKind {
        /** spectral-norm dense layers with ReLU */
        SPECTRAL_RELU,
        /** Björck-orthonormal dense layers with GroupSort */
        BJORCK_GROUPSORT,
        /** AOL dense layers with GroupSort */
        AOL_GROUPSORT
    }

    /**
     * A 1-Lipschitz network on digits, trained with cross-entropy on scaled logits, certified
     * by its margin.
     *
     * @param kind the architecture
     * @param radii the {@code ℓ2} radii
     * @param steps training steps
     * @param seed the seed
     * @return the row
     */
    public static Lipschitz lipschitz(LipschitzKind kind, double[] radii, int steps, long seed) {
        ClientSplits.Labelled[] d = data(Task.DIGITS_MLP, seed);
        Model.Builder b = Model.sequential().input(64).seed(seed);
        switch (kind) {
            case SPECTRAL_RELU -> b.add(SpectralNormDense.spec(64, Activation.RELU))
                    .add(SpectralNormDense.spec(64, Activation.RELU)).add(SpectralNormDense.spec(10,
                            Activation.IDENTITY));
            case BJORCK_GROUPSORT -> b.add(LipschitzLayers.bjorckDense(64, 6)).add(LipschitzLayers.groupSort())
                    .add(LipschitzLayers.bjorckDense(64, 6)).add(LipschitzLayers.groupSort())
                    .add(LipschitzLayers.bjorckDense(10, 6));
            case AOL_GROUPSORT -> b.add(LipschitzLayers.aolDense(64)).add(LipschitzLayers.groupSort())
                    .add(LipschitzLayers.aolDense(64)).add(LipschitzLayers.groupSort())
                    .add(LipschitzLayers.aolDense(10));
        }
        Model m = b.build();
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 1);
        long start = System.nanoTime();
        for (int s = 0; s < steps; s++) {
            Tensor[] bt = batch(d[0], BATCH, r);
            Functional.crossEntropy(m.forward(Var.of(bt[0]), true).mul(8f), bt[1]).backward();
            opt.step(m.parameters());
        }
        double millis = (System.nanoTime() - start) / 1e6;
        Tensor logits = m.forward(Var.of(d[1].inputs())).value();
        double[] radius = LipschitzLayers.certifiedRadius(logits, d[1].y());
        double[] certified = new double[radii.length];
        double[] pgd = new double[radii.length];
        Net net = new Net(x -> m.forward(x), m.parameters(), null);
        for (int k = 0; k < radii.length; k++) {
            for (double rad : radius) {
                certified[k] += rad >= radii[k] ? 1.0 / radius.length : 0;
            }
            pgd[k] = pgdAccuracy(net, d[1], AdversarialAttacks.Norm.L2, (float) radii[k], seed + 3);
        }
        double clean = 0;
        for (double rad : radius) {
            clean += rad > 0 ? 1.0 / radius.length : 0;
        }
        return new Lipschitz(kind.toString(), clean, certified, pgd, millis);
    }

    /**
     * Wave 11 item 11.1.10: a wave 8 defence re-trained at the test budget and evaluated with
     * fixed-step PGD and with {@link AutoAttack}.
     *
     * @param method the training
     * @param clean clean accuracy
     * @param pgd10 PGD-10 robust accuracy
     * @param pgd50 PGD-50 with three restarts
     * @param autoAttack AutoAttack (APGD-CE, targeted APGD-DLR, Square) robust accuracy
     */
    public record Reevaluated(AdversarialTraining.Method method, double clean, double pgd10, double pgd50,
            double autoAttack) {
    }

    /**
     * Re-evaluates one {@link #adversarial} defence on digits with AutoAttack.
     *
     * @param method the training
     * @param epsilon the training and test {@code ℓ∞} budget
     * @param steps training steps
     * @param examples test images evaluated
     * @param seed the seed
     * @return the row
     */
    public static Reevaluated autoAttack(AdversarialTraining.Method method, float epsilon, int steps, int examples,
            long seed) {
        ClientSplits.Labelled[] d = data(Task.DIGITS_MLP, seed);
        Net net = trainAdversarial(Task.DIGITS_MLP, d, method, epsilon, steps, seed);
        float[][] x = Arrays.copyOf(d[1].x(), examples);
        int[] y = Arrays.copyOf(d[1].y(), examples);
        ClientSplits.Labelled test = new ClientSplits.Labelled(x, y, 10);
        Tensor xs = test.inputs();
        Tensor ids = test.ids();
        Tensor p10 = AdversarialAttacks.pgd(net.logits(), xs, ids, AdversarialAttacks.Norm.LINF, epsilon,
                2.5f * epsilon / 10, 10, 1, new SplittableRandom(seed + 3));
        Tensor p50 = AdversarialAttacks.pgd(net.logits(), xs, ids, AdversarialAttacks.Norm.LINF, epsilon,
                2.5f * epsilon / 50, 50, 3, new SplittableRandom(seed + 4));
        AutoAttack.Report rep = AutoAttack.evaluate(net.logits(), x, y, 8, epsilon, 50, 500, seed + 5);
        return new Reevaluated(method, rep.clean(), AdversarialAttacks.accuracy(net.logits(), p10, ids),
                AdversarialAttacks.accuracy(net.logits(), p50, ids), rep.afterSquare());
    }

    /** A test-time corruption of 8x8 images in {@code [0, 1]}, held out of every augmentation. */
    public enum Corruption {
        /** additive Gaussian noise */
        NOISE,
        /** a 3x3 box blur blended in, applied twice at the top severity */
        BLUR,
        /** contrast reduced toward the image mean */
        CONTRAST,
        /** a smooth random displacement field */
        ELASTIC,
        /** 2x2 pixelation blended in, then quantisation to few levels (JPEG-like) */
        QUANTISE
    }

    /** A training scheme for {@link #corruptions}. */
    public enum Augmented {
        /** random translation by up to one pixel */
        STANDARD,
        /** core {@code Augmentation.randAugment(2, 5)} */
        RANDAUGMENT,
        /** {@link TrivialAugment} over translation, resized crop and erasing */
        TRIVIALAUGMENT,
        /** {@link AugMix} over the same operations, with the Jensen-Shannon loss (weight 12) */
        AUGMIX_JSD,
        /** mixup ({@code α = 0.2}) on translated images */
        MIXUP,
        /** PGD adversarial training ({@code ε = 0.1}) */
        PGD_AT
    }

    /**
     * Corruption robustness of one training scheme.
     *
     * @param method the training
     * @param clean clean accuracy
     * @param error test error per corruption and severity (1 to 5)
     * @param corruptionError per corruption, the summed error relative to {@code STANDARD}'s
     * @param mce the mean corruption error relative to {@code STANDARD}
     */
    public record Corrupted(Augmented method, double clean, double[][] error, double[] corruptionError,
            double mce) {
    }

    /**
     * Wave 11 item 11.1.17: training schemes evaluated under five corruptions at five
     * severities on digits.
     *
     * @param steps training steps per model
     * @param seed the seed
     * @return one row per {@link Augmented}, {@code STANDARD} first
     */
    public static List<Corrupted> corruptions(int steps, long seed) {
        ClientSplits.Labelled[] d = data(Task.DIGITS_MLP, seed);
        int kinds = Corruption.values().length;
        float[][][][] corrupted = new float[kinds][5][][];
        for (Corruption c : Corruption.values()) {
            for (int s = 0; s < 5; s++) {
                corrupted[c.ordinal()][s] = corrupt(d[1].x(), c, s + 1, seed + 10L * c.ordinal() + s);
            }
        }
        List<Corrupted> out = new ArrayList<>();
        double[] base = null;
        for (Augmented a : Augmented.values()) {
            Net net = trainAugmented(a, d[0], steps, seed);
            double clean = AdversarialAttacks.accuracy(net.logits(), d[1].inputs(), d[1].ids());
            double[][] err = new double[kinds][5];
            double[] sum = new double[kinds];
            for (int c = 0; c < kinds; c++) {
                for (int s = 0; s < 5; s++) {
                    err[c][s] = 1 - AdversarialAttacks.accuracy(net.logits(), Tensor.of(corrupted[c][s]),
                            d[1].ids());
                    sum[c] += err[c][s];
                }
            }
            if (base == null) {
                base = sum;
            }
            double[] ce = new double[kinds];
            double mce = 0;
            for (int c = 0; c < kinds; c++) {
                ce[c] = sum[c] / Math.max(base[c], 1e-9);
                mce += ce[c] / kinds;
            }
            out.add(new Corrupted(a, clean, err, ce, mce));
        }
        return out;
    }

    /** Digit-safe augmentation operations (no flips, no photometric operations). */
    static List<DoubleFunction<Augmentation>> safeOperations() {
        return List.of(
                s -> (f, o, r) -> {
                },
                s -> translate(Math.max(1, (int) Math.round(2 * s))),
                s -> Augmentation.randomResizedCrop(1, 8, 8, 1 - 0.4 * s, 1),
                s -> Augmentation.randomErasing(1, 8, 8, 1, 0.02, Math.max(0.03, 0.15 * s)));
    }

    /** A random translation by up to {@code max} pixels with zero fill. */
    static Augmentation translate(int max) {
        return (f, o, r) -> {
            int dy = r.nextInt(-max, max + 1);
            int dx = r.nextInt(-max, max + 1);
            float[] src = Arrays.copyOfRange(f, o, o + 64);
            for (int y = 0; y < 8; y++) {
                for (int x = 0; x < 8; x++) {
                    int sy = y - dy;
                    int sx = x - dx;
                    f[o + y * 8 + x] = sy >= 0 && sy < 8 && sx >= 0 && sx < 8 ? src[sy * 8 + sx] : 0;
                }
            }
        };
    }

    static Net trainAugmented(Augmented a, ClientSplits.Labelled train, int steps, long seed) {
        if (a == Augmented.PGD_AT) {
            return trainAdversarial(Task.DIGITS_MLP, new ClientSplits.Labelled[] {train, train},
                    AdversarialTraining.Method.PGD_AT, 0.1f, steps, seed);
        }
        Net net = net(Task.DIGITS_MLP, 10, seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 1);
        Augmentation aug = switch (a) {
            case RANDAUGMENT -> Augmentation.randAugment(1, 8, 8, 2, 5);
            case TRIVIALAUGMENT -> new TrivialAugment(safeOperations());
            default -> translate(1);
        };
        AugMix augMix = new AugMix(safeOperations(), 3, 0.5, 64);
        for (int s = 0; s < steps; s++) {
            Tensor[] b = batch(train, BATCH, r);
            float[][] rows = rows(b[0]);
            for (float[] row : rows) {
                aug.apply(row, 0, r);
            }
            Var loss;
            if (a == Augmented.AUGMIX_JSD) {
                Var clean = net.logits().apply(Var.of(Tensor.of(rows)));
                Var v1 = net.logits().apply(Var.of(Tensor.of(augMix.view(rows, r))));
                Var v2 = net.logits().apply(Var.of(Tensor.of(augMix.view(rows, r))));
                loss = Functional.crossEntropy(clean, b[1]).add(AugMix.jensenShannon(clean, v1, v2).mul(12));
            } else if (a == Augmented.MIXUP) {
                Tensor[] m = BatchMixing.mixup(Tensor.of(rows), oneHot(b[1], 10), 0.2f, r);
                Var logp = net.logits().apply(Var.of(m[0])).logSoftmax(1);
                loss = logp.mul(Var.of(m[1])).sum().mul(-1f / BATCH);
            } else {
                loss = Functional.crossEntropy(net.logits().apply(Var.of(Tensor.of(rows))), b[1]);
            }
            for (Var p : net.parameters()) {
                p.zeroGrad();
            }
            loss.backward();
            opt.step(net.parameters());
        }
        return net;
    }

    private static Tensor oneHot(Tensor ids, int classes) {
        float[] id = ids.toArray();
        float[] out = new float[id.length * classes];
        for (int i = 0; i < id.length; i++) {
            out[i * classes + (int) id[i]] = 1;
        }
        return Tensor.of(Shape.of(id.length, classes), out);
    }

    private static float[][] rows(Tensor x) {
        int n = x.shape().dim(0);
        int d = Math.toIntExact(x.size() / n);
        float[] a = x.toArray();
        float[][] out = new float[n][d];
        for (int i = 0; i < n; i++) {
            System.arraycopy(a, i * d, out[i], 0, d);
        }
        return out;
    }

    /**
     * Applies a corruption to 8x8 images.
     *
     * @param x the images, not changed
     * @param c the corruption
     * @param severity 1 to 5
     * @param seed the seed
     * @return corrupted copies
     */
    public static float[][] corrupt(float[][] x, Corruption c, int severity, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int s = severity - 1;
        float[][] out = new float[x.length][];
        for (int i = 0; i < x.length; i++) {
            float[] img = x[i].clone();
            switch (c) {
                case NOISE -> {
                    double sigma = new double[] {0.1, 0.18, 0.27, 0.38, 0.5}[s];
                    for (int j = 0; j < 64; j++) {
                        img[j] += (float) (sigma * gaussian(r));
                    }
                }
                case BLUR -> {
                    double a = new double[] {0.3, 0.5, 0.7, 0.9, 1.0}[s];
                    float[] b = boxBlur(img);
                    if (severity == 5) {
                        b = boxBlur(b);
                    }
                    for (int j = 0; j < 64; j++) {
                        img[j] = (float) ((1 - a) * img[j] + a * b[j]);
                    }
                }
                case CONTRAST -> {
                    double k = new double[] {0.6, 0.45, 0.32, 0.22, 0.14}[s];
                    double mean = 0;
                    for (float v : img) {
                        mean += v / 64.0;
                    }
                    for (int j = 0; j < 64; j++) {
                        img[j] = (float) (mean + k * (img[j] - mean));
                    }
                }
                case ELASTIC -> img = elastic(img, new double[] {0.6, 1.0, 1.4, 1.8, 2.2}[s], r);
                case QUANTISE -> {
                    double a = new double[] {0.3, 0.5, 0.7, 0.85, 1.0}[s];
                    int levels = new int[] {5, 4, 3, 3, 2}[s];
                    float[] p = pixelate(img);
                    for (int j = 0; j < 64; j++) {
                        double v = Math.clamp((1 - a) * img[j] + a * p[j], 0, 1);
                        img[j] = (float) (Math.round(v * (levels - 1)) / (double) (levels - 1));
                    }
                }
                default -> throw new IllegalArgumentException(c.toString());
            }
            out[i] = img;
        }
        return out;
    }

    private static float[] pixelate(float[] img) {
        float[] p = new float[64];
        for (int y = 0; y < 8; y += 2) {
            for (int x = 0; x < 8; x += 2) {
                float m = (img[y * 8 + x] + img[y * 8 + x + 1] + img[(y + 1) * 8 + x] + img[(y + 1) * 8 + x + 1]) / 4;
                p[y * 8 + x] = m;
                p[y * 8 + x + 1] = m;
                p[(y + 1) * 8 + x] = m;
                p[(y + 1) * 8 + x + 1] = m;
            }
        }
        return p;
    }

    private static double gaussian(SplittableRandom r) {
        return Math.sqrt(-2 * Math.log(1 - r.nextDouble())) * Math.cos(2 * Math.PI * r.nextDouble());
    }

    private static float[] boxBlur(float[] img) {
        float[] b = new float[64];
        for (int y = 0; y < 8; y++) {
            for (int x = 0; x < 8; x++) {
                double sum = 0;
                int count = 0;
                for (int dy = -1; dy <= 1; dy++) {
                    for (int dx = -1; dx <= 1; dx++) {
                        int yy = y + dy;
                        int xx = x + dx;
                        if (yy >= 0 && yy < 8 && xx >= 0 && xx < 8) {
                            sum += img[yy * 8 + xx];
                            count++;
                        }
                    }
                }
                b[y * 8 + x] = (float) (sum / count);
            }
        }
        return b;
    }

    /** A smooth displacement field (twice box-blurred noise scaled to {@code alpha} pixels), bilinear. */
    private static float[] elastic(float[] img, double alpha, SplittableRandom r) {
        float[] fx = new float[64];
        float[] fy = new float[64];
        for (int j = 0; j < 64; j++) {
            fx[j] = (float) r.nextDouble(-1, 1);
            fy[j] = (float) r.nextDouble(-1, 1);
        }
        fx = boxBlur(boxBlur(fx));
        fy = boxBlur(boxBlur(fy));
        double mx = 1e-9;
        for (int j = 0; j < 64; j++) {
            mx = Math.max(mx, Math.max(Math.abs(fx[j]), Math.abs(fy[j])));
        }
        float[] out = new float[64];
        for (int y = 0; y < 8; y++) {
            for (int x = 0; x < 8; x++) {
                double sx = x + alpha * fx[y * 8 + x] / mx;
                double sy = y + alpha * fy[y * 8 + x] / mx;
                int x0 = (int) Math.floor(sx);
                int y0 = (int) Math.floor(sy);
                double tx = sx - x0;
                double ty = sy - y0;
                out[y * 8 + x] = (float) ((1 - ty) * ((1 - tx) * pixel(img, y0, x0) + tx * pixel(img, y0, x0 + 1))
                        + ty * ((1 - tx) * pixel(img, y0 + 1, x0) + tx * pixel(img, y0 + 1, x0 + 1)));
            }
        }
        return out;
    }

    private static float pixel(float[] img, int y, int x) {
        return y >= 0 && y < 8 && x >= 0 && x < 8 ? img[y * 8 + x] : 0;
    }
}
