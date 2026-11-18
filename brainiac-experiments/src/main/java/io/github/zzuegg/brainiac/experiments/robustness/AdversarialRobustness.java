package io.github.zzuegg.brainiac.experiments.robustness;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ClientSplits;
import io.github.zzuegg.brainiac.experimental.data.ShapeImages;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experimental.nn.SpectralNormDense;
import io.github.zzuegg.brainiac.experimental.robust.AdversarialAttacks;
import io.github.zzuegg.brainiac.experimental.robust.AdversarialTraining;
import io.github.zzuegg.brainiac.experimental.robust.BoundPropagation;
import io.github.zzuegg.brainiac.experimental.robust.CertifiedTraining;
import io.github.zzuegg.brainiac.experimental.robust.LipschitzLayers;
import io.github.zzuegg.brainiac.experimental.robust.RandomizedSmoothing;
import io.github.zzuegg.brainiac.experiments.learning.ToyTasks;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.GlobalPool;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.List;
import java.util.SplittableRandom;
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
}
