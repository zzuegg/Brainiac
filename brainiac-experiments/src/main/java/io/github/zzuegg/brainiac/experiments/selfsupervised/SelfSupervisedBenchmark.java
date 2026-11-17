package io.github.zzuegg.brainiac.experiments.selfsupervised;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ShapeImages;
import io.github.zzuegg.brainiac.experimental.nn.PatchEmbedding;
import io.github.zzuegg.brainiac.experimental.nn.PatchMasking;
import io.github.zzuegg.brainiac.experimental.nn.SelfSupervisedLosses;
import io.github.zzuegg.brainiac.experimental.nn.TunableTransformerBlock;
import io.github.zzuegg.brainiac.experimental.train.MomentumTeacher;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.LayerNorm;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.UnaryOperator;

/**
 * Trains an encoder on unlabelled {@link ShapeImages} with one self-supervised method and
 * judges the frozen representation with a linear probe (logistic regression on the
 * features), against the same encoder left at its random initialization.
 *
 * <p>Joint-embedding methods (SimCLR, BYOL, DINO, Barlow Twins, VICReg) train an MLP encoder
 * on two augmented views (random shifts, brightness, noise): they must learn
 * shift-invariant shape features. The masked-modelling methods train a one-block Vision
 * Transformer on 2x2 patches: MAE reconstructs the pixels of 75% hidden patches from the
 * rest; I-JEPA-lite predicts, from the visible patches, the <em>representations</em> that a
 * momentum target encoder gives the hidden ones (no pixels, no augmentations).
 */
public final class SelfSupervisedBenchmark {

    /** The self-supervised methods. */
    public enum Method {
        /** Contrastive NT-Xent between two views. */
        SIMCLR,
        /** Online predictor regressing a momentum target. */
        BYOL,
        /** Self-distillation with centring and sharpening. */
        DINO,
        /** Cross-correlation to identity. */
        BARLOW_TWINS,
        /** Variance, invariance and covariance terms. */
        VICREG,
        /** Masked autoencoder on patches. */
        MAE,
        /** Joint-embedding predictive architecture on patches. */
        IJEPA
    }

    /**
     * The outcome of one run.
     *
     * @param method the method
     * @param firstLoss the self-supervised loss of the first step
     * @param lastLoss the mean loss over the last 10 steps
     * @param probeAccuracy test accuracy of a linear probe on the trained encoder
     * @param randomProbeAccuracy the same for the untrained encoder
     */
    public record Result(Method method, float firstLoss, float lastLoss, float probeAccuracy,
            float randomProbeAccuracy) {
    }

    private static final int SIDE = 8;
    private static final int REP = 32;
    private static final ShapeImages IMAGES = new ShapeImages(SIDE, 0.1f);

    private SelfSupervisedBenchmark() {
    }

    /**
     * Trains and probes one method.
     *
     * @param method the method
     * @param steps training steps of 64 images each
     * @param seed the seed of weights and data
     * @return the result
     */
    public static Result run(Method method, int steps, long seed) {
        return switch (method) {
            case MAE, IJEPA -> runPatches(method, steps, seed);
            default -> runJoint(method, steps, seed);
        };
    }

    private static Model mlp(int in, int hidden, int out, boolean reluOut, long seed) {
        return Model.sequential().input(in).dense(hidden, Activation.RELU)
                .dense(out, reluOut ? Activation.RELU : Activation.IDENTITY).seed(seed).build();
    }

    private static Var flat(Tensor images) {
        return Var.of(images.reshape(images.shape().dim(0), SIDE * SIDE));
    }

    private static Result runJoint(Method method, int steps, long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        Model encoder = mlp(SIDE * SIDE, 128, REP, true, seed);
        float random = probe(x -> encoder.forward(flat(x)).value(), seed);
        int projOut = REP;
        Model projector = mlp(REP, 64, projOut, false, seed + 1);
        Model predictor = mlp(projOut, 64, projOut, false, seed + 2);
        Model targetEncoder = mlp(SIDE * SIDE, 128, REP, true, seed);
        Model targetProjector = mlp(REP, 64, projOut, false, seed + 1);
        List<Var> online = new ArrayList<>(encoder.parameters());
        online.addAll(projector.parameters());
        List<Var> target = new ArrayList<>(targetEncoder.parameters());
        target.addAll(targetProjector.parameters());
        MomentumTeacher teacher = new MomentumTeacher(target, online, method == Method.DINO ? 0.99f : 0.98f);
        List<Var> trained = new ArrayList<>(online);
        if (method == Method.BYOL) {
            trained.addAll(predictor.parameters());
        }
        // DINO's head: L2-normalized bottleneck, then weight-normalized prototypes
        Var prototypes = Var.param(Tensor.normal(Shape.of(projOut, 64), 0, 1, new SplittableRandom(seed + 3)));
        Var targetPrototypes = Var.param(prototypes.value());
        if (method == Method.DINO) {
            trained.add(prototypes);
        }
        MomentumTeacher prototypeTeacher = new MomentumTeacher(List.of(targetPrototypes), List.of(prototypes), 0.99f);
        UnaryOperator<Var> head = z -> dinoHead(z, prototypes);
        UnaryOperator<Var> targetHead = z -> dinoHead(z, targetPrototypes);
        SelfSupervisedLosses.Center center = new SelfSupervisedLosses.Center(64, 0.9f);
        Optimizer opt = Optimizer.adam(method == Method.VICREG ? 1e-3f : 3e-3f);
        float first = 0;
        double tail = 0;
        for (int s = 0; s < steps; s++) {
            Tensor batch = IMAGES.generate(64, rnd.nextLong())[0];
            Tensor a = IMAGES.augment(batch, 2, rnd);
            Tensor b = IMAGES.augment(batch, 2, rnd);
            Var za = projector.forward(encoder.forward(flat(a)));
            Var zb = projector.forward(encoder.forward(flat(b)));
            Var loss = switch (method) {
                case SIMCLR -> SelfSupervisedLosses.ntXent(za, zb, 0.2f);
                case BARLOW_TWINS -> SelfSupervisedLosses.barlowTwins(za, zb, 1f / REP);
                case VICREG -> SelfSupervisedLosses.vicreg(za, zb, 25, 25, 1);
                case BYOL -> {
                    Var ta = targetProjector.forward(targetEncoder.forward(flat(a)));
                    Var tb = targetProjector.forward(targetEncoder.forward(flat(b)));
                    yield SelfSupervisedLosses.byol(predictor.forward(za), tb)
                            .add(SelfSupervisedLosses.byol(predictor.forward(zb), ta));
                }
                case DINO -> {
                    Var ta = targetHead.apply(targetProjector.forward(targetEncoder.forward(flat(a))));
                    Var tb = targetHead.apply(targetProjector.forward(targetEncoder.forward(flat(b))));
                    Var sa = head.apply(za);
                    Var sb = head.apply(zb);
                    Var l = SelfSupervisedLosses.dino(sa, tb, center.value(), 0.1f, 0.04f)
                            .add(SelfSupervisedLosses.dino(sb, ta, center.value(), 0.1f, 0.04f));
                    center.update(Tensor.concat(0, ta.value(), tb.value()));
                    yield l;
                }
                default -> throw new IllegalArgumentException(method + " is not a joint-embedding method");
            };
            float v = loss.value().item();
            first = s == 0 ? v : first;
            tail += s >= steps - 10 ? v / 10 : 0;
            loss.backward();
            opt.step(trained);
            teacher.update();
            prototypeTeacher.update();
        }
        float acc = probe(x -> encoder.forward(flat(x)).value(), seed);
        return new Result(method, first, (float) tail, acc, random);
    }

    /** DINO's head: cosine similarities of the normalized bottleneck to normalized prototypes. */
    private static Var dinoHead(Var z, Var prototypes) {
        Var p = prototypes.div(prototypes.square().sum(0, true).add(1e-8f).sqrt());
        return SelfSupervisedLosses.normalize(z).matmul(p);
    }

    /** A one-block ViT encoder over 2x2 patches. */
    private static final class PatchEncoder {
        final PatchEmbedding embed;
        final TunableTransformerBlock block;
        final LayerNorm norm = LayerNorm.of(REP);

        PatchEncoder(long seed) {
            SplittableRandom r = new SplittableRandom(seed);
            embed = new PatchEmbedding(1, SIDE, SIDE, 2, REP, false, true, r);
            block = TunableTransformerBlock.of(4).ffnFactor(2).build(REP, r);
        }

        Var tokens(Tensor images, int[][] keep) {
            Var t = embed.forward(Var.of(images));
            if (keep != null) {
                t = PatchMasking.gather(t, keep);
            }
            return norm.forward(block.forward(t));
        }

        Tensor features(Tensor images) {
            return tokens(images, null).mean(1, false).value();
        }

        List<Var> parameters() {
            List<Var> p = new ArrayList<>(embed.parameters());
            p.addAll(block.parameters());
            p.addAll(norm.parameters());
            return p;
        }
    }

    private static Result runPatches(Method method, int steps, long seed) {
        SplittableRandom rnd = new SplittableRandom(seed);
        PatchEncoder encoder = new PatchEncoder(seed);
        float random = probe(encoder::features, seed);
        PatchEncoder target = new PatchEncoder(seed);
        MomentumTeacher teacher = new MomentumTeacher(target.parameters(), encoder.parameters(), 0.99f);
        SplittableRandom dr = new SplittableRandom(seed + 7);
        Var maskToken = Var.param(Tensor.zeros(Shape.of(1, 1, REP)));
        Var decoderPositions = Var.param(Tensor.normal(Shape.of(16, REP), 0, 0.02f, dr));
        TunableTransformerBlock decoder = TunableTransformerBlock.of(4).ffnFactor(2).build(REP, dr);
        Var out = Var.param(Tensor.normal(Shape.of(REP, method == Method.MAE ? 4 : REP), 0, 0.1f, dr));
        List<Var> trained = new ArrayList<>(encoder.parameters());
        trained.addAll(List.of(maskToken, decoderPositions, out));
        trained.addAll(decoder.parameters());
        Optimizer opt = Optimizer.adam(3e-3f);
        float first = 0;
        double tail = 0;
        for (int s = 0; s < steps; s++) {
            Tensor batch = IMAGES.generate(64, rnd.nextLong())[0];
            PatchMasking.Mask mask = PatchMasking.random(64, 16, method == Method.MAE ? 0.75f : 0.5f, rnd);
            Var context = encoder.tokens(batch, mask.keep());
            int hidden = mask.masked()[0].length;
            Var fill = maskToken.expand(Shape.of(64, hidden, REP));
            Var full = PatchMasking.unshuffle(context, fill, mask).add(decoderPositions);
            Var predicted = PatchMasking.gather(decoder.forward(full), mask.masked()).matmul(out);
            Var loss;
            if (method == Method.MAE) {
                Var pixels = PatchMasking.gather(PatchEmbedding.patchify(Var.of(batch), 2), mask.masked());
                loss = predicted.sub(pixels).square().mean();
            } else {
                Tensor targets = PatchMasking.gather(Var.of(target.tokens(batch, null).value()), mask.masked()).value();
                loss = predicted.sub(targets).square().mean();
            }
            float v = loss.value().item();
            first = s == 0 ? v : first;
            tail += s >= steps - 10 ? v / 10 : 0;
            loss.backward();
            opt.step(trained);
            if (method == Method.IJEPA) {
                teacher.update();
            }
        }
        float acc = probe(method == Method.IJEPA ? target::features : encoder::features, seed);
        return new Result(method, first, (float) tail, acc, random);
    }

    /**
     * Linear-probe accuracy: logistic regression on standardized frozen features of 384
     * labelled images, tested on 384 others.
     *
     * @param features the frozen encoder, images {@code [n, 1, 8, 8]} to features {@code [n, d]}
     * @param seed the data seed
     * @return the test accuracy
     */
    public static float probe(UnaryOperator<Tensor> features, long seed) {
        Tensor[] train = IMAGES.generate(384, seed + 1000);
        Tensor[] test = IMAGES.generate(384, seed + 2000);
        Tensor ftr = features.apply(train[0]);
        Tensor fte = features.apply(test[0]);
        Tensor mean = ftr.mean(0, true);
        Tensor std = ftr.sub(mean).square().mean(0, true).sqrt().add(1e-4f);
        ftr = ftr.sub(mean).div(std);
        fte = fte.sub(mean).div(std);
        int d = ftr.shape().dim(1);
        Var w = Var.param(Tensor.zeros(Shape.of(d, ShapeImages.CLASSES)));
        Var b = Var.param(Tensor.zeros(Shape.of(ShapeImages.CLASSES)));
        Optimizer opt = Optimizer.adam(0.05f);
        for (int i = 0; i < 200; i++) {
            Functional.crossEntropy(Var.of(ftr).matmul(w).add(b), train[1]).add(w.square().sum().mul(1e-3f))
                    .backward();
            opt.step(List.of(w, b));
        }
        Tensor pred = fte.matmul(w.value()).add(b.value()).argmax(1);
        return pred.equalTo(test[1]).mean().item();
    }
}
