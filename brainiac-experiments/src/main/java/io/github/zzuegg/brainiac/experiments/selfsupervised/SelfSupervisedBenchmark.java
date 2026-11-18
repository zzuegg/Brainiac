package io.github.zzuegg.brainiac.experiments.selfsupervised;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ShapeImages;
import io.github.zzuegg.brainiac.experimental.nn.PatchEmbedding;
import io.github.zzuegg.brainiac.experimental.nn.PatchMasking;
import io.github.zzuegg.brainiac.experimental.nn.SelfSupervisedLosses;
import io.github.zzuegg.brainiac.experimental.nn.TunableTransformerBlock;
import io.github.zzuegg.brainiac.experimental.ml.MlMetrics;
import io.github.zzuegg.brainiac.experimental.ssl.KoLeo;
import io.github.zzuegg.brainiac.experimental.ssl.LeJepa;
import io.github.zzuegg.brainiac.experimental.ssl.MaskedFeaturePrediction;
import io.github.zzuegg.brainiac.experimental.ssl.MomentumContrast;
import io.github.zzuegg.brainiac.experimental.ssl.SimSiam;
import io.github.zzuegg.brainiac.experimental.ssl.SwappedPrediction;
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
import java.util.function.LongFunction;
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
 *
 * <p>The I-JEPA follow-up (wave 7, item 7.6.18) adds {@link Method#IJEPA_FULL} with the
 * paper's recipe pieces that the lite version skips: multi-block target masks with a large
 * context block ({@link SelfSupervisedLosses#multiBlockMask}), a cosine EMA momentum
 * schedule from 0.996 to 1 ({@link SelfSupervisedLosses#cosineMomentum}) and a predictor
 * narrower than the encoder (width 16 against 32). {@link #learningCurve} probes a run at
 * several step counts and reports the representation's effective rank, so methods are
 * compared at matched budgets rather than at one point.
 *
 * <p>Wave 10 (items 10.3.1-10.3.6) adds the {@link Extra} methods without changing the
 * {@link Method} rows: SimSiam with its two collapse ablations, SwAV with and without
 * multi-crop, MoCo v2 and v3, data2vec and iBOT on the patch transformer, and LeJEPA (SIGReg,
 * no teacher, no predictor, no stop-gradient) on both encoders, plus a VICReg-regularised
 * LeJEPA ablation. {@link #report} reruns the existing joint methods with a KoLeo term or
 * another batch size, {@link #evaluate} adds k-NN accuracy, effective rank and uniformity to
 * the linear probe, and {@link #pretrainEncoder} exposes a frozen pretrained encoder to other
 * experiments.
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
        IJEPA,
        /** I-JEPA with multi-block masks, a cosine EMA schedule and a narrow predictor. */
        IJEPA_FULL
    }

    /**
     * One point of a learning curve.
     *
     * @param steps training steps so far
     * @param probeAccuracy the linear-probe accuracy at that point
     * @param effectiveRank the RankMe effective rank of the probed features
     */
    public record CurvePoint(int steps, float probeAccuracy, double effectiveRank) {
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
            case MAE, IJEPA -> runPatches(method, steps, seed, null, null);
            case IJEPA_FULL -> runJepaFull(steps, seed, null, null);
            default -> runJoint(method, steps, seed);
        };
    }

    /**
     * Trains a masked-modelling method once and probes it at several step counts.
     *
     * @param method {@link Method#MAE}, {@link Method#IJEPA} or {@link Method#IJEPA_FULL}
     * @param checkpoints ascending step counts; the run lasts until the last
     * @param seed the seed of weights and data
     * @return one point per checkpoint
     */
    public static List<CurvePoint> learningCurve(Method method, int[] checkpoints, long seed) {
        List<CurvePoint> curve = new ArrayList<>();
        int steps = checkpoints[checkpoints.length - 1];
        switch (method) {
            case MAE, IJEPA -> runPatches(method, steps, seed, checkpoints, curve);
            case IJEPA_FULL -> runJepaFull(steps, seed, checkpoints, curve);
            default -> throw new IllegalArgumentException("curves are for the masked-modelling methods");
        }
        return curve;
    }

    private static void checkpoint(int done, int[] checkpoints, List<CurvePoint> curve, UnaryOperator<Tensor> f,
            long seed) {
        if (checkpoints == null) {
            return;
        }
        boolean hit = false;
        for (int c : checkpoints) {
            hit |= c == done;
        }
        if (hit) {
            curve.add(new CurvePoint(done, probe(f, seed),
                    SelfSupervisedLosses.effectiveRank(f.apply(IMAGES.generate(256, seed + 3000)[0]))));
        }
    }

    private static Model mlp(int in, int hidden, int out, boolean reluOut, long seed) {
        return Model.sequential().input(in).dense(hidden, Activation.RELU)
                .dense(out, reluOut ? Activation.RELU : Activation.IDENTITY).seed(seed).build();
    }

    private static Var flat(Tensor images) {
        return Var.of(images.reshape(images.shape().dim(0), SIDE * SIDE));
    }

    private static Result runJoint(Method method, int steps, long seed) {
        Trained t = trainJoint(method, steps, seed, 0, s -> IMAGES.generate(64, s)[0], true);
        return new Result(method, t.first(), t.tail(), probe(t.features(), seed), t.random());
    }

    /** A trained joint-embedding encoder and its loss trace. */
    private record Trained(UnaryOperator<Tensor> features, float first, float tail, float random, double millis) {
    }

    private static Trained trainJoint(Method method, int steps, long seed, float koleo, LongFunction<Tensor> batches,
            boolean probeRandom) {
        SplittableRandom rnd = new SplittableRandom(seed);
        Model encoder = mlp(SIDE * SIDE, 128, REP, true, seed);
        UnaryOperator<Tensor> features = x -> Var.noGrad(() -> encoder.forward(flat(x))).value();
        float random = probeRandom ? probe(x -> encoder.forward(flat(x)).value(), seed) : 0;
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
        long start = System.nanoTime();
        for (int s = 0; s < steps; s++) {
            Tensor batch = batches.apply(rnd.nextLong());
            Tensor a = IMAGES.augment(batch, 2, rnd);
            Tensor b = IMAGES.augment(batch, 2, rnd);
            Var ha = encoder.forward(flat(a));
            Var za = projector.forward(ha);
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
            if (koleo > 0) {
                loss = loss.add(KoLeo.loss(ha).mul(koleo));
            }
            loss.backward();
            opt.step(trained);
            teacher.update();
            prototypeTeacher.update();
        }
        double millis = (System.nanoTime() - start) / 1e6 / Math.max(1, steps);
        return new Trained(features, first, (float) tail, random, millis);
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

    private static Result runPatches(Method method, int steps, long seed, int[] checkpoints,
            List<CurvePoint> curve) {
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
            checkpoint(s + 1, checkpoints, curve, method == Method.IJEPA ? target::features : encoder::features, seed);
        }
        float acc = probe(method == Method.IJEPA ? target::features : encoder::features, seed);
        return new Result(method, first, (float) tail, acc, random);
    }

    /** I-JEPA with the paper's multi-block masking, cosine EMA schedule and narrow predictor. */
    private static Result runJepaFull(int steps, long seed, int[] checkpoints, List<CurvePoint> curve) {
        SplittableRandom rnd = new SplittableRandom(seed);
        PatchEncoder encoder = new PatchEncoder(seed);
        float random = probe(encoder::features, seed);
        PatchEncoder target = new PatchEncoder(seed);
        MomentumTeacher teacher = new MomentumTeacher(target.parameters(), encoder.parameters(), 0.996f);
        SplittableRandom dr = new SplittableRandom(seed + 7);
        int width = REP / 2;
        Var maskToken = Var.param(Tensor.zeros(Shape.of(1, 1, width)));
        Var positions = Var.param(Tensor.normal(Shape.of(16, width), 0, 0.02f, dr));
        Var in = Var.param(Tensor.normal(Shape.of(REP, width), 0, (float) Math.sqrt(1.0 / REP), dr));
        TunableTransformerBlock predictor = TunableTransformerBlock.of(4).ffnFactor(2).build(width, dr);
        Var out = Var.param(Tensor.normal(Shape.of(width, REP), 0, 0.1f, dr));
        List<Var> trained = new ArrayList<>(encoder.parameters());
        trained.addAll(List.of(maskToken, positions, in, out));
        trained.addAll(predictor.parameters());
        Optimizer opt = Optimizer.adam(3e-3f);
        float first = 0;
        double tail = 0;
        for (int s = 0; s < steps; s++) {
            Tensor batch = IMAGES.generate(64, rnd.nextLong())[0];
            PatchMasking.Mask mask = SelfSupervisedLosses.multiBlockMask(64, 4, 4, 4, 0.15, 0.2,
                    0.85 + 0.15 * rnd.nextDouble(), rnd);
            int nt = mask.masked()[0].length;
            Var context = encoder.tokens(batch, mask.keep()).matmul(in)
                    .add(PatchMasking.gatherRows(positions, mask.keep()));
            Var queries = maskToken.expand(Shape.of(64, nt, width)).add(PatchMasking.gatherRows(positions, mask.masked()));
            Var joint = predictor.forward(Var.concat(1, context, queries));
            int nc = mask.keep()[0].length;
            Var predicted = joint.slice(1, nc, nc + nt).matmul(out);
            Tensor targets = PatchMasking.gather(Var.of(target.tokens(batch, null).value()), mask.masked()).value();
            Var loss = predicted.sub(targets).square().mean();
            float v = loss.value().item();
            first = s == 0 ? v : first;
            tail += s >= steps - 10 ? v / 10 : 0;
            loss.backward();
            opt.step(trained);
            teacher.momentum(SelfSupervisedLosses.cosineMomentum(s, steps, 0.996f, 1f));
            teacher.update();
            checkpoint(s + 1, checkpoints, curve, target::features, seed);
        }
        return new Result(Method.IJEPA_FULL, first, (float) tail, probe(target::features, seed), random);
    }

    // ------------------------------------------------------------------ wave 10 (items 10.3.1-10.3.6)

    /**
     * The wave 10 methods, kept apart from {@link Method} so that its rows stay unchanged.
     * The first seven train the MLP encoder on augmented views like {@link Method#SIMCLR}; the
     * last three train the one-block patch transformer like {@link Method#MAE}.
     */
    public enum Extra {
        /** SimSiam: predictor plus stop-gradient, no negatives and no momentum encoder. */
        SIMSIAM,
        /** SimSiam without the stop-gradient (expected to collapse). */
        SIMSIAM_NO_STOP_GRADIENT,
        /** SimSiam without the predictor (expected to collapse). */
        SIMSIAM_NO_PREDICTOR,
        /** SwAV with two global views. */
        SWAV,
        /** SwAV with two global views plus four local crops. */
        SWAV_MULTI_CROP,
        /** MoCo v2: momentum key encoder and a queue of 512 negatives. */
        MOCO_V2,
        /** MoCo v3: momentum keys, predictor, symmetrised in-batch negatives. */
        MOCO_V3,
        /** LeJEPA on the MLP encoder: two global views, four local crops, SIGReg. */
        LEJEPA,
        /** LeJEPA with VICReg's variance-covariance term in place of SIGReg. */
        LEJEPA_VICREG,
        /** data2vec on the patch transformer: masked regression of instance-normalised teacher layers. */
        DATA2VEC,
        /** iBOT on the patch transformer: masked-patch plus image-level self-distillation. */
        IBOT,
        /** LeJEPA on the patch transformer. */
        LEJEPA_PATCHES
    }

    /**
     * The training settings of a wave 10 run.
     *
     * @param steps training steps
     * @param batch images per step
     * @param seed the seed of weights and data
     * @param koleo the weight of the {@link KoLeo} regulariser on the encoder output (0: off)
     * @param lambda LeJEPA's SIGReg share
     */
    public record Options(int steps, int batch, long seed, float koleo, float lambda) {

        /**
         * Batch 64, no KoLeo, {@code lambda = 0.05}.
         *
         * @param steps training steps
         * @param seed the seed
         * @return the options
         */
        public static Options of(int steps, long seed) {
            return new Options(steps, 64, seed, 0, 0.05f);
        }

        /**
         * @param b images per step
         * @return a copy with that batch size
         */
        public Options batch(int b) {
            return new Options(steps, b, seed, koleo, lambda);
        }

        /**
         * @param w the KoLeo weight
         * @return a copy with that weight
         */
        public Options koleo(float w) {
            return new Options(steps, batch, seed, w, lambda);
        }

        /**
         * @param l the SIGReg share
         * @return a copy with that share
         */
        public Options lambda(float l) {
            return new Options(steps, batch, seed, koleo, l);
        }
    }

    /**
     * How good a frozen representation is.
     *
     * @param probeAccuracy the linear-probe test accuracy ({@link #probe})
     * @param knnAccuracy leave-one-out 5-NN accuracy of the normalised features of 384 test images
     * @param effectiveRank the RankMe effective rank of those features
     * @param uniformity Wang and Isola's uniformity of those features (lower is more spread)
     */
    public record Evaluation(float probeAccuracy, double knnAccuracy, double effectiveRank, double uniformity) {
    }

    /**
     * The outcome of a wave 10 run.
     *
     * @param name the method (and options)
     * @param firstLoss the loss of the first step
     * @param lastLoss the mean loss of the last 10 steps
     * @param evaluation the frozen encoder's scores
     * @param millisPerStep the training wall-clock per step
     * @param extra a method-specific number: SwAV's prototype cluster purity, else {@code NaN}
     */
    public record Report(String name, float firstLoss, float lastLoss, Evaluation evaluation, double millisPerStep,
            double extra) {
    }

    /**
     * Scores a frozen encoder: linear probe, k-NN, effective rank and uniformity.
     *
     * @param features images {@code [n, 1, 8, 8]} to features {@code [n, d]}
     * @param seed the data seed
     * @return the evaluation
     */
    public static Evaluation evaluate(UnaryOperator<Tensor> features, long seed) {
        Tensor[] test = IMAGES.generate(384, seed + 2000);
        Tensor f = features.apply(test[0]);
        int n = f.shape().dim(0);
        int d = f.shape().dim(1);
        float[] v = f.toArray();
        double[][] x = new double[n][d];
        for (int i = 0; i < n; i++) {
            double s = 1e-12;
            for (int j = 0; j < d; j++) {
                s += (double) v[i * d + j] * v[i * d + j];
            }
            s = Math.sqrt(s);
            for (int j = 0; j < d; j++) {
                x[i][j] = v[i * d + j] / s;
            }
        }
        double knn = MlMetrics.knnAccuracy(x, test[1].toIntArray(), 5);
        return new Evaluation(probe(features, seed), knn, SelfSupervisedLosses.effectiveRank(f),
                KoLeo.uniformity(f, 2));
    }

    /**
     * Pretrains an MLP encoder with a joint-embedding method on {@link ShapeImages} (batch 64)
     * and returns it frozen, for downstream experiments (SCAN clustering, robust fine-tuning).
     *
     * @param method a joint-embedding method (SimCLR, BYOL, DINO, Barlow Twins or VICReg)
     * @param steps training steps
     * @param seed the seed
     * @return images {@code [n, 1, 8, 8]} to features {@code [n, 32]}
     */
    public static UnaryOperator<Tensor> pretrainEncoder(Method method, int steps, long seed) {
        return pretrainEncoder(method, s -> IMAGES.generate(64, s)[0], steps, seed);
    }

    /**
     * Pretrains an MLP encoder with a joint-embedding method on any 8x8 single-channel images,
     * with the benchmark's augmentations (shifts, brightness and noise).
     *
     * @param method a joint-embedding method
     * @param batches given a seed, an unlabelled batch {@code [n, 1, 8, 8]}
     * @param steps training steps
     * @param seed the seed
     * @return images {@code [n, 1, 8, 8]} to features {@code [n, 32]}
     */
    public static UnaryOperator<Tensor> pretrainEncoder(Method method, LongFunction<Tensor> batches, int steps,
            long seed) {
        if (method == Method.MAE || method == Method.IJEPA || method == Method.IJEPA_FULL) {
            throw new IllegalArgumentException("pretrainEncoder takes a joint-embedding method: " + method);
        }
        return trainJoint(method, steps, seed, 0, batches, false).features();
    }

    /**
     * Runs an existing joint-embedding method with the wave 10 scores, at any batch size and
     * with an optional KoLeo term (items 10.3.3 and 10.3.5).
     *
     * @param method a joint-embedding method
     * @param options steps, batch, seed and KoLeo weight (lambda is ignored)
     * @return the report
     */
    public static Report report(Method method, Options options) {
        Trained t = trainJoint(method, options.steps(), options.seed(), options.koleo(),
                s -> IMAGES.generate(options.batch(), s)[0], false);
        String name = method + (options.koleo() > 0 ? "+KoLeo" : "")
                + (options.batch() != 64 ? "@b" + options.batch() : "");
        return new Report(name, t.first(), t.tail(), evaluate(t.features(), options.seed()), t.millis(), Double.NaN);
    }

    /**
     * Trains and scores one wave 10 method.
     *
     * @param method the method
     * @param options steps, batch, seed, KoLeo weight and LeJEPA's lambda
     * @return the report
     */
    public static Report run(Extra method, Options options) {
        return switch (method) {
            case DATA2VEC, IBOT, LEJEPA_PATCHES -> runExtraPatches(method, options);
            default -> runExtraJoint(method, options);
        };
    }

    private static String name(Extra method, Options o) {
        return method + (o.koleo() > 0 ? "+KoLeo" : "") + (o.batch() != 64 ? "@b" + o.batch() : "")
                + (method.name().startsWith("LEJEPA") && o.lambda() != 0.05f ? "@l" + o.lambda() : "");
    }

    private static Report runExtraJoint(Extra method, Options o) {
        long seed = o.seed();
        SplittableRandom rnd = new SplittableRandom(seed);
        Model encoder = mlp(SIDE * SIDE, 128, REP, true, seed);
        Model projector = mlp(REP, 64, REP, false, seed + 1);
        Model predictor = mlp(REP, 64, REP, false, seed + 2);
        Model targetEncoder = mlp(SIDE * SIDE, 128, REP, true, seed);
        Model targetProjector = mlp(REP, 64, REP, false, seed + 1);
        List<Var> online = new ArrayList<>(encoder.parameters());
        online.addAll(projector.parameters());
        List<Var> target = new ArrayList<>(targetEncoder.parameters());
        target.addAll(targetProjector.parameters());
        MomentumTeacher teacher = new MomentumTeacher(target, online, 0.99f);
        List<Var> trained = new ArrayList<>(online);
        boolean usesPredictor = method == Extra.SIMSIAM || method == Extra.SIMSIAM_NO_STOP_GRADIENT
                || method == Extra.MOCO_V3;
        if (usesPredictor) {
            trained.addAll(predictor.parameters());
        }
        SwappedPrediction swav = new SwappedPrediction(REP, 30, 0.1f, 0.05, 3, seed + 3);
        if (method == Extra.SWAV || method == Extra.SWAV_MULTI_CROP) {
            trained.addAll(swav.parameters());
        }
        MomentumContrast moco = new MomentumContrast(REP, 512, 0.2f, seed + 4);
        LeJepa lejepa = new LeJepa(o.lambda(), 32,
                method == Extra.LEJEPA_VICREG ? LeJepa.Regularizer.VARIANCE_COVARIANCE : LeJepa.Regularizer.SIGREG);
        Optimizer opt = Optimizer.adam(3e-3f);
        float first = 0;
        double tail = 0;
        long start = System.nanoTime();
        for (int s = 0; s < o.steps(); s++) {
            Tensor batch = IMAGES.generate(o.batch(), rnd.nextLong())[0];
            Tensor a = IMAGES.augment(batch, 2, rnd);
            Tensor b = IMAGES.augment(batch, 2, rnd);
            Var ha = encoder.forward(flat(a));
            Var za = projector.forward(ha);
            Var zb = projector.forward(encoder.forward(flat(b)));
            Var loss = switch (method) {
                case SIMSIAM, SIMSIAM_NO_STOP_GRADIENT -> SimSiam.loss(predictor.forward(za), za,
                        predictor.forward(zb), zb, method == Extra.SIMSIAM);
                case SIMSIAM_NO_PREDICTOR -> SimSiam.loss(za, za, zb, zb, true);
                case SWAV, SWAV_MULTI_CROP -> {
                    List<Var> views = new ArrayList<>(List.of(za, zb));
                    if (method == Extra.SWAV_MULTI_CROP) {
                        for (int c = 0; c < 4; c++) {
                            Tensor crop = SwappedPrediction.localCrops(IMAGES.augment(batch, 1, rnd), 5, rnd);
                            views.add(projector.forward(encoder.forward(flat(crop))));
                        }
                    }
                    yield swav.loss(views, 2);
                }
                case MOCO_V2 -> {
                    Tensor kb = Var.noGrad(() -> targetProjector.forward(targetEncoder.forward(flat(b)))).value();
                    Var l = moco.loss(za, kb);
                    moco.enqueue(kb);
                    yield l;
                }
                case MOCO_V3 -> {
                    Tensor ka = Var.noGrad(() -> targetProjector.forward(targetEncoder.forward(flat(a)))).value();
                    Tensor kb = Var.noGrad(() -> targetProjector.forward(targetEncoder.forward(flat(b)))).value();
                    yield MomentumContrast.symmetricV3(predictor.forward(za), predictor.forward(zb), ka, kb, 0.2f);
                }
                case LEJEPA, LEJEPA_VICREG -> {
                    List<Var> global = List.of(za, zb);
                    List<Var> all = new ArrayList<>(global);
                    for (int c = 0; c < 4; c++) {
                        Tensor crop = SwappedPrediction.localCrops(IMAGES.augment(batch, 1, rnd), 5, rnd);
                        all.add(projector.forward(encoder.forward(flat(crop))));
                    }
                    yield lejepa.loss(global, all, rnd);
                }
                default -> throw new IllegalArgumentException(method + " is not a joint-embedding method");
            };
            float v = loss.value().item();
            first = s == 0 ? v : first;
            tail += s >= o.steps() - 10 ? v / 10 : 0;
            if (o.koleo() > 0) {
                loss = loss.add(KoLeo.loss(ha).mul(o.koleo()));
            }
            loss.backward();
            opt.step(trained);
            teacher.update();
        }
        double millis = (System.nanoTime() - start) / 1e6 / Math.max(1, o.steps());
        UnaryOperator<Tensor> features = x -> Var.noGrad(() -> encoder.forward(flat(x))).value();
        double extra = Double.NaN;
        if (method == Extra.SWAV || method == Extra.SWAV_MULTI_CROP) {
            Tensor[] test = IMAGES.generate(384, seed + 2000);
            int[] assigned = swav.assign(Var.noGrad(() -> projector.forward(encoder.forward(flat(test[0])))).value());
            extra = purity(assigned, test[1].toIntArray(), 30);
        }
        return new Report(name(method, o), first, (float) tail, evaluate(features, seed), millis, extra);
    }

    /** The share of points whose cluster's majority class is their class. */
    private static double purity(int[] cluster, int[] label, int clusters) {
        int[][] counts = new int[clusters][ShapeImages.CLASSES];
        for (int i = 0; i < cluster.length; i++) {
            counts[cluster[i]][label[i]]++;
        }
        int hit = 0;
        for (int[] c : counts) {
            hit += java.util.Arrays.stream(c).max().orElse(0);
        }
        return (double) hit / cluster.length;
    }

    /** The patch encoder's tokens with masked positions replaced by a mask token (iBOT's student input). */
    private static Var maskedTokens(PatchEncoder encoder, Tensor images, PatchMasking.Mask mask, Var maskToken) {
        Var t = encoder.embed.forward(Var.of(images));
        Var kept = PatchMasking.gather(t, mask.keep());
        int hidden = mask.masked()[0].length;
        Var fill = maskToken.expand(Shape.of(images.shape().dim(0), hidden, REP));
        return encoder.norm.forward(encoder.block.forward(PatchMasking.unshuffle(kept, fill, mask)));
    }

    private static Report runExtraPatches(Extra method, Options o) {
        long seed = o.seed();
        int n = o.batch();
        SplittableRandom rnd = new SplittableRandom(seed);
        PatchEncoder encoder = new PatchEncoder(seed);
        PatchEncoder target = new PatchEncoder(seed);
        MomentumTeacher teacher = new MomentumTeacher(target.parameters(), encoder.parameters(), 0.99f);
        SplittableRandom dr = new SplittableRandom(seed + 7);
        Var maskToken = Var.param(Tensor.zeros(Shape.of(1, 1, REP)));
        Var decoderPositions = Var.param(Tensor.normal(Shape.of(16, REP), 0, 0.02f, dr));
        TunableTransformerBlock decoder = TunableTransformerBlock.of(4).ffnFactor(2).build(REP, dr);
        Var out = Var.param(Tensor.normal(Shape.of(REP, REP), 0, 0.1f, dr));
        Var prototypes = Var.param(Tensor.normal(Shape.of(REP, 64), 0, 1, new SplittableRandom(seed + 3)));
        Var targetPrototypes = Var.param(prototypes.value());
        MomentumTeacher prototypeTeacher = new MomentumTeacher(List.of(targetPrototypes), List.of(prototypes), 0.99f);
        SelfSupervisedLosses.Center patchCenter = new SelfSupervisedLosses.Center(64, 0.9f);
        SelfSupervisedLosses.Center imageCenter = new SelfSupervisedLosses.Center(64, 0.9f);
        LeJepa lejepa = new LeJepa(o.lambda(), 32, LeJepa.Regularizer.SIGREG);
        List<Var> trained = new ArrayList<>(encoder.parameters());
        switch (method) {
            case DATA2VEC -> {
                trained.addAll(List.of(maskToken, decoderPositions, out));
                trained.addAll(decoder.parameters());
            }
            case IBOT -> trained.addAll(List.of(maskToken, prototypes));
            default -> {
            }
        }
        Optimizer opt = Optimizer.adam(3e-3f);
        float first = 0;
        double tail = 0;
        long start = System.nanoTime();
        for (int s = 0; s < o.steps(); s++) {
            Tensor batch = IMAGES.generate(n, rnd.nextLong())[0];
            Var loss = switch (method) {
                case DATA2VEC -> {
                    PatchMasking.Mask mask = PatchMasking.random(n, 16, 0.6f, rnd);
                    Var context = encoder.tokens(batch, mask.keep());
                    int hidden = mask.masked()[0].length;
                    Var fill = maskToken.expand(Shape.of(n, hidden, REP));
                    Var full = PatchMasking.unshuffle(context, fill, mask).add(decoderPositions);
                    Var predicted = PatchMasking.gather(decoder.forward(full), mask.masked()).matmul(out);
                    Tensor targets = Var.noGrad(() -> {
                        Var t0 = target.embed.forward(Var.of(batch));
                        Var t1 = target.block.forward(t0);
                        Tensor avg = MaskedFeaturePrediction.data2vecTargets(List.of(t0.value(), t1.value()), 2);
                        return PatchMasking.gather(Var.of(avg), mask.masked());
                    }).value();
                    yield MaskedFeaturePrediction.regression(predicted, targets);
                }
                case IBOT -> {
                    Tensor a = IMAGES.augment(batch, 1, rnd);
                    Tensor b = IMAGES.augment(batch, 1, rnd);
                    PatchMasking.Mask mask = PatchMasking.random(n, 16, 0.4f, rnd);
                    Var student = maskedTokens(encoder, a, mask, maskToken);
                    Var studentPatches = dinoHead(PatchMasking.gather(student, mask.masked()), prototypes);
                    Var studentImage = dinoHead(student.mean(1, false), prototypes);
                    Tensor[] t = teacherOutputs(target, targetPrototypes, a, b, mask);
                    Var l = MaskedFeaturePrediction.ibotPatchLoss(studentPatches, t[0], patchCenter.value(), 0.1f, 0.04f)
                            .add(SelfSupervisedLosses.dino(studentImage, Var.of(t[1]), imageCenter.value(), 0.1f,
                                    0.04f));
                    patchCenter.update(t[0].reshape(Math.toIntExact(t[0].size() / 64), 64));
                    imageCenter.update(t[1]);
                    yield l;
                }
                default -> {
                    List<Var> global = new ArrayList<>();
                    for (int g = 0; g < 2; g++) {
                        global.add(encoder.tokens(IMAGES.augment(batch, 1, rnd), null).mean(1, false));
                    }
                    List<Var> all = new ArrayList<>(global);
                    for (int c = 0; c < 1; c++) {
                        Tensor crop = SwappedPrediction.localCrops(IMAGES.augment(batch, 1, rnd), 5, rnd);
                        all.add(encoder.tokens(crop, null).mean(1, false));
                    }
                    yield lejepa.loss(global, all, rnd);
                }
            };
            float v = loss.value().item();
            first = s == 0 ? v : first;
            tail += s >= o.steps() - 10 ? v / 10 : 0;
            loss.backward();
            opt.step(trained);
            if (method != Extra.LEJEPA_PATCHES) {
                teacher.update();
                prototypeTeacher.update();
            }
        }
        double millis = (System.nanoTime() - start) / 1e6 / Math.max(1, o.steps());
        PatchEncoder probed = method == Extra.LEJEPA_PATCHES ? encoder : target;
        UnaryOperator<Tensor> features = x -> Var.noGrad(() -> Var.of(probed.features(x))).value();
        return new Report(name(method, o), first, (float) tail, evaluate(features, seed), millis, Double.NaN);
    }

    /** The iBOT teacher's patch outputs (view a, masked positions) and image outputs (view b). */
    private static Tensor[] teacherOutputs(PatchEncoder target, Var prototypes, Tensor a, Tensor b,
            PatchMasking.Mask mask) {
        return Var.noGrad(() -> {
            Tensor patches = dinoHead(PatchMasking.gather(target.tokens(a, null), mask.masked()), prototypes).value();
            Tensor image = dinoHead(target.tokens(b, null).mean(1, false), prototypes).value();
            return new Tensor[] {patches, image};
        });
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

    /**
     * The wave 10 Run (long budgets, outside the tests): every new method at 1000 steps for
     * three seeds, MoCo against SimCLR at batch 16, 64 and 256, KoLeo on DINO, BYOL and SimSiam,
     * and LeJEPA's lambda sweep.
     *
     * @param args unused
     */
    public static void main(String[] args) {
        for (long seed = 1; seed <= 3; seed++) {
            for (Extra e : Extra.values()) {
                System.out.println(run(e, Options.of(1000, seed)));
            }
            for (Method m : List.of(Method.SIMCLR, Method.BYOL, Method.DINO, Method.VICREG)) {
                System.out.println(report(m, Options.of(1000, seed)));
            }
        }
        for (int batch : new int[] {16, 64, 256}) {
            System.out.println(report(Method.SIMCLR, Options.of(1000, 1).batch(batch)));
            System.out.println(run(Extra.MOCO_V2, Options.of(1000, 1).batch(batch)));
            System.out.println(run(Extra.MOCO_V3, Options.of(1000, 1).batch(batch)));
        }
        for (Method m : List.of(Method.DINO, Method.BYOL)) {
            System.out.println(report(m, Options.of(1000, 1).koleo(0.1f)));
        }
        System.out.println(run(Extra.SIMSIAM, Options.of(1000, 1).koleo(0.1f)));
        for (float lambda : new float[] {0.01f, 0.02f, 0.05f, 0.1f, 0.2f}) {
            System.out.println(run(Extra.LEJEPA, Options.of(1000, 1).lambda(lambda)));
        }
    }
}
