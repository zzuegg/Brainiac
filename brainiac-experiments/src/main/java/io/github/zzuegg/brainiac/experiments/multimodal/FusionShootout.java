package io.github.zzuegg.brainiac.experiments.multimodal;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.ShapeScenes;
import io.github.zzuegg.brainiac.experimental.llm.HookedDecoder;
import io.github.zzuegg.brainiac.experimental.multimodal.BottleneckFusion;
import io.github.zzuegg.brainiac.experimental.multimodal.FusionDecoder;
import io.github.zzuegg.brainiac.experimental.multimodal.GatedCrossAttentionAdapter;
import io.github.zzuegg.brainiac.experimental.multimodal.PatchProjector;
import io.github.zzuegg.brainiac.experimental.multimodal.QueryTransformer;
import io.github.zzuegg.brainiac.experimental.multimodal.TransfusionModel;
import io.github.zzuegg.brainiac.experimental.nn.VisionTransformer;
import io.github.zzuegg.brainiac.experiments.generative.TalkingDigits;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.SplittableRandom;

/**
 * Wave 9 items 9.4.15 to 9.4.17: connecting modalities.
 *
 * <p><b>Bridges to a frozen LM</b> ({@link #vqa}, 9.4.15): questions about 16x16
 * {@link ShapeScenes} (1-3 glyphs of six classes): {@code HOW <shape>} (how many),
 * {@code IS <shape>} (is there one) and {@code LEFT <shape>} (the shape nearest to its
 * left), half of them about a shape that is present. A {@link FusionDecoder} LM is
 * pretrained on these questions alone, after 16 PAD tokens, and after a 16-slot text
 * description of the scene (slot {@code p} holds the shape whose centre lies in 4x4 patch
 * {@code p}), so that, like a pretrained LLM, it already reads scenes described in words;
 * then it is frozen. A {@link VisionTransformer} is pretrained on single-glyph
 * classification and frozen. Rows: the frozen LM alone (language prior), the frozen LM
 * reading the true description (the ceiling a bridge could reach), the LLaVA-style
 * {@link PatchProjector} (16 visual prefix tokens), the {@link QueryTransformer} (16 query
 * tokens, after a contrastive stage against the LM's embedding of the scene's shape list),
 * the Flamingo-style {@link GatedCrossAttentionAdapter} (resampler and tanh-gated
 * cross-attention after every LM block, behind 16 PAD tokens), and early fusion from
 * scratch (raw pixel patches projected into a fresh decoder, everything trained). Reported:
 * accuracy per question kind, trainable parameters, and the text-only loss of the answering
 * LM afterwards (does fusion damage the LM?).
 *
 * <p><b>Transfusion</b> ({@link #generation}, 9.4.16): {@link DigitGlyphs} 8x8 digits as
 * 16 patches of 2x2 pixels; one {@link TransfusionModel} trained on captioning (a lightly noised image, then
 * the digit token, scored by the text loss only) and generation (digit token, then the image by diffusion) against two
 * Transfusion models trained on one direction each with half the steps, and against the
 * VQ-token early fusion of {@link EarlyFusion} (one model on both directions, and two
 * separate ones). Reported: caption accuracy, the share of generated images whose nearest
 * clean rendering shows the requested digit, and the mean squared distance to it.
 *
 * <p><b>Bottleneck fusion</b> ({@link #audioVisual}, 9.4.17): a digit seen ({@link DigitGlyphs}
 * with flipped pixels, 16 tokens of 2x2) and heard ({@link TalkingDigits} band energies with
 * added noise, 12 frames), each modality noisy enough that neither alone is perfect.
 * {@link BottleneckFusion} with bottleneck tokens, full early fusion and late fusion, each with
 * and without modality dropout, and the unimodal models; accuracy with both modalities and
 * with each missing, and multiply-adds.
 */
public final class FusionShootout {

    private FusionShootout() {
    }

    // ------------------------------------------------------------------ 9.4.15 bridges

    private static final ShapeScenes SCENES = new ShapeScenes(16, 0.1f, 2);
    private static final int Q = 1;
    private static final int HOW = 3;
    private static final int IS = 5;
    private static final int LEFT = 8;
    private static final int SHAPE0 = 10;
    private static final int COUNT0 = SHAPE0 + ShapeScenes.CLASSES;
    private static final int YES = COUNT0 + 6;
    private static final int NO = YES + 1;
    private static final int NONE = NO + 1;
    private static final int VOCAB = NONE + 1;
    private static final int TEXT = 2;
    private static final int DIM = 32;
    private static final int SLOTS = 16;

    /** The bridges of 9.4.15. */
    public enum Bridge {
        /** The frozen LM without the image. */
        TEXT_ONLY,
        /** The frozen LM reading the true 16-slot scene description: the ceiling for the bridges. */
        DESCRIPTION,
        /** A two-layer MLP from patch features to LM input embeddings. */
        PROJECTOR,
        /** Learned queries cross-attending to the patches. */
        QFORMER,
        /** A perceiver resampler and tanh-gated cross-attention inside the LM. */
        GATED,
        /** A fresh decoder over projected raw pixel patches, all trained. */
        EARLY_FUSION
    }

    /**
     * Scores of one bridge.
     *
     * @param bridge the bridge
     * @param accuracy answer accuracy over all questions
     * @param count accuracy on {@code HOW MANY}
     * @param exists accuracy on {@code IS THERE}
     * @param left accuracy on {@code WHAT LEFT OF}
     * @param trainable trainable parameters
     * @param textLoss the answering LM's loss on held-out text-only questions afterwards
     * @param millis training and evaluation time
     */
    public record VqaScores(Bridge bridge, double accuracy, double count, double exists, double left, long trainable,
            double textLoss, long millis) {
    }

    /** One question: its kind (0 count, 1 exists, 2 left), tokens and answer. */
    private record Question(int kind, int[] text, int answer) {
    }

    /** The frozen parts shared by every bridge, and the data. */
    public static final class VqaSetup {
        private final FusionDecoder lm;
        private final VisionTransformer vit;
        private final ShapeScenes.Scene[] train;
        private final Tensor trainFeatures;
        private final ShapeScenes.Scene[] test;
        private final Tensor testFeatures;
        private final long seed;

        private VqaSetup(FusionDecoder lm, VisionTransformer vit, ShapeScenes.Scene[] train,
                ShapeScenes.Scene[] test, long seed) {
            this.lm = lm;
            this.vit = vit;
            this.train = train;
            this.test = test;
            this.trainFeatures = features(vit, train);
            this.testFeatures = features(vit, test);
            this.seed = seed;
        }

        /** @return the frozen LM's text-only loss on held-out questions */
        public double textLoss() {
            return FusionShootout.textLoss(lm, seed + 7);
        }
    }

    private static Tensor features(VisionTransformer vit, ShapeScenes.Scene[] s) {
        return Var.noGrad(() -> vit.tokens(Var.of(ShapeScenes.images(s)), false)).value();
    }

    private static Question question(ShapeScenes.Scene s, SplittableRandom r) {
        int[] count = new int[ShapeScenes.CLASSES];
        for (int c : s.classes()) {
            count[c]++;
        }
        List<Integer> unique = new ArrayList<>();
        for (int c = 0; c < count.length; c++) {
            if (count[c] == 1) {
                unique.add(c);
            }
        }
        int kind = r.nextInt(3);
        if (kind == 2 && !unique.isEmpty()) {
            int c = unique.get(r.nextInt(unique.size()));
            int self = -1;
            for (int i = 0; i < s.count(); i++) {
                self = s.classes()[i] == c ? i : self;
            }
            float cx = (s.boxes()[4 * self] + s.boxes()[4 * self + 2]) / 2;
            int best = -1;
            float bestX = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < s.count(); i++) {
                float x = (s.boxes()[4 * i] + s.boxes()[4 * i + 2]) / 2;
                if (i != self && x < cx && x > bestX) {
                    bestX = x;
                    best = i;
                }
            }
            return new Question(2, new int[] {LEFT, SHAPE0 + c},
                    best < 0 ? NONE : SHAPE0 + s.classes()[best]);
        }
        // half the questions ask about a shape that is present, so the prior is not the answer
        int c = r.nextBoolean() ? s.classes()[r.nextInt(s.count())] : r.nextInt(ShapeScenes.CLASSES);
        if (kind == 1) {
            return new Question(1, new int[] {IS, SHAPE0 + c}, count[c] > 0 ? YES : NO);
        }
        return new Question(0, new int[] {HOW, SHAPE0 + c}, COUNT0 + Math.min(5, count[c]));
    }

    /** Text embeddings after {@code prefix} pad embeddings. */
    private static Var padded(FusionDecoder lm, int[][] text, int prefix) {
        int[][] ids = new int[text.length][prefix + text[0].length];
        for (int i = 0; i < text.length; i++) {
            System.arraycopy(text[i], 0, ids[i], prefix, text[i].length);
        }
        return lm.embed(ids);
    }

    /**
     * The scene as {@link #SLOTS} words, one per 4x4 patch: the shape token of the glyph
     * whose box centre lies in that patch, else PAD. The text the LM is pretrained to read.
     */
    private static int[] description(ShapeScenes.Scene s) {
        int[] slots = new int[SLOTS];
        for (int i = 0; i < s.count(); i++) {
            int cx = Math.min(3, (int) ((s.boxes()[4 * i] + s.boxes()[4 * i + 2]) / 2 / 4));
            int cy = Math.min(3, (int) ((s.boxes()[4 * i + 1] + s.boxes()[4 * i + 3]) / 2 / 4));
            slots[cy * 4 + cx] = SHAPE0 + s.classes()[i];
        }
        return slots;
    }

    private static int[][] answerTargets(int b, int length, int[] answers) {
        int[][] t = new int[b][length];
        for (int i = 0; i < b; i++) {
            Arrays.fill(t[i], -1);
            t[i][length - 1] = answers[i];
        }
        return t;
    }

    /**
     * Questions about random scenes as text: {@code mode} 0 the question alone, 1 after
     * {@link #SLOTS} PAD tokens, 2 after the scene's {@link #description}.
     */
    private static Var textOnlyLoss(FusionDecoder lm, int mode, int n, SplittableRandom r, boolean training) {
        ShapeScenes.Scene[] s = SCENES.scenes(n, 1, 3, ShapeScenes.Layout.SPARSE, r.nextLong());
        int prefix = mode == 0 ? 0 : SLOTS;
        int[][] ids = new int[n][prefix + TEXT];
        int[] ans = new int[n];
        for (int i = 0; i < n; i++) {
            Question q = question(s[i], r);
            if (mode == 2) {
                System.arraycopy(description(s[i]), 0, ids[i], 0, SLOTS);
            }
            System.arraycopy(q.text(), 0, ids[i], prefix, TEXT);
            ans[i] = q.answer();
        }
        Var logits = lm.logits(lm.embed(ids), null, null, training);
        return FusionDecoder.loss(logits, answerTargets(n, prefix + TEXT, ans));
    }

    private static double textLoss(FusionDecoder lm, long seed) {
        return Var.noGrad(() -> textOnlyLoss(lm, 0, 256, new SplittableRandom(seed), false)).value().item();
    }

    /**
     * Pretrains and freezes the LM and the vision encoder, and draws the scenes.
     *
     * @param lmSteps LM steps of 32 text questions (in turn: alone, after 16 PAD tokens, after the
     *     16-slot scene description)
     * @param visionSteps ViT steps of 32 single-glyph scenes
     * @param trainScenes scenes for bridge training
     * @param testScenes held-out scenes
     * @param seed the seed
     * @return the setup
     */
    public static VqaSetup vqaSetup(int lmSteps, int visionSteps, int trainScenes, int testScenes, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        FusionDecoder lm = new FusionDecoder(VOCAB, DIM, 2, 4, 32, seed + 1);
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int s = 0; s < lmSteps; s++) {
            textOnlyLoss(lm, s % 4 < 2 ? 2 : s % 4 - 2, 32, r, true).backward();
            opt.step(lm.parameters());
        }
        VisionTransformer vit = VisionTransformer.of(4, DIM, 2, 4).classToken(false).classes(ShapeScenes.CLASSES)
                .build(1, 16, 16, new SplittableRandom(seed + 2));
        Optimizer vo = Optimizer.adam(3e-3f);
        for (int s = 0; s < visionSteps; s++) {
            ShapeScenes.Scene[] one = SCENES.scenes(32, 1, 1, ShapeScenes.Layout.SPARSE, r.nextLong());
            float[] y = new float[one.length];
            for (int i = 0; i < y.length; i++) {
                y[i] = one[i].classes()[0];
            }
            Functional.crossEntropy(vit.forward(Var.of(ShapeScenes.images(one)), true),
                    Tensor.of(Shape.of(y.length), y)).backward();
            vo.step(vit.parameters());
        }
        return new VqaSetup(lm, vit, SCENES.scenes(trainScenes, 1, 3, ShapeScenes.Layout.SPARSE, seed + 3),
                SCENES.scenes(testScenes, 1, 3, ShapeScenes.Layout.SPARSE, seed + 4), seed);
    }

    /** A trained bridge: answer logits at the last position. */
    private interface Answerer {
        Var logits(Tensor patches, Tensor pixels, int[][] scenes, int[][] text, boolean training);

        List<Var> trainable();
    }

    private static Tensor rows(Tensor t, int[] idx) {
        int per = (int) (t.shape().size() / t.shape().dim(0));
        float[] all = t.toArray();
        float[] out = new float[idx.length * per];
        for (int i = 0; i < idx.length; i++) {
            System.arraycopy(all, idx[i] * per, out, i * per, per);
        }
        return Tensor.of(t.shape().withDim(0, idx.length), out);
    }

    /** Raw 4x4 pixel patches {@code [n, 16, 16]} of 16x16 scenes. */
    private static Tensor pixels(ShapeScenes.Scene[] s, int[] idx) {
        float[] out = new float[idx.length * 256];
        for (int i = 0; i < idx.length; i++) {
            float[] img = s[idx[i]].image();
            for (int p = 0; p < 16; p++) {
                int px = (p % 4) * 4;
                int py = (p / 4) * 4;
                for (int k = 0; k < 16; k++) {
                    out[i * 256 + p * 16 + k] = img[(py + k / 4) * 16 + px + k % 4];
                }
            }
        }
        return Tensor.of(Shape.of(idx.length, 16, 16), out);
    }

    private static long count(List<Var> ps) {
        long n = 0;
        for (Var p : ps) {
            n += p.value().shape().size();
        }
        return n;
    }

    private static Answerer bridge(Bridge b, VqaSetup setup, QueryTransformer qf, long seed) {
        FusionDecoder lm = setup.lm;
        return switch (b) {
            case TEXT_ONLY -> new Answerer() {
                @Override
                public Var logits(Tensor patches, Tensor pixels, int[][] scenes, int[][] text, boolean training) {
                    return lm.logits(lm.embed(text), null, null, training);
                }

                @Override
                public List<Var> trainable() {
                    return List.of();
                }
            };
            case DESCRIPTION -> new Answerer() {
                @Override
                public Var logits(Tensor patches, Tensor pixels, int[][] scenes, int[][] text, boolean training) {
                    int[][] ids = new int[text.length][];
                    for (int i = 0; i < text.length; i++) {
                        ids[i] = Arrays.copyOf(scenes[i], SLOTS + text[i].length);
                        System.arraycopy(text[i], 0, ids[i], SLOTS, text[i].length);
                    }
                    return lm.logits(lm.embed(ids), null, null, training);
                }

                @Override
                public List<Var> trainable() {
                    return List.of();
                }
            };
            case PROJECTOR -> {
                PatchProjector proj = new PatchProjector(DIM, DIM, true, seed);
                yield new Answerer() {
                    @Override
                    public Var logits(Tensor patches, Tensor pixels, int[][] scenes, int[][] text, boolean training) {
                        return lm.logits(Var.concat(1, proj.forward(Var.of(patches)), lm.embed(text)), null, null,
                                training);
                    }

                    @Override
                    public List<Var> trainable() {
                        return proj.parameters();
                    }
                };
            }
            case QFORMER -> {
                yield new Answerer() {
                    @Override
                    public Var logits(Tensor patches, Tensor pixels, int[][] scenes, int[][] text, boolean training) {
                        return lm.logits(Var.concat(1, qf.prefix(Var.of(patches)), lm.embed(text)), null, null,
                                training);
                    }

                    @Override
                    public List<Var> trainable() {
                        return qf.parameters();
                    }
                };
            }
            case GATED -> {
                GatedCrossAttentionAdapter g = new GatedCrossAttentionAdapter(DIM, DIM, 8, 1, lm.layers(), 1, 4, seed);
                yield new Answerer() {
                    @Override
                    public Var logits(Tensor patches, Tensor pixels, int[][] scenes, int[][] text, boolean training) {
                        return lm.logits(padded(lm, text, SLOTS), null, g.hook(g.resample(Var.of(patches))),
                                training);
                    }

                    @Override
                    public List<Var> trainable() {
                        return g.parameters();
                    }
                };
            }
            case EARLY_FUSION -> {
                FusionDecoder fresh = new FusionDecoder(VOCAB, DIM, 2, 4, 32, seed);
                PatchProjector proj = new PatchProjector(16, DIM, false, seed + 1);
                yield new Answerer() {
                    @Override
                    public Var logits(Tensor patches, Tensor pixels, int[][] scenes, int[][] text, boolean training) {
                        if (pixels == null) {
                            return fresh.logits(fresh.embed(text), null, null, training);
                        }
                        return fresh.logits(Var.concat(1, proj.forward(Var.of(pixels)), fresh.embed(text)), null, null,
                                training);
                    }

                    @Override
                    public List<Var> trainable() {
                        List<Var> p = new ArrayList<>(fresh.parameters());
                        p.addAll(proj.parameters());
                        return p;
                    }
                };
            }
        };
    }

    /**
     * Trains one bridge and scores it.
     *
     * @param setup the frozen parts and data
     * @param bridge the bridge
     * @param steps steps of 32 questions (the Q-Former also gets as many contrastive steps first)
     * @param seed the seed
     * @return the scores
     */
    public static VqaScores vqa(VqaSetup setup, Bridge bridge, int steps, long seed) {
        long start = System.currentTimeMillis();
        QueryTransformer qf = bridge == Bridge.QFORMER ? new QueryTransformer(DIM, DIM, DIM, DIM, SLOTS, 1, 4, seed) : null;
        Answerer a = bridge(bridge, setup, qf, seed);
        SplittableRandom r = new SplittableRandom(seed + 1);
        Optimizer opt = Optimizer.adam(3e-3f);
        int n = setup.train.length;
        if (qf != null) {
            qformerContrastive(setup, qf, steps / 3, r);
        }
        if (!a.trainable().isEmpty()) {
            for (int s = 0; s < steps; s++) {
                int[] idx = r.ints(32, 0, n).toArray();
                int[][] text = new int[idx.length][];
                int[] ans = new int[idx.length];
                for (int i = 0; i < idx.length; i++) {
                    Question q = question(setup.train[idx[i]], r);
                    text[i] = q.text();
                    ans[i] = q.answer();
                }
                Var logits = a.logits(rows(setup.trainFeatures, idx), pixels(setup.train, idx), null, text, true);
                FusionDecoder.loss(logits, answerTargets(idx.length, logits.shape().dim(1), ans)).backward();
                opt.step(a.trainable());
                setup.lm.zeroGrad();
            }
        }
        double[] hit = new double[3];
        double[] tot = new double[3];
        SplittableRandom tr = new SplittableRandom(seed + 2);
        ShapeScenes.Scene[] test = setup.test;
        for (int from = 0; from < test.length; from += 64) {
            int[] idx = new int[Math.min(64, test.length - from)];
            for (int i = 0; i < idx.length; i++) {
                idx[i] = from + i;
            }
            int[][] text = new int[idx.length][];
            int[][] described = new int[idx.length][];
            Question[] qs = new Question[idx.length];
            for (int i = 0; i < idx.length; i++) {
                qs[i] = question(test[idx[i]], tr);
                text[i] = qs[i].text();
                described[i] = description(test[idx[i]]);
            }
            Tensor logits = Var.noGrad(() -> a.logits(rows(setup.testFeatures, idx), pixels(test, idx), described,
                    text, false)).value();
            int len = logits.shape().dim(1);
            float[] l = logits.toArray();
            for (int i = 0; i < idx.length; i++) {
                int off = (i * len + len - 1) * VOCAB;
                int best = 0;
                for (int k = 1; k < VOCAB; k++) {
                    best = l[off + k] > l[off + best] ? k : best;
                }
                hit[qs[i].kind()] += best == qs[i].answer() ? 1 : 0;
                tot[qs[i].kind()]++;
            }
        }
        double after = Var.noGrad(() -> {
            SplittableRandom lr = new SplittableRandom(setup.seed + 7);
            ShapeScenes.Scene[] s = SCENES.scenes(256, 1, 3, ShapeScenes.Layout.SPARSE, lr.nextLong());
            int[][] text = new int[s.length][];
            int[] ans = new int[s.length];
            for (int i = 0; i < s.length; i++) {
                Question q = question(s[i], lr);
                text[i] = q.text();
                ans[i] = q.answer();
            }
            Var logits = bridge == Bridge.EARLY_FUSION ? a.logits(null, null, null, text, false)
                    : setup.lm.logits(setup.lm.embed(text), null, null, false);
            return FusionDecoder.loss(logits, answerTargets(s.length, TEXT, ans));
        }).value().item();
        return new VqaScores(bridge, (hit[0] + hit[1] + hit[2]) / (tot[0] + tot[1] + tot[2]), hit[0] / tot[0],
                hit[1] / tot[1], tot[2] == 0 ? Double.NaN : hit[2] / tot[2], count(a.trainable()), after,
                System.currentTimeMillis() - start);
    }

    /** The Q-Former's image-text contrastive stage against the frozen LM's mean hidden state of the shape list. */
    private static void qformerContrastive(VqaSetup setup, QueryTransformer qf, int steps, SplittableRandom r) {
        Optimizer opt = Optimizer.adam(3e-3f);
        int n = setup.train.length;
        for (int s = 0; s < steps; s++) {
            int[] idx = r.ints(32, 0, n).toArray();
            int[][] caption = new int[idx.length][4];
            for (int i = 0; i < idx.length; i++) {
                int[] cls = setup.train[idx[i]].classes().clone();
                Arrays.sort(cls);
                for (int k = 0; k < Math.min(3, cls.length); k++) {
                    caption[i][1 + k] = SHAPE0 + cls[k];
                }
                caption[i][0] = Q;
            }
            Var text = Var.noGrad(() -> setup.lm.hidden(setup.lm.embed(caption), null, null, false).mean(1, false));
            qf.contrastiveLoss(Var.of(rows(setup.trainFeatures, idx)), Var.of(text.value()), 0.1f).backward();
            opt.step(qf.parameters());
        }
    }

    // ------------------------------------------------------------------ 9.4.16 Transfusion

    /**
     * Scores of one captioning-and-generation setup.
     *
     * @param method the setup
     * @param captionAccuracy the share of clean renderings captioned with the right digit
     * @param generationAccuracy the share of generated images whose nearest clean rendering
     *     shows the requested digit
     * @param distance the mean squared distance per pixel of generated images to that rendering
     * @param parameters the trainable parameters (summed over separate models)
     * @param millis training and evaluation time
     */
    public record GenerationScores(String method, double captionAccuracy, double generationAccuracy, double distance,
            long parameters, long millis) {
    }

    private static List<float[]> renders(List<Integer> labels) {
        List<float[]> out = new ArrayList<>();
        for (int d = 0; d < DigitGlyphs.CLASSES; d++) {
            for (int x = 0; x <= DigitGlyphs.SIDE - 3; x++) {
                for (int y = 0; y <= DigitGlyphs.SIDE - 5; y++) {
                    out.add(DigitGlyphs.render(d, x, y));
                    labels.add(d);
                }
            }
        }
        return out;
    }

    /** An 8x8 image as 16 row-major patches of 2x2 pixels. */
    static float[] patchify(float[] img) {
        float[] out = new float[img.length];
        for (int p = 0; p < 16; p++) {
            int px = (p % 4) * 2;
            int py = (p / 4) * 2;
            for (int k = 0; k < 4; k++) {
                out[p * 4 + k] = img[(py + k / 2) * DigitGlyphs.SIDE + px + k % 2];
            }
        }
        return out;
    }

    /** The inverse of {@link #patchify}. */
    static float[] unpatchify(float[] patches) {
        float[] out = new float[patches.length];
        for (int p = 0; p < 16; p++) {
            int px = (p % 4) * 2;
            int py = (p / 4) * 2;
            for (int k = 0; k < 4; k++) {
                out[(py + k / 2) * DigitGlyphs.SIDE + px + k % 2] = patches[p * 4 + k];
            }
        }
        return out;
    }

    private static TransfusionModel transfusion(long seed) {
        return new TransfusionModel(DigitGlyphs.CLASSES, 4, 16, DIM, 2, 4, 24, 100, seed);
    }

    private static List<TransfusionModel.Sequence> captionBatch(List<float[]> renders, List<Integer> labels, int n,
            SplittableRandom r) {
        List<TransfusionModel.Sequence> out = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            int k = r.nextInt(renders.size());
            out.add(new TransfusionModel.Sequence(new int[0], patchify(renders.get(k)), new int[] {labels.get(k)}, 50));
        }
        return out;
    }

    private static List<TransfusionModel.Sequence> generationBatch(List<float[]> renders, List<Integer> labels, int n,
            SplittableRandom r) {
        List<TransfusionModel.Sequence> out = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            int k = r.nextInt(renders.size());
            out.add(new TransfusionModel.Sequence(new int[] {labels.get(k)}, patchify(renders.get(k)), new int[0],
                    100));
        }
        return out;
    }

    private static double captionAccuracy(TransfusionModel m, List<float[]> renders, List<Integer> labels) {
        List<int[]> before = new ArrayList<>();
        float[][] images = new float[renders.size()][];
        for (int i = 0; i < images.length; i++) {
            before.add(new int[0]);
            images[i] = patchify(renders.get(i));
        }
        int[] got = m.caption(before, images);
        int hit = 0;
        for (int i = 0; i < got.length; i++) {
            hit += got[i] == labels.get(i) ? 1 : 0;
        }
        return (double) hit / got.length;
    }

    /** {share showing the requested digit, mean squared distance to the nearest rendering}. */
    private static double[] generationAccuracy(TransfusionModel m, List<float[]> renders, List<Integer> labels,
            int perDigit, long seed) {
        List<int[]> prompts = new ArrayList<>();
        for (int i = 0; i < perDigit * DigitGlyphs.CLASSES; i++) {
            prompts.add(new int[] {i % DigitGlyphs.CLASSES});
        }
        float[][] gen = m.generate(prompts, 20, new SplittableRandom(seed));
        int hit = 0;
        double dist = 0;
        for (int i = 0; i < gen.length; i++) {
            float[] img = unpatchify(gen[i]);
            int best = 0;
            double bestD = Double.MAX_VALUE;
            for (int k = 0; k < renders.size(); k++) {
                double d = 0;
                for (int p = 0; p < img.length; p++) {
                    double e = img[p] - renders.get(k)[p];
                    d += e * e;
                }
                if (d < bestD) {
                    bestD = d;
                    best = k;
                }
            }
            hit += labels.get(best) == i % DigitGlyphs.CLASSES ? 1 : 0;
            dist += bestD / img.length / gen.length;
        }
        return new double[] {(double) hit / gen.length, dist};
    }

    /** The captioning-and-generation arrangements of 9.4.16. */
    public enum Arrangement {
        /** One Transfusion model trained on both directions. */
        TRANSFUSION_SHARED("Transfusion (one model, both directions)"),
        /** A Transfusion captioner and a Transfusion generator, half the steps each. */
        TRANSFUSION_SEPARATE("Transfusion (separate captioner and generator)"),
        /** One VQ-token early-fusion decoder trained on both directions. */
        VQ_SHARED("VQ-token early fusion (one model)"),
        /** A VQ-token captioner and generator, half the steps each. */
        VQ_SEPARATE("VQ-token early fusion (separate models)");

        private final String label;

        Arrangement(String label) {
            this.label = label;
        }

        /** @return the row label */
        public String label() {
            return label;
        }
    }

    /**
     * Captioning and generation at equal steps of 32 sequences: one Transfusion model on both
     * directions (16 + 16 per step), two Transfusion models on one direction each (half the
     * steps each), and the same two arrangements of the VQ-token {@link EarlyFusion}.
     *
     * @param steps the total steps of each arrangement
     * @param perDigit generated samples per digit
     * @param seed the seed
     * @return the four rows
     */
    public static List<GenerationScores> generation(int steps, int perDigit, long seed) {
        List<GenerationScores> out = new ArrayList<>();
        for (Arrangement a : Arrangement.values()) {
            out.add(generation(a, steps, perDigit, seed));
        }
        return out;
    }

    /**
     * One arrangement of {@link #generation(int, int, long)}.
     *
     * @param arrangement the arrangement
     * @param steps its total steps of 32 sequences
     * @param perDigit generated samples per digit
     * @param seed the seed
     * @return the row
     */
    public static GenerationScores generation(Arrangement arrangement, int steps, int perDigit, long seed) {
        List<Integer> labels = new ArrayList<>();
        List<float[]> renders = renders(labels);
        long start = System.currentTimeMillis();
        SplittableRandom r = new SplittableRandom(seed);
        switch (arrangement) {
            case TRANSFUSION_SHARED -> {
                TransfusionModel both = transfusion(seed);
                Optimizer opt = Optimizer.adam(TRANSFUSION_LR);
                for (int s = 0; s < steps; s++) {
                    both.loss(captionBatch(renders, labels, 16, r), 0f, r)
                            .add(both.loss(generationBatch(renders, labels, 16, r), 1f, r)).backward();
                    opt.step(both.parameters());
                }
                double[] g = generationAccuracy(both, renders, labels, perDigit, seed + 1);
                return new GenerationScores(arrangement.label(), captionAccuracy(both, renders, labels), g[0], g[1],
                        count(both.parameters()), System.currentTimeMillis() - start);
            }
            case TRANSFUSION_SEPARATE -> {
                TransfusionModel cap = transfusion(seed + 2);
                TransfusionModel gen = transfusion(seed + 3);
                Optimizer oc = Optimizer.adam(TRANSFUSION_LR);
                Optimizer og = Optimizer.adam(TRANSFUSION_LR);
                for (int s = 0; s < steps / 2; s++) {
                    cap.loss(captionBatch(renders, labels, 32, r), 0f, r).backward();
                    oc.step(cap.parameters());
                    gen.loss(generationBatch(renders, labels, 32, r), 1f, r).backward();
                    og.step(gen.parameters());
                }
                double[] g = generationAccuracy(gen, renders, labels, perDigit, seed + 1);
                return new GenerationScores(arrangement.label(), captionAccuracy(cap, renders, labels), g[0], g[1],
                        count(cap.parameters()) + count(gen.parameters()), System.currentTimeMillis() - start);
            }
            case VQ_SHARED -> {
                EarlyFusion ef = EarlyFusion.create(seed);
                HookedDecoder vq = ef.train(EarlyFusion.Mode.BOTH, steps, seed + 4);
                double[] g = ef.generationAccuracy(vq, perDigit, seed + 1);
                return new GenerationScores(arrangement.label(), ef.captionAccuracy(vq), g[0], g[1],
                        count(vq.parameters()), System.currentTimeMillis() - start);
            }
            default -> {
                EarlyFusion ef = EarlyFusion.create(seed);
                HookedDecoder vqCap = ef.train(EarlyFusion.Mode.CAPTION, steps / 2, seed + 5);
                HookedDecoder vqGen = ef.train(EarlyFusion.Mode.GENERATE, steps / 2, seed + 6);
                double[] g = ef.generationAccuracy(vqGen, perDigit, seed + 1);
                return new GenerationScores(arrangement.label(), ef.captionAccuracy(vqCap), g[0], g[1],
                        count(vqCap.parameters()) + count(vqGen.parameters()), System.currentTimeMillis() - start);
            }
        }
    }

    /** The Transfusion models' Adam learning rate. */
    private static final float TRANSFUSION_LR = 4e-3f;

    // ------------------------------------------------------------------ 9.4.17 bottleneck fusion

    private static final int FRAMES = 12;

    /**
     * Scores of one audio-visual model.
     *
     * @param method the model
     * @param both accuracy with both modalities (for a unimodal model, with its one modality)
     * @param imageOnly accuracy with the audio missing
     * @param audioOnly accuracy with the image missing
     * @param macs multiply-adds per example
     */
    public record AvScores(String method, double both, double imageOnly, double audioOnly, long macs) {
    }

    /** Image tokens {@code [n, 16, 4]}, audio tokens {@code [n, FRAMES, BANDS]} and labels. */
    private record AvData(Tensor image, Tensor audio, Tensor labels) {
        List<Tensor> inputs() {
            return List.of(image, audio);
        }

        AvData rows(int[] idx) {
            return new AvData(FusionShootout.rows(image, idx), FusionShootout.rows(audio, idx),
                    FusionShootout.rows(labels, idx));
        }
    }

    private static AvData avData(int n, double flip, double audioNoise, SplittableRandom r) {
        float[] img = new float[n * 64];
        float[] aud = new float[n * FRAMES * TalkingDigits.BANDS];
        float[] y = new float[n];
        for (int i = 0; i < n; i++) {
            int d = r.nextInt(DigitGlyphs.CLASSES);
            y[i] = d;
            System.arraycopy(patchify(DigitGlyphs.sample(d, flip, r)), 0, img, i * 64, 64);
            float[] wave = TalkingDigits.speak(new int[] {d}, r);
            for (int k = 0; k < wave.length; k++) {
                wave[k] += (float) (audioNoise * r.nextGaussian());
            }
            float[][] f = TalkingDigits.features(wave, FRAMES);
            for (int t = 0; t < FRAMES; t++) {
                System.arraycopy(f[t], 0, aud, (i * FRAMES + t) * TalkingDigits.BANDS, TalkingDigits.BANDS);
            }
        }
        return new AvData(Tensor.of(Shape.of(n, 16, 4), img),
                Tensor.of(Shape.of(n, FRAMES, TalkingDigits.BANDS), aud), Tensor.of(Shape.of(n), y));
    }

    private static BottleneckFusion fusion(BottleneckFusion.Mode mode, long seed) {
        return new BottleneckFusion(new int[] {16, FRAMES}, new int[] {4, TalkingDigits.BANDS}, DIM, 2, 1, 2,
                DigitGlyphs.CLASSES, mode, seed);
    }

    /** Training and test examples of the audio-visual comparison. */
    private static final int AV_TRAIN = 600;
    private static final int AV_TEST = 300;

    /**
     * The audio-visual comparison.
     *
     * @param steps Adam steps of 32 examples per model
     * @param flip the image pixel flip probability
     * @param audioNoise the standard deviation of noise added to the waveform
     * @param seed the seed
     * @return bottleneck, early and late fusion without and with modality dropout (0.5), then
     *     the image-only and audio-only models
     */
    public static List<AvScores> audioVisual(int steps, double flip, double audioNoise, long seed) {
        List<AvScores> out = new ArrayList<>();
        for (BottleneckFusion.Mode mode : BottleneckFusion.Mode.values()) {
            out.addAll(audioVisual(mode, steps, flip, audioNoise, seed));
        }
        out.addAll(audioVisual(null, steps, flip, audioNoise, seed));
        return out;
    }

    /**
     * Part of the audio-visual comparison on the same data.
     *
     * @param mode the fusion, trained without and with modality dropout (0.5); null for the
     *     image-only and audio-only models
     * @param steps Adam steps of 32 examples per model
     * @param flip the image pixel flip probability
     * @param audioNoise the standard deviation of noise added to the waveform
     * @param seed the seed
     * @return two rows
     */
    public static List<AvScores> audioVisual(BottleneckFusion.Mode mode, int steps, double flip, double audioNoise,
            long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        AvData train = avData(AV_TRAIN, flip, audioNoise, r);
        AvData test = avData(AV_TEST, flip, audioNoise, r);
        List<AvScores> out = new ArrayList<>();
        if (mode != null) {
            for (double dropout : new double[] {0, 0.5}) {
                BottleneckFusion m = fusion(mode, seed + 1);
                Optimizer opt = Optimizer.adam(AV_LR);
                SplittableRandom tr = new SplittableRandom(seed + 2);
                for (int s = 0; s < steps; s++) {
                    AvData b = train.rows(tr.ints(32, 0, AV_TRAIN).toArray());
                    m.loss(b.inputs(), b.labels(), dropout, tr).backward();
                    opt.step(m.parameters());
                }
                String name = mode.name().toLowerCase(Locale.ROOT) + " fusion"
                        + (dropout > 0 ? " + modality dropout" : "");
                out.add(new AvScores(name, m.accuracy(test.inputs(), test.labels(), new boolean[] {true, true}),
                        m.accuracy(test.inputs(), test.labels(), new boolean[] {true, false}),
                        m.accuracy(test.inputs(), test.labels(), new boolean[] {false, true}), m.macs()));
            }
            return out;
        }
        for (int keep = 0; keep < 2; keep++) {
            boolean[] present = {keep == 0, keep == 1};
            BottleneckFusion m = fusion(BottleneckFusion.Mode.LATE, seed + 5 + keep);
            Optimizer opt = Optimizer.adam(AV_LR);
            SplittableRandom tr = new SplittableRandom(seed + 2);
            for (int s = 0; s < steps; s++) {
                AvData b = train.rows(tr.ints(32, 0, AV_TRAIN).toArray());
                Functional.crossEntropy(m.logits(b.inputs(), present, true), b.labels()).backward();
                opt.step(m.parameters());
                m.parameters().forEach(Var::zeroGrad);
            }
            double acc = m.accuracy(test.inputs(), test.labels(), present);
            out.add(new AvScores(keep == 0 ? "image only" : "audio only", acc, keep == 0 ? acc : Double.NaN,
                    keep == 1 ? acc : Double.NaN, m.macs() / 2));
        }
        return out;
    }

    /** The audio-visual models' Adam learning rate. */
    private static final float AV_LR = 4e-3f;
}
