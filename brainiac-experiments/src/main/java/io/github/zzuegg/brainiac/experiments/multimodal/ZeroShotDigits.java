package io.github.zzuegg.brainiac.experiments.multimodal;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.ContrastivePairLoss;
import io.github.zzuegg.brainiac.experimental.nn.SelfSupervisedLosses;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
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
 * CLIP and SigLIP at toy scale: an image tower (MLP over 8x8 {@link DigitGlyphs}) and a text
 * tower (MLP over the bag of words of a caption) trained on image-caption pairs with a
 * {@link ContrastivePairLoss}. No class labels are ever given; the classes exist only as
 * words. Zero-shot classification embeds one caption per class and picks the closest; it
 * uses the phrasing "the digit 7", which never appeared in training (only "seven",
 * "digit seven", "number 7" and "a seven" did).
 *
 * <p>The baseline is the usual way to get a classifier from unlabeled images: SimCLR
 * ({@code SelfSupervisedLosses.ntXent}) on augmented views (one-pixel shifts and pixel
 * noise) of the same image tower, then a linear probe trained on a few labeled images per
 * class.
 */
public final class ZeroShotDigits {
    private static final int EMBED = 32;
    private static final double NOISE = 0.05;
    private static final int TRAIN_TEMPLATES = 4;
    private static final int ZERO_SHOT_TEMPLATE = 4;
    private final Map<String, Integer> words = new LinkedHashMap<>();

    private ZeroShotDigits() {
        for (int d = 0; d < DigitGlyphs.CLASSES; d++) {
            for (String c : DigitGlyphs.captions(d)) {
                for (String w : c.split(" ")) {
                    words.putIfAbsent(w, words.size());
                }
            }
        }
    }

    /**
     * Creates the experiment.
     *
     * @return the experiment
     */
    public static ZeroShotDigits create() {
        return new ZeroShotDigits();
    }

    private static Model imageTower(long seed) {
        return Model.sequential().input(DigitGlyphs.PIXELS).dense(128, Activation.SILU).dense(EMBED).seed(seed).build();
    }

    private Tensor bagOfWords(List<String> captions) {
        float[] f = new float[captions.size() * words.size()];
        for (int i = 0; i < captions.size(); i++) {
            String[] ws = captions.get(i).split(" ");
            for (String w : ws) {
                f[i * words.size() + words.get(w)] += 1f / ws.length;
            }
        }
        return Tensor.of(Shape.of(captions.size(), words.size()), f);
    }

    private static Tensor images(List<float[]> rows) {
        float[] f = new float[rows.size() * DigitGlyphs.PIXELS];
        for (int i = 0; i < rows.size(); i++) {
            System.arraycopy(rows.get(i), 0, f, i * DigitGlyphs.PIXELS, DigitGlyphs.PIXELS);
        }
        return Tensor.of(Shape.of(rows.size(), DigitGlyphs.PIXELS), f);
    }

    /** A labeled test set: {@code images, labels}. */
    private static Object[] testSet(int n, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        List<float[]> imgs = new ArrayList<>();
        int[] labels = new int[n];
        for (int i = 0; i < n; i++) {
            labels[i] = r.nextInt(DigitGlyphs.CLASSES);
            imgs.add(DigitGlyphs.sample(labels[i], NOISE, r));
        }
        return new Object[] {imgs, labels};
    }

    private static double accuracy(float[] scores, int[] labels, int classes) {
        int hit = 0;
        for (int i = 0; i < labels.length; i++) {
            int best = 0;
            for (int c = 1; c < classes; c++) {
                if (scores[i * classes + c] > scores[i * classes + best]) {
                    best = c;
                }
            }
            hit += best == labels[i] ? 1 : 0;
        }
        return (double) hit / labels.length;
    }

    /**
     * Trains the two towers on image-caption pairs and classifies held-out images zero-shot.
     *
     * @param loss the contrastive loss (CLIP or SigLIP)
     * @param steps optimizer steps of 64 pairs
     * @param seed the seed
     * @return the zero-shot accuracy on 500 test images, with the unseen caption phrasing
     */
    @SuppressWarnings("unchecked")
    public double contrastive(ContrastivePairLoss loss, int steps, long seed) {
        Model image = imageTower(seed);
        Model text = Model.sequential().input(words.size()).dense(64, Activation.SILU).dense(EMBED).seed(seed + 1)
                .build();
        List<Var> params = new ArrayList<>(image.parameters());
        params.addAll(text.parameters());
        params.addAll(loss.parameters());
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 2);
        for (int s = 0; s < steps; s++) {
            List<float[]> imgs = new ArrayList<>();
            List<String> caps = new ArrayList<>();
            for (int i = 0; i < 64; i++) {
                int d = r.nextInt(DigitGlyphs.CLASSES);
                imgs.add(DigitGlyphs.sample(d, NOISE, r));
                caps.add(DigitGlyphs.captions(d)[r.nextInt(TRAIN_TEMPLATES)]);
            }
            Var l = loss.compute(image.forward(Var.of(images(imgs)), true), text.forward(Var.of(bagOfWords(caps)), true));
            l.backward();
            opt.step(params);
        }
        Object[] test = testSet(500, seed + 3);
        List<String> prompts = new ArrayList<>();
        for (int d = 0; d < DigitGlyphs.CLASSES; d++) {
            prompts.add(DigitGlyphs.captions(d)[ZERO_SHOT_TEMPLATE]);
        }
        float[] scores = loss.logits(image.forward(Var.of(images((List<float[]>) test[0]))),
                text.forward(Var.of(bagOfWords(prompts)))).value().toArray();
        return accuracy(scores, (int[]) test[1], DigitGlyphs.CLASSES);
    }

    private static float[] shift(float[] img, int dx, int dy) {
        float[] out = new float[img.length];
        java.util.Arrays.fill(out, -1);
        for (int y = 0; y < DigitGlyphs.SIDE; y++) {
            for (int x = 0; x < DigitGlyphs.SIDE; x++) {
                int sx = x - dx;
                int sy = y - dy;
                if (sx >= 0 && sy >= 0 && sx < DigitGlyphs.SIDE && sy < DigitGlyphs.SIDE) {
                    out[y * DigitGlyphs.SIDE + x] = img[sy * DigitGlyphs.SIDE + sx];
                }
            }
        }
        return out;
    }

    private static float[] augment(float[] img, SplittableRandom r) {
        float[] out = shift(img, r.nextInt(3) - 1, r.nextInt(3) - 1);
        for (int i = 0; i < out.length; i++) {
            if (r.nextDouble() < NOISE) {
                out[i] = -out[i];
            }
        }
        return out;
    }

    /**
     * The baseline: SimCLR on unlabeled images, then a linear probe on a few labeled ones.
     *
     * @param steps SimCLR steps of 64 images
     * @param labelsPerClass labeled images per class for the probe
     * @param seed the seed
     * @return the probe's accuracy on the same 500 test images
     */
    @SuppressWarnings("unchecked")
    public double simclrLinearProbe(int steps, int labelsPerClass, long seed) {
        Model image = imageTower(seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 2);
        for (int s = 0; s < steps; s++) {
            List<float[]> a = new ArrayList<>();
            List<float[]> b = new ArrayList<>();
            for (int i = 0; i < 64; i++) {
                float[] img = DigitGlyphs.sample(r.nextInt(DigitGlyphs.CLASSES), 0, r);
                a.add(augment(img, r));
                b.add(augment(img, r));
            }
            Var l = SelfSupervisedLosses.ntXent(image.forward(Var.of(images(a)), true),
                    image.forward(Var.of(images(b)), true), 0.2f);
            l.backward();
            opt.step(image.parameters());
        }
        List<float[]> labeled = new ArrayList<>();
        float[] ids = new float[labelsPerClass * DigitGlyphs.CLASSES];
        for (int d = 0; d < DigitGlyphs.CLASSES; d++) {
            for (int k = 0; k < labelsPerClass; k++) {
                ids[labeled.size()] = d;
                labeled.add(DigitGlyphs.sample(d, NOISE, r));
            }
        }
        Tensor features = image.forward(Var.of(images(labeled))).value();
        Model probe = Model.sequential().input(EMBED).dense(DigitGlyphs.CLASSES).seed(seed + 4).build();
        Optimizer po = Optimizer.adam(1e-2f);
        for (int s = 0; s < 300; s++) {
            Var l = Functional.crossEntropy(probe.forward(Var.of(features), true), Tensor.of(Shape.of(ids.length), ids));
            l.backward();
            po.step(probe.parameters());
        }
        Object[] test = testSet(500, seed + 3);
        float[] scores = probe.forward(Var.of(image.forward(Var.of(images((List<float[]>) test[0]))).value()))
                .value().toArray();
        return accuracy(scores, (int[]) test[1], DigitGlyphs.CLASSES);
    }
}
