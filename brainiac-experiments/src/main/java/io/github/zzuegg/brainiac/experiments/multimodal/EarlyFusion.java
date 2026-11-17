package io.github.zzuegg.brainiac.experiments.multimodal;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.llm.HookedDecoder;
import io.github.zzuegg.brainiac.experimental.nn.PatchVqTokenizer;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Chameleon-style early fusion (Chameleon Team, <a href="https://arxiv.org/abs/2405.09818">2405.09818</a>):
 * images become discrete tokens ({@link PatchVqTokenizer}, 2x2 tiles of an 8x8
 * {@link DigitGlyphs} image, 16 tokens) in the same vocabulary as text, and one causal
 * transformer ({@link HookedDecoder}) models mixed sequences in both directions:
 *
 * <pre>
 * captioning:  [IMG] i1 ... i16 [TXT] digit
 * generation:  [TXT] digit [IMG] i1 ... i16
 * </pre>
 *
 * <p>The baseline is two separate models of the same size, one per direction, each trained
 * on half the steps, so the total compute and the examples per direction are equal.
 * Captioning is scored by the accuracy of the predicted digit token; generation by whether
 * the decoded sample's nearest clean rendering shows the requested digit.
 */
@SuppressWarnings("exports")
public final class EarlyFusion {
    private final PatchVqTokenizer tokenizer;
    private final int codes;
    private final int textBase;
    private final int imgToken;
    private final int txtToken;
    private final int vocab;
    private final List<float[]> renders = new ArrayList<>();
    private final List<Integer> renderLabels = new ArrayList<>();

    private EarlyFusion(long seed) {
        tokenizer = PatchVqTokenizer.of(DigitGlyphs.SIDE, 2, 16, seed);
        for (int d = 0; d < DigitGlyphs.CLASSES; d++) {
            for (int x = 0; x <= DigitGlyphs.SIDE - 3; x++) {
                for (int y = 0; y <= DigitGlyphs.SIDE - 5; y++) {
                    renders.add(DigitGlyphs.render(d, x, y));
                    renderLabels.add(d);
                }
            }
        }
        tokenizer.fit(renders, 10);
        codes = tokenizer.vocabulary();
        textBase = codes;
        imgToken = codes + DigitGlyphs.CLASSES;
        txtToken = imgToken + 1;
        vocab = txtToken + 1;
    }

    /**
     * Creates the experiment and fits the image tokenizer on every clean rendering.
     *
     * @param seed the seed
     * @return the experiment
     */
    public static EarlyFusion create(long seed) {
        return new EarlyFusion(seed);
    }

    /** Which directions a model is trained on. */
    public enum Mode {
        /** Image to digit only. */
        CAPTION,
        /** Digit to image only. */
        GENERATE,
        /** Both, in one model. */
        BOTH
    }

    private int[] captionSequence(int[] image, int digit) {
        int[] s = new int[image.length + 3];
        s[0] = imgToken;
        System.arraycopy(image, 0, s, 1, image.length);
        s[image.length + 1] = txtToken;
        s[image.length + 2] = textBase + digit;
        return s;
    }

    private int[] generationSequence(int[] image, int digit) {
        int[] s = new int[image.length + 3];
        s[0] = txtToken;
        s[1] = textBase + digit;
        s[2] = imgToken;
        System.arraycopy(image, 0, s, 3, image.length);
        return s;
    }

    /**
     * Trains a fresh model.
     *
     * @param mode the directions
     * @param steps optimizer steps of 32 sequences
     * @param seed the seed
     * @return the model
     */
    public HookedDecoder train(Mode mode, int steps, long seed) {
        HookedDecoder model = HookedDecoder.create(HookedDecoder.Settings.tiny(vocab), seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 1);
        for (int s = 0; s < steps; s++) {
            int[][] batch = new int[32][];
            for (int i = 0; i < batch.length; i++) {
                int k = r.nextInt(renders.size());
                int[] image = tokenizer.encode(renders.get(k));
                boolean caption = mode == Mode.CAPTION || mode == Mode.BOTH && i % 2 == 0;
                batch[i] = caption ? captionSequence(image, renderLabels.get(k))
                        : generationSequence(image, renderLabels.get(k));
            }
            Var loss = model.loss(batch, true);
            loss.backward();
            opt.step(model.parameters());
            model.zeroGrad();
        }
        return model;
    }

    /**
     * Captioning accuracy: the most likely digit token after {@code [TXT]}, on noisy-free
     * renderings at every position.
     *
     * @param model the model
     * @return the share of correct digits
     */
    public double captionAccuracy(HookedDecoder model) {
        int n = renders.size();
        int[][] in = new int[n][];
        for (int k = 0; k < n; k++) {
            int[] s = captionSequence(tokenizer.encode(renders.get(k)), 0);
            in[k] = java.util.Arrays.copyOf(s, s.length - 1);
        }
        float[] logits = model.forward(in).logits().value().toArray();
        int len = in[0].length;
        int hit = 0;
        for (int k = 0; k < n; k++) {
            int off = ((k + 1) * len - 1) * vocab;
            int best = 0;
            for (int d = 1; d < DigitGlyphs.CLASSES; d++) {
                if (logits[off + textBase + d] > logits[off + textBase + best]) {
                    best = d;
                }
            }
            hit += best == renderLabels.get(k) ? 1 : 0;
        }
        return (double) hit / n;
    }

    /**
     * Generation accuracy: samples images of every digit and checks the digit of the nearest
     * clean rendering.
     *
     * @param model the model
     * @param perDigit samples per digit
     * @param seed the sampling seed
     * @return {@code {share showing the requested digit, mean squared distance per pixel to the nearest rendering}}
     */
    public double[] generationAccuracy(HookedDecoder model, int perDigit, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        int n = perDigit * DigitGlyphs.CLASSES;
        int len = tokenizer.tokens();
        int[][] seq = new int[n][3 + len];
        for (int i = 0; i < n; i++) {
            seq[i][0] = txtToken;
            seq[i][1] = textBase + i % DigitGlyphs.CLASSES;
            seq[i][2] = imgToken;
        }
        for (int pos = 0; pos < len; pos++) {
            int[][] in = new int[n][];
            for (int i = 0; i < n; i++) {
                in[i] = java.util.Arrays.copyOf(seq[i], 3 + pos);
            }
            float[] logits = model.forward(in).logits().value().toArray();
            int width = 3 + pos;
            for (int i = 0; i < n; i++) {
                int off = ((i + 1) * width - 1) * vocab;
                double max = Double.NEGATIVE_INFINITY;
                for (int c = 0; c < codes; c++) {
                    max = Math.max(max, logits[off + c]);
                }
                double[] p = new double[codes];
                double sum = 0;
                for (int c = 0; c < codes; c++) {
                    p[c] = Math.exp(logits[off + c] - max);
                    sum += p[c];
                }
                double u = r.nextDouble() * sum;
                int pick = codes - 1;
                for (int c = 0; c < codes; c++) {
                    u -= p[c];
                    if (u <= 0) {
                        pick = c;
                        break;
                    }
                }
                seq[i][3 + pos] = pick;
            }
        }
        int hit = 0;
        double dist = 0;
        for (int i = 0; i < n; i++) {
            float[] img = tokenizer.decode(java.util.Arrays.copyOfRange(seq[i], 3, 3 + len));
            double best = Double.MAX_VALUE;
            int label = -1;
            for (int k = 0; k < renders.size(); k++) {
                double d = 0;
                float[] ref = renders.get(k);
                for (int j = 0; j < img.length; j++) {
                    d += (img[j] - ref[j]) * (img[j] - ref[j]);
                }
                if (d < best) {
                    best = d;
                    label = renderLabels.get(k);
                }
            }
            hit += label == i % DigitGlyphs.CLASSES ? 1 : 0;
            dist += best / img.length;
        }
        return new double[] {(double) hit / n, dist / n};
    }
}
