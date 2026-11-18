package io.github.zzuegg.brainiac.experiments.document;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.GlyphDocuments;
import io.github.zzuegg.brainiac.experimental.data.GlyphDocuments.Line;
import io.github.zzuegg.brainiac.experimental.data.GlyphDocuments.Style;
import io.github.zzuegg.brainiac.experimental.document.LineRecognizer;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.train.Optimizer;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Wave 9 item 9.4.10: line OCR on {@link GlyphDocuments} lines (lexicon words and numbers in
 * the 3x5 font, 1-2 blank columns between glyphs, 30% of neighbouring glyphs touching, 3%
 * pixel noise, 30% of lines with a one-pixel baseline step).
 *
 * <p>Rows: the CRNN of {@link LineRecognizer} (convolutions, bidirectional GRU, CTC), the
 * same with a transformer encoder instead of the GRU, and the classic pipeline: segment the
 * line at blank columns (isolated noise pixels removed first, components wider than one
 * glyph split evenly by width, runs of three blank columns read as spaces), then classify
 * each piece with an MLP trained on single glyphs segmented the same way. Scores: character
 * error rate (edit distance over reference length), word error rate before and after lexicon
 * snapping, and the character error rate on lines with and without touching glyphs.
 */
public final class GlyphOcr {

    /** Classes of the CTC models: the blank plus every character. */
    public static final int CLASSES = GlyphDocuments.CHARSET.length() + 1;
    private static final int CROP = 5;

    private GlyphOcr() {
    }

    /** The recognisers compared. */
    public enum Method {
        /** CNN + bidirectional GRU + CTC. */
        CRNN,
        /** CNN + transformer encoder + CTC. */
        TRANSFORMER_CTC,
        /** Blank-column segmentation and a per-glyph classifier. */
        SEGMENTATION
    }

    /**
     * Scores of one recogniser.
     *
     * @param method the recogniser
     * @param cer character error rate
     * @param wer word error rate
     * @param werLexicon word error rate after snapping words to the lexicon (distance 1)
     * @param cerTouching character error rate on lines with touching glyphs
     * @param cerSeparated character error rate on the other lines
     * @param millis training and evaluation time
     */
    public record Scores(Method method, double cer, double wer, double werLexicon, double cerTouching,
            double cerSeparated, long millis) {
    }

    /**
     * Random lines.
     *
     * @param n lines
     * @param minGlyphs fewest characters
     * @param maxGlyphs most characters
     * @param style the drawing style
     * @param random the source
     * @return the lines
     */
    public static List<Line> lines(int n, int minGlyphs, int maxGlyphs, Style style, SplittableRandom random) {
        List<Line> out = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            out.add(GlyphDocuments.line(GlyphDocuments.randomText(minGlyphs, maxGlyphs, random), style, random));
        }
        return out;
    }

    /**
     * Trains a CTC line recogniser.
     *
     * @param encoder the sequence model
     * @param steps Adam steps of 16 lines
     * @param minGlyphs fewest characters per line
     * @param maxGlyphs most characters per line
     * @param seed the seed
     * @return the trained recogniser
     */
    public static LineRecognizer trainRecognizer(LineRecognizer.Encoder encoder, int steps, int minGlyphs,
            int maxGlyphs, long seed) {
        LineRecognizer rec = new LineRecognizer(GlyphDocuments.LINE_HEIGHT, CLASSES, 32, encoder, seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        SplittableRandom r = new SplittableRandom(seed + 1);
        for (int s = 0; s < steps; s++) {
            List<Line> batch = lines(16, minGlyphs, maxGlyphs, Style.hard(), r);
            int[][] labels = new int[batch.size()][];
            for (int i = 0; i < labels.length; i++) {
                labels[i] = batch.get(i).labels();
            }
            rec.loss(GlyphDocuments.batch(batch, maxWidth(batch)), labels).backward();
            opt.step(rec.parameters());
        }
        return rec;
    }

    private static int maxWidth(List<Line> lines) {
        int w = 0;
        for (Line l : lines) {
            w = Math.max(w, l.width());
        }
        return w;
    }

    /**
     * Reads lines with a CTC recogniser, in batches of 32.
     *
     * @param rec the recogniser
     * @param lines the lines
     * @return the texts
     */
    public static String[] read(LineRecognizer rec, List<Line> lines) {
        String[] out = new String[lines.size()];
        for (int from = 0; from < lines.size(); from += 32) {
            List<Line> part = lines.subList(from, Math.min(lines.size(), from + 32));
            String[] t = rec.read(GlyphDocuments.batch(part, maxWidth(part)), GlyphDocuments.CHARSET);
            System.arraycopy(t, 0, out, from, t.length);
        }
        return out;
    }

    /**
     * Runs one recogniser.
     *
     * @param method the recogniser
     * @param steps training steps (batches of 16 lines, or 64 glyphs for the segmenter)
     * @param minGlyphs fewest characters per line
     * @param maxGlyphs most characters per line
     * @param testLines held-out lines
     * @param seed the seed
     * @return the scores
     */
    public static Scores run(Method method, int steps, int minGlyphs, int maxGlyphs, int testLines, long seed) {
        long start = System.currentTimeMillis();
        List<Line> test = lines(testLines, minGlyphs, maxGlyphs, Style.hard(), new SplittableRandom(seed + 99));
        String[] hyp;
        if (method == Method.SEGMENTATION) {
            Segmenter seg = new Segmenter(seed);
            seg.train(steps, seed + 1);
            hyp = new String[test.size()];
            for (int i = 0; i < hyp.length; i++) {
                hyp[i] = seg.read(test.get(i));
            }
        } else {
            LineRecognizer rec = trainRecognizer(method == Method.CRNN ? LineRecognizer.Encoder.BIGRU
                    : LineRecognizer.Encoder.TRANSFORMER, steps, minGlyphs, maxGlyphs, seed);
            hyp = read(rec, test);
        }
        double[] errT = new double[2];
        double[] lenT = new double[2];
        double wer = 0;
        double werLex = 0;
        for (int i = 0; i < hyp.length; i++) {
            Line l = test.get(i);
            int k = l.touching() ? 0 : 1;
            errT[k] += LineRecognizer.editDistance(hyp[i], l.text());
            lenT[k] += l.text().length();
            wer += LineRecognizer.wordErrorRate(hyp[i], l.text()) / hyp.length;
            werLex += LineRecognizer.wordErrorRate(LineRecognizer.snapToLexicon(hyp[i], GlyphDocuments.LEXICON, 1),
                    l.text()) / hyp.length;
        }
        double cer = (errT[0] + errT[1]) / (lenT[0] + lenT[1]);
        return new Scores(method, cer, wer, werLex, lenT[0] == 0 ? Double.NaN : errT[0] / lenT[0],
                lenT[1] == 0 ? Double.NaN : errT[1] / lenT[1], System.currentTimeMillis() - start);
    }

    /**
     * The segmentation baseline: blank-column segmentation and a glyph classifier.
     */
    public static final class Segmenter {

        private final Model classifier;

        /**
         * Creates an untrained segmenter.
         *
         * @param seed the classifier seed
         */
        public Segmenter(long seed) {
            this.classifier = Model.sequential().input(GlyphDocuments.LINE_HEIGHT * CROP).dense(64, Activation.RELU)
                    .dense(GlyphDocuments.CHARSET.length() - 1, Activation.IDENTITY).seed(seed).build();
        }

        /**
         * Trains the classifier on single glyphs rendered in the hard style and segmented as
         * at test time (the largest component).
         *
         * @param steps Adam steps of 64 glyphs
         * @param seed the seed
         */
        public void train(int steps, long seed) {
            Optimizer opt = Optimizer.adam(3e-3f);
            SplittableRandom r = new SplittableRandom(seed);
            int f = GlyphDocuments.LINE_HEIGHT * CROP;
            for (int s = 0; s < steps; s++) {
                float[] x = new float[64 * f];
                float[] y = new float[64];
                for (int i = 0; i < 64; i++) {
                    int c = 1 + r.nextInt(GlyphDocuments.CHARSET.length() - 1);
                    Line l = GlyphDocuments.line(String.valueOf(GlyphDocuments.CHARSET.charAt(c)), Style.hard(), r);
                    float[] clean = denoise(l);
                    int[] best = {0, 0};
                    for (int[] p : pieces(clean, l.width())) {
                        if (p[0] >= 0 && p[1] - p[0] > best[1] - best[0]) {
                            best = p;
                        }
                    }
                    System.arraycopy(crop(clean, l.width(), best[0], best[1]), 0, x, i * f, f);
                    y[i] = c - 1;
                }
                Functional.crossEntropy(classifier.forward(Var.of(Tensor.of(Shape.of(64, f), x)), true),
                        Tensor.of(Shape.of(64), y)).backward();
                opt.step(classifier.parameters());
            }
        }

        /**
         * Reads a line.
         *
         * @param line the line
         * @return the text
         */
        public String read(Line line) {
            float[] clean = denoise(line);
            List<int[]> ps = pieces(clean, line.width());
            StringBuilder sb = new StringBuilder();
            List<float[]> crops = new ArrayList<>();
            List<Integer> where = new ArrayList<>();
            for (int[] p : ps) {
                if (p[0] < 0) {
                    sb.append(' ');
                } else {
                    crops.add(crop(clean, line.width(), p[0], p[1]));
                    where.add(sb.length());
                    sb.append('?');
                }
            }
            if (!crops.isEmpty()) {
                int f = GlyphDocuments.LINE_HEIGHT * CROP;
                float[] x = new float[crops.size() * f];
                for (int i = 0; i < crops.size(); i++) {
                    System.arraycopy(crops.get(i), 0, x, i * f, f);
                }
                float[] pred = Var.noGrad(() -> classifier.forward(Var.of(Tensor.of(Shape.of(crops.size(), f), x))))
                        .value().argmax(1).toArray();
                for (int i = 0; i < pred.length; i++) {
                    sb.setCharAt(where.get(i), GlyphDocuments.CHARSET.charAt((int) pred[i] + 1));
                }
            }
            return sb.toString().trim().replaceAll(" +", " ");
        }

        /** Removes ink pixels with no other ink within two pixels. */
        private static float[] denoise(Line l) {
            int h = GlyphDocuments.LINE_HEIGHT;
            int w = l.width();
            float[] p = l.pixels();
            float[] out = new float[p.length];
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    if (p[y * w + x] < 0.5f) {
                        continue;
                    }
                    boolean near = false;
                    for (int dy = -2; dy <= 2 && !near; dy++) {
                        for (int dx = -2; dx <= 2; dx++) {
                            int yy = y + dy;
                            int xx = x + dx;
                            if ((dx != 0 || dy != 0) && yy >= 0 && yy < h && xx >= 0 && xx < w
                                    && p[yy * w + xx] > 0.5f) {
                                near = true;
                                break;
                            }
                        }
                    }
                    out[y * w + x] = near ? 1 : 0;
                }
            }
            return out;
        }

        /** Column ranges {@code [from, to)} of glyph pieces in order; {@code {-1, -1}} is a space. */
        private static List<int[]> pieces(float[] img, int w) {
            int h = GlyphDocuments.LINE_HEIGHT;
            boolean[] ink = new boolean[w];
            for (int x = 0; x < w; x++) {
                for (int y = 0; y < h; y++) {
                    ink[x] |= img[y * w + x] > 0.5f;
                }
            }
            List<int[]> out = new ArrayList<>();
            int x = 0;
            int lastEnd = -1;
            while (x < w) {
                if (!ink[x]) {
                    x++;
                    continue;
                }
                int from = x;
                while (x < w && ink[x]) {
                    x++;
                }
                if (lastEnd >= 0 && from - lastEnd >= 3) {
                    out.add(new int[] {-1, -1});
                }
                lastEnd = x;
                int width = x - from;
                int n = (int) Math.max(1, Math.round((width + 1) / 4.0));
                for (int i = 0; i < n; i++) {
                    out.add(new int[] {from + i * width / n, from + (i + 1) * width / n});
                }
            }
            return out;
        }

        /** The piece centred in a {@code LINE_HEIGHT x CROP} window. */
        private static float[] crop(float[] img, int w, int from, int to) {
            int h = GlyphDocuments.LINE_HEIGHT;
            float[] out = new float[h * CROP];
            int width = Math.min(CROP, to - from);
            int off = (CROP - width) / 2;
            for (int y = 0; y < h; y++) {
                for (int i = 0; i < width; i++) {
                    out[y * CROP + off + i] = img[y * w + from + i];
                }
            }
            return out;
        }
    }
}
