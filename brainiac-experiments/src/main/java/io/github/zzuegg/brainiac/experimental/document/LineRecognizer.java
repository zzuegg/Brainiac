package io.github.zzuegg.brainiac.experimental.document;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.CtcLoss;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Recurrent;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Line OCR without character segmentation: a convolutional-recurrent network trained with
 * CTC (CRNN, Shi et al., TPAMI 2017). Two 3x3 convolutions turn a text-line image
 * {@code [batch, 1, height, width]} into one feature vector per pixel column, a projection
 * and a bidirectional GRU (core {@link Recurrent}) add left and right context, and a linear
 * layer emits per-column class scores (class 0 is the CTC blank), trained with
 * {@link CtcLoss} so the alignment between columns and characters is never labelled.
 *
 * <p>Decoding is best path (most likely class per column, repeats merged, blanks dropped)
 * or lexicon-constrained as in the paper: each decoded word is replaced by the lexicon word
 * at the smallest edit distance when that distance is small enough.
 *
 * <p>With {@link Encoder#TRANSFORMER} the GRU is replaced by two non-causal core
 * {@link TransformerBlock}s over the columns with sinusoidal positions, the baseline that
 * trades recurrence for attention.
 *
 * <p>As a {@link Layer} it maps line images to the column logits {@code [batch, width, classes]}.
 *
 * <p>Paper: <a href="https://arxiv.org/abs/1507.05717">arXiv:1507.05717</a>.
 */
public final class LineRecognizer implements Layer {

    /** The sequence model on top of the convolutions. */
    public enum Encoder {
        /** A bidirectional GRU (CRNN). */
        BIGRU,
        /** Two transformer encoder blocks. */
        TRANSFORMER
    }

    private final int height;
    private final int classes;
    private final int hidden;
    private final Encoder encoder;
    private final Var conv1;
    private final Var conv1b;
    private final Var conv2;
    private final Var conv2b;
    private final Var project;
    private final Var projectB;
    private final Layer rnn;
    private final List<Layer> blocks = new ArrayList<>();
    private final Var out;
    private final Var outB;

    /**
     * Creates the recogniser.
     *
     * @param height the line height in pixels
     * @param classes output classes including the blank (0)
     * @param hidden the sequence model's width (even)
     * @param encoder the sequence model
     * @param seed the seed
     */
    public LineRecognizer(int height, int classes, int hidden, Encoder encoder, long seed) {
        this.height = height;
        this.classes = classes;
        this.hidden = hidden;
        this.encoder = encoder;
        SplittableRandom r = new SplittableRandom(seed);
        this.conv1 = Var.param(Tensor.normal(Shape.of(16, 1, 3, 3), 0, (float) Math.sqrt(2.0 / 9), r));
        this.conv1b = Var.param(Tensor.zeros(Shape.of(16)));
        this.conv2 = Var.param(Tensor.normal(Shape.of(16, 16, 3, 3), 0, (float) Math.sqrt(2.0 / 144), r));
        this.conv2b = Var.param(Tensor.zeros(Shape.of(16)));
        int features = 16 * height;
        this.project = Var.param(Tensor.normal(Shape.of(features, hidden), 0, (float) Math.sqrt(2.0 / features), r));
        this.projectB = Var.param(Tensor.zeros(Shape.of(hidden)));
        if (encoder == Encoder.BIGRU) {
            this.rnn = Recurrent.gru(hidden / 2).returnSequences(true).bidirectional(Recurrent.Merge.CONCAT)
                    .create(Shape.of(8, hidden), r);
        } else {
            this.rnn = null;
            for (int i = 0; i < 2; i++) {
                blocks.add(TransformerBlock.of(4).ffnSize(2 * hidden).create(Shape.of(8, hidden), r));
            }
        }
        this.out = Var.param(Tensor.normal(Shape.of(hidden, classes), 0, (float) Math.sqrt(1.0 / hidden), r));
        this.outB = Var.param(Tensor.zeros(Shape.of(classes)));
    }

    private static Var same(Var x, Var w, Var b) {
        return Functional.conv(x, w, b, new int[] {1, 1}, new int[] {1, 1}, new int[] {1, 1}, new int[] {1, 1}, 1);
    }

    @Override
    public Var forward(Var images) {
        return forward(images, false);
    }

    @Override
    public Var forward(Var images, boolean training) {
        Shape s = images.shape();
        if (s.rank() != 4 || s.dim(1) != 1 || s.dim(2) != height) {
            throw new IllegalArgumentException("expects [batch, 1, " + height + ", width], got " + s);
        }
        int b = s.dim(0);
        int w = s.dim(3);
        Var h = same(same(images, conv1, conv1b).relu(), conv2, conv2b).relu();
        Var cols = h.permute(0, 3, 1, 2).reshape(b, w, 16 * height).matmul(project).add(projectB).relu();
        Var seq;
        if (encoder == Encoder.BIGRU) {
            seq = rnn.forward(cols, training);
        } else {
            seq = cols.add(Functional.sinusoidal(w, hidden, 10000f, cols.value().device()));
            for (Layer blk : blocks) {
                seq = blk.forward(seq, training);
            }
        }
        return seq.matmul(out).add(outB);
    }

    @Override
    public Shape outputShape(Shape input) {
        return Shape.of(input.dim(input.rank() - 1), classes);
    }

    /**
     * The CTC loss of a batch.
     *
     * @param images {@code [batch, 1, height, width]}
     * @param labels per line its label ids {@code 1..classes-1}
     * @return the mean loss
     */
    public Var loss(Tensor images, int[][] labels) {
        return CtcLoss.loss(forward(Var.of(images), true), labels);
    }

    /**
     * Best-path decoding.
     *
     * @param images {@code [batch, 1, height, width]}
     * @return per line the decoded label ids
     */
    public int[][] decode(Tensor images) {
        return CtcLoss.greedyDecode(Var.noGrad(() -> forward(Var.of(images))).value());
    }

    /**
     * Best-path decoding to text.
     *
     * @param images {@code [batch, 1, height, width]}
     * @param alphabet the characters of labels {@code 1..}: label {@code i} is
     *     {@code alphabet.charAt(i - 1)}
     * @return per line the text
     */
    public String[] read(Tensor images, String alphabet) {
        int[][] ids = decode(images);
        String[] out = new String[ids.length];
        for (int i = 0; i < ids.length; i++) {
            StringBuilder sb = new StringBuilder();
            for (int c : ids[i]) {
                if (c >= 1 && c <= alphabet.length()) {
                    sb.append(alphabet.charAt(c - 1));
                }
            }
            out[i] = sb.toString().trim().replaceAll(" +", " ");
        }
        return out;
    }

    /**
     * Lexicon-constrained correction: every word is replaced by the nearest lexicon word
     * when the edit distance is at most {@code maxDistance}; numbers are left as read.
     *
     * @param text decoded text
     * @param lexicon the allowed words
     * @param maxDistance the largest edit distance corrected
     * @return the corrected text
     */
    public static String snapToLexicon(String text, String[] lexicon, int maxDistance) {
        String[] words = text.split(" ");
        StringBuilder sb = new StringBuilder();
        for (String w : words) {
            String best = w;
            if (!w.isEmpty() && !w.chars().allMatch(Character::isDigit)) {
                int bestD = maxDistance + 1;
                for (String l : lexicon) {
                    int d = editDistance(w, l);
                    if (d < bestD) {
                        bestD = d;
                        best = l;
                    }
                }
            }
            if (!sb.isEmpty()) {
                sb.append(' ');
            }
            sb.append(best);
        }
        return sb.toString();
    }

    /**
     * Levenshtein distance.
     *
     * @param a one string
     * @param b another
     * @return the fewest insertions, deletions and substitutions turning {@code a} into {@code b}
     */
    public static int editDistance(String a, String b) {
        int[] prev = new int[b.length() + 1];
        int[] cur = new int[b.length() + 1];
        for (int j = 0; j <= b.length(); j++) {
            prev[j] = j;
        }
        for (int i = 1; i <= a.length(); i++) {
            cur[0] = i;
            for (int j = 1; j <= b.length(); j++) {
                int sub = prev[j - 1] + (a.charAt(i - 1) == b.charAt(j - 1) ? 0 : 1);
                cur[j] = Math.min(sub, Math.min(prev[j] + 1, cur[j - 1] + 1));
            }
            int[] t = prev;
            prev = cur;
            cur = t;
        }
        return prev[b.length()];
    }

    /**
     * Word error rate between two texts: the word-level edit distance over the reference's
     * word count.
     *
     * @param hypothesis the decoded text
     * @param reference the true text
     * @return the word error rate
     */
    public static double wordErrorRate(String hypothesis, String reference) {
        String[] h = hypothesis.isEmpty() ? new String[0] : hypothesis.split(" ");
        String[] r = reference.split(" ");
        int[][] d = new int[h.length + 1][r.length + 1];
        for (int i = 0; i <= h.length; i++) {
            d[i][0] = i;
        }
        for (int j = 0; j <= r.length; j++) {
            d[0][j] = j;
        }
        for (int i = 1; i <= h.length; i++) {
            for (int j = 1; j <= r.length; j++) {
                d[i][j] = Math.min(d[i - 1][j - 1] + (h[i - 1].equals(r[j - 1]) ? 0 : 1),
                        Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1));
            }
        }
        return (double) d[h.length][r.length] / r.length;
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(List.of(conv1, conv1b, conv2, conv2b, project, projectB));
        if (rnn != null) {
            p.addAll(rnn.parameters());
        }
        for (Layer blk : blocks) {
            p.addAll(blk.parameters());
        }
        p.add(out);
        p.add(outB);
        return p;
    }

    @Override
    public String toString() {
        return "LineRecognizer(" + encoder + ", " + hidden + " wide)";
    }
}
