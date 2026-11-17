package io.github.zzuegg.brainiac.experiments.generative;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.nn.PatchVqTokenizer;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Model;
import io.github.zzuegg.brainiac.nn.TransformerBlock;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.random.RandomGenerator;

/** Token transformers and image-token scoring shared by the token-based image generators. */
final class TokenModels {
    private TokenModels() {
    }

    /** A transformer from {@code [batch, length]} ids to {@code [batch, length, vocab]} logits. */
    static Model transformer(int length, int vocab, int width, int layers, boolean causal, long seed) {
        Model.Builder b = Model.sequential().input(length).embedding(vocab, width).positionalEncoding();
        for (int i = 0; i < layers; i++) {
            b.add(TransformerBlock.of(2).causal(causal).rotary(100f));
        }
        return b.dense(vocab).seed(seed).build();
    }

    static Tensor ids(int[][] rows) {
        int len = rows[0].length;
        float[] f = new float[rows.length * len];
        for (int i = 0; i < rows.length; i++) {
            for (int j = 0; j < len; j++) {
                f[i * len + j] = rows[i][j];
            }
        }
        return Tensor.of(Shape.of(rows.length, len), f);
    }

    /** Next-token loss of a causal model: the input is the sequence shifted right behind {@code bos}. */
    static Var autoregressiveLoss(Model model, int[][] seqs, int bos, int vocab) {
        int b = seqs.length;
        int len = seqs[0].length;
        int[][] in = new int[b][len];
        float[] target = new float[b * len];
        for (int i = 0; i < b; i++) {
            for (int j = 0; j < len; j++) {
                in[i][j] = j == 0 ? bos : seqs[i][j - 1];
                target[i * len + j] = seqs[i][j];
            }
        }
        Var logits = model.forward(Var.of(ids(in)), true);
        return Functional.crossEntropy(logits.reshape(b * len, vocab), Tensor.of(Shape.of(b * len), target));
    }

    /** Samples from a causal model one token per network call; tokens at or above {@code limit} are never drawn. */
    static int[][] sampleAutoregressive(Model model, int count, int len, int bos, int vocab, int limit,
            float temperature, RandomGenerator random) {
        int[][] seq = new int[count][len + 1];
        for (int[] s : seq) {
            s[0] = bos;
        }
        for (int pos = 0; pos < len; pos++) {
            int[][] in = new int[count][len];
            for (int i = 0; i < count; i++) {
                for (int j = 0; j < len; j++) {
                    in[i][j] = j <= pos ? seq[i][j] : bos;
                }
            }
            float[] logits = model.forward(Var.of(ids(in)), false).value().toArray();
            for (int i = 0; i < count; i++) {
                seq[i][pos + 1] = draw(logits, (i * len + pos) * vocab, limit, temperature, random);
            }
        }
        int[][] out = new int[count][];
        for (int i = 0; i < count; i++) {
            out[i] = java.util.Arrays.copyOfRange(seq[i], 1, len + 1);
        }
        return out;
    }

    static int draw(float[] logits, int offset, int limit, float temperature, RandomGenerator random) {
        double max = Double.NEGATIVE_INFINITY;
        for (int v = 0; v < limit; v++) {
            max = Math.max(max, logits[offset + v] / temperature);
        }
        double[] p = new double[limit];
        double sum = 0;
        for (int v = 0; v < limit; v++) {
            p[v] = Math.exp(logits[offset + v] / temperature - max);
            sum += p[v];
        }
        double u = random.nextDouble() * sum;
        for (int v = 0; v < limit; v++) {
            u -= p[v];
            if (u <= 0) {
                return v;
            }
        }
        return limit - 1;
    }

    /** A tokenizer for the 8x8 {@link ToyImages}: 2x2 tiles, 16 codes, fitted on every image. */
    static PatchVqTokenizer toyImageTokenizer(long seed) {
        PatchVqTokenizer t = PatchVqTokenizer.of(ToyImages.SIDE, 2, 16, seed);
        List<float[]> all = new ArrayList<>();
        for (int c = 0; c < ToyImages.CLASSES; c++) {
            all.addAll(ToyImages.all(c));
        }
        t.fit(all, 10);
        return t;
    }

    /**
     * Scores generated images against the enumerable data set.
     *
     * @return {@code {mean squared distance per pixel to the nearest real image, share that is
     *     a real image, distinct real images produced, classes covered}}
     */
    static double[] score(List<float[]> images) {
        double dist = 0;
        int exact = 0;
        Set<String> distinct = new HashSet<>();
        Set<Integer> classes = new HashSet<>();
        for (float[] img : images) {
            double[] n = ToyImages.nearest(img);
            dist += n[1] / ToyImages.PIXELS;
            if (n[1] < 0.5) {
                exact++;
                classes.add((int) n[0]);
                StringBuilder key = new StringBuilder();
                for (float v : img) {
                    key.append(v > 0 ? '1' : '0');
                }
                distinct.add(key.toString());
            }
        }
        return new double[] {dist / images.size(), (double) exact / images.size(), distinct.size(), classes.size()};
    }
}
