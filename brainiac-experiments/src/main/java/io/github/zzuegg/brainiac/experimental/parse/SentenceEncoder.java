package io.github.zzuegg.brainiac.experimental.parse;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.Recurrent;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Word encoders for the parsers: a bidirectional LSTM, or Kitaev and Klein's self-attentive
 * encoder whose content and position streams are kept apart, so each attention score is
 * {@code q_c . k_c + q_p . k_p} (content-to-content plus position-to-position) and each stream is
 * updated only from its own values. Position embeddings are fixed sinusoids, so the encoder runs
 * on sentences longer than any it was trained on. The output is {@code [batch, steps, width]};
 * for the BiLSTM the first half is the forward and the second half the backward direction.
 */
final class SentenceEncoder {

    /** The encoder type. */
    enum Kind {
        BILSTM, FACTORED_ATTENTION
    }

    private final Kind kind;
    private final int width;
    private final Var embedding;
    private final Layer lstm;
    private final List<Var[]> blocks = new ArrayList<>();
    private final List<Var> params = new ArrayList<>();

    SentenceEncoder(Kind kind, int vocab, int embed, int units, int layers, SplittableRandom r) {
        this.kind = kind;
        this.width = 2 * units;
        int contentWidth = kind == Kind.BILSTM ? embed : units;
        embedding = Var.param(Tensor.uniform(Shape.of(vocab, contentWidth), -0.5f, 0.5f, r));
        params.add(embedding);
        if (kind == Kind.BILSTM) {
            Recurrent.Spec spec = Recurrent.lstm(units).returnSequences(true).bidirectional(Recurrent.Merge.CONCAT);
            lstm = spec.create(Shape.of(1, embed), r);
            params.addAll(lstm.parameters());
        } else {
            lstm = null;
            int h = units;
            for (int l = 0; l < layers; l++) {
                // per stream: q, k, v, out, ffn1, ffn2
                Var[] b = new Var[12];
                for (int s = 0; s < 2; s++) {
                    b[6 * s] = matrix(r, h, h);
                    b[6 * s + 1] = matrix(r, h, h);
                    b[6 * s + 2] = matrix(r, h, h);
                    b[6 * s + 3] = matrix(r, h, h);
                    b[6 * s + 4] = matrix(r, h, 2 * h);
                    b[6 * s + 5] = matrix(r, 2 * h, h);
                }
                blocks.add(b);
                params.addAll(List.of(b));
            }
        }
    }

    static Var matrix(SplittableRandom r, int in, int out) {
        float limit = (float) Math.sqrt(6.0 / (in + out));
        return Var.param(Tensor.uniform(Shape.of(in, out), -limit, limit, r));
    }

    int width() {
        return width;
    }

    List<Var> parameters() {
        return params;
    }

    Kind kind() {
        return kind;
    }

    /**
     * @param words {@code [batch][steps]} token ids, all rows the same length
     * @return {@code [batch, steps, width]}
     */
    Var encode(int[][] words) {
        int b = words.length;
        int t = words[0].length;
        float[] ids = new float[b * t];
        for (int i = 0; i < b; i++) {
            for (int j = 0; j < t; j++) {
                ids[i * t + j] = words[i][j];
            }
        }
        Var x = embedding.gather(0, Tensor.of(Shape.of(b, t), ids));
        if (kind == Kind.BILSTM) {
            return lstm.forward(x);
        }
        int h = width / 2;
        Var c = x;
        Var p = Var.of(sinusoid(t, h)).reshape(1, t, h).expand(Shape.of(b, t, h));
        int heads = 2;
        int dk = h / heads;
        float scale = (float) (1 / Math.sqrt(2.0 * dk));
        for (Var[] w : blocks) {
            Var q = Var.concat(-1, split(c.matmul(w[0]), heads), split(p.matmul(w[6]), heads));
            Var k = Var.concat(-1, split(c.matmul(w[1]), heads), split(p.matmul(w[7]), heads));
            Var v = Var.concat(-1, split(c.matmul(w[2]), heads), split(p.matmul(w[8]), heads));
            Var o = Functional.attention(q, k, v, false, scale).permute(0, 2, 1, 3);   // [b, t, heads, 2dk]
            Var oc = o.slice(3, 0, dk).reshape(b, t, h);
            Var op = o.slice(3, dk, 2 * dk).reshape(b, t, h);
            c = norm(c.add(oc.matmul(w[3])));
            p = norm(p.add(op.matmul(w[9])));
            c = norm(c.add(c.matmul(w[4]).relu().matmul(w[5])));
            p = norm(p.add(p.matmul(w[10]).relu().matmul(w[11])));
        }
        return Var.concat(-1, c, p);
    }

    private static Var split(Var x, int heads) {
        int b = x.shape().dim(0);
        int t = x.shape().dim(1);
        int d = x.shape().dim(2) / heads;
        return x.reshape(b, t, heads, d).permute(0, 2, 1, 3);
    }

    private static Var norm(Var x) {
        return x.div(x.square().mean(-1, true).add(1e-5f).sqrt());
    }

    static Tensor sinusoid(int steps, int dim) {
        float[] f = new float[steps * dim];
        for (int t = 0; t < steps; t++) {
            for (int i = 0; i < dim / 2; i++) {
                double angle = t / Math.pow(100, 2.0 * i / dim);
                f[t * dim + 2 * i] = (float) Math.sin(angle);
                f[t * dim + 2 * i + 1] = (float) Math.cos(angle);
            }
        }
        return Tensor.of(Shape.of(steps, dim), f);
    }

    /**
     * Samples a batch of sentences of one length (so no padding is needed): a random sentence
     * and others of its length, with replacement.
     */
    static int[] sameLengthBatch(int[] lengths, int batch, SplittableRandom r) {
        int first = r.nextInt(lengths.length);
        int[] same = java.util.stream.IntStream.range(0, lengths.length).filter(i -> lengths[i] == lengths[first])
                .toArray();
        int[] out = new int[batch];
        out[0] = first;
        for (int i = 1; i < batch; i++) {
            out[i] = same[r.nextInt(same.length)];
        }
        return out;
    }
}
