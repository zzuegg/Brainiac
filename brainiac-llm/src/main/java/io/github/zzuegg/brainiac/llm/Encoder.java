package io.github.zzuegg.brainiac.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.io.PretrainedTokenizer;
import io.github.zzuegg.brainiac.io.WeightSource;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.tensor.spi.ActivationOp;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * A pretrained BERT-style encoder from a Hugging Face folder: BERT and its descendants
 * (MiniLM, BGE, GTE, E5 in their BERT form) and RoBERTa / XLM-RoBERTa, with sentence
 * embeddings by {@code [CLS]} or mean pooling, and the sequence-classification head of
 * cross-encoder rerankers ({@code BertForSequenceClassification},
 * {@code XLMRobertaForSequenceClassification}, such as bge-reranker or ms-marco MiniLM).
 * Results match transformers.
 *
 * <pre>
 * Encoder bge = Encoder.load(Path.of("bge-small-en-v1.5"), Device.best(), DType.FLOAT32);
 * PretrainedTokenizer tok = PretrainedTokenizer.load(Path.of("bge-small-en-v1.5"));
 * float[][] vectors = bge.embed(tok, List.of("a cat", "a kitten"), Encoder.Pooling.CLS, true);
 *
 * Encoder reranker = Encoder.load(Path.of("bge-reranker-base"), Device.best(), DType.FLOAT32);
 * float[] scores = reranker.rerank(rerankerTok, "where do cats sleep?", documents);
 * </pre>
 */
public final class Encoder {

    /** How token vectors become one vector per text. */
    public enum Pooling {
        /** The first token's vector ({@code [CLS]} or {@code <s>}), as BGE uses. */
        CLS,
        /** The mean over the real (unpadded) tokens, as sentence-transformers' MiniLM and E5 use. */
        MEAN
    }

    private record Layer(Tensor q, Tensor qb, Tensor k, Tensor kb, Tensor v, Tensor vb, Tensor o, Tensor ob,
            Tensor norm1, Tensor norm1b, Tensor up, Tensor upb, Tensor down, Tensor downb, Tensor norm2,
            Tensor norm2b) {
    }

    private final boolean roberta;
    private final int hidden;
    private final int heads;
    private final float eps;
    private final int padId;
    private final ActivationOp activation;
    private final Tensor words;
    private final Tensor positions;
    private final Tensor tokenTypes;
    private final Tensor embedNorm;
    private final Tensor embedNormBias;
    private final List<Layer> layers = new ArrayList<>();
    /** BERT's pooler, or RoBERTa's classifier.dense; null when absent. */
    private final Tensor pooler;
    private final Tensor poolerBias;
    private final Tensor classifier;
    private final Tensor classifierBias;
    private final Device device;

    private Encoder(Map<String, Object> config, WeightSource w, Device device, DType dtype) {
        String type = Json.string(config, "model_type", "bert");
        roberta = type.equals("roberta") || type.equals("xlm-roberta") || type.equals("camembert");
        if (!roberta && !type.equals("bert")) {
            throw new IllegalArgumentException("not a BERT or RoBERTa encoder: model_type " + type);
        }
        this.device = device;
        hidden = (int) Json.integer(config, "hidden_size");
        heads = (int) Json.integer(config, "num_attention_heads");
        eps = (float) Json.number(config, "layer_norm_eps", 1e-12);
        padId = (int) Json.integer(config, "pad_token_id", roberta ? 1 : 0);
        String act = Json.string(config, "hidden_act", "gelu");
        activation = switch (act) {
            case "gelu" -> ActivationOp.GELU;
            case "gelu_new", "gelu_pytorch_tanh" -> ActivationOp.GELU_TANH;
            case "relu" -> ActivationOp.RELU;
            default -> throw new IllegalArgumentException("hidden_act " + act + " is not supported");
        };
        String base = w.contains("bert.embeddings.word_embeddings.weight") ? "bert."
                : w.contains("roberta.embeddings.word_embeddings.weight") ? "roberta." : "";
        java.util.function.Function<String, Tensor> t = name -> w.tensor(base + name, device, dtype);
        words = t.apply("embeddings.word_embeddings.weight");
        positions = t.apply("embeddings.position_embeddings.weight");
        tokenTypes = w.contains(base + "embeddings.token_type_embeddings.weight")
                ? t.apply("embeddings.token_type_embeddings.weight") : null;
        embedNorm = t.apply("embeddings.LayerNorm.weight");
        embedNormBias = t.apply("embeddings.LayerNorm.bias");
        int n = (int) Json.integer(config, "num_hidden_layers");
        for (int i = 0; i < n; i++) {
            String p = "encoder.layer." + i + ".";
            layers.add(new Layer(
                    t.apply(p + "attention.self.query.weight").transpose(), t.apply(p + "attention.self.query.bias"),
                    t.apply(p + "attention.self.key.weight").transpose(), t.apply(p + "attention.self.key.bias"),
                    t.apply(p + "attention.self.value.weight").transpose(), t.apply(p + "attention.self.value.bias"),
                    t.apply(p + "attention.output.dense.weight").transpose(), t.apply(p + "attention.output.dense.bias"),
                    t.apply(p + "attention.output.LayerNorm.weight"), t.apply(p + "attention.output.LayerNorm.bias"),
                    t.apply(p + "intermediate.dense.weight").transpose(), t.apply(p + "intermediate.dense.bias"),
                    t.apply(p + "output.dense.weight").transpose(), t.apply(p + "output.dense.bias"),
                    t.apply(p + "output.LayerNorm.weight"), t.apply(p + "output.LayerNorm.bias")));
        }
        if (w.contains("classifier.out_proj.weight")) {   // RoBERTa's classification head
            pooler = w.tensor("classifier.dense.weight", device, dtype).transpose();
            poolerBias = w.tensor("classifier.dense.bias", device, dtype);
            classifier = w.tensor("classifier.out_proj.weight", device, dtype).transpose();
            classifierBias = w.tensor("classifier.out_proj.bias", device, dtype);
        } else {
            boolean pooled = w.contains(base + "pooler.dense.weight");
            pooler = pooled ? t.apply("pooler.dense.weight").transpose() : null;
            poolerBias = pooled ? t.apply("pooler.dense.bias") : null;
            boolean head = w.contains("classifier.weight");
            classifier = head ? w.tensor("classifier.weight", device, dtype).transpose() : null;
            classifierBias = head ? w.tensor("classifier.bias", device, dtype) : null;
        }
    }

    /**
     * Loads an encoder from a folder with {@code config.json} and safetensors weights.
     *
     * @param dir the model folder
     * @param device where the weights go
     * @param dtype the weights' type
     * @return the encoder
     */
    public static Encoder load(Path dir, Device device, DType dtype) {
        try (WeightSource w = WeightSource.open(dir)) {
            return new Encoder(Json.object(Json.read(dir.resolve("config.json"))), w, device, dtype);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** @return the width of the token vectors */
    public int hiddenSize() {
        return hidden;
    }

    /** @return whether the model has a sequence-classification head, as rerankers do */
    public boolean hasClassifier() {
        return classifier != null;
    }

    /**
     * The final token vectors.
     *
     * @param ids {@code [batch][steps]} token ids, padded on the right
     * @param mask {@code [batch][steps]} 1 for real tokens, 0 for padding
     * @param types {@code [batch][steps]} segment ids (0 for the first text, 1 for the
     *     second of a pair), or {@code null} for all 0
     * @return {@code [batch, steps, hidden]}
     */
    public Tensor hidden(int[][] ids, int[][] mask, int[][] types) {
        return Var.noGrad(() -> encode(ids, mask, types)).value();
    }

    private Var encode(int[][] ids, int[][] mask, int[][] types) {
        int batch = ids.length;
        int steps = ids[0].length;
        float[] tok = new float[batch * steps];
        float[] pos = new float[batch * steps];
        float[] seg = new float[batch * steps];
        float[] keep = new float[batch * steps];
        for (int b = 0; b < batch; b++) {
            int counted = 0;
            for (int s = 0; s < steps; s++) {
                int i = b * steps + s;
                tok[i] = ids[b][s];
                keep[i] = mask[b][s];
                seg[i] = types == null ? 0 : types[b][s];
                if (roberta) {   // positions count real tokens from padding_idx + 1; padding sits at padding_idx
                    pos[i] = ids[b][s] != padId ? padId + ++counted : padId;
                } else {
                    pos[i] = s;
                }
            }
        }
        Tensor x = words.gather(0, idx(tok)).add(positions.gather(0, idx(pos)));
        if (tokenTypes != null) {
            x = x.add(tokenTypes.gather(0, idx(seg)));
        }
        Var h = norm(Var.of(x), embedNorm, embedNormBias).reshape(batch, steps, hidden);
        Tensor bias = Functional.keyMask(Tensor.of(Shape.of(batch, steps), keep, device)).to(x.dtype());
        int d = hidden / heads;
        float scale = (float) (1 / Math.sqrt(d));
        for (Layer l : layers) {
            Var flat = h.reshape(batch * steps, hidden);
            Var q = split(linear(flat, l.q, l.qb), batch, steps, d);
            Var k = split(linear(flat, l.k, l.kb), batch, steps, d);
            Var v = split(linear(flat, l.v, l.vb), batch, steps, d);
            Var a = Functional.attention(q, k, v, bias, scale, 0, null).permute(0, 2, 1, 3).reshape(batch * steps,
                    hidden);
            Var h1 = norm(linear(a, l.o, l.ob).add(flat), l.norm1, l.norm1b);
            Var up = linear(h1, l.up, l.upb).activation(activation, 0);
            h = norm(linear(up, l.down, l.downb).add(h1), l.norm2, l.norm2b).reshape(batch, steps, hidden);
        }
        return h;
    }

    private Tensor idx(float[] values) {
        return Tensor.of(Shape.of(values.length), values, device);
    }

    private Var split(Var x, int batch, int steps, int d) {
        return x.reshape(batch, steps, heads, d).permute(0, 2, 1, 3);
    }

    private static Var linear(Var x, Tensor w, Tensor b) {
        return x.matmul(w).add(b);
    }

    private Var norm(Var x, Tensor w, Tensor b) {
        Var centered = x.sub(x.mean(-1, true));
        return centered.div(centered.square().mean(-1, true).add(eps).sqrt()).mul(w).add(b);
    }

    /**
     * Sentence embeddings.
     *
     * @param ids {@code [batch][steps]} token ids, padded on the right
     * @param mask {@code [batch][steps]} 1 for real tokens
     * @param pooling how token vectors become one
     * @param normalize whether to scale each vector to unit length, for cosine similarity
     * @return {@code [batch][hidden]}
     */
    public float[][] embed(int[][] ids, int[][] mask, Pooling pooling, boolean normalize) {
        int batch = ids.length;
        int steps = ids[0].length;
        float[] h = hidden(ids, mask, null).to(DType.FLOAT32).toArray();
        float[][] out = new float[batch][hidden];
        for (int b = 0; b < batch; b++) {
            if (pooling == Pooling.CLS) {
                System.arraycopy(h, b * steps * hidden, out[b], 0, hidden);
            } else {
                int count = 0;
                for (int s = 0; s < steps; s++) {
                    if (mask[b][s] != 0) {
                        count++;
                        for (int j = 0; j < hidden; j++) {
                            out[b][j] += h[(b * steps + s) * hidden + j];
                        }
                    }
                }
                for (int j = 0; j < hidden; j++) {
                    out[b][j] /= Math.max(1, count);
                }
            }
            if (normalize) {
                double n = 0;
                for (float f : out[b]) {
                    n += f * f;
                }
                float inv = (float) (1 / Math.max(1e-12, Math.sqrt(n)));
                for (int j = 0; j < hidden; j++) {
                    out[b][j] *= inv;
                }
            }
        }
        return out;
    }

    /**
     * Embeds texts, tokenized with the model's tokenizer (special tokens added, padded per batch).
     *
     * @param tokenizer the model's tokenizer
     * @param texts the texts
     * @param pooling how token vectors become one
     * @param normalize whether to scale to unit length
     * @return one vector per text
     */
    public float[][] embed(PretrainedTokenizer tokenizer, List<String> texts, Pooling pooling, boolean normalize) {
        List<int[]> seqs = new ArrayList<>();
        for (String t : texts) {
            seqs.add(tokenizer.encode(t, true));
        }
        int[][][] b = pad(seqs, null);
        return embed(b[0], b[1], pooling, normalize);
    }

    /**
     * The classification head's outputs: BERT's pooler (dense and tanh on {@code [CLS]})
     * then its classifier, or RoBERTa's dense, tanh and projection on {@code <s>}.
     *
     * @param ids {@code [batch][steps]} token ids
     * @param mask {@code [batch][steps]} 1 for real tokens
     * @param types {@code [batch][steps]} segment ids, or {@code null}
     * @return {@code [batch][labels]} logits; a reranker has one label, its relevance score
     */
    public float[][] classify(int[][] ids, int[][] mask, int[][] types) {
        if (classifier == null) {
            throw new IllegalStateException("this encoder has no classification head");
        }
        int batch = ids.length;
        int steps = ids[0].length;
        Tensor logits = Var.noGrad(() -> {
            Var cls = encode(ids, mask, types).reshape(batch * steps, hidden)
                    .gather(0, idx(firstTokens(batch, steps)));
            Var pooled = pooler != null ? linear(cls, pooler, poolerBias).tanh() : cls;
            return linear(pooled, classifier, classifierBias);
        }).value();
        int labels = logits.shape().dim(1);
        float[] f = logits.to(DType.FLOAT32).toArray();
        float[][] out = new float[batch][labels];
        for (int b = 0; b < batch; b++) {
            System.arraycopy(f, b * labels, out[b], 0, labels);
        }
        return out;
    }

    private static float[] firstTokens(int batch, int steps) {
        float[] f = new float[batch];
        for (int b = 0; b < batch; b++) {
            f[b] = b * steps;
        }
        return f;
    }

    /**
     * Scores how well each document answers a query, as a cross-encoder reranker reads the
     * pair: {@code [CLS] query [SEP] document [SEP]} (RoBERTa: {@code <s> query </s></s>
     * document </s>}). Higher is more relevant.
     *
     * @param tokenizer the reranker's tokenizer
     * @param query the query
     * @param documents the candidates
     * @return one score per document
     */
    public float[] rerank(PretrainedTokenizer tokenizer, String query, List<String> documents) {
        int[] q = tokenizer.encode(query, false);
        int cls = special(tokenizer, roberta ? "<s>" : "[CLS]");
        int sep = special(tokenizer, roberta ? "</s>" : "[SEP]");
        List<int[]> seqs = new ArrayList<>();
        List<int[]> segs = new ArrayList<>();
        for (String doc : documents) {
            int[] d = tokenizer.encode(doc, false);
            int middle = roberta ? 2 : 1;
            int[] s = new int[q.length + d.length + 2 + middle];
            int[] seg = new int[s.length];
            int at = 0;
            s[at++] = cls;
            for (int x : q) {
                s[at++] = x;
            }
            for (int i = 0; i < middle; i++) {
                s[at++] = sep;
            }
            int second = at;
            for (int x : d) {
                s[at++] = x;
            }
            s[at] = sep;
            if (!roberta) {
                java.util.Arrays.fill(seg, second, s.length, 1);
            }
            seqs.add(s);
            segs.add(seg);
        }
        int[][][] b = pad(seqs, segs);
        float[][] logits = classify(b[0], b[1], b[2]);
        float[] out = new float[logits.length];
        for (int i = 0; i < out.length; i++) {
            out[i] = logits[i][0];
        }
        return out;
    }

    private static int special(PretrainedTokenizer tokenizer, String token) {
        return tokenizer.id(token).orElseThrow(() -> new IllegalArgumentException("the tokenizer has no " + token));
    }

    /** {ids, mask, types} padded on the right with {@code padId}. */
    private int[][][] pad(List<int[]> seqs, List<int[]> segs) {
        int longest = 1;
        for (int[] s : seqs) {
            longest = Math.max(longest, s.length);
        }
        int n = seqs.size();
        int[][] ids = new int[n][longest];
        int[][] mask = new int[n][longest];
        int[][] types = new int[n][longest];
        for (int i = 0; i < n; i++) {
            int[] s = seqs.get(i);
            java.util.Arrays.fill(ids[i], padId);
            System.arraycopy(s, 0, ids[i], 0, s.length);
            java.util.Arrays.fill(mask[i], 0, s.length, 1);
            if (segs != null) {
                System.arraycopy(segs.get(i), 0, types[i], 0, s.length);
            }
        }
        return new int[][][] {ids, mask, types};
    }

    @Override
    public String toString() {
        return "Encoder[" + (roberta ? "roberta" : "bert") + ", " + layers.size() + " layers, " + hidden + " hidden, "
                + heads + " heads" + (classifier != null ? ", classifier" : "") + " on " + device + "]";
    }
}
