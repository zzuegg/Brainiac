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
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

/**
 * A pretrained T5 encoder-decoder from a Hugging Face folder: T5, T5 v1.1 and Flan-T5 (ReLU
 * or gated-GELU feed-forward), mT5, and the encoder alone (as diffusion models use T5 for
 * their text). Text in, text out, with greedy decoding and a key/value cache. Results match
 * transformers.
 *
 * <pre>
 * T5 flan = T5.load(Path.of("flan-t5-base"), Device.best(), DType.FLOAT32);
 * PretrainedTokenizer tok = PretrainedTokenizer.load(Path.of("flan-t5-base"));
 * String german = flan.generate(tok, "translate English to German: How old are you?", 40);
 * </pre>
 */
public final class T5 {

    private record Attention(Tensor q, Tensor k, Tensor v, Tensor o) {
    }

    private record Block(Tensor norm1, Attention self, Tensor norm2, Attention cross, Tensor normFf, Tensor wi,
            Tensor wi1, Tensor wo) {
    }

    private final Device device;
    private final DType dtype;
    private final int dModel;
    private final int heads;
    private final int dKv;
    private final float eps;
    private final boolean gated;
    private final ActivationOp act;
    private final int buckets;
    private final int maxDistance;
    private final Tensor shared;
    private final Tensor encoderBias;
    private final Tensor decoderBias;
    private final List<Block> encoder = new ArrayList<>();
    private final List<Block> decoder = new ArrayList<>();
    private final Tensor encoderNorm;
    private final Tensor decoderNorm;
    private final Tensor lmHead;
    private final boolean tied;
    private final int start;
    private final int eos;
    private final int pad;

    private T5(Map<String, Object> c, WeightSource w, Device device, DType dtype) {
        String type = Json.string(c, "model_type", "t5");
        if (!type.equals("t5") && !type.equals("mt5") && !type.equals("umt5")) {
            throw new IllegalArgumentException("not a T5 model: model_type " + type);
        }
        if (type.equals("umt5")) {
            throw new IllegalArgumentException("umT5 (a position bias per layer) is not supported");
        }
        this.device = device;
        this.dtype = dtype;
        dModel = (int) Json.integer(c, "d_model");
        heads = (int) Json.integer(c, "num_heads");
        dKv = (int) Json.integer(c, "d_kv");
        eps = (float) Json.number(c, "layer_norm_epsilon", 1e-6);
        String ff = Json.string(c, "feed_forward_proj", "relu");
        gated = ff.startsWith("gated-");
        String a = gated ? ff.substring("gated-".length()) : ff;
        act = switch (a) {
            case "relu" -> ActivationOp.RELU;
            case "gelu" -> ActivationOp.GELU_TANH;   // T5's "gelu" is gelu_new
            case "gelu_new", "gelu_pytorch_tanh" -> ActivationOp.GELU_TANH;
            case "silu" -> ActivationOp.SILU;
            default -> throw new IllegalArgumentException("feed_forward_proj " + ff + " is not supported");
        };
        buckets = (int) Json.integer(c, "relative_attention_num_buckets", 32);
        maxDistance = (int) Json.integer(c, "relative_attention_max_distance", 128);
        // a tied head scales its input by d_model^-1/2; transformers 5 writes the flag itself
        tied = c.get("scale_decoder_outputs") instanceof Boolean b ? b
                : !Boolean.FALSE.equals(c.get("tie_word_embeddings"));
        start = (int) Json.integer(c, "decoder_start_token_id", 0);
        eos = (int) Json.integer(c, "eos_token_id", 1);
        pad = (int) Json.integer(c, "pad_token_id", 0);
        Function<String, Tensor> t = name -> w.tensor(name, device, dtype);
        Function<String, Tensor> lin = name -> t.apply(name).transpose();
        shared = w.contains("shared.weight") ? t.apply("shared.weight") : t.apply("encoder.embed_tokens.weight");
        int layers = (int) Json.integer(c, "num_layers");
        int decoderLayers = (int) Json.integer(c, "num_decoder_layers", layers);
        for (int i = 0; i < layers; i++) {
            String p = "encoder.block." + i + ".layer.";
            encoder.add(new Block(t.apply(p + "0.layer_norm.weight"), attention(lin, p + "0.SelfAttention."), null, null,
                    t.apply(p + "1.layer_norm.weight"),
                    lin.apply(p + "1.DenseReluDense." + (gated ? "wi_0" : "wi") + ".weight"),
                    gated ? lin.apply(p + "1.DenseReluDense.wi_1.weight") : null,
                    lin.apply(p + "1.DenseReluDense.wo.weight")));
        }
        boolean hasDecoder = w.contains("decoder.block.0.layer.0.SelfAttention.q.weight");
        for (int i = 0; hasDecoder && i < decoderLayers; i++) {
            String p = "decoder.block." + i + ".layer.";
            decoder.add(new Block(t.apply(p + "0.layer_norm.weight"), attention(lin, p + "0.SelfAttention."),
                    t.apply(p + "1.layer_norm.weight"), attention(lin, p + "1.EncDecAttention."),
                    t.apply(p + "2.layer_norm.weight"),
                    lin.apply(p + "2.DenseReluDense." + (gated ? "wi_0" : "wi") + ".weight"),
                    gated ? lin.apply(p + "2.DenseReluDense.wi_1.weight") : null,
                    lin.apply(p + "2.DenseReluDense.wo.weight")));
        }
        encoderBias = w.tensor("encoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight", Device.cpu(),
                DType.FLOAT32);
        encoderNorm = t.apply("encoder.final_layer_norm.weight");
        if (hasDecoder) {
            decoderBias = w.tensor("decoder.block.0.layer.0.SelfAttention.relative_attention_bias.weight", Device.cpu(),
                    DType.FLOAT32);
            decoderNorm = t.apply("decoder.final_layer_norm.weight");
            lmHead = w.contains("lm_head.weight") ? lin.apply("lm_head.weight") : shared.transpose();
        } else {
            decoderBias = decoderNorm = lmHead = null;
        }
    }

    private static Attention attention(Function<String, Tensor> lin, String p) {
        return new Attention(lin.apply(p + "q.weight"), lin.apply(p + "k.weight"), lin.apply(p + "v.weight"),
                lin.apply(p + "o.weight"));
    }

    /**
     * Loads a model from a folder with {@code config.json} and weights; a folder with only an
     * encoder ({@code T5EncoderModel}) loads too, for {@link #encode}.
     *
     * @param dir the model folder
     * @param device where the weights go
     * @param dtype the weights' type
     * @return the model
     */
    public static T5 load(Path dir, Device device, DType dtype) {
        try (WeightSource w = WeightSource.open(dir)) {
            return new T5(Json.object(Json.read(dir.resolve("config.json"))), w, device, dtype);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** @return the width of the hidden states */
    public int hiddenSize() {
        return dModel;
    }

    // ------------------------------------------------------------------ encoder

    /**
     * The encoder's final hidden states.
     *
     * @param ids {@code [batch][steps]} token ids, padded on the right
     * @param mask {@code [batch][steps]} 1 for real tokens, or {@code null} for all real
     * @return {@code [batch, steps, d_model]}
     */
    public Tensor encode(int[][] ids, int[][] mask) {
        return Var.noGrad(() -> {
            int b = ids.length;
            int n = ids[0].length;
            Tensor x = embed(ids).reshape(b * n, dModel);
            Tensor bias = positionBias(encoderBias, n, n, 0, true).add(keyMask(mask, b, n));
            for (Block l : encoder) {
                Tensor h = norm(x, l.norm1);
                x = x.add(attend(l.self, h, h, b, n, n, bias, null));
                x = x.add(feedForward(l, norm(x, l.normFf)));
            }
            return norm(x, encoderNorm).reshape(b, n, dModel);
        });
    }

    private Tensor embed(int[][] ids) {
        int b = ids.length;
        int n = ids[0].length;
        float[] flat = new float[b * n];
        for (int i = 0; i < b; i++) {
            for (int s = 0; s < n; s++) {
                flat[i * n + s] = ids[i][s];
            }
        }
        return shared.gather(0, Tensor.of(Shape.of(flat.length), flat, device));
    }

    /** {@code [1 or batch, 1, 1, keys]}: 0 for real keys, a large negative number for padding. */
    private Tensor keyMask(int[][] mask, int b, int n) {
        float[] m = new float[b * n];
        if (mask != null) {
            for (int i = 0; i < b; i++) {
                for (int s = 0; s < n; s++) {
                    m[i * n + s] = mask[i][s] != 0 ? 0 : -1e9f;
                }
            }
        }
        return Tensor.of(Shape.of(b, 1, 1, n), m, device).to(dtype);
    }

    // ------------------------------------------------------------------ decoder

    /**
     * The decoder's logits for whole target sequences (teacher forcing): scores a known
     * answer, or trains.
     *
     * @param encoded the {@linkplain #encode encoder's states}
     * @param mask the encoder's padding mask, or {@code null}
     * @param decoderIds {@code [batch][steps]}, starting with the decoder start token (the pad token)
     * @return {@code [batch, steps, vocab]}
     */
    public Tensor logits(Tensor encoded, int[][] mask, int[][] decoderIds) {
        requireDecoder();
        return Var.noGrad(() -> {
            int b = decoderIds.length;
            int t = decoderIds[0].length;
            int n = encoded.shape().dim(1);
            Tensor memory = encoded.reshape(b * n, dModel);
            Tensor x = embed(decoderIds).reshape(b * t, dModel);
            Tensor self = positionBias(decoderBias, t, t, 0, false).add(Functional.causalMask(t, t, device).to(dtype)
                    .reshape(1, 1, t, t));
            Tensor cross = keyMask(mask, b, n);
            for (Block l : decoder) {
                Tensor h = norm(x, l.norm1);
                x = x.add(attend(l.self, h, h, b, t, t, self, null));
                x = x.add(attend(l.cross, norm(x, l.norm2), memory, b, t, n, cross, null));
                x = x.add(feedForward(l, norm(x, l.normFf)));
            }
            return head(x).reshape(b, t, lmHead.shape().dim(1));
        });
    }

    private Tensor head(Tensor x) {
        Tensor h = norm(x, decoderNorm);
        if (tied) {
            h = h.mul((float) (1 / Math.sqrt(dModel)));
        }
        return h.matmul(lmHead);
    }

    /**
     * Greedy decoding with a key/value cache, for a batch.
     *
     * @param ids {@code [batch][steps]} encoder token ids, padded on the right
     * @param mask their padding mask, or {@code null} for none
     * @param maxNewTokens the most tokens to generate per row
     * @return each row's tokens, without the start token, up to and without the end token
     */
    public int[][] generate(int[][] ids, int[][] mask, int maxNewTokens) {
        requireDecoder();
        return Var.noGrad(() -> {
            int b = ids.length;
            int n = ids[0].length;
            Tensor memory = encode(ids, mask).reshape(b * n, dModel);
            Tensor cross = keyMask(mask, b, n);
            // the encoder's keys and values, once per layer
            Tensor[][] crossKv = new Tensor[decoder.size()][];
            for (int i = 0; i < decoder.size(); i++) {
                Attention a = decoder.get(i).cross;
                crossKv[i] = new Tensor[] {split(memory.matmul(a.k), b, n), split(memory.matmul(a.v), b, n)};
            }
            Tensor[][] cache = new Tensor[decoder.size()][2];
            int[] next = new int[b];
            Arrays.fill(next, start);
            boolean[] done = new boolean[b];
            List<List<Integer>> out = new ArrayList<>();
            for (int i = 0; i < b; i++) {
                out.add(new ArrayList<>());
            }
            for (int step = 0; step < maxNewTokens; step++) {
                int[][] col = new int[b][1];
                for (int i = 0; i < b; i++) {
                    col[i][0] = next[i];
                }
                Tensor x = embed(col).reshape(b, dModel);
                Tensor self = positionBias(decoderBias, 1, step + 1, step, false);
                for (int i = 0; i < decoder.size(); i++) {
                    Block l = decoder.get(i);
                    Tensor h = norm(x, l.norm1);
                    Tensor k = split(h.matmul(l.self.k), b, 1);
                    Tensor v = split(h.matmul(l.self.v), b, 1);
                    cache[i][0] = cache[i][0] == null ? k : Tensor.concat(2, cache[i][0], k);
                    cache[i][1] = cache[i][1] == null ? v : Tensor.concat(2, cache[i][1], v);
                    x = x.add(attend(l.self, h, null, b, 1, step + 1, self, cache[i]));
                    x = x.add(attend(l.cross, norm(x, l.norm2), null, b, 1, n, cross, crossKv[i]));
                    x = x.add(feedForward(l, norm(x, l.normFf)));
                }
                float[] best = head(x).argmax(-1).toArray();
                boolean all = true;
                for (int i = 0; i < b; i++) {
                    next[i] = (int) best[i];
                    if (!done[i]) {
                        if (next[i] == eos) {
                            done[i] = true;
                        } else {
                            out.get(i).add(next[i]);
                        }
                    }
                    all &= done[i];
                }
                if (all) {
                    break;
                }
            }
            return out.stream().map(l -> l.stream().mapToInt(Integer::intValue).toArray()).toArray(int[][]::new);
        });
    }

    /**
     * Generates an answer to one text.
     *
     * @param tokenizer the model's tokenizer
     * @param text the input, with its task prefix for T5 ("summarize: ...")
     * @param maxNewTokens the most tokens to generate
     * @return the decoded answer
     */
    public String generate(PretrainedTokenizer tokenizer, String text, int maxNewTokens) {
        int[] ids = tokenizer.encode(text, true);   // with the closing </s> T5 was trained with
        return tokenizer.decode(generate(new int[][] {ids}, null, maxNewTokens)[0]);
    }

    private void requireDecoder() {
        if (lmHead == null) {
            throw new IllegalStateException("this T5 folder holds only an encoder");
        }
    }

    // ------------------------------------------------------------------ layers

    /**
     * Multi-head attention without score scaling (T5 folds it into the weights). With
     * {@code kv} the keys and values are given ({@code [batch, heads, keys, d_kv]}); otherwise
     * they come from {@code memory}.
     */
    private Tensor attend(Attention a, Tensor x, Tensor memory, int b, int queries, int keys, Tensor bias,
            Tensor[] kv) {
        Tensor q = split(x.matmul(a.q), b, queries);
        Tensor k = kv != null ? kv[0] : split(memory.matmul(a.k), b, keys);
        Tensor v = kv != null ? kv[1] : split(memory.matmul(a.v), b, keys);
        Tensor full = bias.shape().dim(0) == b ? bias : bias.expand(Shape.of(b, bias.shape().dim(1),
                bias.shape().dim(2), bias.shape().dim(3)));
        if (full.shape().dim(1) != heads) {
            full = full.expand(Shape.of(b, heads, queries, keys));
        }
        Tensor o = Functional.attention(Var.of(q), Var.of(k), Var.of(v), full, 1f, 0, null).value();
        return o.permute(0, 2, 1, 3).reshape(b * queries, heads * dKv).matmul(a.o);
    }

    private Tensor split(Tensor x, int b, int n) {
        return x.reshape(b, n, heads, dKv).permute(0, 2, 1, 3);
    }

    private Tensor feedForward(Block l, Tensor x) {
        Tensor h = x.matmul(l.wi).activation(act, 0);
        if (l.wi1 != null) {
            h = h.mul(x.matmul(l.wi1));
        }
        return h.matmul(l.wo);
    }

    /** T5's norm: scaled by the root mean square, without centering or bias. */
    private Tensor norm(Tensor x, Tensor w) {
        Tensor f = x.to(DType.FLOAT32);
        Tensor r = f.mul(f).mean(-1, true).add(eps).sqrt();
        return f.div(r).to(dtype).mul(w);
    }

    /**
     * The relative position bias {@code [1, heads, queries, keys]} for queries starting at
     * {@code offset}: each pair's distance in log-spaced buckets, looked up in {@code table}.
     */
    private Tensor positionBias(Tensor table, int queries, int keys, int offset, boolean bidirectional) {
        float[] t = table.toArray();
        float[] out = new float[heads * queries * keys];
        for (int qi = 0; qi < queries; qi++) {
            for (int ki = 0; ki < keys; ki++) {
                int bucket = bucket(ki - (qi + offset), bidirectional);
                for (int h = 0; h < heads; h++) {
                    out[(h * queries + qi) * keys + ki] = t[bucket * heads + h];
                }
            }
        }
        return Tensor.of(Shape.of(1, heads, queries, keys), out, device).to(dtype);
    }

    private int bucket(int relative, boolean bidirectional) {
        int n = buckets;
        int result = 0;
        int distance;
        if (bidirectional) {
            n /= 2;
            if (relative > 0) {
                result += n;
            }
            distance = Math.abs(relative);
        } else {
            distance = -Math.min(relative, 0);
        }
        int exact = n / 2;
        if (distance < exact) {
            return result + distance;
        }
        float scaled = (float) Math.log(distance / (float) exact);
        int large = exact + (int) (scaled / (float) Math.log((double) maxDistance / exact) * (n - exact));
        return result + Math.min(large, n - 1);
    }

    @Override
    public String toString() {
        return "T5[" + encoder.size() + "+" + decoder.size() + " layers, d_model " + dModel + ", " + heads + " heads"
                + (gated ? ", gated" : "") + "]";
    }
}
