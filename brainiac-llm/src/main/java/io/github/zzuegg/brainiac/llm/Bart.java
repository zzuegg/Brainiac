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
 * A pretrained BART-family encoder-decoder from a Hugging Face folder: BART (summarization,
 * as {@code bart-large-cnn}), mBART (normalization before each block) and Marian (the
 * Opus-MT translation models: sine positions, scaled embeddings). Text in, text out, with
 * greedy decoding and a key/value cache. Results match transformers.
 *
 * <pre>
 * Bart opus = Bart.load(Path.of("opus-mt-en-de"), Device.best(), DType.FLOAT32);
 * PretrainedTokenizer tok = PretrainedTokenizer.load(Path.of("opus-mt-en-de"));
 * String german = opus.generate(tok, "How old are you?", 40);
 * </pre>
 */
public final class Bart {

    private record Linear(Tensor w, Tensor b) {

        Tensor apply(Tensor x) {
            return x.matmul(w).add(b);
        }
    }

    private record Norm(Tensor w, Tensor b) {
    }

    private record Attention(Linear q, Linear k, Linear v, Linear o) {
    }

    private record Layer(Attention self, Norm selfNorm, Attention cross, Norm crossNorm, Linear fc1, Linear fc2,
            Norm ffNorm) {
    }

    private final Device device;
    private final DType dtype;
    private final String type;
    private final int dModel;
    private final int encoderHeads;
    private final int decoderHeads;
    private final ActivationOp act;
    private final boolean preNorm;
    private final float embedScale;
    private final float eps = 1e-5f;
    private final Tensor encoderTokens;
    private final Tensor decoderTokens;
    /** Learned positions (offset by 2, as BART stores them), or {@code null} for sine positions. */
    private final Tensor encoderPositions;
    private final Tensor decoderPositions;
    private final Norm encoderEmbedNorm;
    private final Norm decoderEmbedNorm;
    private final Norm encoderFinal;
    private final Norm decoderFinal;
    private final List<Layer> encoder = new ArrayList<>();
    private final List<Layer> decoder = new ArrayList<>();
    private final Tensor lmHead;
    private final Tensor logitsBias;
    private final int start;
    /** the token every generation must begin with ({@code forced_bos_token_id}), or -1 */
    private final int forcedFirst;
    private final int eos;

    private Bart(Map<String, Object> c, WeightSource w, Device device, DType dtype) {
        type = Json.string(c, "model_type", "bart");
        if (!List.of("bart", "mbart", "marian").contains(type)) {
            throw new IllegalArgumentException("not a BART, mBART or Marian model: model_type " + type);
        }
        this.device = device;
        this.dtype = dtype;
        dModel = (int) Json.integer(c, "d_model");
        encoderHeads = (int) Json.integer(c, "encoder_attention_heads");
        decoderHeads = (int) Json.integer(c, "decoder_attention_heads");
        String a = Json.string(c, "activation_function", "gelu");
        act = switch (a) {
            case "gelu" -> ActivationOp.GELU;
            case "gelu_new", "gelu_pytorch_tanh" -> ActivationOp.GELU_TANH;
            case "relu" -> ActivationOp.RELU;
            case "silu", "swish" -> ActivationOp.SILU;
            default -> throw new IllegalArgumentException("activation_function " + a + " is not supported");
        };
        preNorm = type.equals("mbart");
        embedScale = Boolean.TRUE.equals(c.get("scale_embedding")) ? (float) Math.sqrt(dModel) : 1f;
        start = (int) Json.integer(c, "decoder_start_token_id", 2);
        forcedFirst = c.get("forced_bos_token_id") instanceof Number f ? f.intValue() : -1;
        eos = (int) Json.integer(c, "eos_token_id", 2);
        String root = w.contains("model.shared.weight") || w.contains("model.encoder.layers.0.fc1.weight") ? "model."
                : "";
        Function<String, Tensor> t = name -> w.tensor(root + name, device, dtype);
        Function<String, Linear> lin = name -> new Linear(t.apply(name + ".weight").transpose(),
                t.apply(name + ".bias"));
        Function<String, Norm> norm = name -> w.contains(root + name + ".weight")
                ? new Norm(t.apply(name + ".weight"), t.apply(name + ".bias")) : null;
        // tied checkpoints (bart-large-cnn) may store the shared table under one name only
        Tensor shared = w.contains(root + "shared.weight") ? t.apply("shared.weight")
                : w.contains(root + "decoder.embed_tokens.weight") ? t.apply("decoder.embed_tokens.weight")
                : t.apply("encoder.embed_tokens.weight");
        // one table for both sides, unless Marian keeps the decoder's apart
        encoderTokens = shared;
        decoderTokens = Boolean.FALSE.equals(c.get("share_encoder_decoder_embeddings"))
                && w.contains(root + "decoder.embed_tokens.weight") ? t.apply("decoder.embed_tokens.weight") : shared;
        boolean sine = type.equals("marian");
        encoderPositions = sine ? null : t.apply("encoder.embed_positions.weight");
        decoderPositions = sine ? null : t.apply("decoder.embed_positions.weight");
        encoderEmbedNorm = norm.apply("encoder.layernorm_embedding");
        decoderEmbedNorm = norm.apply("decoder.layernorm_embedding");
        encoderFinal = preNorm ? norm.apply("encoder.layer_norm") : null;
        decoderFinal = preNorm ? norm.apply("decoder.layer_norm") : null;
        for (int i = 0; i < (int) Json.integer(c, "encoder_layers"); i++) {
            String p = "encoder.layers." + i + ".";
            encoder.add(new Layer(attention(lin, p + "self_attn."), norm.apply(p + "self_attn_layer_norm"), null, null,
                    lin.apply(p + "fc1"), lin.apply(p + "fc2"), norm.apply(p + "final_layer_norm")));
        }
        for (int i = 0; i < (int) Json.integer(c, "decoder_layers"); i++) {
            String p = "decoder.layers." + i + ".";
            decoder.add(new Layer(attention(lin, p + "self_attn."), norm.apply(p + "self_attn_layer_norm"),
                    attention(lin, p + "encoder_attn."), norm.apply(p + "encoder_attn_layer_norm"),
                    lin.apply(p + "fc1"), lin.apply(p + "fc2"), norm.apply(p + "final_layer_norm")));
        }
        lmHead = w.contains("lm_head.weight") ? w.tensor("lm_head.weight", device, dtype).transpose()
                : decoderTokens.transpose();
        logitsBias = w.contains("final_logits_bias") ? w.tensor("final_logits_bias", device, dtype) : null;
    }

    private static Attention attention(Function<String, Linear> lin, String p) {
        return new Attention(lin.apply(p + "q_proj"), lin.apply(p + "k_proj"), lin.apply(p + "v_proj"),
                lin.apply(p + "out_proj"));
    }

    /**
     * Loads a model from a folder with {@code config.json} and weights.
     *
     * @param dir the model folder
     * @param device where the weights go
     * @param dtype the weights' type
     * @return the model
     */
    public static Bart load(Path dir, Device device, DType dtype) {
        try (WeightSource w = WeightSource.open(dir)) {
            return new Bart(Json.object(Json.read(dir.resolve("config.json"))), w, device, dtype);
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
            Tensor x = embed(encoderTokens, encoderPositions, ids, 0).reshape(b * n, dModel);
            x = maybeNorm(x, encoderEmbedNorm);
            Tensor bias = keyMask(mask, b, n);
            for (Layer l : encoder) {
                x = block(x, l.selfNorm, h -> attend(l.self, encoderHeads, h, h, b, n, n, bias, null));
                x = block(x, l.ffNorm, h -> l.fc2.apply(l.fc1.apply(h).activation(act, 0)));
            }
            return maybeNorm(x, encoderFinal).reshape(b, n, dModel);
        });
    }

    /** A residual block, normalized before (mBART) or after (BART, Marian). */
    private Tensor block(Tensor x, Norm norm, Function<Tensor, Tensor> f) {
        return preNorm ? x.add(f.apply(norm(x, norm))) : norm(x.add(f.apply(x)), norm);
    }

    private Tensor embed(Tensor tokens, Tensor positions, int[][] ids, int offset) {
        int b = ids.length;
        int n = ids[0].length;
        float[] flat = new float[b * n];
        for (int i = 0; i < b; i++) {
            for (int s = 0; s < n; s++) {
                flat[i * n + s] = ids[i][s];
            }
        }
        Tensor x = tokens.gather(0, Tensor.of(Shape.of(flat.length), flat, device));
        if (embedScale != 1) {
            x = x.mul(embedScale);
        }
        Tensor pos;
        if (positions != null) {
            float[] p = new float[n];
            for (int s = 0; s < n; s++) {
                p[s] = s + offset + 2;
            }
            pos = positions.gather(0, Tensor.of(Shape.of(n), p, device));
        } else {
            pos = sine(n, offset);
        }
        return x.reshape(b, n, dModel).add(pos.reshape(1, n, dModel)).reshape(b * n, dModel);
    }

    /** Marian's positions: sines in the first half of each vector, cosines in the second. */
    private Tensor sine(int n, int offset) {
        float[] out = new float[n * dModel];
        int half = (dModel + 1) / 2;
        for (int s = 0; s < n; s++) {
            int pos = s + offset;
            for (int j = 0; j < dModel; j++) {
                int k = j < half ? j : j - half;
                double angle = pos / Math.pow(10000, 2.0 * k / dModel);
                out[s * dModel + j] = (float) (j < half ? Math.sin(angle) : Math.cos(angle));
            }
        }
        return Tensor.of(Shape.of(n, dModel), out, device).to(dtype);
    }

    /** {@code [batch, 1, 1, keys]}: 0 for real keys, a large negative number for padding. */
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
     * The decoder's logits for whole target sequences (teacher forcing).
     *
     * @param encoded the {@linkplain #encode encoder's states}
     * @param mask the encoder's padding mask, or {@code null}
     * @param decoderIds {@code [batch][steps]}, starting with the decoder start token
     * @return {@code [batch, steps, vocab]}
     */
    public Tensor logits(Tensor encoded, int[][] mask, int[][] decoderIds) {
        return Var.noGrad(() -> {
            int b = decoderIds.length;
            int t = decoderIds[0].length;
            int n = encoded.shape().dim(1);
            Tensor memory = encoded.reshape(b * n, dModel);
            Tensor x = maybeNorm(embed(decoderTokens, decoderPositions, decoderIds, 0), decoderEmbedNorm);
            Tensor self = Functional.causalMask(t, t, device).to(dtype).reshape(1, 1, t, t);
            Tensor cross = keyMask(mask, b, n);
            for (Layer l : decoder) {
                x = block(x, l.selfNorm, h -> attend(l.self, decoderHeads, h, h, b, t, t, self, null));
                x = block(x, l.crossNorm, h -> attend(l.cross, decoderHeads, h, memory, b, t, n, cross, null));
                x = block(x, l.ffNorm, h -> l.fc2.apply(l.fc1.apply(h).activation(act, 0)));
            }
            return head(x).reshape(b, t, lmHead.shape().dim(1));
        });
    }

    private Tensor head(Tensor x) {
        Tensor logits = maybeNorm(x, decoderFinal).matmul(lmHead);
        return logitsBias == null ? logits : logits.add(logitsBias);
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
        return Var.noGrad(() -> {
            int b = ids.length;
            int n = ids[0].length;
            Tensor memory = encode(ids, mask).reshape(b * n, dModel);
            Tensor cross = keyMask(mask, b, n);
            Tensor[][] crossKv = new Tensor[decoder.size()][];
            for (int i = 0; i < decoder.size(); i++) {
                Attention a = decoder.get(i).cross;
                crossKv[i] = new Tensor[] {split(a.k.apply(memory), decoderHeads, b, n),
                        split(a.v.apply(memory), decoderHeads, b, n)};
            }
            Tensor[][] cache = new Tensor[decoder.size()][2];
            int[] next = new int[b];
            Arrays.fill(next, start);
            boolean[] done = new boolean[b];
            List<List<Integer>> out = new ArrayList<>();
            for (int i = 0; i < b; i++) {
                out.add(new ArrayList<>());
            }
            Tensor none = Tensor.zeros(Shape.of(1, 1, 1, 1), dtype, device);
            for (int step = 0; step < maxNewTokens; step++) {
                int[][] col = new int[b][1];
                for (int i = 0; i < b; i++) {
                    col[i][0] = next[i];
                }
                Tensor x = maybeNorm(embed(decoderTokens, decoderPositions, col, step), decoderEmbedNorm);
                int keys = step + 1;
                for (int i = 0; i < decoder.size(); i++) {
                    Layer l = decoder.get(i);
                    Tensor[] c = cache[i];
                    Tensor[] kv = crossKv[i];
                    x = block(x, l.selfNorm, h -> {
                        Tensor k = split(l.self.k.apply(h), decoderHeads, b, 1);
                        Tensor v = split(l.self.v.apply(h), decoderHeads, b, 1);
                        c[0] = c[0] == null ? k : Tensor.concat(2, c[0], k);
                        c[1] = c[1] == null ? v : Tensor.concat(2, c[1], v);
                        return attend(l.self, decoderHeads, h, null, b, 1, keys, none, c);
                    });
                    x = block(x, l.crossNorm, h -> attend(l.cross, decoderHeads, h, null, b, 1, n, cross, kv));
                    x = block(x, l.ffNorm, h -> l.fc2.apply(l.fc1.apply(h).activation(act, 0)));
                }
                float[] best = head(x).argmax(-1).toArray();
                if (step == 0 && forcedFirst >= 0) {
                    // as transformers: the first token is forced (bart-large-cnn's <s>) and,
                    // being a special token, is not part of the text
                    Arrays.fill(next, forcedFirst);
                    continue;
                }
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
     * Generates an answer to one text (a summary, or a translation).
     *
     * @param tokenizer the model's tokenizer
     * @param text the input
     * @param maxNewTokens the most tokens to generate
     * @return the decoded answer
     */
    public String generate(PretrainedTokenizer tokenizer, String text, int maxNewTokens) {
        int[] ids = tokenizer.encode(text, true);   // with the end token the model was trained with
        return tokenizer.decode(generate(new int[][] {ids}, null, maxNewTokens)[0]);
    }

    // ------------------------------------------------------------------ layers

    /**
     * Multi-head attention scaled by {@code 1/sqrt(head size)}. With {@code kv} the keys and
     * values are given ({@code [batch, heads, keys, head size]}); otherwise they come from
     * {@code memory}.
     */
    private Tensor attend(Attention a, int heads, Tensor x, Tensor memory, int b, int queries, int keys, Tensor bias,
            Tensor[] kv) {
        int d = dModel / heads;
        Tensor q = split(a.q.apply(x), heads, b, queries);
        Tensor k = kv != null ? kv[0] : split(a.k.apply(memory), heads, b, keys);
        Tensor v = kv != null ? kv[1] : split(a.v.apply(memory), heads, b, keys);
        Tensor full = bias.expand(Shape.of(b, heads, queries, keys));
        Tensor o = Functional.attention(Var.of(q), Var.of(k), Var.of(v), full, (float) (1 / Math.sqrt(d)), 0, null)
                .value();
        return a.o.apply(o.permute(0, 2, 1, 3).reshape(b * queries, dModel));
    }

    private Tensor split(Tensor x, int heads, int b, int n) {
        return x.reshape(b, n, heads, dModel / heads).permute(0, 2, 1, 3);
    }

    private Tensor maybeNorm(Tensor x, Norm n) {
        return n == null ? x : norm(x, n);
    }

    private Tensor norm(Tensor x, Norm n) {
        Tensor f = x.to(DType.FLOAT32);
        Tensor centered = f.sub(f.mean(-1, true));
        Tensor r = centered.mul(centered).mean(-1, true).add(eps).sqrt();
        return centered.div(r).to(dtype).mul(n.w).add(n.b);
    }

    @Override
    public String toString() {
        return "Bart[" + type + ", " + encoder.size() + "+" + decoder.size() + " layers, d_model " + dModel + "]";
    }
}
