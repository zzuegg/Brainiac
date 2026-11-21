package io.github.zzuegg.brainiac.audio;

import io.github.zzuegg.brainiac.audio.ChatterboxNets.Lin;
import io.github.zzuegg.brainiac.audio.ChatterboxNets.Norm;
import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.io.WeightSource;
import io.github.zzuegg.brainiac.llm.Llama;
import io.github.zzuegg.brainiac.llm.LlamaConfig;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CancellationException;
import java.util.function.BooleanSupplier;
import java.util.random.RandomGenerator;

import static io.github.zzuegg.brainiac.audio.ChatterboxNets.layerNorm;
import static io.github.zzuegg.brainiac.audio.ChatterboxNets.lin;
import static io.github.zzuegg.brainiac.audio.ChatterboxNets.norm;
import static io.github.zzuegg.brainiac.audio.ChatterboxNets.rows;

/**
 * Chatterbox's T3 ("token to token"): a Llama that reads the voice (speaker embedding, a
 * perceiver summary of the reference's speech tokens, the emotion exaggeration), the text and
 * the speech tokens so far, each through its own embedding and learned positions, and writes
 * S3 speech tokens, guided away from a text-free copy of itself.
 */
final class ChatterboxT3 {

    static final int START_SPEECH = 6561;
    static final int STOP_SPEECH = 6562;
    private static final int HEAD_DIM = 64;
    private static final int PERCEIVER_HEADS = 4;
    private static final int PLACEHOLDER = 0;

    /**
     * How T3 picks tokens.
     *
     * @param temperature softmax temperature; 0 picks the most likely token
     * @param minP tokens below this share of the top probability are dropped
     * @param topP nucleus sampling share (1 keeps all)
     * @param repetitionPenalty divides the logits of tokens already written
     * @param cfgWeight classifier-free guidance: how far the text pulls the logits
     * @param maxTokens at most this many speech tokens
     */
    record Sampling(float temperature, float minP, float topP, float repetitionPenalty, float cfgWeight,
            int maxTokens) {
        static Sampling defaults() {
            return new Sampling(0.8f, 0.05f, 1f, 1.2f, 0.5f, 1000);
        }

        static Sampling greedy(int maxTokens, float cfgWeight, float repetitionPenalty) {
            return new Sampling(0, 0, 1, repetitionPenalty, cfgWeight, maxTokens);
        }
    }

    private final Device device;
    private final DType dtype;
    private final Llama llama;
    private final int hidden;
    private final Var textEmb;
    private final Var speechEmb;
    private final Var textPos;
    private final Var speechPos;
    private final Lin spkrEnc;
    private final Var emotion;
    private final Var query;
    private final Norm pNorm;
    private final Lin pq;
    private final Lin pk;
    private final Lin pv;
    private final Lin pOut;

    ChatterboxT3(WeightSource source, Weights w) {
        device = w.device();
        dtype = w.dtype();
        Map<String, String> alias = new HashMap<>();
        for (String n : source.names()) {
            if (n.startsWith("tfmr.") && !n.startsWith("tfmr.embed_tokens.")) {
                alias.put("model." + n.substring("tfmr.".length()), n);
            }
        }
        // the Llama reads embeddings only; the speech head is its output layer
        alias.put("model.embed_tokens.weight", "speech_emb.weight");
        alias.put("lm_head.weight", "speech_head.weight");
        hidden = source.shape("speech_emb.weight").dim(1);
        llama = Llama.load(new Qwen3Speech.Alias(source, alias), config(source), device, dtype);
        textEmb = w.read("text_emb.weight");
        speechEmb = w.read("speech_emb.weight");
        textPos = w.read("text_pos_emb.emb.weight");
        speechPos = w.read("speech_pos_emb.emb.weight");
        spkrEnc = lin(w, "cond_enc.spkr_enc");
        emotion = w.contains("cond_enc.emotion_adv_fc.weight")
                ? w.read("cond_enc.emotion_adv_fc.weight").reshape(1, hidden) : null;
        String p = "cond_enc.perceiver.";
        query = w.read(p + "pre_attention_query");
        pNorm = norm(w, p + "attn.norm");
        pq = lin(w, p + "attn.to_q");
        pk = lin(w, p + "attn.to_k");
        pv = lin(w, p + "attn.to_v");
        pOut = lin(w, p + "attn.proj_out");
    }

    /** Llama_520M's settings, sized from the weights. */
    private static LlamaConfig config(WeightSource s) {
        int layers = 0;
        while (s.contains("tfmr.layers." + layers + ".self_attn.q_proj.weight")) {
            layers++;
        }
        int d = s.shape("speech_emb.weight").dim(1);
        Map<String, Object> m = new HashMap<>();
        m.put("model_type", "llama");
        m.put("vocab_size", s.shape("speech_head.weight").dim(0));
        m.put("hidden_size", d);
        m.put("intermediate_size", s.shape("tfmr.layers.0.mlp.gate_proj.weight").dim(0));
        m.put("num_hidden_layers", layers);
        m.put("num_attention_heads", s.shape("tfmr.layers.0.self_attn.q_proj.weight").dim(0) / HEAD_DIM);
        m.put("num_key_value_heads", s.shape("tfmr.layers.0.self_attn.k_proj.weight").dim(0) / HEAD_DIM);
        m.put("head_dim", HEAD_DIM);
        m.put("rms_norm_eps", 1e-5);
        m.put("rope_theta", 500000.0);
        m.put("max_position_embeddings", 131072);
        m.put("hidden_act", "silu");
        m.put("tie_word_embeddings", false);
        m.put("rope_scaling", Map.of("factor", 8.0, "high_freq_factor", 4.0, "low_freq_factor", 1.0,
                "original_max_position_embeddings", 8192, "rope_type", "llama3"));
        return LlamaConfig.fromHuggingFace(m);
    }

    /** The text vocabulary's size. */
    int textVocabulary() {
        return textEmb.shape().dim(0);
    }

    /**
     * The voice conditioning: the speaker, a perceiver's summary of the reference's speech
     * tokens, and the emotion exaggeration, as {@code [steps, hidden]}.
     */
    Var cond(float[] speaker, int[] promptTokens, float exaggeration) {
        return Var.noGrad(() -> {
            Var spk = spkrEnc.apply(Var.of(Tensor.of(Shape.of(1, speaker.length), speaker, device).to(dtype)));
            List<Var> parts = new ArrayList<>();
            parts.add(spk);
            if (promptTokens != null && promptTokens.length > 0) {
                Var pe = rows(speechEmb, promptTokens).add(speechPos.slice(0, 0, promptTokens.length));
                Var q = query.reshape(query.shape().dim(1), hidden);
                Var pre = perceive(q, pe);
                parts.add(perceive(pre, pre));
            }
            if (emotion != null) {
                parts.add(emotion.mul(exaggeration));
            }
            return Vocos.settle(Var.concat(0, parts.toArray(Var[]::new)));
        });
    }

    private Var perceive(Var x1, Var x2) {
        Var a = layerNorm(x1, pNorm, 1e-5f);
        Var b = layerNorm(x2, pNorm, 1e-5f);
        int n = x1.shape().dim(0);
        int m = x2.shape().dim(0);
        Var h = ChatterboxNets.attend(pq.apply(a).reshape(1, n, hidden), pk.apply(b).reshape(1, m, hidden),
                pv.apply(b).reshape(1, m, hidden), PERCEIVER_HEADS);
        return x1.add(pOut.apply(h.reshape(n, hidden)));
    }

    /**
     * Speech tokens for text.
     *
     * @param text the text tokens with start and stop tokens
     * @param cond {@link #cond}
     * @param s the sampling settings
     * @param random the random source (unused when greedy)
     * @param cancelled asked before each token
     * @return the tokens written, ending with {@link #STOP_SPEECH} unless cut at the maximum
     */
    int[] generate(int[] text, Var cond, Sampling s, RandomGenerator random, BooleanSupplier cancelled) {
        return Var.noGrad(() -> {
            boolean guided = s.cfgWeight() != 0;
            int rowsN = guided ? 2 : 1;
            Var te = rows(textEmb, text);
            Var tp = textPos.slice(0, 0, text.length);
            Var bos = rows(speechEmb, START_SPEECH).add(speechPos.slice(0, 0, 1));
            Var row = Var.concat(0, cond, te.add(tp), bos, bos);
            Var embeds = guided ? Var.concat(0, row, Var.concat(0, cond, tp, bos, bos)) : row;
            int steps = row.shape().dim(0);
            int[][] tokens = new int[rowsN][steps];
            Llama.State state = llama.newState(rowsN);
            try {
                Tensor logits = llama.next(state, tokens, PLACEHOLDER, Vocos.settle(embeds).value());
                List<Integer> out = new ArrayList<>();
                Set<Integer> seen = new HashSet<>();
                seen.add(START_SPEECH);
                int[][] one = new int[rowsN][1];
                // the learned positions bound the length (4100 in the real model)
                int limit = Math.min(s.maxTokens(), speechPos.shape().dim(0) - 1);
                for (int i = 0; i < limit; i++) {
                    if (cancelled.getAsBoolean()) {
                        throw new CancellationException("speech synthesis cancelled");
                    }
                    float[] all = logits.to(DType.FLOAT32).toArray();
                    int vocab = all.length / rowsN;
                    float[] l = new float[vocab];
                    for (int v = 0; v < vocab; v++) {
                        float c = all[v];
                        l[v] = guided ? c + s.cfgWeight() * (c - all[vocab + v]) : c;
                    }
                    int next = pick(l, seen, s, random);
                    out.add(next);
                    seen.add(next);
                    if (next == STOP_SPEECH) {
                        break;
                    }
                    Var e = rows(speechEmb, next).add(speechPos.slice(0, i + 1, i + 2));
                    Var in = guided ? Var.concat(0, e, e) : e;
                    logits = llama.next(state, one, PLACEHOLDER, Vocos.settle(in).value());
                }
                return out.stream().mapToInt(Integer::intValue).toArray();
            } finally {
                state.close();
            }
        });
    }

    /** The repetition penalty, temperature, min-p and top-p, then a draw (or the arg max). */
    static int pick(float[] l, Set<Integer> seen, Sampling s, RandomGenerator random) {
        float pen = s.repetitionPenalty();
        if (pen != 1) {
            for (int t : seen) {
                if (t < l.length) {
                    l[t] = l[t] < 0 ? l[t] * pen : l[t] / pen;
                }
            }
        }
        int best = 0;
        for (int i = 1; i < l.length; i++) {
            if (l[i] > l[best]) {
                best = i;
            }
        }
        if (s.temperature() <= 0) {
            return best;
        }
        double[] p = new double[l.length];
        double max = l[best] / s.temperature();
        double sum = 0;
        for (int i = 0; i < l.length; i++) {
            p[i] = Math.exp(l[i] / s.temperature() - max);
            sum += p[i];
        }
        // min-p: drop tokens less likely than minP times the most likely
        double floor = s.minP() * (1 / sum);
        for (int i = 0; i < p.length; i++) {
            p[i] /= sum;
            if (p[i] < floor) {
                p[i] = 0;
            }
        }
        if (s.topP() < 1) {
            Integer[] order = new Integer[p.length];
            for (int i = 0; i < order.length; i++) {
                order[i] = i;
            }
            java.util.Arrays.sort(order, (a, b) -> Double.compare(p[a], p[b]));
            double total = 0;
            for (double v : p) {
                total += v;
            }
            // as transformers: drop the least likely tokens whose cumulative share stays within 1 - top_p
            double cum = 0;
            for (int i = 0; i < order.length - 1; i++) {
                cum += p[order[i]] / total;
                if (cum <= 1 - s.topP()) {
                    p[order[i]] = 0;
                } else {
                    break;
                }
            }
        }
        double total = 0;
        for (double v : p) {
            total += v;
        }
        double r = random.nextDouble() * total;
        for (int i = 0; i < p.length; i++) {
            r -= p[i];
            if (r < 0 && p[i] > 0) {
                return i;
            }
        }
        return best;
    }

}
