package io.github.zzuegg.brainiac.llm;

import io.github.zzuegg.brainiac.io.Gguf;
import io.github.zzuegg.brainiac.io.WeightSource;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/**
 * Reads llama.cpp's multimodal projector files ({@code mmproj-*.gguf}): their settings
 * become the transformers-style configs {@link VisionLanguageModel} reads, and their tensor
 * names the Hugging Face ones. LLaVA ({@code mlp} projector, CLIP tower) and Gemma 3
 * ({@code gemma3} projector, SigLIP tower) are understood.
 */
final class Mmproj {

    /** The model config, the image processor config, and the weights under Hugging Face names. */
    record Parts(Map<String, Object> config, Map<String, Object> processor, WeightSource weights) {
    }

    private Mmproj() {
    }

    private static Object meta(Map<String, Object> m, String key, Object otherwise) {
        Object v = m.get(key);
        return v == null ? otherwise : v;
    }

    private static int integer(Map<String, Object> m, String key, int otherwise) {
        return ((Number) meta(m, key, otherwise)).intValue();
    }

    static Parts read(Gguf g, int imageToken) {
        Map<String, Object> m = g.metadata();
        String projector = String.valueOf(meta(m, "clip.vision.projector_type", meta(m, "clip.projector_type", "mlp")));
        boolean gemma = projector.equals("gemma3");
        if (!gemma && !projector.equals("mlp") && !projector.equals("llava")) {
            throw new UnsupportedOperationException("mmproj projector " + projector
                    + " is not supported; LLaVA (mlp) and Gemma 3 are");
        }
        Map<String, Object> vision = new LinkedHashMap<>();
        vision.put("model_type", gemma ? "siglip_vision_model" : "clip_vision_model");
        vision.put("hidden_size", integer(m, "clip.vision.embedding_length", 768));
        vision.put("intermediate_size", integer(m, "clip.vision.feed_forward_length", 3072));
        vision.put("num_hidden_layers", integer(m, "clip.vision.block_count", 12));
        vision.put("num_attention_heads", integer(m, "clip.vision.attention.head_count", 12));
        vision.put("layer_norm_eps", ((Number) meta(m, "clip.vision.attention.layer_norm_epsilon",
                gemma ? 1e-6 : 1e-5)).doubleValue());
        vision.put("image_size", integer(m, "clip.vision.image_size", gemma ? 896 : 336));
        vision.put("patch_size", integer(m, "clip.vision.patch_size", 14));
        boolean gelu = Boolean.TRUE.equals(m.get("clip.use_gelu"));
        vision.put("hidden_act", gemma ? "gelu_pytorch_tanh" : gelu ? "gelu" : "quick_gelu");

        Map<String, Object> processor = new LinkedHashMap<>();
        int size = (int) vision.get("image_size");
        if (gemma) {
            processor.put("size", Map.of("height", size, "width", size));
            processor.put("resample", 2);
        } else {
            processor.put("size", Map.of("shortest_edge", size));
            processor.put("do_center_crop", true);
            processor.put("crop_size", Map.of("height", size, "width", size));
            processor.put("resample", 3);
        }
        Object mean = m.get("clip.vision.image_mean");
        Object std = m.get("clip.vision.image_std");
        processor.put("image_mean", mean instanceof List<?> l ? l : gemma ? List.of(0.5, 0.5, 0.5)
                : List.of(0.48145466, 0.4578275, 0.40821073));
        processor.put("image_std", std instanceof List<?> l ? l : gemma ? List.of(0.5, 0.5, 0.5)
                : List.of(0.26862954, 0.26130258, 0.27577711));

        Map<String, Object> config = new LinkedHashMap<>();
        config.put("model_type", gemma ? "gemma3" : "llava");
        config.put("vision_config", vision);
        config.put("image_token_index", imageToken);
        if (gemma) {
            int patches = size / (int) vision.get("patch_size");
            config.put("mm_tokens_per_image", integer(m, "clip.vision.mm_tokens_per_image",
                    Math.min(256, patches * patches)));
        } else {
            config.put("projector_hidden_act", "gelu");
            config.put("vision_feature_select_strategy", "default");
            // with the final norm kept the file holds the whole tower and LLaVA reads the
            // second-to-last layer; without it the last layer was already cut off
            Object layer = m.get("clip.vision.feature_layer");
            if (layer instanceof List<?> l && l.size() == 1) {
                layer = l.get(0);
            }
            config.put("vision_feature_layer", layer instanceof Number n ? n.intValue()
                    : g.contains("v.post_ln.weight") ? -2 : -1);
        }
        return new Parts(config, processor, g.renamed(names(g)));
    }

    /** GGUF names to Hugging Face ones; the feed-forward pair is told apart by shape. */
    private static java.util.function.UnaryOperator<String> names(Gguf g) {
        Map<String, String> map = new HashMap<>();
        TreeMap<Integer, String> mlp = new TreeMap<>();
        for (String n : g.names()) {
            String h = name(n, g);
            if (h != null) {
                map.put(n, h);
            }
            if (n.matches("mm\\.\\d+\\.weight")) {
                mlp.put(Integer.parseInt(n.split("\\.")[1]), n.substring(0, n.length() - ".weight".length()));
            }
        }
        // a two-layer projector: mm.0 and mm.2 (or mm.1 and mm.2) are linear_1 and linear_2
        List<String> layers = new ArrayList<>(mlp.values());
        for (int i = 0; i < layers.size() && i < 2; i++) {
            String target = "projector.linear_" + (i + 1);
            map.put(layers.get(i) + ".weight", target + ".weight");
            map.put(layers.get(i) + ".bias", target + ".bias");
        }
        return map::get;
    }

    private static String name(String n, Gguf g) {
        String v = "vision_model.";
        String e = v + "embeddings.";
        switch (n) {
            case "v.patch_embd.weight":
                return e + "patch_embedding.weight";
            case "v.patch_embd.bias":
                return e + "patch_embedding.bias";
            case "v.class_embd":
                return e + "class_embedding";
            case "v.position_embd.weight":
                return e + "position_embedding.weight";
            case "mm.input_projection.weight":
                return "projector.mm_input_projection_weight";
            case "mm.soft_emb_norm.weight":
                return "projector.mm_soft_emb_norm.weight";
            default:
                break;
        }
        for (String suffix : new String[] {".weight", ".bias"}) {
            if (n.equals("v.pre_ln" + suffix)) {
                return v + "pre_layrnorm" + suffix;
            }
            if (n.equals("v.post_ln" + suffix)) {
                return v + "post_layernorm" + suffix;
            }
        }
        if (!n.startsWith("v.blk.")) {
            return null;
        }
        String[] parts = n.split("\\.");
        if (parts.length != 5) {
            return null;
        }
        String p = v + "encoder.layers." + parts[2] + ".";
        String suffix = "." + parts[4];
        String what = switch (parts[3]) {
            case "attn_q" -> "self_attn.q_proj";
            case "attn_k" -> "self_attn.k_proj";
            case "attn_v" -> "self_attn.v_proj";
            case "attn_out" -> "self_attn.out_proj";
            case "ln1" -> "layer_norm1";
            case "ln2" -> "layer_norm2";
            case "ffn_up", "ffn_down" -> {
                // old converters swapped the names: fc1 widens to the intermediate size
                String w = "v.blk." + parts[2] + "." + parts[3] + ".weight";
                int out = g.shape(w).dim(0);
                int in = g.shape(w).dim(1);
                yield out > in ? "mlp.fc1" : "mlp.fc2";
            }
            default -> null;
        };
        return what == null ? null : p + what + suffix;
    }
}
