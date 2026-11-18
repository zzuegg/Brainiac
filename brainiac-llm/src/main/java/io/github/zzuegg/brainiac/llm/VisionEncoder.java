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

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;

import javax.imageio.ImageIO;

/**
 * A pretrained Vision Transformer from a Hugging Face folder: CLIP and SigLIP (image and text
 * towers, for image embeddings and zero-shot classification), ViT image classifiers
 * ({@code ViTForImageClassification}) and DINOv2 image features. The folder's
 * {@code preprocessor_config.json} turns images into pixels as transformers does. Results
 * match transformers.
 *
 * <pre>
 * VisionEncoder clip = VisionEncoder.load(Path.of("clip-vit-base-patch32"), Device.best(), DType.FLOAT32);
 * PretrainedTokenizer tok = PretrainedTokenizer.load(Path.of("clip-vit-base-patch32"));
 * Tensor pixels = clip.pixels(Path.of("cat.jpg")).reshape(1, 3, 224, 224);
 * float[][] p = clip.zeroShot(pixels, tok, List.of("a photo of a cat", "a photo of a dog"));
 * </pre>
 */
public final class VisionEncoder {

    /** The model families. */
    public enum Kind {
        /** OpenAI CLIP ({@code model_type} {@code clip}). */
        CLIP,
        /** Google SigLIP ({@code siglip}): sigmoid image-text scores, attention-pooled images. */
        SIGLIP,
        /** ViT ({@code vit}), with its classification head when present. */
        VIT,
        /** Meta DINOv2 ({@code dinov2}) image features. */
        DINOV2
    }

    private record Block(Tensor norm1, Tensor norm1b, Tensor q, Tensor qb, Tensor k, Tensor kb, Tensor v, Tensor vb,
            Tensor o, Tensor ob, Tensor norm2, Tensor norm2b, Tensor up, Tensor upb, Tensor down, Tensor downb,
            Tensor scale1, Tensor scale2) {
    }

    /** A stack of pre-norm transformer blocks. */
    private record Tower(List<Block> blocks, int hidden, int heads, float eps, String act) {
    }

    private final Kind kind;
    private final Device device;
    private final DType dtype;
    private final Tower vision;
    private final int patch;
    private final Tensor patchWeight;
    private final Tensor patchBias;
    private final Tensor cls;
    private final Tensor positions;
    private final Tensor preNorm;
    private final Tensor preNormBias;
    private final Tensor finalNorm;
    private final Tensor finalNormBias;
    private final Tensor projection;
    private final Map<String, Tensor> head = new HashMap<>();
    private final Tensor classifier;
    private final Tensor classifierBias;
    private final List<String> labels;
    private final Tower text;
    private final Tensor tokens;
    private final Tensor textPositions;
    private final Tensor textNorm;
    private final Tensor textNormBias;
    private final Tensor textProjection;
    private final Tensor textProjectionBias;
    private final int eos;
    private final int pad;
    private final int textLength;
    private final float logitScale;
    private final float logitBias;
    private final ImagePreprocessor preprocessor;
    private final Map<Long, Tensor> interpolated = new HashMap<>();

    private VisionEncoder(Map<String, Object> processor, Map<String, Object> config, WeightSource w, Device device,
            DType dtype)
            throws IOException {
        this.device = device;
        this.dtype = dtype;
        String type = Json.string(config, "model_type", "");
        kind = switch (type) {
            case "clip", "clip_vision_model" -> Kind.CLIP;
            case "siglip", "siglip_vision_model" -> Kind.SIGLIP;
            case "vit" -> Kind.VIT;
            case "dinov2" -> Kind.DINOV2;
            default -> throw new IllegalArgumentException("not a CLIP, SigLIP, ViT or DINOv2 model: model_type " + type);
        };
        // CLIP and SigLIP with their text towers, or their image towers alone (as in LLaVA)
        boolean pair = type.equals("clip") || type.equals("siglip");
        boolean towers = kind == Kind.CLIP || kind == Kind.SIGLIP;
        Map<String, Object> vc = towers ? defaults(pair ? config.get("vision_config") : config, kind, false) : config;
        String vp = towers ? "vision_model." : kind == Kind.VIT && w.contains("vit.embeddings.cls_token") ? "vit." : "";
        Function<String, Tensor> t = name -> w.tensor(name, device, dtype);
        patch = (int) Json.integer(vc, "patch_size", 16);
        int imageSize = (int) Json.integer(vc, "image_size", 224);
        String embed = vp + "embeddings.";
        Tensor conv = towers ? t.apply(embed + "patch_embedding.weight")
                : t.apply(embed + "patch_embeddings.projection.weight");
        int hidden = conv.shape().dim(0);
        patchWeight = conv.reshape(hidden, (int) (conv.size() / hidden)).transpose();
        String biasName = towers ? embed + "patch_embedding.bias" : embed + "patch_embeddings.projection.bias";
        patchBias = w.contains(biasName) ? t.apply(biasName) : null;
        if (kind == Kind.CLIP) {
            cls = t.apply(embed + "class_embedding").reshape(1, hidden);
        } else if (kind != Kind.SIGLIP) {
            cls = t.apply(embed + "cls_token").reshape(1, hidden);
        } else {
            cls = null;
        }
        Tensor pos = towers ? t.apply(embed + "position_embedding.weight") : t.apply(embed + "position_embeddings");
        positions = pos.reshape(pos.shape().dim(pos.shape().rank() - 2), hidden);
        vision = tower(w, t, vc, vp, kind == Kind.VIT || kind == Kind.DINOV2);
        if (kind == Kind.CLIP) {
            preNorm = t.apply(vp + "pre_layrnorm.weight");
            preNormBias = t.apply(vp + "pre_layrnorm.bias");
        } else {
            preNorm = preNormBias = null;
        }
        String fin = towers ? vp + "post_layernorm." : vp + "layernorm.";
        // llama.cpp's LLaVA projector files leave out the final norm, which the features skip anyway
        finalNorm = w.contains(fin + "weight") ? t.apply(fin + "weight") : null;
        finalNormBias = w.contains(fin + "bias") ? t.apply(fin + "bias") : null;
        projection = kind == Kind.CLIP && w.contains("visual_projection.weight")
                ? t.apply("visual_projection.weight").transpose() : null;
        if (kind == Kind.SIGLIP && w.contains(vp + "head.probe")) {
            String h = vp + "head.";
            Tensor in = t.apply(h + "attention.in_proj_weight");
            Tensor inBias = t.apply(h + "attention.in_proj_bias");
            head.put("probe", t.apply(h + "probe").reshape(1, hidden));
            head.put("q", in.slice(0, 0, hidden).transpose());
            head.put("k", in.slice(0, hidden, 2 * hidden).transpose());
            head.put("v", in.slice(0, 2 * hidden, 3 * hidden).transpose());
            head.put("qb", inBias.slice(0, 0, hidden));
            head.put("kb", inBias.slice(0, hidden, 2 * hidden));
            head.put("vb", inBias.slice(0, 2 * hidden, 3 * hidden));
            head.put("o", t.apply(h + "attention.out_proj.weight").transpose());
            head.put("ob", t.apply(h + "attention.out_proj.bias"));
            head.put("norm", t.apply(h + "layernorm.weight"));
            head.put("normb", t.apply(h + "layernorm.bias"));
            head.put("up", t.apply(h + "mlp.fc1.weight").transpose());
            head.put("upb", t.apply(h + "mlp.fc1.bias"));
            head.put("down", t.apply(h + "mlp.fc2.weight").transpose());
            head.put("downb", t.apply(h + "mlp.fc2.bias"));
        }
        boolean classifies = kind == Kind.VIT && w.contains("classifier.weight");
        classifier = classifies ? t.apply("classifier.weight").transpose() : null;
        classifierBias = classifies ? t.apply("classifier.bias") : null;
        List<String> names = new ArrayList<>();
        if (config.get("id2label") instanceof Map<?, ?> m) {
            for (int i = 0; i < m.size(); i++) {
                Object label = m.get(String.valueOf(i));
                names.add(label == null ? "LABEL_" + i : String.valueOf(label));
            }
        }
        labels = List.copyOf(names);
        if (pair) {
            Map<String, Object> tc = defaults(config.get("text_config"), kind, true);
            text = tower(w, t, tc, "text_model.", false);
            tokens = t.apply("text_model.embeddings.token_embedding.weight");
            textPositions = t.apply("text_model.embeddings.position_embedding.weight");
            textNorm = t.apply("text_model.final_layer_norm.weight");
            textNormBias = t.apply("text_model.final_layer_norm.bias");
            if (kind == Kind.CLIP) {
                textProjection = t.apply("text_projection.weight").transpose();
                textProjectionBias = null;
            } else {
                textProjection = t.apply("text_model.head.weight").transpose();
                textProjectionBias = t.apply("text_model.head.bias");
            }
            eos = (int) Json.integer(tc, "eos_token_id", 2);
            pad = (int) Json.integer(tc, "pad_token_id", kind == Kind.SIGLIP ? 1 : 0);
            textLength = (int) Json.integer(tc, "max_position_embeddings", 64);
            logitScale = w.tensor("logit_scale", Device.cpu(), DType.FLOAT32).reshape(1).toArray()[0];
            logitBias = w.contains("logit_bias")
                    ? w.tensor("logit_bias", Device.cpu(), DType.FLOAT32).reshape(1).toArray()[0] : 0;
        } else {
            text = null;
            tokens = textPositions = textNorm = textNormBias = textProjection = textProjectionBias = null;
            eos = pad = textLength = 0;
            logitScale = logitBias = 0;
        }
        preprocessor = new ImagePreprocessor(processor, imageSize);
    }

    /**
     * Fills in what transformers' CLIP and SigLIP configs leave out because it equals their
     * defaults: many hub configs list only the values that differ.
     */
    private static Map<String, Object> defaults(Object section, Kind kind, boolean text) {
        Map<String, Object> m = new java.util.HashMap<>(section == null ? Map.of() : Json.object(section));
        Map<String, Object> d;
        if (kind == Kind.CLIP) {
            d = text ? Map.of("hidden_size", 512, "intermediate_size", 2048, "num_hidden_layers", 12,
                    "num_attention_heads", 8, "max_position_embeddings", 77, "eos_token_id", 49407, "pad_token_id", 1)
                    : Map.of("hidden_size", 768, "intermediate_size", 3072, "num_hidden_layers", 12,
                    "num_attention_heads", 12, "image_size", 224, "patch_size", 32);
        } else {
            d = text ? Map.of("hidden_size", 768, "intermediate_size", 3072, "num_hidden_layers", 12,
                    "num_attention_heads", 12, "max_position_embeddings", 64, "eos_token_id", 1, "pad_token_id", 1)
                    : Map.of("hidden_size", 768, "intermediate_size", 3072, "num_hidden_layers", 12,
                    "num_attention_heads", 12, "image_size", 224, "patch_size", 16);
        }
        d.forEach(m::putIfAbsent);
        return m;
    }

    private Tower tower(WeightSource w, Function<String, Tensor> t, Map<String, Object> c, String prefix,
            boolean vitNames) {
        int n = (int) Json.integer(c, "num_hidden_layers");
        int heads = (int) Json.integer(c, "num_attention_heads");
        float eps = (float) Json.number(c, "layer_norm_eps", kind == Kind.CLIP ? 1e-5 : kind == Kind.VIT ? 1e-12 : 1e-6);
        String act = Json.string(c, "hidden_act",
                kind == Kind.CLIP ? "quick_gelu" : kind == Kind.SIGLIP ? "gelu_pytorch_tanh" : "gelu");
        List<Block> blocks = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (vitNames) {
                String p = prefix + "encoder.layer." + i + ".";
                String a = p + "attention.attention.";
                boolean dino = kind == Kind.DINOV2;
                blocks.add(new Block(
                        t.apply(p + (dino ? "norm1" : "layernorm_before") + ".weight"),
                        t.apply(p + (dino ? "norm1" : "layernorm_before") + ".bias"),
                        t.apply(a + "query.weight").transpose(), t.apply(a + "query.bias"),
                        t.apply(a + "key.weight").transpose(), t.apply(a + "key.bias"),
                        t.apply(a + "value.weight").transpose(), t.apply(a + "value.bias"),
                        t.apply(p + "attention.output.dense.weight").transpose(), t.apply(p + "attention.output.dense.bias"),
                        t.apply(p + (dino ? "norm2" : "layernorm_after") + ".weight"),
                        t.apply(p + (dino ? "norm2" : "layernorm_after") + ".bias"),
                        t.apply(p + (dino ? "mlp.fc1" : "intermediate.dense") + ".weight").transpose(),
                        t.apply(p + (dino ? "mlp.fc1" : "intermediate.dense") + ".bias"),
                        t.apply(p + (dino ? "mlp.fc2" : "output.dense") + ".weight").transpose(),
                        t.apply(p + (dino ? "mlp.fc2" : "output.dense") + ".bias"),
                        w.contains(p + "layer_scale1.lambda1") ? t.apply(p + "layer_scale1.lambda1") : null,
                        w.contains(p + "layer_scale2.lambda1") ? t.apply(p + "layer_scale2.lambda1") : null));
            } else {
                String p = prefix + "encoder.layers." + i + ".";
                String a = p + "self_attn.";
                blocks.add(new Block(t.apply(p + "layer_norm1.weight"), t.apply(p + "layer_norm1.bias"),
                        t.apply(a + "q_proj.weight").transpose(), t.apply(a + "q_proj.bias"),
                        t.apply(a + "k_proj.weight").transpose(), t.apply(a + "k_proj.bias"),
                        t.apply(a + "v_proj.weight").transpose(), t.apply(a + "v_proj.bias"),
                        t.apply(a + "out_proj.weight").transpose(), t.apply(a + "out_proj.bias"),
                        t.apply(p + "layer_norm2.weight"), t.apply(p + "layer_norm2.bias"),
                        t.apply(p + "mlp.fc1.weight").transpose(), t.apply(p + "mlp.fc1.bias"),
                        t.apply(p + "mlp.fc2.weight").transpose(), t.apply(p + "mlp.fc2.bias"), null, null));
            }
        }
        int hidden = blocks.isEmpty() ? (int) Json.integer(c, "hidden_size") : blocks.getFirst().norm1.shape().dim(0);
        return new Tower(List.copyOf(blocks), hidden, heads, eps, act);
    }

    /**
     * Loads a model from a folder with {@code config.json}, weights and (for preprocessing)
     * {@code preprocessor_config.json}.
     *
     * @param dir the model folder
     * @param device where the weights go
     * @param dtype the weights' type
     * @return the model
     */
    public static VisionEncoder load(Path dir, Device device, DType dtype) {
        try (WeightSource w = WeightSource.open(dir)) {
            return new VisionEncoder(processorConfig(dir), Json.object(Json.read(dir.resolve("config.json"))), w,
                    device, dtype);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** @return the folder's {@code preprocessor_config.json}, or an empty map */
    static Map<String, Object> processorConfig(Path dir) throws IOException {
        Path pre = dir.resolve("preprocessor_config.json");
        return Files.isRegularFile(pre) ? Json.object(Json.read(pre)) : Map.of();
    }

    /** @return the model family */
    public Kind kind() {
        return kind;
    }

    /** @return the classifier's labels by index (ViT), empty without one */
    public List<String> labels() {
        return labels;
    }

    // ------------------------------------------------------------------ images

    /**
     * Turns an image into the model's input as its processor does: resized, cropped,
     * rescaled and normalized.
     *
     * @param image the image
     * @return {@code [3, height, width]} on the model's device
     */
    public Tensor pixels(BufferedImage image) {
        int[] size = new int[2];
        float[] values = preprocessor.apply(image, size);
        return Tensor.of(Shape.of(3, size[0], size[1]), values, device).to(dtype);
    }

    /**
     * Reads an image file (any format ImageIO reads: PNG, JPEG, BMP, GIF) and
     * {@linkplain #pixels(BufferedImage) preprocesses} it.
     *
     * @param file the image
     * @return {@code [3, height, width]}
     */
    public Tensor pixels(Path file) {
        try {
            BufferedImage image = ImageIO.read(file.toFile());
            if (image == null) {
                throw new IllegalArgumentException(file + " is not an image ImageIO can read");
            }
            return pixels(image);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /**
     * The image tower's final token vectors (CLIP's before its post-norm, as transformers'
     * {@code last_hidden_state}).
     *
     * @param pixels {@code [batch, 3, height, width]}
     * @return {@code [batch, tokens, hidden]}
     */
    public Tensor hidden(Tensor pixels) {
        return Var.noGrad(() -> visionStates(pixels)).value();
    }

    /**
     * One vector per image: CLIP's projected embedding, SigLIP's attention-pooled one, the
     * class token for ViT and DINOv2.
     *
     * @param pixels {@code [batch, 3, height, width]}
     * @param normalize whether to scale each to unit length
     * @return {@code [batch][dim]}
     */
    public float[][] embedImages(Tensor pixels, boolean normalize) {
        return rows(Var.noGrad(() -> pooledImages(pixels)).value(), normalize);
    }

    /**
     * The classifier's logits (ViT).
     *
     * @param pixels {@code [batch, 3, height, width]}
     * @return {@code [batch][labels]}
     */
    public float[][] classify(Tensor pixels) {
        if (classifier == null) {
            throw new IllegalStateException("this model has no classification head");
        }
        return rows(Var.noGrad(() -> visionStates(pixels).slice(1, 0, 1).reshape(pixels.shape().dim(0),
                vision.hidden).matmul(classifier).add(classifierBias)).value(), false);
    }

    /**
     * Loads the image tower of a larger model (a vision-language model's) from its weights.
     *
     * @param config the tower's config ({@code model_type} {@code clip_vision_model} or
     *     {@code siglip_vision_model})
     * @param preprocessor the processor config, or an empty map
     * @param w the weights, named from {@code vision_model.}
     */
    static VisionEncoder tower(Map<String, Object> config, Map<String, Object> preprocessor, WeightSource w,
            Device device, DType dtype) {
        try {
            return new VisionEncoder(preprocessor, config, w, device, dtype);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** @return the width of the image tower's states */
    int width() {
        return vision.hidden;
    }

    /** @return the image tower's block count */
    int depth() {
        return vision.blocks.size();
    }

    /**
     * The image tower's states after {@code layer} blocks, as transformers' {@code
     * hidden_states[layer]}: 0 for the embeddings, negative from the end ({@code -1} after the
     * last block, before any final norm).
     */
    Var features(Tensor pixels, int layer) {
        int n = vision.blocks.size();
        int stop = layer < 0 ? n + 1 + layer : layer;
        if (stop < 0 || stop > n) {
            throw new IllegalArgumentException("layer " + layer + " of a " + n + "-block tower");
        }
        return run(vision, embeddings(pixels), false, stop);
    }

    /** The image tower's final states, after its final norm (SigLIP's {@code last_hidden_state}). */
    Var states(Tensor pixels) {
        return visionStates(pixels);
    }

    private Var embeddings(Tensor pixels) {
        int b = pixels.shape().dim(0);
        int h = pixels.shape().dim(2) / patch;
        int w = pixels.shape().dim(3) / patch;
        int d = vision.hidden;
        Tensor x = pixels.to(dtype).reshape(b, 3, h, patch, w, patch).permute(0, 2, 4, 1, 3, 5)
                .reshape(b * h * w, 3 * patch * patch).matmul(patchWeight);
        if (patchBias != null) {
            x = x.add(patchBias);
        }
        x = x.reshape(b, h * w, d);
        if (cls != null) {
            x = Tensor.concat(1, cls.reshape(1, 1, d).expand(Shape.of(b, 1, d)), x);
        }
        x = x.add(positions(h, w));
        Var s = Var.of(x);
        if (preNorm != null) {
            s = norm(s, preNorm, preNormBias, vision.eps);
        }
        return s;
    }

    private Var visionStates(Tensor pixels) {
        Var s = run(vision, embeddings(pixels), false);
        if (kind == Kind.CLIP || finalNorm == null) {
            return s;
        }
        return norm(s, finalNorm, finalNormBias, vision.eps);
    }

    /** The position embeddings for an {@code h x w} grid: DINOv2 interpolates them bicubically. */
    private Tensor positions(int h, int w) {
        int extra = cls != null ? 1 : 0;
        int n = positions.shape().dim(0) - extra;
        if (n == h * w && (kind != Kind.DINOV2 || h == w)) {
            return positions;
        }
        if (kind != Kind.DINOV2) {
            throw new IllegalArgumentException("the model takes " + n + " patches, the image has " + h + " x " + w);
        }
        return interpolated.computeIfAbsent((long) h << 32 | w, key -> {
            int side = (int) Math.round(Math.sqrt(n));
            int d = vision.hidden;
            float[] grid = positions.slice(0, extra, extra + n).to(DType.FLOAT32).toArray();
            float[] out = new float[(h * w + extra) * d];
            float[] first = positions.slice(0, 0, extra).to(DType.FLOAT32).toArray();
            System.arraycopy(first, 0, out, 0, first.length);
            double[][] wy = cubicWeights(side, h);
            double[][] wx = cubicWeights(side, w);
            for (int y = 0; y < h; y++) {
                for (int x = 0; x < w; x++) {
                    for (int c = 0; c < d; c++) {
                        double v = 0;
                        for (int i = 0; i < 4; i++) {
                            int sy = (int) wy[y][4 + i];
                            for (int j = 0; j < 4; j++) {
                                int sx = (int) wx[x][4 + j];
                                v += wy[y][i] * wx[x][j] * grid[(sy * side + sx) * d + c];
                            }
                        }
                        out[(extra + y * w + x) * d + c] = (float) v;
                    }
                }
            }
            return Tensor.of(Shape.of(h * w + extra, d), out, device).to(dtype);
        });
    }

    /** PyTorch's bicubic weights (a = -0.75, align_corners false): 4 weights, then 4 clamped indices. */
    private static double[][] cubicWeights(int in, int out) {
        double[][] r = new double[out][8];
        double scale = (double) in / out;
        double a = -0.75;
        for (int i = 0; i < out; i++) {
            double src = (i + 0.5) * scale - 0.5;
            int base = (int) Math.floor(src);
            double t = src - base;
            double[] wts = {
                ((a * (t + 1) - 5 * a) * (t + 1) + 8 * a) * (t + 1) - 4 * a,
                ((a + 2) * t - (a + 3)) * t * t + 1,
                ((a + 2) * (1 - t) - (a + 3)) * (1 - t) * (1 - t) + 1,
                ((a * (2 - t) - 5 * a) * (2 - t) + 8 * a) * (2 - t) - 4 * a};
            for (int k = 0; k < 4; k++) {
                r[i][k] = wts[k];
                r[i][4 + k] = Math.max(0, Math.min(in - 1, base - 1 + k));
            }
        }
        return r;
    }

    private Var pooledImages(Tensor pixels) {
        int b = pixels.shape().dim(0);
        int d = vision.hidden;
        Var s = visionStates(pixels);
        return switch (kind) {
            case CLIP -> norm(s.slice(1, 0, 1).reshape(b, d), finalNorm, finalNormBias, vision.eps).matmul(projection);
            case SIGLIP -> head.isEmpty() ? s.mean(1, false) : attentionPool(s, b);
            default -> s.slice(1, 0, 1).reshape(b, d);
        };
    }

    /** SigLIP's head: a learned probe attends over the tokens, then a residual MLP. */
    private Var attentionPool(Var s, int b) {
        int d = vision.hidden;
        int heads = vision.heads;
        int dh = d / heads;
        int n = s.shape().dim(1);
        Var flat = s.reshape(b * n, d);
        Var q = Var.of(head.get("probe").matmul(head.get("q")).add(head.get("qb")).reshape(1, 1, heads, dh)
                .permute(0, 2, 1, 3).expand(Shape.of(b, heads, 1, dh)));
        Var k = flat.matmul(head.get("k")).add(head.get("kb")).reshape(b, n, heads, dh).permute(0, 2, 1, 3);
        Var v = flat.matmul(head.get("v")).add(head.get("vb")).reshape(b, n, heads, dh).permute(0, 2, 1, 3);
        Var a = Functional.attention(q, k, v, false, (float) (1 / Math.sqrt(dh))).permute(0, 2, 1, 3).reshape(b, d);
        Var o = a.matmul(head.get("o")).add(head.get("ob"));
        Var m = norm(o, head.get("norm"), head.get("normb"), vision.eps);
        Var up = activate(m.matmul(head.get("up")).add(head.get("upb")), vision.act);
        return o.add(up.matmul(head.get("down")).add(head.get("downb")));
    }

    // ------------------------------------------------------------------ text

    /**
     * Text embeddings from token ids (CLIP and SigLIP): CLIP pools at each row's end-of-text
     * token, SigLIP at the last position, so SigLIP rows should be padded to the length it
     * was trained with (see {@link #embedTexts(PretrainedTokenizer, List, boolean)}).
     *
     * @param ids {@code [batch][steps]} token ids
     * @param normalize whether to scale each to unit length
     * @return {@code [batch][dim]}
     */
    public float[][] embedTexts(int[][] ids, boolean normalize) {
        return rows(Var.noGrad(() -> pooledTexts(ids)).value(), normalize);
    }

    /**
     * Tokenizes and embeds texts: CLIP pads to the longest, SigLIP to its full length.
     *
     * @param tokenizer the model's tokenizer
     * @param texts the texts
     * @param normalize whether to scale each to unit length
     * @return {@code [texts][dim]}
     */
    public float[][] embedTexts(PretrainedTokenizer tokenizer, List<String> texts, boolean normalize) {
        return embedTexts(tokenize(tokenizer, texts), normalize);
    }

    private int[][] tokenize(PretrainedTokenizer tokenizer, List<String> texts) {
        requireText();
        List<int[]> encoded = new ArrayList<>();
        int longest = 1;
        for (String s : texts) {
            int[] e = tokenizer.encode(s, true);   // with the start and end tokens the pooling looks for
            if (e.length > textLength) {
                e = Arrays.copyOf(e, textLength);
                if (kind == Kind.CLIP) {
                    e[textLength - 1] = eos;
                }
            }
            encoded.add(e);
            longest = Math.max(longest, e.length);
        }
        int width = kind == Kind.SIGLIP ? textLength : longest;
        int[][] ids = new int[texts.size()][width];
        for (int i = 0; i < ids.length; i++) {
            Arrays.fill(ids[i], pad);
            int[] e = encoded.get(i);
            System.arraycopy(e, 0, ids[i], 0, e.length);
        }
        return ids;
    }

    private Var pooledTexts(int[][] ids) {
        requireText();
        int b = ids.length;
        int steps = ids[0].length;
        int d = text.hidden;
        float[] tok = new float[b * steps];
        float[] pos = new float[b * steps];
        float[] at = new float[b];
        for (int i = 0; i < b; i++) {
            int end = -1;
            for (int s = 0; s < steps; s++) {
                tok[i * steps + s] = ids[i][s];
                pos[i * steps + s] = s;
                if (end < 0 && ids[i][s] == eos) {
                    end = s;
                }
            }
            if (kind == Kind.CLIP && eos == 2) {   // old CLIP configs: the end token has the largest id
                end = 0;
                for (int s = 1; s < steps; s++) {
                    if (ids[i][s] > ids[i][end]) {
                        end = s;
                    }
                }
            }
            at[i] = i * steps + (kind == Kind.SIGLIP || end < 0 ? steps - 1 : end);
        }
        Tensor x = tokens.gather(0, Tensor.of(Shape.of(b * steps), tok, device))
                .add(textPositions.gather(0, Tensor.of(Shape.of(b * steps), pos, device)));
        Var s = run(text, Var.of(x.reshape(b, steps, d)), kind == Kind.CLIP);
        s = norm(s, textNorm, textNormBias, text.eps).reshape(b * steps, d);
        Var pooled = s.gather(0, Tensor.of(Shape.of(b), at, device));
        Var out = pooled.matmul(textProjection);
        return textProjectionBias != null ? out.add(textProjectionBias) : out;
    }

    /**
     * Image-text scores as the model was trained with: CLIP's scaled cosine similarities (a
     * softmax over texts gives probabilities), SigLIP's logits (a sigmoid gives each pair's
     * probability).
     *
     * @param pixels {@code [images, 3, height, width]}
     * @param ids {@code [texts][steps]} token ids
     * @return {@code [images][texts]}
     */
    public float[][] logits(Tensor pixels, int[][] ids) {
        float[][] images = embedImages(pixels, true);
        float[][] texts = embedTexts(ids, true);
        float scale = (float) Math.exp(logitScale);
        float[][] out = new float[images.length][texts.length];
        for (int i = 0; i < images.length; i++) {
            for (int j = 0; j < texts.length; j++) {
                double dot = 0;
                for (int k = 0; k < images[i].length; k++) {
                    dot += images[i][k] * texts[j][k];
                }
                out[i][j] = (float) (dot * scale + logitBias);
            }
        }
        return out;
    }

    /**
     * Zero-shot classification: how well each image matches each label text (such as
     * "a photo of a cat"). CLIP's are a softmax over the labels, SigLIP's independent
     * sigmoids.
     *
     * @param pixels {@code [images, 3, height, width]}
     * @param tokenizer the model's tokenizer
     * @param labels the label texts
     * @return {@code [images][labels]} probabilities
     */
    public float[][] zeroShot(Tensor pixels, PretrainedTokenizer tokenizer, List<String> labels) {
        float[][] l = logits(pixels, tokenize(tokenizer, labels));
        for (float[] row : l) {
            if (kind == Kind.SIGLIP) {
                for (int j = 0; j < row.length; j++) {
                    row[j] = (float) (1 / (1 + Math.exp(-row[j])));
                }
            } else {
                float max = Float.NEGATIVE_INFINITY;
                for (float v : row) {
                    max = Math.max(max, v);
                }
                double sum = 0;
                for (int j = 0; j < row.length; j++) {
                    row[j] = (float) Math.exp(row[j] - max);
                    sum += row[j];
                }
                for (int j = 0; j < row.length; j++) {
                    row[j] /= (float) sum;
                }
            }
        }
        return l;
    }

    private void requireText() {
        if (text == null) {
            throw new IllegalStateException("a " + kind + " model has no text tower");
        }
    }

    // ------------------------------------------------------------------ blocks

    private Var run(Tower tower, Var x, boolean causal) {
        return run(tower, x, causal, tower.blocks.size());
    }

    private Var run(Tower tower, Var x, boolean causal, int stop) {
        int b = x.shape().dim(0);
        int n = x.shape().dim(1);
        int d = tower.hidden;
        int dh = d / tower.heads;
        float scale = (float) (1 / Math.sqrt(dh));
        Var h = x.reshape(b * n, d);
        for (Block l : tower.blocks.subList(0, stop)) {
            Var a = norm(h, l.norm1, l.norm1b, tower.eps);
            Var q = a.matmul(l.q).add(l.qb).reshape(b, n, tower.heads, dh).permute(0, 2, 1, 3);
            Var k = a.matmul(l.k).add(l.kb).reshape(b, n, tower.heads, dh).permute(0, 2, 1, 3);
            Var v = a.matmul(l.v).add(l.vb).reshape(b, n, tower.heads, dh).permute(0, 2, 1, 3);
            Var att = Functional.attention(q, k, v, causal, scale).permute(0, 2, 1, 3).reshape(b * n, d);
            Var o = att.matmul(l.o).add(l.ob);
            h = h.add(l.scale1 != null ? o.mul(l.scale1) : o);
            Var m = norm(h, l.norm2, l.norm2b, tower.eps);
            Var up = activate(m.matmul(l.up).add(l.upb), tower.act);
            Var down = up.matmul(l.down).add(l.downb);
            h = h.add(l.scale2 != null ? down.mul(l.scale2) : down);
        }
        return h.reshape(b, n, d);
    }

    private static Var activate(Var x, String act) {
        return switch (act) {
            case "quick_gelu" -> x.mul(x.mul(1.702f).sigmoid());
            case "gelu" -> x.activation(ActivationOp.GELU, 0);
            case "gelu_new", "gelu_pytorch_tanh", "gelu_fast" -> x.activation(ActivationOp.GELU_TANH, 0);
            case "relu" -> x.activation(ActivationOp.RELU, 0);
            default -> throw new IllegalArgumentException("hidden_act " + act + " is not supported");
        };
    }

    private static Var norm(Var x, Tensor w, Tensor b, float eps) {
        Var centered = x.sub(x.mean(-1, true));
        return centered.div(centered.square().mean(-1, true).add(eps).sqrt()).mul(w).add(b);
    }

    private static float[][] rows(Tensor t, boolean normalize) {
        int b = t.shape().dim(0);
        int d = (int) (t.size() / b);
        float[] v = t.to(DType.FLOAT32).toArray();
        float[][] out = new float[b][d];
        for (int i = 0; i < b; i++) {
            System.arraycopy(v, i * d, out[i], 0, d);
            if (normalize) {
                double n = 0;
                for (float f : out[i]) {
                    n += f * f;
                }
                float inv = (float) (1 / Math.sqrt(Math.max(n, 1e-24)));
                for (int j = 0; j < d; j++) {
                    out[i][j] *= inv;
                }
            }
        }
        return out;
    }

    @Override
    public String toString() {
        return "VisionEncoder[" + kind + ", " + vision.blocks.size() + " layers, " + vision.hidden + " wide"
                + (text != null ? ", text tower" : "") + "]";
    }
}
