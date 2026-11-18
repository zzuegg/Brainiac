package io.github.zzuegg.brainiac.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.io.WeightSource;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.tensor.spi.ActivationOp;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Map;

/**
 * A vision-language model from a Hugging Face folder: LLaVA 1.5 and Gemma 3, where an
 * image tower (CLIP or SigLIP), a projector and a decoder ({@link Llama}, any family it
 * loads) answer questions about images. Each image becomes a run of placeholder tokens in the
 * prompt, whose embeddings are the projected image features. Results match transformers.
 *
 * <p>The model is a {@link LanguageModel}, so a {@link Generator} drives it: {@link #prompt}
 * expands a prompt's image tokens and queues the images' features, which the next call that
 * reads placeholders takes.
 *
 * <pre>
 * VisionLanguageModel llava = VisionLanguageModel.load(Path.of("llava-1.5-7b-hf"), Device.best(), DType.FLOAT16);
 * PretrainedTokenizer tok = PretrainedTokenizer.load(Path.of("llava-1.5-7b-hf"));
 * Generator g = new Generator(llava, tok);
 * int[] ids = llava.prompt(tok.encode("USER: &lt;image&gt;\nWhat is in the picture? ASSISTANT:"),
 *         List.of(ImageIO.read(new File("cat.png"))));
 * String answer = g.generate(new int[][] {ids}, Generator.Options.greedy(100), null).getFirst().text();
 * </pre>
 */
public final class VisionLanguageModel implements LanguageModel {

    /** The model families. */
    public enum Kind {
        /** LLaVA 1.5 ({@code llava}): a CLIP or SigLIP layer's patches through a two-layer MLP. */
        LLAVA,
        /**
         * Gemma 3 ({@code gemma3}): SigLIP patches average-pooled to 256 tokens, normalized and
         * projected; an image's tokens see each other both ways.
         */
        GEMMA3
    }

    private final Kind kind;
    private final Llama language;
    private final VisionEncoder vision;
    private final Tensor proj1;
    private final Tensor proj1Bias;
    private final Tensor proj2;
    private final Tensor proj2Bias;
    private final ActivationOp act;
    private final Tensor softNorm;
    private final float softEps;
    private final int tokensPerImage;
    private final int imageToken;
    private final int featureLayer;
    private final boolean dropClass;
    private final Device device;
    private final DType dtype;
    /** Image features queued by {@link #prompt}, taken by the calls that read placeholders. */
    private final Deque<Tensor> pending = new ArrayDeque<>();

    private VisionLanguageModel(Map<String, Object> c, Map<String, Object> processor, WeightSource raw, Device device,
            DType dtype) {
        String type = Json.string(c, "model_type", "");
        kind = switch (type) {
            case "llava" -> Kind.LLAVA;
            case "gemma3" -> Kind.GEMMA3;
            default -> throw new IllegalArgumentException("not a LLaVA or Gemma 3 model: model_type " + type);
        };
        this.device = device;
        this.dtype = dtype;
        WeightSource w = raw.renamed(VisionLanguageModel::canonical);
        Map<String, Object> textConfig = new java.util.LinkedHashMap<>(Json.object(c.get("text_config")));
        if (kind == Kind.GEMMA3) {
            textConfig.putIfAbsent("model_type", "gemma3_text");
        }
        language = Llama.load(w.renamed(n -> n.startsWith("model.") || n.startsWith("lm_head.") ? n : null),
                LlamaConfig.fromHuggingFace(textConfig), device, dtype);
        Map<String, Object> visionConfig = new java.util.LinkedHashMap<>(Json.object(c.get("vision_config")));
        visionConfig.putIfAbsent("model_type", kind == Kind.GEMMA3 ? "siglip_vision_model" : "clip_vision_model");
        vision = VisionEncoder.tower(visionConfig, processor,
                w.renamed(n -> n.startsWith("vision_model.") ? n : null), device, dtype);
        if (kind == Kind.GEMMA3) {
            // x / rms(x) * (1 + w), then one matrix stored [vision, text]
            softNorm = w.tensor("projector.mm_soft_emb_norm.weight", Device.cpu(), DType.FLOAT32).add(1f).to(device);
            softEps = (float) Json.number(visionConfig, "layer_norm_eps", 1e-6);
            proj1 = w.tensor("projector.mm_input_projection_weight", device, dtype);
            proj1Bias = proj2 = proj2Bias = null;
            act = null;
            tokensPerImage = (int) Json.integer(c, "mm_tokens_per_image", 256);
            imageToken = (int) Json.integer(c, "image_token_index", Json.integer(c, "image_token_id", 262144));
            featureLayer = -1;
            dropClass = false;
            return;
        }
        softNorm = null;
        softEps = 0;
        tokensPerImage = 0;
        proj1 = w.tensor("projector.linear_1.weight", device, dtype).transpose();
        proj2 = w.tensor("projector.linear_2.weight", device, dtype).transpose();
        proj1Bias = w.contains("projector.linear_1.bias") ? w.tensor("projector.linear_1.bias", device, dtype) : null;
        proj2Bias = w.contains("projector.linear_2.bias") ? w.tensor("projector.linear_2.bias", device, dtype) : null;
        String a = Json.string(c, "projector_hidden_act", "gelu");
        act = switch (a) {
            case "gelu" -> ActivationOp.GELU;
            case "gelu_pytorch_tanh", "gelu_new" -> ActivationOp.GELU_TANH;
            case "silu" -> ActivationOp.SILU;
            case "relu" -> ActivationOp.RELU;
            default -> throw new IllegalArgumentException("projector_hidden_act " + a + " is not supported");
        };
        imageToken = (int) Json.integer(c, "image_token_index", Json.integer(c, "image_token_id", 32000));
        if (!(c.get("vision_feature_layer") instanceof Number) && c.get("vision_feature_layer") != null) {
            throw new IllegalArgumentException("features from several layers are not supported");
        }
        featureLayer = (int) Json.integer(c, "vision_feature_layer", -2);
        dropClass = !"full".equals(Json.string(c, "vision_feature_select_strategy", "default"));
    }

    /** Hub names, old and new layouts, to {@code model.}, {@code lm_head.}, {@code vision_model.}, {@code projector.}. */
    private static String canonical(String n) {
        if (n.startsWith("model.vision_tower.") || n.startsWith("model.multi_modal_projector.")
                || n.startsWith("model.language_model.")) {
            n = n.substring("model.".length());
            if (n.startsWith("language_model.")) {
                return "model." + n.substring("language_model.".length());
            }
        }
        if (n.startsWith("language_model.model.")) {
            return n.substring("language_model.".length());
        }
        if (n.startsWith("language_model.lm_head.")) {
            return n.substring("language_model.".length());
        }
        if (n.startsWith("vision_tower.vision_model.")) {
            return n.substring("vision_tower.".length());
        }
        if (n.startsWith("vision_tower.")) {
            return "vision_model." + n.substring("vision_tower.".length());
        }
        if (n.startsWith("multi_modal_projector.")) {
            return "projector." + n.substring("multi_modal_projector.".length());
        }
        return n;
    }

    /**
     * Loads a model from a folder with {@code config.json}, weights and
     * {@code preprocessor_config.json}.
     *
     * @param dir the model folder
     * @param device where the weights go
     * @param dtype the weights' type
     * @return the model
     */
    public static VisionLanguageModel load(Path dir, Device device, DType dtype) {
        try (WeightSource w = WeightSource.open(dir)) {
            return new VisionLanguageModel(Json.object(Json.read(dir.resolve("config.json"))),
                    VisionEncoder.processorConfig(dir), w, device, dtype);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** @return the model family */
    public Kind kind() {
        return kind;
    }

    /** @return the decoder */
    public Llama language() {
        return language;
    }

    /** @return the token that stands for one image feature in a prompt */
    public int imageToken() {
        return imageToken;
    }

    /**
     * Preprocesses an image as the model's processor does.
     *
     * @param image the image
     * @return {@code [3, height, width]}
     */
    public Tensor pixels(BufferedImage image) {
        return vision.pixels(image);
    }

    /**
     * The projected features of images, one row per placeholder token.
     *
     * @param pixels {@code [batch, 3, height, width]}
     * @return {@code [batch * tokens per image, hidden]} in the decoder's width
     */
    public Tensor imageFeatures(Tensor pixels) {
        if (kind == Kind.GEMMA3) {
            return Var.noGrad(() -> Var.of(gemmaFeatures(pixels))).value();
        }
        return Var.noGrad(() -> {
            Var f = vision.features(pixels, featureLayer);
            if (dropClass) {
                f = f.slice(1, 1, f.shape().dim(1));
            }
            int rows = f.shape().dim(0) * f.shape().dim(1);
            Tensor h = f.value().reshape(rows, vision.width()).matmul(proj1);
            if (proj1Bias != null) {
                h = h.add(proj1Bias);
            }
            h = h.activation(act, 0).matmul(proj2);
            return Var.of(proj2Bias != null ? h.add(proj2Bias) : h);
        }).value();
    }

    /** Gemma 3: the patch grid average-pooled to {@code tokensPerImage}, normalized, projected. */
    private Tensor gemmaFeatures(Tensor pixels) {
        Tensor s = vision.states(pixels).value();
        int b = s.shape().dim(0);
        int n = s.shape().dim(1);
        int d = s.shape().dim(2);
        int side = (int) Math.round(Math.sqrt(n));
        int out = (int) Math.round(Math.sqrt(tokensPerImage));
        int k = side / out;
        // [b, side, side, d] -> [b, out, k, out, k, d] -> mean over both k
        Tensor pooled = s.to(DType.FLOAT32).reshape(b, out, k, out, k, d).mean(4, false).mean(2, false)
                .reshape(b * out * out, d);
        Tensor r = pooled.mul(pooled).mean(-1, true).add(softEps).sqrt();
        return pooled.div(r).mul(softNorm).to(dtype).matmul(proj1);
    }

    /**
     * Tokenizes a prompt with images marked as the model's processor marks them ({@code
     * <image>} for LLaVA, {@code <start_of_image>} for Gemma 3), expands each image into its
     * tokens and queues the images' features, as {@link #prompt(int[], List)} does.
     *
     * @param tokenizer the model's tokenizer
     * @param text the prompt, with one marker per image
     * @param images the images, in prompt order
     * @return the expanded prompt
     */
    public int[] prompt(io.github.zzuegg.brainiac.io.PretrainedTokenizer tokenizer, String text,
            List<BufferedImage> images) {
        return prompt(tokenizer.encode(markImages(text)), images);
    }

    /**
     * The text as the model's processor rewrites its image markers before tokenizing: for
     * Gemma 3, each {@code <start_of_image>} becomes the image's opening, one image token
     * and its closing; for LLaVA the text stays as it is.
     *
     * @param text a prompt with one marker per image
     * @return the text to tokenize
     */
    public String markImages(String text) {
        if (kind == Kind.GEMMA3) {
            return text.replace("<start_of_image>", "\n\n<start_of_image><image_soft_token><end_of_image>\n\n");
        }
        return text;
    }

    /**
     * Readies a prompt with images: each image token in {@code ids} becomes as many as the
     * image has features, and the images' features are queued for the call that reads them.
     *
     * @param ids the tokenized prompt, one image token per image
     * @param images the images, in prompt order
     * @return the expanded prompt
     */
    public int[] prompt(int[] ids, List<BufferedImage> images) {
        int count = 0;
        for (int id : ids) {
            count += id == imageToken ? 1 : 0;
        }
        if (count != images.size()) {
            throw new IllegalArgumentException("the prompt has " + count + " image tokens for " + images.size()
                    + " images");
        }
        List<Integer> out = new ArrayList<>();
        int image = 0;
        for (int id : ids) {
            if (id != imageToken) {
                out.add(id);
                continue;
            }
            Tensor px = pixels(images.get(image++));
            Tensor f = imageFeatures(px.reshape(1, px.shape().dim(0), px.shape().dim(1), px.shape().dim(2)));
            pending.add(f);
            for (int i = 0; i < f.shape().dim(0); i++) {
                out.add(imageToken);
            }
        }
        return out.stream().mapToInt(Integer::intValue).toArray();
    }

    /**
     * Logits for whole sequences (teacher forcing) whose image tokens are already expanded.
     *
     * @param ids {@code [batch][steps]}
     * @param pixels {@code [images, 3, height, width]}, in the order their tokens appear
     * @return {@code [batch, steps, vocab]}
     */
    public Tensor logits(int[][] ids, Tensor pixels) {
        Tensor f = imageFeatures(pixels);
        return Var.noGrad(() -> language.forward(ids, imageToken, f, kind == Kind.GEMMA3)).value();
    }

    @Override
    public int vocabSize() {
        return language.vocabSize();
    }

    @Override
    public State newState(int batch) {
        return language.newState(batch);
    }

    @Override
    public Tensor next(State state, int[][] tokens, boolean[][] padding) {
        int count = 0;
        for (int[] row : tokens) {
            for (int t : row) {
                count += t == imageToken ? 1 : 0;
            }
        }
        if (count == 0 || pending.isEmpty()) {
            // a generated image token, with no image waiting, reads as an ordinary token
            return language.next(state, tokens, padding);
        }
        if (padding != null) {
            for (boolean[] row : padding) {
                for (boolean p : row) {
                    if (p) {
                        throw new IllegalArgumentException("prompts with images are read without padding");
                    }
                }
            }
        }
        // the queued features, as many rows as placeholders
        List<Tensor> parts = new ArrayList<>();
        int have = 0;
        while (have < count) {
            Tensor f = pending.poll();
            if (f == null) {
                throw new IllegalStateException("the tokens hold " + count + " image tokens; prepare them with prompt()");
            }
            int take = Math.min(f.shape().dim(0), count - have);
            parts.add(f.slice(0, 0, take));
            if (take < f.shape().dim(0)) {
                pending.addFirst(f.slice(0, take, f.shape().dim(0)));
            }
            have += take;
        }
        Tensor rows = parts.size() == 1 ? parts.getFirst() : Tensor.concat(0, parts.toArray(Tensor[]::new));
        return language.next(state, tokens, imageToken, rows, kind == Kind.GEMMA3);
    }

    @Override
    public String toString() {
        return "VisionLanguageModel[" + kind + ", " + vision.kind() + " " + vision.depth() + " blocks, " + language + "]";
    }
}
