package io.github.zzuegg.brainiac.diffusion;

import io.github.zzuegg.brainiac.io.WeightSource;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.diffusion.AutoencoderKL;
import io.github.zzuegg.brainiac.nn.diffusion.ClipTextEncoder;
import io.github.zzuegg.brainiac.nn.diffusion.UNet;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.function.UnaryOperator;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Single-file checkpoints in the original Stable Diffusion layout ({@code model.diffusion_model.*},
 * {@code first_stage_model.*}, {@code cond_stage_model.*} or {@code conditioner.embedders.*}),
 * as image tools share them. Their weight names are translated to diffusers' on the fly.
 */
final class SingleFile {

    private SingleFile() {
    }

    private static final String UNET = "model.diffusion_model.";
    private static final String VAE = "first_stage_model.";

    static Optional<StableDiffusion.Kind> detect(WeightSource w) {
        if (!w.contains(UNET + "input_blocks.0.0.weight")) {
            return Optional.empty();
        }
        if (w.contains("conditioner.embedders.1.model.token_embedding.weight")
                && w.contains("conditioner.embedders.0.transformer.text_model.embeddings.token_embedding.weight")) {
            return Optional.of(StableDiffusion.Kind.SDXL);
        }
        if (w.contains("cond_stage_model.model.token_embedding.weight")) {
            return Optional.of(StableDiffusion.Kind.SD2);
        }
        if (w.contains("cond_stage_model.transformer.text_model.embeddings.token_embedding.weight")) {
            return Optional.of(StableDiffusion.Kind.SD1);
        }
        return Optional.empty();
    }

    /**
     * What a checkpoint does not say about its architecture: group-norm groups and attention
     * head sizes. Stable Diffusion uses 32 groups, 8 heads in 1.x and 64-wide heads in 2.x and XL.
     */
    record Arch(int unetGroups, int vaeGroups, int sd1Heads, int headDim, int clipHeadDim, float vaeScaling) {
        static final Arch STANDARD = new Arch(32, 32, 8, 64, 64, 0);
    }

    /** Reads the UNet's shape from its weights. */
    static UNet.Settings unetConfig(WeightSource w, Arch arch) {
        int levels = 1;
        for (int i = 1; w.contains(UNET + "input_blocks." + i + ".0.in_layers.0.weight")
                || w.contains(UNET + "input_blocks." + i + ".0.op.weight"); i++) {
            if (w.contains(UNET + "input_blocks." + i + ".0.op.weight")) {
                levels++;
            }
        }
        int outputs = 0;
        while (w.contains(UNET + "output_blocks." + outputs + ".0.out_layers.3.weight")) {
            outputs++;
        }
        int per = outputs / levels - 1;
        int[] channels = new int[levels];
        boolean[] cross = new boolean[levels];
        int[] heads = new int[levels];
        int[] depth = new int[levels];
        int context = 0;
        boolean linear = false;
        for (int i = 0; i < levels; i++) {
            String block = UNET + "input_blocks." + (1 + i * (per + 1)) + ".";
            channels[i] = w.shape(block + "0.out_layers.3.weight").dim(0);
            cross[i] = w.contains(block + "1.norm.weight");
            depth[i] = 1;
            if (cross[i]) {
                int d = 0;
                while (w.contains(block + "1.transformer_blocks." + d + ".norm1.weight")) {
                    d++;
                }
                depth[i] = d;
                context = w.shape(block + "1.transformer_blocks.0.attn2.to_k.weight").dim(1);
                linear = w.shape(block + "1.proj_in.weight").rank() == 2;
            }
            heads[i] = linear ? Math.max(1, channels[i] / arch.headDim()) : arch.sd1Heads();
        }
        int addTimeDim = 0;
        int pooled = 0;
        if (w.contains(UNET + "label_emb.0.0.weight")) {
            int in = w.shape(UNET + "label_emb.0.0.weight").dim(1);
            String proj = "conditioner.embedders.1.model.text_projection";
            pooled = w.contains(proj) ? w.shape(proj).dim(1) : 1280;
            addTimeDim = (in - pooled) / 6;
        }
        return new UNet.Settings(w.shape(UNET + "input_blocks.0.0.weight").dim(1),
                w.shape(UNET + "out.2.weight").dim(0), channels, cross, per, heads, depth, context, linear,
                arch.unetGroups(), 1e-5f, addTimeDim, pooled, addTimeDim > 0 ? 6 : 0, true, 0);
    }

    /** Reads the VAE's shape from its weights. */
    static AutoencoderKL.Settings vaeConfig(WeightSource w, Arch arch, float scaling) {
        List<Integer> channels = new ArrayList<>();
        while (w.contains(VAE + "encoder.down." + channels.size() + ".block.0.conv1.weight")) {
            channels.add(w.shape(VAE + "encoder.down." + channels.size() + ".block.0.conv1.weight").dim(0));
        }
        int layers = 0;
        while (w.contains(VAE + "encoder.down.0.block." + layers + ".conv1.weight")) {
            layers++;
        }
        return new AutoencoderKL.Settings(3, channels.stream().mapToInt(Integer::intValue).toArray(), layers,
                w.shape(VAE + "decoder.conv_in.weight").dim(1), arch.vaeGroups(), 1e-6f, scaling, 0,
                w.contains(VAE + "post_quant_conv.weight"), false);
    }

    static StableDiffusion load(Path file, Device device, DType dtype) throws IOException {
        return load(file, device, dtype, Arch.STANDARD);
    }

    static StableDiffusion load(Path file, Device device, DType dtype, Arch arch) throws IOException {
        WeightSource source = WeightSource.open(file);
        List<WeightSource> sources = new ArrayList<>(List.of(source));
        List<Tensor> all = new ArrayList<>();
        try {
            StableDiffusion.Kind kind = detect(source).orElseThrow(() -> new IllegalArgumentException(file
                    + " is not a Stable Diffusion 1.x, 2.x or XL checkpoint this library can run"));
            Path vocab = tokenizerFolder(file);
            UNet.Settings uc = unetConfig(source, arch);
            Weights w = new Weights(source, "", device, dtype, all);
            // OpenCLIP pads with "!" (id 0), OpenAI's CLIP with the end token
            ClipTextEncoder enc1;
            ClipTokenizer tok2 = null;
            ClipTextEncoder enc2 = null;
            switch (kind) {
                case SD1 -> enc1 = clipL(w.at("cond_stage_model.transformer."), arch);
                // Stable Diffusion 2 conditions on the penultimate layer of OpenCLIP ViT-H
                case SD2 -> enc1 = Pretrained.openClip(w.at("cond_stage_model.model."), 1, arch.clipHeadDim());
                default -> {
                    enc1 = clipL(w.at("conditioner.embedders.0.transformer."), arch);
                    enc2 = Pretrained.openClip(w.at("conditioner.embedders.1.model."), 0, arch.clipHeadDim());
                    tok2 = ClipTokenizer.load(vocab, enc2.settings().positions(), "!");
                }
            }
            // OpenCLIP pads with "!" (id 0), OpenAI's CLIP with the end token
            ClipTokenizer tok1 = ClipTokenizer.load(vocab, enc1.settings().positions(),
                    kind == StableDiffusion.Kind.SD2 ? "!" : "<|endoftext|>");
            UNet unet = Pretrained.unet(w.renamed(unetNames(uc)), uc);
            float scaling = arch.vaeScaling() > 0 ? arch.vaeScaling()
                    : kind == StableDiffusion.Kind.SDXL ? 0.13025f : 0.18215f;
            AutoencoderKL.Settings vc = vaeConfig(source, arch, scaling);
            AutoencoderKL vae = Pretrained.vae(new Weights(source, "", device, DType.FLOAT32, all)
                    .renamed(vaeNames(vc.channels().length)), vc);
            // a 768 Stable Diffusion 2 model predicts velocity; single files do not say which they are
            NoiseSchedule.Prediction prediction = kind == StableDiffusion.Kind.SD2 && !isBase(file)
                    ? NoiseSchedule.Prediction.V_PREDICTION : NoiseSchedule.Prediction.EPSILON;
            int nativeSize = switch (kind) {
                case SD1 -> 512;
                case SD2 -> prediction == NoiseSchedule.Prediction.V_PREDICTION ? 768 : 512;
                case SDXL -> 1024;
            };
            return new StableDiffusion(kind, device, sources, all, tok1, enc1, tok2, enc2, unet, vae,
                    NoiseSchedule.stableDiffusion(prediction), kind == StableDiffusion.Kind.SDXL, nativeSize);
        } catch (IOException | RuntimeException e) {
            for (Tensor t : all) {
                t.release();
            }
            source.close();
            throw e;
        }
    }

    private static boolean isBase(Path file) {
        String name = file.getFileName().toString().toLowerCase(java.util.Locale.ROOT);
        return name.contains("base") || name.contains("512");
    }

    private static ClipTextEncoder clipL(Weights w, Arch arch) {
        Weights t = w.at("text_model.");
        int layers = 0;
        while (t.has("encoder.layers." + layers + ".layer_norm1.weight")) {
            layers++;
        }
        int[] embed = t.shape("embeddings.token_embedding.weight").toArray();
        ClipTextEncoder.Settings c = new ClipTextEncoder.Settings(embed[0], embed[1], layers,
                embed[1] / arch.clipHeadDim(), t.shape("encoder.layers.0.mlp.fc1.weight").dim(0),
                t.shape("embeddings.position_embedding.weight").dim(0), Activation.QUICK_GELU, 1e-5f, -1, 0);
        return Pretrained.clip(w, c, layers);
    }

    /** Where the CLIP vocabulary for a single file is: a {@code tokenizer} folder beside it, or the shared cache. */
    static Path tokenizerFolder(Path file) throws IOException {
        Path parent = file.toAbsolutePath().getParent();
        for (Path p : new Path[] {parent.resolve("tokenizer"), StableDiffusion.tokenizerCache()}) {
            if (Files.isRegularFile(p.resolve("vocab.json")) && Files.isRegularFile(p.resolve("merges.txt"))
                    || Files.isRegularFile(p.resolve("tokenizer.json"))) {
                return p;
            }
        }
        throw new IOException("single-file checkpoints do not include the CLIP tokenizer: put vocab.json and "
                + "merges.txt of openai/clip-vit-large-patch14 into " + StableDiffusion.tokenizerCache()
                + " (StableDiffusion.downloadTokenizer does this)");
    }

    // ------------------------------------------------------------------------------- name mapping

    private static final Pattern DOWN = Pattern.compile("down_blocks\\.(\\d+)\\.(resnets|attentions)\\.(\\d+)\\.(.*)");
    private static final Pattern DOWNSAMPLE = Pattern.compile("down_blocks\\.(\\d+)\\.downsamplers\\.0\\.conv\\.(.*)");
    private static final Pattern UP = Pattern.compile("up_blocks\\.(\\d+)\\.(resnets|attentions)\\.(\\d+)\\.(.*)");
    private static final Pattern UPSAMPLE = Pattern.compile("up_blocks\\.(\\d+)\\.upsamplers\\.0\\.conv\\.(.*)");
    private static final Pattern MID = Pattern.compile("mid_block\\.(resnets|attentions)\\.(\\d+)\\.(.*)");

    /** Diffusers UNet names to the original ones. */
    static UnaryOperator<String> unetNames(UNet.Settings c) {
        int per = c.layersPerBlock();
        boolean[] cross = c.attention();
        int levels = c.channels().length;
        return name -> {
            Matcher m;
            String ldm;
            if ((m = DOWN.matcher(name)).matches()) {
                int block = 1 + Integer.parseInt(m.group(1)) * (per + 1) + Integer.parseInt(m.group(3));
                ldm = "input_blocks." + block + (m.group(2).equals("resnets") ? ".0." + resnet(m.group(4))
                        : ".1." + m.group(4));
            } else if ((m = DOWNSAMPLE.matcher(name)).matches()) {
                int block = (Integer.parseInt(m.group(1)) + 1) * (per + 1);
                ldm = "input_blocks." + block + ".0.op." + m.group(2);
            } else if ((m = MID.matcher(name)).matches()) {
                int i = Integer.parseInt(m.group(2));
                ldm = m.group(1).equals("resnets") ? "middle_block." + (i * 2) + "." + resnet(m.group(3))
                        : "middle_block.1." + m.group(3);
            } else if ((m = UP.matcher(name)).matches()) {
                int block = Integer.parseInt(m.group(1)) * (per + 1) + Integer.parseInt(m.group(3));
                ldm = "output_blocks." + block + (m.group(2).equals("resnets") ? ".0." + resnet(m.group(4))
                        : ".1." + m.group(4));
            } else if ((m = UPSAMPLE.matcher(name)).matches()) {
                int up = Integer.parseInt(m.group(1));
                int block = up * (per + 1) + per;
                boolean attention = cross[levels - 1 - up];
                ldm = "output_blocks." + block + (attention ? ".2." : ".1.") + "conv." + m.group(2);
            } else {
                ldm = switch (name.substring(0, name.lastIndexOf('.'))) {
                    case "conv_in" -> "input_blocks.0.0" + name.substring(name.lastIndexOf('.'));
                    case "time_embedding.linear_1" -> "time_embed.0" + name.substring(name.lastIndexOf('.'));
                    case "time_embedding.linear_2" -> "time_embed.2" + name.substring(name.lastIndexOf('.'));
                    case "add_embedding.linear_1" -> "label_emb.0.0" + name.substring(name.lastIndexOf('.'));
                    case "add_embedding.linear_2" -> "label_emb.0.2" + name.substring(name.lastIndexOf('.'));
                    case "conv_norm_out" -> "out.0" + name.substring(name.lastIndexOf('.'));
                    case "conv_out" -> "out.2" + name.substring(name.lastIndexOf('.'));
                    default -> name;
                };
            }
            return UNET + ldm;
        };
    }

    private static String resnet(String rest) {
        int dot = rest.lastIndexOf('.');
        String part = rest.substring(0, dot);
        String tail = rest.substring(dot);
        return switch (part) {
            case "norm1" -> "in_layers.0";
            case "conv1" -> "in_layers.2";
            case "time_emb_proj" -> "emb_layers.1";
            case "norm2" -> "out_layers.0";
            case "conv2" -> "out_layers.3";
            case "conv_shortcut" -> "skip_connection";
            default -> part;
        } + tail;
    }

    private static final Pattern VAE_BLOCK = Pattern.compile(
            "(encoder|decoder)\\.(down|up)_blocks\\.(\\d+)\\.resnets\\.(\\d+)\\.(.*)");
    private static final Pattern VAE_SAMPLE = Pattern.compile(
            "(encoder|decoder)\\.(down|up)_blocks\\.(\\d+)\\.(downsamplers|upsamplers)\\.0\\.conv\\.(.*)");
    private static final Pattern VAE_MID = Pattern.compile("(encoder|decoder)\\.mid_block\\.(resnets|attentions)\\.(\\d+)\\.(.*)");

    /** Diffusers VAE names to the original ones. */
    static UnaryOperator<String> vaeNames(int levels) {
        return name -> {
            Matcher m;
            String ldm;
            if ((m = VAE_BLOCK.matcher(name)).matches()) {
                int i = level(m.group(2), Integer.parseInt(m.group(3)), levels);
                ldm = m.group(1) + "." + m.group(2) + "." + i + ".block." + m.group(4) + "."
                        + m.group(5).replace("conv_shortcut", "nin_shortcut");
            } else if ((m = VAE_SAMPLE.matcher(name)).matches()) {
                int i = level(m.group(2), Integer.parseInt(m.group(3)), levels);
                ldm = m.group(1) + "." + m.group(2) + "." + i + (m.group(4).equals("downsamplers") ? ".downsample.conv."
                        : ".upsample.conv.") + m.group(5);
            } else if ((m = VAE_MID.matcher(name)).matches()) {
                String rest = m.group(4);
                if (m.group(2).equals("resnets")) {
                    ldm = m.group(1) + ".mid.block_" + (Integer.parseInt(m.group(3)) + 1) + "."
                            + rest.replace("conv_shortcut", "nin_shortcut");
                } else {
                    String part = rest.substring(0, rest.lastIndexOf('.'));
                    String tail = rest.substring(rest.lastIndexOf('.'));
                    ldm = m.group(1) + ".mid.attn_1." + switch (part) {
                        case "group_norm" -> "norm";
                        case "to_q" -> "q";
                        case "to_k" -> "k";
                        case "to_v" -> "v";
                        case "to_out.0" -> "proj_out";
                        default -> part;
                    } + tail;
                }
            } else {
                ldm = name.replace("conv_norm_out", "norm_out");
            }
            return VAE + ldm;
        };
    }

    private static int level(String direction, int i, int levels) {
        return direction.equals("up") ? levels - 1 - i : i;
    }
}
