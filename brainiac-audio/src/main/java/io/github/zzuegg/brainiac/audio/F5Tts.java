package io.github.zzuegg.brainiac.audio;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.io.WeightSource;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.ConvNeXt1d;
import io.github.zzuegg.brainiac.nn.DitBlock;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.IntConsumer;
import java.util.random.RandomGenerator;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * F5-TTS (Chen et al., 2024): a Diffusion Transformer that fills in a mel spectrogram by flow
 * matching. Given the mel frames of a reference recording, the reference's transcript plus
 * the new text as characters, and a total length, it integrates a learned velocity field
 * from noise to speech that continues the reference voice, with classifier-free guidance.
 * Loads {@code SWivid/F5-TTS} checkpoints ({@code F5TTS_v1_Base}: a {@code .safetensors} and
 * {@code vocab.txt}); {@link F5Speech} turns it into a text-to-speech model with
 * {@link Vocos}.
 */
public final class F5Tts implements AutoCloseable {

    /**
     * The network's shape, read from the weights.
     *
     * @param dim the transformer width
     * @param depth the transformer blocks
     * @param heads the attention heads
     * @param textDim the text embedding width
     * @param mels the mel bands
     * @param textBlocks the ConvNeXt blocks over the text
     */
    public record Config(int dim, int depth, int heads, int textDim, int mels, int textBlocks) {
    }

    private static final int TIME_FEATURES = 256;
    private static final Map<Integer, int[]> EPSS = Map.of(
            5, new int[] {0, 2, 4, 8, 16, 32},
            6, new int[] {0, 2, 4, 6, 8, 16, 32},
            7, new int[] {0, 2, 4, 6, 8, 16, 24, 32},
            10, new int[] {0, 2, 4, 6, 8, 12, 16, 20, 24, 28, 32},
            12, new int[] {0, 2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32},
            16, new int[] {0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32});

    private final Config config;
    private final Map<String, Integer> vocab;
    private final Weights weights;
    private final Device device;
    private final DType dtype;
    private final Var textTable;
    private final List<ConvNeXt1d> textBlocks = new ArrayList<>();
    private final Var projX;
    private final Var projCond;
    private final Var projText;
    private final Var projB;
    private final Var pos1W;
    private final Var pos1B;
    private final Var pos2W;
    private final Var pos2B;
    private final Var time1W;
    private final Var time1B;
    private final Var time2W;
    private final Var time2B;
    private final List<DitBlock> blocks = new ArrayList<>();
    private final Var outModW;
    private final Var outModB;
    private final Var outW;
    private final Var outB;

    private F5Tts(WeightSource source, Map<String, Integer> vocab, int dimHead, Device device, DType dtype) {
        this.vocab = Map.copyOf(vocab);
        this.device = device;
        this.dtype = dtype;
        this.weights = new Weights(source, device, dtype);
        try {
            Weights w = weights;
            String p = prefix(source);
            textTable = w.read(p + "text_embed.text_embed.weight");
            int textDim = textTable.shape().dim(1);
            for (int i = 0; w.contains(p + "text_embed.text_blocks." + i + ".dwconv.weight"); i++) {
                String b = p + "text_embed.text_blocks." + i + ".";
                int hidden = w.shape(b + "pwconv1.weight").dim(0);
                textBlocks.add(new ConvNeXt1d(1e-6f, w.read(b + "dwconv.weight"), w.read(b + "dwconv.bias"),
                        w.read(b + "norm.weight"), w.read(b + "norm.bias"), w.linear(b + "pwconv1.weight"),
                        w.read(b + "pwconv1.bias"), w.read(b + "grn.gamma", hidden), w.read(b + "grn.beta", hidden),
                        w.linear(b + "pwconv2.weight"), w.read(b + "pwconv2.bias"), null));
            }
            Shape proj = w.shape(p + "input_embed.proj.weight");          // [dim, 2 mels + textDim]
            int dim = proj.dim(0);
            int mels = (proj.dim(1) - textDim) / 2;
            float[] pw = w.floats(p + "input_embed.proj.weight");
            projX = part(pw, dim, proj.dim(1), 0, mels);
            projCond = part(pw, dim, proj.dim(1), mels, 2 * mels);
            projText = part(pw, dim, proj.dim(1), 2 * mels, 2 * mels + textDim);
            projB = w.read(p + "input_embed.proj.bias");
            pos1W = w.read(p + "input_embed.conv_pos_embed.conv1d.0.weight");
            pos1B = w.read(p + "input_embed.conv_pos_embed.conv1d.0.bias");
            pos2W = w.read(p + "input_embed.conv_pos_embed.conv1d.2.weight");
            pos2B = w.read(p + "input_embed.conv_pos_embed.conv1d.2.bias");
            time1W = w.linear(p + "time_embed.time_mlp.0.weight");
            time1B = w.read(p + "time_embed.time_mlp.0.bias");
            time2W = w.linear(p + "time_embed.time_mlp.2.weight");
            time2B = w.read(p + "time_embed.time_mlp.2.bias");
            int inner = w.shape(p + "transformer_blocks.0.attn.to_q.weight").dim(0);
            if (inner % dimHead != 0) {
                throw new IllegalArgumentException("attention width " + inner + " is not a multiple of the head size "
                        + dimHead + "; set dim_head in config.json");
            }
            DitBlock.Settings s = new DitBlock.Settings(inner / dimHead, DitBlock.Rotary.INTERLEAVED, 10_000f,
                    Activation.GELU_TANH, 1e-6f);
            for (int i = 0; w.contains(p + "transformer_blocks." + i + ".attn.to_q.weight"); i++) {
                String b = p + "transformer_blocks." + i + ".";
                blocks.add(new DitBlock(s, w.linear(b + "attn_norm.linear.weight"), w.read(b + "attn_norm.linear.bias"),
                        w.linear(b + "attn.to_q.weight"), w.read(b + "attn.to_q.bias"),
                        w.linear(b + "attn.to_k.weight"), w.read(b + "attn.to_k.bias"),
                        w.linear(b + "attn.to_v.weight"), w.read(b + "attn.to_v.bias"),
                        w.linear(b + "attn.to_out.0.weight"), w.read(b + "attn.to_out.0.bias"),
                        w.linear(b + "ff.ff.0.0.weight"), w.read(b + "ff.ff.0.0.bias"),
                        w.linear(b + "ff.ff.2.weight"), w.read(b + "ff.ff.2.bias")));
            }
            if (w.contains(p + "long_skip_connection.weight")) {
                throw new IllegalArgumentException("F5-TTS checkpoints with a long skip connection are not supported");
            }
            outModW = w.linear(p + "norm_out.linear.weight");
            outModB = w.read(p + "norm_out.linear.bias");
            outW = w.linear(p + "proj_out.weight");
            outB = w.read(p + "proj_out.bias");
            config = new Config(dim, blocks.size(), s.heads(), textDim, mels, textBlocks.size());
            if (textTable.shape().dim(0) != vocab.size() + 1) {
                throw new IllegalArgumentException("vocab.txt has " + vocab.size() + " characters but the model "
                        + (textTable.shape().dim(0) - 1));
            }
        } catch (RuntimeException e) {
            weights.close();
            throw e;
        }
    }

    private Var part(float[] w, int rows, int cols, int from, int to) {
        float[] t = new float[(to - from) * rows];
        for (int o = 0; o < rows; o++) {
            for (int i = from; i < to; i++) {
                t[(i - from) * rows + o] = w[o * cols + i];
            }
        }
        Tensor x = Tensor.of(Shape.of(to - from, rows), t, device).to(dtype);
        Tensor.evaluate(List.of(x));
        return Var.of(x);
    }

    private static String prefix(WeightSource source) {
        for (String p : new String[] {"ema_model.transformer.", "transformer.", "model.transformer.", ""}) {
            if (source.contains(p + "time_embed.time_mlp.0.weight")) {
                return p;
            }
        }
        throw new IllegalArgumentException("not an F5-TTS checkpoint: no time_embed.time_mlp.0.weight");
    }

    /**
     * Loads an F5-TTS folder: the first {@code .safetensors} (or {@code .pt}) checkpoint and
     * {@code vocab.txt}; an optional {@code config.json} may set {@code dim_head} (default 64).
     *
     * @param dir the folder
     * @param device where it runs
     * @param dtype the type of its weights
     * @return the model
     */
    public static F5Tts load(Path dir, Device device, DType dtype) {
        Path ckpt = checkpoint(dir);
        if (ckpt == null) {
            throw new IllegalArgumentException(dir + " has no F5-TTS checkpoint (.safetensors) with vocab.txt");
        }
        Map<String, Integer> vocab = readVocab(dir.resolve("vocab.txt"));
        int dimHead = 64;
        Path cfg = dir.resolve("config.json");
        if (Files.isRegularFile(cfg)) {
            try {
                Object v = Json.object(Json.read(cfg)).get("dim_head");
                if (v instanceof Number n) {
                    dimHead = n.intValue();
                }
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }
        try (WeightSource source = WeightSource.open(ckpt)) {
            return new F5Tts(source, vocab, dimHead, device, dtype);
        }
    }

    /**
     * Whether a folder holds an F5-TTS checkpoint (directly or in one subfolder) and its
     * {@code vocab.txt}. A folder that also holds a Vocos vocoder folder is a collection of
     * models rather than an F5 model, so its subfolders are not searched.
     *
     * @param dir a folder
     * @return the folder with the checkpoint, or {@code null}
     */
    public static Path find(Path dir) {
        if (checkpoint(dir) != null) {
            return dir;
        }
        try (var subs = Files.list(dir)) {
            List<Path> dirs = subs.filter(Files::isDirectory).sorted().toList();
            if (dirs.stream().anyMatch(Vocos::isVocos)) {
                return null;
            }
            return dirs.stream().filter(d -> checkpoint(d) != null).findFirst().orElse(null);
        } catch (IOException e) {
            return null;
        }
    }

    private static Path checkpoint(Path dir) {
        if (!Files.isRegularFile(dir.resolve("vocab.txt"))) {
            return null;
        }
        try (var files = Files.list(dir)) {
            List<Path> all = files.filter(f -> {
                String n = f.getFileName().toString();
                return n.endsWith(".safetensors") || n.endsWith(".pt");
            }).sorted().toList();
            for (Path f : all) {
                try (WeightSource s = WeightSource.open(f)) {
                    prefix(s);
                    return f;
                } catch (RuntimeException e) {
                    // not an F5 checkpoint
                }
            }
            return null;
        } catch (IOException e) {
            return null;
        }
    }

    private static Map<String, Integer> readVocab(Path file) {
        try {
            String text = Files.readString(file, StandardCharsets.UTF_8);
            Map<String, Integer> v = new HashMap<>();
            int i = 0;
            int start = 0;
            for (int at = text.indexOf('\n'); at >= 0; at = text.indexOf('\n', start)) {
                v.putIfAbsent(text.substring(start, at), i++);
                start = at + 1;
            }
            if (start < text.length()) {
                v.putIfAbsent(text.substring(start), i);
            }
            return v;
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    /** @return the network's shape */
    public Config config() {
        return config;
    }

    // ---- text ----------------------------------------------------------------------------

    private static final Pattern SEGMENT = Pattern.compile("[A-Za-z0-9]+(?:[._\\-][A-Za-z0-9]+)*%?|[._\\-]+|.",
            Pattern.DOTALL);

    /**
     * The characters F5-TTS reads for a text, as its {@code convert_char_to_pinyin} spells
     * text without Chinese: {@code ;} becomes {@code ,}, curly quotes straight ones, and a
     * space goes before a word that directly follows punctuation (as jieba's segmentation
     * makes it). Chinese characters are not converted to pinyin.
     *
     * @param text the text
     * @return one string per character the model reads
     */
    public static List<String> spell(String text) {
        String t = text.replace(';', ',').replace('“', '"').replace('”', '"').replace('‘', '\'')
                .replace('’', '\'');
        List<String> out = new ArrayList<>();
        Matcher m = SEGMENT.matcher(t);
        while (m.find()) {
            String seg = m.group();
            boolean ascii = seg.chars().allMatch(c -> c < 128);
            if (ascii && seg.length() > 1 && !out.isEmpty() && " :'\"".indexOf(out.getLast().charAt(0)) < 0) {
                out.add(" ");
            }
            seg.codePoints().forEach(c -> out.add(new String(Character.toChars(c))));
        }
        return out;
    }

    /**
     * Token ids for a text: {@link #spell} through {@code vocab.txt}, unknown characters 0.
     *
     * @param text the text
     * @return the ids
     */
    public int[] encode(String text) {
        return spell(text).stream().mapToInt(c -> vocab.getOrDefault(c, 0)).toArray();
    }

    /** @return the characters the model knows, with their ids */
    public Map<String, Integer> vocab() {
        return vocab;
    }

    // ---- network -------------------------------------------------------------------------

    /** The parts of the input that stay the same over the sampling steps. */
    private record Context(Var condPart, int length) {
    }

    /** The text embedding {@code [1, length, textDim]}: drop for the unconditioned pass. */
    private Var textEmbedding(int[] ids, int length, boolean drop) {
        float[] idx = new float[length];
        boolean[] filler = new boolean[length];
        for (int i = 0; i < length; i++) {
            int tok = i < ids.length ? ids[i] + 1 : 0;
            filler[i] = tok == 0;
            idx[i] = drop ? 0 : tok;
        }
        int td = textTable.shape().dim(1);
        Var x = textTable.gather(0, Tensor.of(Shape.of(length), idx, device)).reshape(1, length, td);
        int half = td / 2;
        float[] pos = new float[length * td];
        float[] mask = new float[length];
        for (int t = 0; t < length; t++) {
            for (int i = 0; i < half; i++) {
                float freq = (float) (1.0 / Math.pow(10_000.0, (float) (2 * i) / td));
                float angle = t * freq;
                pos[t * td + i] = (float) Math.cos(angle);
                pos[t * td + half + i] = (float) Math.sin(angle);
            }
            mask[t] = filler[t] ? 0 : 1;
        }
        Var keep = Var.of(Tensor.of(Shape.of(1, length, 1), mask, device).to(dtype));
        x = x.add(Var.of(Tensor.of(Shape.of(1, length, td), pos, device).to(dtype))).mul(keep);
        for (ConvNeXt1d b : textBlocks) {
            x = Vocos.settle(b.forward(x.permute(0, 2, 1)).permute(0, 2, 1).mul(keep));
        }
        return x;
    }

    /**
     * The constant part of the input projection for the conditioned and the unconditioned pass,
     * {@code [2, length, dim]}.
     */
    private Context context(float[][] cond, int[] ids, int length) {
        int mels = config.mels();
        float[] c = new float[length * mels];
        for (int t = 0; t < Math.min(cond.length, length); t++) {
            System.arraycopy(cond[t], 0, c, t * mels, mels);
        }
        Var condVar = Var.of(Tensor.of(Shape.of(1, length, mels), c, device).to(dtype));
        Var withCond = condVar.matmul(projCond).add(textEmbedding(ids, length, false).matmul(projText));
        Var without = textEmbedding(ids, length, true).matmul(projText);
        return new Context(Vocos.settle(Var.concat(0, withCond, without).add(projB)), length);
    }

    private Var timeEmbedding(float time) {
        int half = TIME_FEATURES / 2;
        float[] f = new float[TIME_FEATURES];
        double step = Math.log(10_000) / (half - 1);
        for (int i = 0; i < half; i++) {
            float freq = (float) Math.exp(i * -step);
            float arg = 1000f * time * freq;
            f[i] = (float) Math.sin(arg);
            f[half + i] = (float) Math.cos(arg);
        }
        Var t = Var.of(Tensor.of(Shape.of(1, TIME_FEATURES), f, device).to(dtype));
        Var e = Activation.SILU.forward(t.matmul(time1W).add(time1B)).matmul(time2W).add(time2B);
        return Var.concat(0, e, e);
    }

    /** Both passes' velocities {@code [2, length, mels]} at {@code x} {@code [1, length, mels]}. */
    private Var velocity(Var x, Context ctx, float time) {
        int len = ctx.length();
        int dim = config.dim();
        Var h = x.to(dtype).matmul(projX).add(ctx.condPart());             // [2, length, dim]
        Var c = h.permute(0, 2, 1);
        int groups = dim / pos1W.shape().dim(1);
        int k = pos1W.shape().dim(2);
        c = Activation.MISH.forward(Functional.conv(c, pos1W, pos1B, new int[] {1}, new int[] {k / 2},
                new int[] {k / 2}, new int[] {1}, groups));
        c = Activation.MISH.forward(Functional.conv(c, pos2W, pos2B, new int[] {1}, new int[] {k / 2},
                new int[] {k / 2}, new int[] {1}, groups));
        h = Vocos.settle(c.permute(0, 2, 1).add(h));
        Var t = timeEmbedding(time);
        for (DitBlock b : blocks) {
            h = Vocos.settle(b.forward(h, t));
        }
        Var e = Activation.SILU.forward(t).matmul(outModW).add(outModB);
        h = Functional.adaLayerNorm(h, e.slice(1, 0, dim), e.slice(1, dim, 2 * dim), 1e-6f);
        return Vocos.settle(h.matmul(outW).add(outB).to(DType.FLOAT32).reshape(2, len, config.mels()));
    }

    /**
     * One evaluation of the network, as its PyTorch {@code forward} with {@code cfg_infer}:
     * the velocity with the conditions, then without the reference audio and text.
     *
     * @param x the noisy mel {@code [length][mels]}
     * @param cond the reference mel frames, zero beyond the reference
     * @param ids the text's ids
     * @param time the flow time in [0, 1]
     * @return {@code [2][length][mels]}
     */
    public float[][][] velocity(float[][] x, float[][] cond, int[] ids, float time) {
        int len = x.length;
        Context ctx = context(cond, ids, len);
        float[] v = velocity(Var.of(Tensor.of(Shape.of(1, len, config.mels()), flatten(x), device)), ctx, time)
                .value().toArray();
        float[][][] out = new float[2][len][config.mels()];
        for (int b = 0; b < 2; b++) {
            for (int t = 0; t < len; t++) {
                System.arraycopy(v, (b * len + t) * config.mels(), out[b][t], 0, config.mels());
            }
        }
        return out;
    }

    /**
     * The flow's time steps: uniform, or for 5, 6, 7, 10, 12 and 16 steps the empirically
     * pruned schedules of F5-TTS v1, then bent by sway sampling.
     *
     * @param steps the steps
     * @param sway the sway coefficient (-1 by default; 0 for none)
     * @return {@code steps + 1} times from 0 to 1
     */
    static float[] times(int steps, float sway) {
        float[] t = new float[steps + 1];
        int[] epss = EPSS.get(steps);
        for (int i = 0; i <= steps; i++) {
            t[i] = epss != null ? epss[i] / 32f : (float) i / steps;
            t[i] = (float) (t[i] + sway * (Math.cos(Math.PI / 2 * t[i]) - 1 + t[i]));
        }
        return t;
    }

    /**
     * Fills in a mel spectrogram: the reference frames stay, the rest is sampled by Euler
     * steps along the flow with classifier-free guidance.
     *
     * @param cond the reference mel frames {@code [frames][mels]}
     * @param ids the token ids of the reference transcript and the new text together
     * @param length the total frames, at least the reference's and the text's plus one
     * @param steps the Euler steps (32 by default)
     * @param guidance the classifier-free guidance strength (2 by default)
     * @param sway the sway sampling coefficient (-1 by default)
     * @param noise draws the start noise ({@code nextGaussian})
     * @param progress told each finished step, or {@code null}
     * @return {@code [length][mels]}
     */
    public float[][] sample(float[][] cond, int[] ids, int length, int steps, float guidance, float sway,
            RandomGenerator noise, IntConsumer progress) {
        int mels = config.mels();
        length = Math.max(length, Math.max(ids.length, cond.length) + 1);
        Context ctx = context(cond, ids, length);
        float[] y = new float[length * mels];
        for (int i = 0; i < y.length; i++) {
            y[i] = (float) noise.nextGaussian();
        }
        float[] t = times(steps, sway);
        Var x = Var.of(Tensor.of(Shape.of(1, length, mels), y, device));
        for (int s = 0; s < steps; s++) {
            Var v = velocity(x, ctx, t[s]);
            Var pred = v.slice(0, 0, 1);
            Var guided = pred.add(pred.sub(v.slice(0, 1, 2)).mul(guidance));
            x = Vocos.settle(x.add(guided.mul(t[s + 1] - t[s])));
            if (progress != null) {
                progress.accept(s + 1);
            }
        }
        float[] out = x.value().toArray();
        float[][] mel = new float[length][mels];
        for (int f = 0; f < length; f++) {
            if (f < cond.length) {
                mel[f] = cond[f].clone();
            } else {
                System.arraycopy(out, f * mels, mel[f], 0, mels);
            }
        }
        return mel;
    }

    private static float[] flatten(float[][] x) {
        int w = x[0].length;
        float[] f = new float[x.length * w];
        for (int i = 0; i < x.length; i++) {
            System.arraycopy(x[i], 0, f, i * w, w);
        }
        return f;
    }

    @Override
    public void close() {
        weights.close();
    }
}
