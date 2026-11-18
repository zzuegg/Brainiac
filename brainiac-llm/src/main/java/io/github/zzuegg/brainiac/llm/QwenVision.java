package io.github.zzuegg.brainiac.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.io.WeightSource;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;
import io.github.zzuegg.brainiac.tensor.spi.ActivationOp;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;

/**
 * The image side of Qwen2-VL and Qwen2.5-VL: images resized to a multiple of the merged
 * patch size within a pixel budget, cut into patches (each image twice in time, as a
 * two-frame video), a vision transformer with 2D rotary positions (Qwen2.5: RMS norms, gated
 * MLPs, attention within windows except in a few blocks), and a merger that turns each
 * 2 x 2 patches into one token of the decoder's width.
 */
final class QwenVision {

    /** An image ready for the tower: its patches {@code [patches, 3 * frames * patch^2]} and grid in patches. */
    record Patches(Tensor values, int height, int width) {
    }

    private record Block(Tensor norm1, Tensor norm1b, Tensor qkv, Tensor qkvb, Tensor proj, Tensor projb, Tensor norm2,
            Tensor norm2b, Tensor gate, Tensor gateb, Tensor up, Tensor upb, Tensor down, Tensor downb) {
    }

    private final boolean v25;
    private final Device device;
    private final DType dtype;
    private final int hidden;
    private final int heads;
    private final int patch;
    private final int merge;
    private final int frames;
    private final int window;
    private final Set<Integer> fullBlocks;
    private final double theta;
    private final Tensor patchWeight;
    private final List<Block> blocks = new ArrayList<>();
    private final Tensor mergeNorm;
    private final Tensor mergeNormBias;
    private final Tensor merge1;
    private final Tensor merge1b;
    private final Tensor merge2;
    private final Tensor merge2b;
    private final int minPixels;
    private final int maxPixels;
    private final double[] mean;
    private final double[] std;
    private final double rescale;
    private final int resample;

    QwenVision(boolean v25, Map<String, Object> c, Map<String, Object> processor, WeightSource w, Device device,
            DType dtype) {
        this.v25 = v25;
        this.device = device;
        this.dtype = dtype;
        hidden = (int) Json.integer(c, v25 ? "hidden_size" : "embed_dim");
        heads = (int) Json.integer(c, "num_heads");
        patch = (int) Json.integer(c, "patch_size", 14);
        merge = (int) Json.integer(c, "spatial_merge_size", 2);
        frames = (int) Json.integer(c, "temporal_patch_size", 2);
        window = v25 ? (int) Json.integer(c, "window_size", 112) / merge / patch : 0;
        Set<Integer> full = new java.util.HashSet<>();
        if (c.get("fullatt_block_indexes") instanceof List<?> l) {
            for (Object o : l) {
                full.add(((Number) o).intValue());
            }
        }
        fullBlocks = full;
        Map<String, Object> rope = c.get("rope_parameters") instanceof Map<?, ?> m ? Json.object(m) : Map.of();
        theta = Json.number(rope, "rope_theta", 10000);
        Function<String, Tensor> t = name -> w.tensor(name, device, dtype);
        Function<String, Tensor> opt = name -> w.contains(name) ? t.apply(name) : null;
        Tensor conv = t.apply("visual.patch_embed.proj.weight");
        patchWeight = conv.reshape(hidden, (int) (conv.size() / hidden)).transpose();
        int depth = (int) Json.integer(c, "depth");
        for (int i = 0; i < depth; i++) {
            String p = "visual.blocks." + i + ".";
            blocks.add(v25
                    ? new Block(t.apply(p + "norm1.weight"), null, lin(t, p + "attn.qkv"), t.apply(p + "attn.qkv.bias"),
                            lin(t, p + "attn.proj"), t.apply(p + "attn.proj.bias"), t.apply(p + "norm2.weight"), null,
                            lin(t, p + "mlp.gate_proj"), t.apply(p + "mlp.gate_proj.bias"), lin(t, p + "mlp.up_proj"),
                            t.apply(p + "mlp.up_proj.bias"), lin(t, p + "mlp.down_proj"), t.apply(p + "mlp.down_proj.bias"))
                    : new Block(t.apply(p + "norm1.weight"), t.apply(p + "norm1.bias"), lin(t, p + "attn.qkv"),
                            t.apply(p + "attn.qkv.bias"), lin(t, p + "attn.proj"), t.apply(p + "attn.proj.bias"),
                            t.apply(p + "norm2.weight"), t.apply(p + "norm2.bias"), null, null, lin(t, p + "mlp.fc1"),
                            t.apply(p + "mlp.fc1.bias"), lin(t, p + "mlp.fc2"), t.apply(p + "mlp.fc2.bias")));
        }
        mergeNorm = t.apply("visual.merger.ln_q.weight");
        mergeNormBias = opt.apply("visual.merger.ln_q.bias");
        merge1 = lin(t, "visual.merger.mlp.0");
        merge1b = t.apply("visual.merger.mlp.0.bias");
        merge2 = lin(t, "visual.merger.mlp.2");
        merge2b = t.apply("visual.merger.mlp.2.bias");
        Map<String, Object> size = processor.get("size") instanceof Map<?, ?> m ? Json.object(m) : Map.of();
        minPixels = (int) Json.integer(processor, "min_pixels", Json.integer(size, "shortest_edge", 56 * 56));
        maxPixels = (int) Json.integer(processor, "max_pixels", Json.integer(size, "longest_edge", 28 * 28 * 1280));
        mean = triple(processor.get("image_mean"), new double[] {0.48145466, 0.4578275, 0.40821073});
        std = triple(processor.get("image_std"), new double[] {0.26862954, 0.26130258, 0.27577711});
        rescale = Json.number(processor, "rescale_factor", 1 / 255.0);
        resample = (int) Json.integer(processor, "resample", 3);
    }

    private static Tensor lin(Function<String, Tensor> t, String name) {
        return t.apply(name + ".weight").transpose();
    }

    private static double[] triple(Object v, double[] otherwise) {
        if (v instanceof List<?> l && l.size() == 3) {
            return new double[] {((Number) l.get(0)).doubleValue(), ((Number) l.get(1)).doubleValue(),
                    ((Number) l.get(2)).doubleValue()};
        }
        return otherwise;
    }

    /** @return how many tokens an image of this grid (in patches) becomes */
    int tokens(Patches p) {
        return p.height() / merge * (p.width() / merge);
    }

    /** @return the grid in tokens, {height, width} */
    int[] tokenGrid(Patches p) {
        return new int[] {p.height() / merge, p.width() / merge};
    }

    // ------------------------------------------------------------------ preprocessing

    /** Qwen's resize target: multiples of {@code factor}, the area within the pixel budget. */
    private int[] smartResize(int h, int w) {
        int factor = patch * merge;
        long hb = Math.max(factor, Math.round(roundHalfEven((double) h / factor)) * factor);
        long wb = Math.max(factor, Math.round(roundHalfEven((double) w / factor)) * factor);
        if (hb * wb > maxPixels) {
            double beta = Math.sqrt((double) h * w / maxPixels);
            hb = Math.max(factor, (long) Math.floor(h / beta / factor) * factor);
            wb = Math.max(factor, (long) Math.floor(w / beta / factor) * factor);
        } else if (hb * wb < minPixels) {
            double beta = Math.sqrt((double) minPixels / ((double) h * w));
            hb = (long) Math.ceil(h * beta / factor) * factor;
            wb = (long) Math.ceil(w * beta / factor) * factor;
        }
        return new int[] {(int) hb, (int) wb};
    }

    private static double roundHalfEven(double v) {
        return Math.rint(v);
    }

    /** Resizes, normalizes and cuts an image into patches, in the tower's order. */
    Patches patches(BufferedImage image) {
        int h = image.getHeight();
        int w = image.getWidth();
        int[] size = smartResize(h, w);
        double[][] rgb = ImagePreprocessor.channels(image);
        if (size[0] != h || size[1] != w) {
            ImagePreprocessor.resize(rgb, h, w, size[0], size[1], resample);
        }
        h = size[0];
        w = size[1];
        int gh = h / patch;
        int gw = w / patch;
        int per = 3 * frames * patch * patch;
        float[] out = new float[gh * gw * per];
        int row = 0;
        // merged blocks row by row, the patches of a block row by row, values (channel, frame, y, x)
        for (int by = 0; by < gh / merge; by++) {
            for (int bx = 0; bx < gw / merge; bx++) {
                for (int my = 0; my < merge; my++) {
                    for (int mx = 0; mx < merge; mx++) {
                        int py = by * merge + my;
                        int px = bx * merge + mx;
                        int at = row++ * per;
                        for (int c = 0; c < 3; c++) {
                            for (int f = 0; f < frames; f++) {
                                for (int y = 0; y < patch; y++) {
                                    for (int x = 0; x < patch; x++) {
                                        double v = rgb[c][(py * patch + y) * w + px * patch + x] * rescale;
                                        out[at++] = (float) ((v - mean[c]) / std[c]);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return new Patches(Tensor.of(Shape.of(gh * gw, per), out, device), gh, gw);
    }

    // ------------------------------------------------------------------ tower

    /**
     * The merged features of one image.
     *
     * @return {@code [tokens, decoder width]}
     */
    Tensor features(Patches p) {
        return Var.noGrad(() -> Var.of(run(p))).value();
    }

    private Tensor run(Patches p) {
        int n = p.height() * p.width();
        int hd = hidden / heads;
        Tensor x = p.values().to(dtype).matmul(patchWeight);
        Tensor[] rope = rotary(p);
        Tensor windowMask = window > 0 ? windows(p) : null;
        float scale = (float) (1 / Math.sqrt(hd));
        for (int i = 0; i < blocks.size(); i++) {
            Block b = blocks.get(i);
            Tensor h = norm(x, b.norm1, b.norm1b);
            Tensor qkv = h.matmul(b.qkv).add(b.qkvb).reshape(n, 3, heads, hd);
            Tensor q = rotate(qkv.slice(1, 0, 1).reshape(n, heads, hd), rope);
            Tensor k = rotate(qkv.slice(1, 1, 2).reshape(n, heads, hd), rope);
            Tensor v = qkv.slice(1, 2, 3).reshape(n, heads, hd).permute(1, 0, 2).reshape(1, heads, n, hd);
            Var a;
            if (windowMask != null && !fullBlocks.contains(i)) {
                a = Functional.attention(Var.of(q), Var.of(k), Var.of(v), windowMask.expand(Shape.of(1, heads, n, n)),
                        scale, 0, null);
            } else {
                a = Functional.attention(Var.of(q), Var.of(k), Var.of(v), false, scale);
            }
            Tensor o = a.value().permute(0, 2, 1, 3).reshape(n, hidden).matmul(b.proj).add(b.projb);
            x = x.add(o);
            h = norm(x, b.norm2, b.norm2b);
            Tensor m;
            if (v25) {
                m = h.matmul(b.gate).add(b.gateb).activation(ActivationOp.SILU, 0).mul(h.matmul(b.up).add(b.upb))
                        .matmul(b.down).add(b.downb);
            } else {
                m = quickGelu(h.matmul(b.up).add(b.upb)).matmul(b.down).add(b.downb);
            }
            x = x.add(m);
        }
        Tensor merged = norm(x, mergeNorm, mergeNormBias).reshape(n / (merge * merge), hidden * merge * merge);
        return merged.matmul(merge1).add(merge1b).activation(ActivationOp.GELU, 0).matmul(merge2).add(merge2b);
    }

    private static Tensor quickGelu(Tensor x) {
        return x.mul(x.mul(1.702f).activation(ActivationOp.SIGMOID, 0));
    }

    /** RMS norm (Qwen2.5) or layer norm (Qwen2, with a bias), in float32. */
    private Tensor norm(Tensor x, Tensor w, Tensor b) {
        Tensor f = x.to(DType.FLOAT32);
        if (b == null) {
            Tensor r = f.mul(f).mean(-1, true).add(1e-6f).sqrt();
            return f.div(r).to(dtype).mul(w);
        }
        Tensor centered = f.sub(f.mean(-1, true));
        Tensor r = centered.mul(centered).mean(-1, true).add(1e-6f).sqrt();
        return centered.div(r).to(dtype).mul(w).add(b);
    }

    /** cos and sin {@code [n, 1, head]} of each patch: height then width frequencies, both halves. */
    private Tensor[] rotary(Patches p) {
        int hd = hidden / heads;
        int quarter = hd / 4;
        double[] inv = new double[quarter];
        for (int i = 0; i < quarter; i++) {
            inv[i] = 1 / Math.pow(theta, (2.0 * i) / (hd / 2.0));
        }
        int n = p.height() * p.width();
        float[] cos = new float[n * hd];
        float[] sin = new float[n * hd];
        int row = 0;
        for (int by = 0; by < p.height() / merge; by++) {
            for (int bx = 0; bx < p.width() / merge; bx++) {
                for (int my = 0; my < merge; my++) {
                    for (int mx = 0; mx < merge; mx++) {
                        int y = by * merge + my;
                        int x = bx * merge + mx;
                        for (int i = 0; i < quarter; i++) {
                            float ay = (float) y * (float) inv[i];
                            float ax = (float) x * (float) inv[i];
                            for (int half = 0; half < 2; half++) {
                                int at = row * hd + half * (hd / 2);
                                cos[at + i] = (float) Math.cos(ay);
                                sin[at + i] = (float) Math.sin(ay);
                                cos[at + quarter + i] = (float) Math.cos(ax);
                                sin[at + quarter + i] = (float) Math.sin(ax);
                            }
                        }
                        row++;
                    }
                }
            }
        }
        return new Tensor[] {Tensor.of(Shape.of(n, 1, hd), cos, device), Tensor.of(Shape.of(n, 1, hd), sin, device)};
    }

    /** {@code x cos + rotate_half(x) sin} in float32, as {@code [1, heads, n, head]}. */
    private Tensor rotate(Tensor x, Tensor[] rope) {
        int n = x.shape().dim(0);
        int hd = x.shape().dim(2);
        Tensor f = x.to(DType.FLOAT32);
        Tensor halfTurn = Tensor.concat(2, f.slice(2, hd / 2, hd).mul(-1f), f.slice(2, 0, hd / 2));
        Tensor r = f.mul(rope[0]).add(halfTurn.mul(rope[1])).to(dtype);
        return r.permute(1, 0, 2).reshape(1, heads, n, hd);
    }

    /** {@code [1, 1, n, n]}: 0 between patches of one window of {@code window x window} tokens, else -1e9. */
    private Tensor windows(Patches p) {
        int bw = p.width() / merge;
        int n = p.height() * p.width();
        int unit = merge * merge;
        int across = (bw + window - 1) / window;
        int[] id = new int[n];
        for (int i = 0; i < n; i++) {
            int block = i / unit;
            id[i] = block / bw / window * across + block % bw / window;
        }
        float[] m = new float[n * n];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (id[i] != id[j]) {
                    m[i * n + j] = -1e9f;
                }
            }
        }
        return Tensor.of(Shape.of(1, 1, n, n), m, device).to(dtype);
    }
}
