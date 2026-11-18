package io.github.zzuegg.brainiac.experimental.multimodal;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Transfusion (Zhou et al., ICLR 2025): one transformer trained on mixed sequences of text
 * tokens and continuous image patches. Text positions get the next-token loss; image patches
 * get a diffusion loss (predict the noise added at a random timestep, DDPM style with a
 * cosine schedule); the two are summed with a weight. Attention is causal over the whole
 * sequence but bidirectional among the patches of one image, whose span is marked by
 * {@code BOI} and {@code EOI} tokens. Patches enter through a linear layer plus a learned
 * timestep embedding and leave through another linear layer. The same model captions images
 * (image, then text) and generates them (text, then {@code BOI} and denoising of the image
 * span with deterministic DDIM steps).
 *
 * <p>The trunk is a {@link FusionDecoder} with a custom attention bias. By default the patches
 * are raw pixel tiles entering and leaving through linear layers (wave 9). With
 * {@link UNetPatches} (wave 10, the paper's setting) the image is a VAE latent (for example
 * from {@link ConvVae}), one patch per latent position, embedded by a small U-Net down block
 * (two 3x3 convolutions over the latent grid, then a linear map to the model width) and
 * un-embedded by the matching up block (a linear map back, concatenated with the down block's
 * features as a skip connection, then two 3x3 convolutions); the paper reports that these
 * layers matter a lot at small scale. {@link #forwardMacs} counts the multiply-adds of one
 * sequence for comparisons at matched compute.
 *
 * <p>Ids: the text vocabulary is {@code 0..textVocab-1}; {@code BOI = textVocab},
 * {@code EOI = textVocab + 1}, {@code BOS = textVocab + 2}.
 *
 * <p>Paper: <a href="https://arxiv.org/abs/2408.11039">arXiv:2408.11039</a>.
 */
public final class TransfusionModel {

    /**
     * One training sequence: text, an image, more text.
     *
     * @param before the text before the image (after {@code BOS})
     * @param image the image's patches, {@code patches * patchDim} values
     * @param after the text after the image
     * @param maxNoise the largest diffusion timestep to sample for this image (for example
     *     half the schedule when the image conditions a caption)
     */
    public record Sequence(int[] before, float[] image, int[] after, int maxNoise) {
    }

    /**
     * U-Net patch layers over a latent grid.
     *
     * @param grid the latent side (patches are {@code grid * grid} positions)
     * @param channels latent channels (the patch size)
     * @param hidden U-Net channels
     */
    public record UNetPatches(int grid, int channels, int hidden) {
    }

    private final int textVocab;
    private final int patchDim;
    private final int patches;
    private final int steps;
    private final FusionDecoder trunk;
    private final Var patchIn;
    private final Var patchInB;
    private final Var patchOut;
    private final Var patchOutB;
    private final Var time;
    private final double[] alphaBar;
    private final UNetPatches unet;
    private final Var down1;
    private final Var down1b;
    private final Var down2;
    private final Var down2b;
    private final Var downProj;
    private final Var downProjB;
    private final Var upProj;
    private final Var upProjB;
    private final Var up1;
    private final Var up1b;
    private final Var up2;
    private final Var up2b;

    /**
     * Creates the model.
     *
     * @param textVocab text tokens
     * @param patchDim values per patch
     * @param patches patches per image
     * @param dim the width
     * @param layers blocks
     * @param heads attention heads
     * @param maxLength the longest sequence
     * @param diffusionSteps timesteps of the noise schedule
     * @param seed the seed
     */
    public TransfusionModel(int textVocab, int patchDim, int patches, int dim, int layers, int heads, int maxLength,
            int diffusionSteps, long seed) {
        this(textVocab, patchDim, patches, dim, layers, heads, maxLength, diffusionSteps, null, seed);
    }

    /**
     * Creates the model, optionally with U-Net patch layers over a latent grid. Without them
     * it is exactly the model of the other constructor; the U-Net weights are drawn afterwards.
     *
     * @param textVocab text tokens
     * @param patchDim values per patch (the latent channels with U-Net patches)
     * @param patches patches per image ({@code grid * grid} with U-Net patches)
     * @param dim the width
     * @param layers blocks
     * @param heads attention heads
     * @param maxLength the longest sequence
     * @param diffusionSteps timesteps of the noise schedule
     * @param unet the U-Net patch layers, or null for linear patch layers
     * @param seed the seed
     */
    public TransfusionModel(int textVocab, int patchDim, int patches, int dim, int layers, int heads, int maxLength,
            int diffusionSteps, UNetPatches unet, long seed) {
        if (unet != null && (unet.channels() != patchDim || unet.grid() * unet.grid() != patches)) {
            throw new IllegalArgumentException("U-Net patches need patchDim = channels and patches = grid^2");
        }
        SplittableRandom r = new SplittableRandom(seed);
        this.textVocab = textVocab;
        this.patchDim = patchDim;
        this.patches = patches;
        this.steps = diffusionSteps;
        this.trunk = new FusionDecoder(textVocab + 3, dim, layers, heads, maxLength, r.nextLong());
        this.patchIn = Var.param(Tensor.normal(Shape.of(patchDim, dim), 0, (float) Math.sqrt(1.0 / patchDim), r));
        this.patchInB = Var.param(Tensor.zeros(Shape.of(dim)));
        this.patchOut = Var.param(Tensor.normal(Shape.of(dim, patchDim), 0, (float) Math.sqrt(1.0 / dim), r));
        this.patchOutB = Var.param(Tensor.zeros(Shape.of(patchDim)));
        this.time = Var.param(Tensor.normal(Shape.of(diffusionSteps + 1, dim), 0, 0.3f, r));
        this.alphaBar = new double[diffusionSteps + 1];
        double s = 0.008;
        double f0 = Math.pow(Math.cos(s / (1 + s) * Math.PI / 2), 2);
        for (int t = 0; t <= diffusionSteps; t++) {
            double f = Math.pow(Math.cos(((double) t / diffusionSteps + s) / (1 + s) * Math.PI / 2), 2);
            alphaBar[t] = Math.max(1e-4, Math.min(1, f / f0));
        }
        this.unet = unet;
        if (unet != null) {
            int c = unet.channels();
            int h = unet.hidden();
            this.down1 = ConvVae.conv(r, c, h, 3);
            this.down1b = ConvVae.zeros(h);
            this.down2 = ConvVae.conv(r, h, h, 3);
            this.down2b = ConvVae.zeros(h);
            this.downProj = Var.param(Tensor.normal(Shape.of(h, dim), 0, (float) Math.sqrt(1.0 / h), r));
            this.downProjB = Var.param(Tensor.zeros(Shape.of(dim)));
            this.upProj = Var.param(Tensor.normal(Shape.of(dim, h), 0, (float) Math.sqrt(1.0 / dim), r));
            this.upProjB = Var.param(Tensor.zeros(Shape.of(h)));
            this.up1 = ConvVae.conv(r, 2 * h, h, 3);
            this.up1b = ConvVae.zeros(h);
            this.up2 = Var.param(Tensor.normal(Shape.of(c, h, 3, 3), 0, (float) Math.sqrt(0.1 / (h * 9)), r));
            this.up2b = ConvVae.zeros(c);
        } else {
            this.down1 = null;
            this.down1b = null;
            this.down2 = null;
            this.down2b = null;
            this.downProj = null;
            this.downProjB = null;
            this.upProj = null;
            this.upProjB = null;
            this.up1 = null;
            this.up1b = null;
            this.up2 = null;
            this.up2b = null;
        }
    }

    /** The image tokens {@code [b, patches, dim]} and the U-Net skip features (or null). */
    private Var[] embedImage(Var noisy) {
        int b = noisy.shape().dim(0);
        if (unet == null) {
            return new Var[] {noisy.matmul(patchIn).add(patchInB), null};
        }
        int g = unet.grid();
        Var grid = noisy.reshape(b, g, g, patchDim).permute(0, 3, 1, 2);
        Var d = ConvVae.silu(ConvVae.apply(grid, down1, down1b, 1));
        d = ConvVae.silu(ConvVae.apply(d, down2, down2b, 1)).add(d);
        Var tokens = d.reshape(b, unet.hidden(), patches).permute(0, 2, 1).matmul(downProj).add(downProjB);
        return new Var[] {tokens, d};
    }

    /** The predicted noise {@code [b, patches, patchDim]} of the image span's hidden states. */
    private Var unembed(Var h, Var skip) {
        if (unet == null) {
            return h.matmul(patchOut).add(patchOutB);
        }
        int b = h.shape().dim(0);
        int g = unet.grid();
        Var u = h.matmul(upProj).add(upProjB).permute(0, 2, 1).reshape(b, unet.hidden(), g, g);
        u = ConvVae.silu(ConvVae.apply(Var.concat(1, u, skip), up1, up1b, 1));
        u = ConvVae.apply(u, up2, up2b, 1);
        return u.permute(0, 2, 3, 1).reshape(b, patches, patchDim);
    }

    /**
     * Multiply-adds of one forward pass over a sequence: the trunk (projections, MLPs,
     * attention, output head at every position) and the patch layers.
     *
     * @param textTokens text tokens in the sequence besides {@code BOS}, {@code BOI}, {@code EOI}
     * @return multiply-adds
     */
    public long forwardMacs(int textTokens) {
        long length = textTokens + 3L + patches;
        long d = trunk.dim();
        long perToken = trunk.layers() * (4 * d * d + 2 * d * 4 * d) + d * trunk.vocab();
        long attention = trunk.layers() * length * length * d * 2;
        long patchLayers;
        if (unet == null) {
            patchLayers = 2L * patches * patchDim * d;
        } else {
            long c = unet.channels();
            long h = unet.hidden();
            patchLayers = patches * (9 * c * h + 9 * h * h + h * d + d * h + 9 * 2 * h * h + 9 * h * c);
        }
        return length * perToken + attention + patchLayers;
    }

    /** @return the begin-of-image id */
    public int boi() {
        return textVocab;
    }

    /** @return the end-of-image id */
    public int eoi() {
        return textVocab + 1;
    }

    /** @return the begin-of-sequence id */
    public int bos() {
        return textVocab + 2;
    }

    /** The attention bias: causal, plus bidirectional within the image span. */
    private Tensor mask(int a, int length) {
        int start = a + 2;              // BOS, before..., BOI at a + 1, patches from a + 2
        int end = start + patches;      // exclusive
        float[] m = new float[length * length];
        for (int i = 0; i < length; i++) {
            for (int j = 0; j < length; j++) {
                boolean allowed = j <= i || i >= start && i < end && j >= start && j < end;
                m[i * length + j] = allowed ? 0 : -1e9f;
            }
        }
        return Tensor.of(Shape.of(1, 1, length, length), m);
    }

    /** Runs the trunk; returns {hidden states, the layout's token ids (-1 at patches)}. */
    private Var hidden(List<int[]> before, float[][] noisy, int[] t, List<int[]> after, boolean training) {
        int b = before.size();
        int a = before.get(0).length;
        int c = after.get(0).length;
        int length = 1 + a + 1 + patches + 1 + c;
        int[][] lead = new int[b][a + 2];
        int[][] tail = new int[b][c + 1];
        for (int i = 0; i < b; i++) {
            lead[i][0] = bos();
            System.arraycopy(before.get(i), 0, lead[i], 1, a);
            lead[i][a + 1] = boi();
            tail[i][0] = eoi();
            System.arraycopy(after.get(i), 0, tail[i], 1, c);
        }
        float[] flat = new float[b * patches * patchDim];
        float[] ts = new float[b];
        for (int i = 0; i < b; i++) {
            System.arraycopy(noisy[i], 0, flat, i * patches * patchDim, patches * patchDim);
            ts[i] = t[i];
        }
        Var[] embedded = embedImage(Var.of(Tensor.of(Shape.of(b, patches, patchDim), flat)));
        lastSkip = embedded[1];
        Var img = embedded[0].add(time.gather(0, Tensor.of(Shape.of(b), ts)).reshape(b, 1, trunk.dim()));
        Var x = Var.concat(1, trunk.embed(lead), img, trunk.embed(tail));
        return trunk.hidden(x, mask(a, length), null, training);
    }

    /** The U-Net skip features of the last {@link #hidden} call (null without U-Net patches). */
    private Var lastSkip;

    /**
     * The training loss of a batch with one layout (equal text lengths before and after).
     *
     * @param batch the sequences
     * @param diffusionWeight the weight of the diffusion loss
     * @param random the source of timesteps and noise
     * @return {@code lm loss + weight * diffusion loss}
     */
    public Var loss(List<Sequence> batch, float diffusionWeight, SplittableRandom random) {
        int b = batch.size();
        int a = batch.get(0).before().length;
        int c = batch.get(0).after().length;
        int n = patches * patchDim;
        float[][] noisy = new float[b][n];
        float[] eps = new float[b * n];
        int[] t = new int[b];
        List<int[]> before = new ArrayList<>();
        List<int[]> after = new ArrayList<>();
        for (int i = 0; i < b; i++) {
            Sequence s = batch.get(i);
            before.add(s.before());
            after.add(s.after());
            t[i] = 1 + random.nextInt(Math.max(1, Math.min(steps, s.maxNoise())));
            double sa = Math.sqrt(alphaBar[t[i]]);
            double sn = Math.sqrt(1 - alphaBar[t[i]]);
            for (int k = 0; k < n; k++) {
                float e = (float) random.nextGaussian();
                eps[i * n + k] = e;
                noisy[i][k] = (float) (sa * s.image()[k] + sn * e);
            }
        }
        Var h = hidden(before, noisy, t, after, true);
        int length = h.shape().dim(1);
        int start = a + 2;
        Var pred = unembed(h.slice(1, start, start + patches), lastSkip);
        Var diffusion = pred.sub(Tensor.of(Shape.of(b, patches, patchDim), eps)).square().mean();
        int[][] targets = new int[b][length];
        for (int i = 0; i < b; i++) {
            java.util.Arrays.fill(targets[i], -1);
            for (int k = 0; k < a; k++) {
                targets[i][k] = before.get(i)[k];         // position k predicts before[k]
            }
            targets[i][a] = boi();                       // the last text token predicts BOI
            for (int k = 0; k < c; k++) {
                targets[i][start + patches + k] = after.get(i)[k];   // EOI and later predict after[k]
            }
        }
        Var logits = trunk.project(h);
        Var lm = FusionDecoder.loss(logits, targets);
        return lm.add(diffusion.mul(diffusionWeight));
    }

    /**
     * Captions images: the most likely text token after {@code BOS before BOI image EOI}.
     *
     * @param before text before each image (may be empty arrays)
     * @param images clean images, {@code patches * patchDim} values each
     * @return per image the most likely next text token
     */
    public int[] caption(List<int[]> before, float[][] images) {
        int b = images.length;
        List<int[]> after = new ArrayList<>();
        for (int i = 0; i < b; i++) {
            after.add(new int[0]);
        }
        int[] nearlyClean = new int[b];
        java.util.Arrays.fill(nearlyClean, 1);
        float[] logits = Var.noGrad(() -> {
            Var h = hidden(before, images, nearlyClean, after, false);
            int last = h.shape().dim(1) - 1;
            return trunk.project(h.slice(1, last, last + 1));
        }).value().toArray();
        int v = textVocab + 3;
        int[] out = new int[b];
        for (int i = 0; i < b; i++) {
            for (int k = 1; k < textVocab; k++) {
                out[i] = logits[i * v + k] > logits[i * v + out[i]] ? k : out[i];
            }
        }
        return out;
    }

    /**
     * Generates images after text prompts by deterministic DDIM sampling.
     *
     * @param prompts the text before each image (equal lengths)
     * @param samplingSteps denoising steps (strided over the schedule)
     * @param random the source of the initial noise
     * @return the images
     */
    public float[][] generate(List<int[]> prompts, int samplingSteps, SplittableRandom random) {
        int b = prompts.size();
        int n = patches * patchDim;
        int a = prompts.get(0).length;
        float[][] x = new float[b][n];
        for (int i = 0; i < b; i++) {
            for (int k = 0; k < n; k++) {
                x[i][k] = (float) random.nextGaussian();
            }
        }
        List<int[]> after = new ArrayList<>();
        for (int i = 0; i < b; i++) {
            after.add(new int[0]);
        }
        int start = a + 2;
        for (int s = samplingSteps; s >= 1; s--) {
            int t = (int) Math.round((double) s * steps / samplingSteps);
            int prev = (int) Math.round((double) (s - 1) * steps / samplingSteps);
            int[] ts = new int[b];
            java.util.Arrays.fill(ts, t);
            float[][] cur = x;
            float[] eps = Var.noGrad(() -> unembed(hidden(prompts, cur, ts, after, false)
                    .slice(1, start, start + patches), lastSkip)).value().toArray();
            double sa = Math.sqrt(alphaBar[t]);
            double sn = Math.sqrt(1 - alphaBar[t]);
            double pa = Math.sqrt(alphaBar[prev]);
            double pn = Math.sqrt(1 - alphaBar[prev]);
            float[][] next = new float[b][n];
            for (int i = 0; i < b; i++) {
                for (int k = 0; k < n; k++) {
                    double e = eps[i * n + k];
                    double x0 = (x[i][k] - sn * e) / sa;
                    x0 = Math.max(-1.5, Math.min(1.5, x0));
                    next[i][k] = (float) (prev == 0 ? x0 : pa * x0 + pn * e);
                }
            }
            x = next;
        }
        return x;
    }

    /** @return every trainable value */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(trunk.parameters());
        if (unet == null) {
            p.addAll(List.of(patchIn, patchInB, patchOut, patchOutB, time));
        } else {
            p.add(time);
            p.addAll(List.of(down1, down1b, down2, down2b, downProj, downProjB, upProj, upProjB, up1, up1b, up2, up2b));
        }
        return p;
    }

    @Override
    public String toString() {
        return "TransfusionModel(" + trunk + ", " + patches + " patches" + (unet != null ? ", U-Net patch layers" : "")
                + ")";
    }
}
