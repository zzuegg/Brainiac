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
 * <p>The trunk is a {@link FusionDecoder} with a custom attention bias. The paper encodes
 * patches from a VAE latent; here the patches are raw pixel tiles, which for tiny images
 * plays the same role.
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
        Var img = Var.of(Tensor.of(Shape.of(b, patches, patchDim), flat)).matmul(patchIn).add(patchInB)
                .add(time.gather(0, Tensor.of(Shape.of(b), ts)).reshape(b, 1, trunk.dim()));
        Var x = Var.concat(1, trunk.embed(lead), img, trunk.embed(tail));
        return trunk.hidden(x, mask(a, length), null, training);
    }

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
        Var pred = h.slice(1, start, start + patches).matmul(patchOut).add(patchOutB);
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
            float[] eps = Var.noGrad(() -> hidden(prompts, cur, ts, after, false).slice(1, start, start + patches)
                    .matmul(patchOut).add(patchOutB)).value().toArray();
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
        p.addAll(List.of(patchIn, patchInB, patchOut, patchOutB, time));
        return p;
    }

    @Override
    public String toString() {
        return "TransfusionModel(" + trunk + ", " + patches + " patches)";
    }
}
