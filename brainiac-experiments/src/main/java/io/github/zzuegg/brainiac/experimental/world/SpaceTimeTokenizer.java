package io.github.zzuegg.brainiac.experimental.world;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.audio.FiniteScalarQuantizer;
import io.github.zzuegg.brainiac.experimental.nn.SimpleMlp;
import io.github.zzuegg.brainiac.experimental.nn.VectorQuantizer;
import io.github.zzuegg.brainiac.nn.Functional;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.function.UnaryOperator;

/**
 * A MAGVIT-style 3-D video tokenizer (Yu et al., CVPR 2023,
 * <a href="https://arxiv.org/abs/2212.05199">arXiv:2212.05199</a>; the causal first frame of
 * MAGVIT-v2, <a href="https://arxiv.org/abs/2310.05737">arXiv:2310.05737</a>). A clip of
 * {@code 1 + s k} frames is compressed in space and time: the first frame alone is cut into
 * {@code patch x patch} patches, and every following group of {@code s} frames into
 * {@code s x patch x patch} tubelets (a 3-D convolution whose kernel equals its stride), each
 * encoded by an MLP to a latent vector and quantized to one discrete token. Because the first
 * frame is tokenized by itself, an image is a valid one-frame clip and image-only conditioning
 * (predict a video from its first frame) works on the same token canvas; because a tubelet
 * covers {@code s} frames, the static background and the smooth motion inside it cost one token
 * instead of {@code s}, so a clip needs about {@code s} times fewer tokens than per-frame
 * tokenization ({@code s = 1}, the 2-D baseline).
 *
 * <p>The quantizer is either {@link FiniteScalarQuantizer} (no codebook, no auxiliary loss) or a
 * {@link VectorQuantizer} with codebook and commitment losses and dead-code restarts. Tokens
 * are ordered first-frame patches, then each tubelet group's patches, so
 * {@link #frameOfToken} tells which frames a token covers, the basis of the multi-task masks of
 * {@link VideoMaskTasks}.
 */
public final class SpaceTimeTokenizer {

    private final int side;
    private final int patch;
    private final int stride;
    private final int dim;
    private final SimpleMlp firstEncoder;
    private final SimpleMlp firstDecoder;
    private final SimpleMlp encoder;
    private final SimpleMlp decoder;
    private final FiniteScalarQuantizer fsq;
    private final VectorQuantizer vq;

    private SpaceTimeTokenizer(int side, int patch, int stride, int hidden, int dim, FiniteScalarQuantizer fsq,
            VectorQuantizer vq, long seed) {
        this.side = side;
        this.patch = patch;
        this.stride = stride;
        this.dim = dim;
        this.fsq = fsq;
        this.vq = vq;
        int pp = patch * patch;
        UnaryOperator<Var> silu = v -> v.mul(v.sigmoid());
        firstEncoder = SimpleMlp.of(seed, silu, pp, hidden, dim);
        firstDecoder = SimpleMlp.of(seed + 1, silu, dim, hidden, pp);
        encoder = SimpleMlp.of(seed + 2, silu, stride * pp, hidden, dim);
        decoder = SimpleMlp.of(seed + 3, silu, dim, hidden, stride * pp);
    }

    /**
     * A tokenizer with finite scalar quantization.
     *
     * @param side the frame side
     * @param patch the patch side
     * @param stride frames per tubelet ({@code 1} for per-frame tokens)
     * @param hidden the MLP width
     * @param levels the FSQ levels (the latent size is their count)
     * @param seed the seed
     * @return the tokenizer
     */
    public static SpaceTimeTokenizer fsq(int side, int patch, int stride, int hidden, int[] levels, long seed) {
        return new SpaceTimeTokenizer(side, patch, stride, hidden, levels.length, new FiniteScalarQuantizer(levels),
                null, seed);
    }

    /**
     * A tokenizer with a learned codebook.
     *
     * @param side the frame side
     * @param patch the patch side
     * @param stride frames per tubelet
     * @param hidden the MLP width
     * @param codes the codebook size
     * @param dim the code vector size
     * @param seed the seed
     * @return the tokenizer
     */
    public static SpaceTimeTokenizer vq(int side, int patch, int stride, int hidden, int codes, int dim, long seed) {
        return new SpaceTimeTokenizer(side, patch, stride, hidden, dim, null, VectorQuantizer.of(codes, dim, 0.25f,
                seed + 4), seed);
    }

    /** @return the number of distinct tokens */
    public int vocabulary() {
        return fsq != null ? fsq.codebookSize() : vq.codebook().shape().dim(0);
    }

    /** @return frames per tubelet */
    public int stride() {
        return stride;
    }

    /** @return patches per frame */
    public int patchesPerFrame() {
        return (side / patch) * (side / patch);
    }

    /**
     * @param frames a clip length {@code 1 + stride k}
     * @return its token count
     */
    public int tokens(int frames) {
        check(frames);
        return patchesPerFrame() * (1 + (frames - 1) / stride);
    }

    /**
     * The first frame a token covers.
     *
     * @param token the token index
     * @return frame 0 for the first-frame tokens, else {@code 1 + stride g} for group {@code g}
     */
    public int frameOfToken(int token) {
        int g = token / patchesPerFrame();
        return g == 0 ? 0 : 1 + stride * (g - 1);
    }

    private void check(int frames) {
        if (frames < 1 || (frames - 1) % stride != 0) {
            throw new IllegalArgumentException("clips need 1 + " + stride + " k frames, got " + frames);
        }
    }

    /** @return the trainable parameters */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(firstEncoder.parameters());
        p.addAll(firstDecoder.parameters());
        p.addAll(encoder.parameters());
        p.addAll(decoder.parameters());
        if (vq != null) {
            p.addAll(vq.parameters());
        }
        return p;
    }

    /** The latents {@code [n * tokens, dim]} of clips {@code [n, T, side, side]}. */
    private Var latents(Var clips) {
        int n = clips.shape().dim(0);
        int t = clips.shape().dim(1);
        check(t);
        int p = patchesPerFrame();
        Var first = Functional.patchify(clips.slice(1, 0, 1), patch);   // [n, P, pp]
        List<Var> parts = new ArrayList<>();
        parts.add(firstEncoder.forward(first.reshape(n * p, patch * patch)).reshape(n, p, dim));
        int groups = (t - 1) / stride;
        if (groups > 0) {
            Var rest = clips.slice(1, 1, t).reshape(n * groups, stride, side, side);
            Var tubes = Functional.patchify(rest, patch);   // [n * groups, P, stride * pp]
            parts.add(encoder.forward(tubes.reshape(n * groups * p, stride * patch * patch)).reshape(n, groups * p,
                    dim));
        }
        return Var.concat(1, parts.toArray(new Var[0])).reshape(n * p * (1 + groups), dim);
    }

    /** Decodes quantized latents {@code [n * tokens, dim]} to clips. */
    private Var decodeLatents(Var q, int n, int frames) {
        int p = patchesPerFrame();
        int groups = (frames - 1) / stride;
        int per = p * (1 + groups);
        Var all = q.reshape(n, per, dim);
        Var first = firstDecoder.forward(all.slice(1, 0, p).reshape(n * p, dim)).reshape(n, p, patch * patch);
        List<Var> frames0 = new ArrayList<>();
        frames0.add(Functional.unpatchify(first, 1, side, side, patch));
        if (groups > 0) {
            Var tubes = decoder.forward(all.slice(1, p, per).reshape(n * groups * p, dim)).reshape(n * groups, p,
                    stride * patch * patch);
            frames0.add(Functional.unpatchify(tubes, stride, side, side, patch).reshape(n, groups * stride, side,
                    side));
        }
        return Var.concat(1, frames0.toArray(new Var[0]));
    }

    /**
     * The training loss: reconstruction MSE, plus the codebook and commitment terms for VQ.
     *
     * @param clips {@code [n, T, side, side]}
     * @return the loss
     */
    public Var loss(Tensor clips) {
        int n = clips.shape().dim(0);
        int t = clips.shape().dim(1);
        Var z = latents(Var.of(clips));
        Var q;
        Var aux = null;
        if (fsq != null) {
            q = fsq.quantize(z).output();
        } else {
            VectorQuantizer.Result r = vq.quantize(z);
            q = r.output();
            aux = r.loss();
        }
        Var rec = decodeLatents(q, n, t).sub(clips).square().mean();
        return aux == null ? rec : rec.add(aux);
    }

    /**
     * Re-seeds unused codebook entries from current latents (VQ only; FSQ has none).
     *
     * @param clips clips to draw latents from
     * @return the codes restarted
     */
    public int restartDeadCodes(Tensor clips) {
        if (vq == null) {
            return 0;
        }
        return vq.restartDeadCodes(Var.noGrad(() -> latents(Var.of(clips))).value());
    }

    /**
     * Tokenizes clips.
     *
     * @param clips {@code [n, T, side, side]}
     * @return {@code [n][tokens(T)]}
     */
    public int[][] encode(Tensor clips) {
        int n = clips.shape().dim(0);
        Tensor z = Var.noGrad(() -> latents(Var.of(clips))).value();
        int[] codes = fsq != null ? fsq.codes(z) : vq.nearest(z);
        int per = codes.length / n;
        int[][] out = new int[n][per];
        for (int i = 0; i < n; i++) {
            System.arraycopy(codes, i * per, out[i], 0, per);
        }
        return out;
    }

    /**
     * Decodes tokens.
     *
     * @param tokens {@code [n][tokens(frames)]}
     * @param frames the clip length
     * @return {@code [n, frames, side, side]}
     */
    public Tensor decode(int[][] tokens, int frames) {
        int n = tokens.length;
        int per = tokens[0].length;
        int[] flat = new int[n * per];
        for (int i = 0; i < n; i++) {
            System.arraycopy(tokens[i], 0, flat, i * per, per);
        }
        return Var.noGrad(() -> {
            Var q = fsq != null ? Var.of(fsq.lookup(flat)) : vq.lookup(flat);
            return decodeLatents(q, n, frames);
        }).value();
    }

    /**
     * Encodes and decodes.
     *
     * @param clips {@code [n, T, side, side]}
     * @return the reconstruction
     */
    public Tensor reconstruct(Tensor clips) {
        return decode(encode(clips), clips.shape().dim(1));
    }

    /**
     * The codebook usage of token sequences.
     *
     * @param tokens token ids
     * @return counts per code
     */
    public long[] usage(int[][] tokens) {
        long[] u = new long[vocabulary()];
        for (int[] row : tokens) {
            for (int c : row) {
                u[c]++;
            }
        }
        return u;
    }

    /**
     * The peak signal-to-noise ratio of a reconstruction, for signals in about {@code [0, 1]}.
     *
     * @param estimate the reconstruction
     * @param truth the clips
     * @return PSNR in dB
     */
    public static double psnr(Tensor estimate, Tensor truth) {
        float[] a = estimate.toArray();
        float[] b = truth.toArray();
        double mse = 0;
        for (int i = 0; i < a.length; i++) {
            double e = a[i] - b[i];
            mse += e * e;
        }
        mse /= a.length;
        return 10 * Math.log10(1 / Math.max(mse, 1e-12));
    }

    @Override
    public String toString() {
        return "SpaceTimeTokenizer(stride " + stride + ", " + (fsq != null ? fsq : "VQ " + vocabulary()) + ")";
    }
}
