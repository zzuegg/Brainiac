package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.List;
import java.util.SplittableRandom;

/**
 * Learned context compression into a few special tokens: Activation Beacon (Zhang et al. 2024,
 * <a href="https://arxiv.org/abs/2401.03462">arXiv:2401.03462</a>) and gist tokens (Mu et al.,
 * NeurIPS 2023, <a href="https://arxiv.org/abs/2304.08467">arXiv:2304.08467</a>). A context is
 * cut into chunks; {@code beacons} learned tokens are appended to each chunk and, through an
 * attention mask, read only that chunk. Their outputs are the chunk's compressed memory, so
 * only {@code beacons} entries per chunk are kept as the window slides (a fixed compression
 * ratio {@code chunk / beacons}), and later tokens see the chunk only through them. Trained end
 * to end with the downstream loss, the beacons learn what to keep.
 * <ul>
 * <li>{@link Mode#BEACON}: chunks of {@code chunk} tokens, each compressed on its own;</li>
 * <li>{@link Mode#GIST}: the whole prompt is one chunk and its {@code beacons} gist tokens
 *     replace it, so a prompt's memory can be cached at a fixed size.</li>
 * </ul>
 * As a {@link Layer} it maps {@code [batch, tokens, features]} to
 * {@code [batch, chunks * beacons, features]}. Simplification: the paper's beacons of a chunk
 * also attend to earlier chunks' beacons and run in every layer of the LLM; here one attention
 * layer compresses each chunk independently.
 */
public final class BeaconCompressor implements Layer {

    /** How the context is cut. */
    public enum Mode {
        /** Fixed-size chunks, each with its own beacons. */
        BEACON,
        /** The whole input is one chunk, compressed into gist tokens. */
        GIST
    }

    private final Mode mode;
    private final int chunk;
    private final int beacons;
    private final int features;
    private final Var tokens;
    private final Var wq;
    private final Var wk;
    private final Var wv;
    private final Var wo;

    private BeaconCompressor(Mode mode, int chunk, int beacons, int features, SplittableRandom random) {
        if (beacons < 1 || features < 1 || mode == Mode.BEACON && chunk < beacons) {
            throw new IllegalArgumentException("need beacons >= 1 and, for BEACON, chunk >= beacons");
        }
        this.mode = mode;
        this.chunk = chunk;
        this.beacons = beacons;
        this.features = features;
        float s = (float) (1 / Math.sqrt(features));
        this.tokens = param(random, 0.01f, beacons, features);   // near-uniform reads at the start
        this.wq = param(random, s, features, features);
        this.wk = param(random, s, features, features);
        this.wv = param(random, s, features, features);
        this.wo = param(random, s, features, features);
    }

    private static Var param(SplittableRandom r, float scale, int... dims) {
        int n = 1;
        for (int d : dims) {
            n *= d;
        }
        float[] f = new float[n];
        for (int i = 0; i < n; i++) {
            f[i] = (float) (r.nextGaussian() * scale);
        }
        return Var.param(Tensor.of(Shape.of(dims), f));
    }

    /**
     * Activation Beacon compression.
     *
     * @param chunk tokens per chunk
     * @param beacons beacons per chunk
     * @param features the feature size
     * @param random the initialisation source
     * @return the compressor
     */
    public static BeaconCompressor beacon(int chunk, int beacons, int features, SplittableRandom random) {
        return new BeaconCompressor(Mode.BEACON, chunk, beacons, features, random);
    }

    /**
     * Gist-token compression of a whole prompt.
     *
     * @param gists gist tokens
     * @param features the feature size
     * @param random the initialisation source
     * @return the compressor
     */
    public static BeaconCompressor gist(int gists, int features, SplittableRandom random) {
        return new BeaconCompressor(Mode.GIST, 0, gists, features, random);
    }

    /**
     * A spec for {@code Model.sequential()}.
     *
     * @param mode the mode
     * @param chunk tokens per chunk ({@code BEACON})
     * @param beacons beacons per chunk
     * @return the spec
     */
    public static LayerSpec spec(Mode mode, int chunk, int beacons) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                int n = input.dim(1);
                int chunks = mode == Mode.GIST ? 1 : n / chunk;
                return Shape.of(input.dim(0), chunks * beacons, input.dim(2));
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new BeaconCompressor(mode, chunk, beacons, input.dim(input.rank() - 1), random);
            }
        };
    }

    /** @return the mode */
    public Mode mode() {
        return mode;
    }

    /**
     * @param tokens the context length
     * @return the compressed length
     */
    public int compressedLength(int tokens) {
        return mode == Mode.GIST ? beacons : tokens / chunk * beacons;
    }

    @Override
    public Var forward(Var input) {
        int batch = input.shape().dim(0);
        int n = input.shape().dim(1);
        int len = mode == Mode.GIST ? n : chunk;
        if (n % len != 0) {
            throw new IllegalArgumentException("context of " + n + " tokens is not a multiple of the chunk " + len);
        }
        int chunks = n / len;
        Var x = input.reshape(batch * chunks, 1, len, features);
        Var k = x.matmul(wk);
        Var v = x.matmul(wv);
        // scores [chunks, 1, len, beacons]: every beacon query against the chunk's keys
        Var scores = k.matmul(tokens.matmul(wq).transpose()).mul((float) (1 / Math.sqrt(features)));
        Var weights = scores.sub(scores.logsumexp(2, true)).exp();
        Var read = weights.permute(0, 1, 3, 2).matmul(v);
        // a residual over the chunk's read (the beacon embeddings only steer what is read)
        Var out = read.add(read.matmul(wo));
        return out.reshape(batch, chunks * beacons, features);
    }

    @Override
    public List<Var> parameters() {
        return List.of(tokens, wq, wk, wv, wo);
    }

    @Override
    public Shape outputShape(Shape inputShape) {
        return Shape.of(inputShape.dim(0), compressedLength(inputShape.dim(1)), features);
    }

    @Override
    public String toString() {
        return mode == Mode.GIST ? "Gist(" + beacons + ")" : "Beacon(" + chunk + " -> " + beacons + ")";
    }
}
