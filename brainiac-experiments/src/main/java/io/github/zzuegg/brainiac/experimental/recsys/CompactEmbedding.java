package io.github.zzuegg.brainiac.experimental.recsys;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Compact embedding tables for huge ID vocabularies, as a {@link Layer} from {@code [batch]} (or
 * {@code [batch, 1]}) integer IDs stored as floats to {@code [batch, dim]} vectors. ID tables
 * dominate the memory of ranking models; these trade a little accuracy for far fewer bytes.
 * <ul>
 *   <li>{@link Kind#FULL}: one row per ID (the baseline).</li>
 *   <li>{@link Kind#HASHED} (the hashing trick, Weinberger et al., ICML 2009,
 *       <a href="https://arxiv.org/abs/0902.2206">arXiv:0902.2206</a>): IDs are hashed into fewer
 *       buckets ({@link #hash}), accepting collisions.</li>
 *   <li>{@link Kind#QUOTIENT_REMAINDER} (Shi et al., KDD 2020, <a href="https://arxiv.org/abs/1909.02107">
 *       arXiv:1909.02107</a>): two small tables indexed by {@code id / m} and {@code id mod m},
 *       combined by element-wise product, so no two IDs share both rows: every ID keeps a unique
 *       vector with about {@code 2 sqrt(V)} rows instead of {@code V}.</li>
 *   <li>{@link Kind#FREQUENCY_THRESHOLD}: the most frequent IDs keep their own rows, the rest share
 *       one out-of-vocabulary row.</li>
 *   <li>{@link Kind#MIXED_DIMENSION} (Ginart et al. 2021): frequent IDs get wide vectors, the tail
 *       narrow ones, each projected to {@code dim}.</li>
 *   <li>{@link Kind#LOW_RANK}: a factored table {@code E = A B} with {@code A} of rank {@code r} (the
 *       {@code LowRankDense} factorisation applied to a lookup table).</li>
 * </ul>
 * IDs are expected in frequency order (ID 0 the most frequent) for the frequency-based kinds; pass
 * a remapped ID otherwise.
 */
public final class CompactEmbedding implements Layer {

    /** The table layout. */
    public enum Kind {
        /** One row per ID. */
        FULL,
        /** Hashed buckets. */
        HASHED,
        /** Quotient-remainder product of two tables. */
        QUOTIENT_REMAINDER,
        /** Frequent IDs plus one shared OOV row. */
        FREQUENCY_THRESHOLD,
        /** Wide vectors for frequent IDs, narrow ones for the tail. */
        MIXED_DIMENSION,
        /** A rank-limited factored table. */
        LOW_RANK
    }

    private final Kind kind;
    private final int vocabulary;
    private final int dim;
    private final int size;
    private final Var a;
    private final Var b;
    private final Var projection;
    private final Var tailProjection;

    private CompactEmbedding(Kind kind, int vocabulary, int dim, int size, int rank, SplittableRandom random) {
        this.kind = kind;
        this.vocabulary = vocabulary;
        this.dim = dim;
        this.size = size;
        float s = 0.1f;
        switch (kind) {
            case FULL -> {
                a = RecsysOps.normal(random, s, vocabulary, dim);
                b = null;
                projection = null;
                tailProjection = null;
            }
            case HASHED -> {
                a = RecsysOps.normal(random, s, size, dim);
                b = null;
                projection = null;
                tailProjection = null;
            }
            case QUOTIENT_REMAINDER -> {
                a = Var.param(Tensor.normal(Shape.of((vocabulary + size - 1) / size, dim), 1, 0.1f, random));
                b = RecsysOps.normal(random, s, size, dim);
                projection = null;
                tailProjection = null;
            }
            case FREQUENCY_THRESHOLD -> {
                a = RecsysOps.normal(random, s, size + 1, dim);
                b = null;
                projection = null;
                tailProjection = null;
            }
            case MIXED_DIMENSION -> {
                a = RecsysOps.normal(random, s, size, dim);
                b = RecsysOps.normal(random, s, vocabulary - size, rank);
                projection = null;
                tailProjection = RecsysOps.matrix(random, rank, dim);
            }
            case LOW_RANK -> {
                a = RecsysOps.normal(random, s, vocabulary, rank);
                b = null;
                projection = RecsysOps.matrix(random, rank, dim);
                tailProjection = null;
            }
            default -> throw new IllegalArgumentException(kind.toString());
        }
    }

    /**
     * @param vocabulary IDs
     * @param dim vector width
     * @param seed the seed
     * @return a full table
     */
    public static CompactEmbedding full(int vocabulary, int dim, long seed) {
        return new CompactEmbedding(Kind.FULL, vocabulary, dim, vocabulary, 0, new SplittableRandom(seed));
    }

    /**
     * @param vocabulary IDs
     * @param dim vector width
     * @param buckets hash buckets
     * @param seed the seed
     * @return a hashed table
     */
    public static CompactEmbedding hashed(int vocabulary, int dim, int buckets, long seed) {
        return new CompactEmbedding(Kind.HASHED, vocabulary, dim, buckets, 0, new SplittableRandom(seed));
    }

    /**
     * @param vocabulary IDs
     * @param dim vector width
     * @param remainders the remainder table's rows {@code m} (the quotient table has {@code V / m})
     * @param seed the seed
     * @return a quotient-remainder table
     */
    public static CompactEmbedding quotientRemainder(int vocabulary, int dim, int remainders, long seed) {
        return new CompactEmbedding(Kind.QUOTIENT_REMAINDER, vocabulary, dim, remainders, 0,
                new SplittableRandom(seed));
    }

    /**
     * @param vocabulary IDs
     * @param dim vector width
     * @param kept the most frequent IDs that keep their own row
     * @param seed the seed
     * @return a thresholded table
     */
    public static CompactEmbedding frequencyThreshold(int vocabulary, int dim, int kept, long seed) {
        return new CompactEmbedding(Kind.FREQUENCY_THRESHOLD, vocabulary, dim, kept, 0, new SplittableRandom(seed));
    }

    /**
     * @param vocabulary IDs
     * @param dim vector width (and the frequent IDs' width)
     * @param frequent the most frequent IDs that get full-width rows
     * @param tailDim the tail's narrow width
     * @param seed the seed
     * @return a mixed-dimension table
     */
    public static CompactEmbedding mixedDimension(int vocabulary, int dim, int frequent, int tailDim, long seed) {
        return new CompactEmbedding(Kind.MIXED_DIMENSION, vocabulary, dim, frequent, tailDim,
                new SplittableRandom(seed));
    }

    /**
     * @param vocabulary IDs
     * @param dim vector width
     * @param rank the factor rank
     * @param seed the seed
     * @return a low-rank table
     */
    public static CompactEmbedding lowRank(int vocabulary, int dim, int rank, long seed) {
        return new CompactEmbedding(Kind.LOW_RANK, vocabulary, dim, vocabulary, rank, new SplittableRandom(seed));
    }

    /**
     * @param vocabulary IDs
     * @param dim width
     * @return a spec for a full table (other kinds through the factories)
     */
    public static LayerSpec spec(int vocabulary, int dim) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(dim);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new CompactEmbedding(Kind.FULL, vocabulary, dim, vocabulary, 0, random);
            }
        };
    }

    /**
     * The feature-hashing function shared by the hashed embedding and hashed linear models: a
     * 64-bit mix (SplitMix64 finaliser) of the ID and a seed, reduced to a bucket.
     *
     * @param id the raw ID or feature key
     * @param seed the hash seed
     * @param buckets the number of buckets
     * @return the bucket in {@code [0, buckets)}
     */
    public static int hash(long id, long seed, int buckets) {
        long z = id * 0x9E3779B97F4A7C15L + seed * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        z = z ^ (z >>> 31);
        return (int) Math.floorMod(z, (long) buckets);
    }

    /**
     * The sign hash of the hashing trick ({@code +1} or {@code -1}), which makes collisions cancel
     * in expectation in hashed linear models.
     *
     * @param id the ID
     * @param seed the seed
     * @return {@code +1} or {@code -1}
     */
    public static int sign(long id, long seed) {
        return (hash(id, seed ^ 0x5DEECE66DL, 2) << 1) - 1;
    }

    private static Tensor index(int[] rows) {
        float[] f = new float[rows.length];
        for (int i = 0; i < f.length; i++) {
            f[i] = rows[i];
        }
        return Tensor.of(Shape.of(rows.length), f);
    }

    @Override
    public Var forward(Var input) {
        float[] raw = input.value().toArray();
        int n = raw.length;
        int[] ids = new int[n];
        for (int i = 0; i < n; i++) {
            ids[i] = (int) raw[i];
        }
        return lookup(ids);
    }

    /**
     * Looks up IDs.
     *
     * @param ids the IDs
     * @return {@code [ids, dim]} vectors
     */
    public Var lookup(int[] ids) {
        int n = ids.length;
        int[] r1 = new int[n];
        int[] r2 = new int[n];
        switch (kind) {
            case FULL, LOW_RANK -> {
                Var rows = a.gather(0, index(ids));
                return kind == Kind.FULL ? rows : rows.matmul(projection);
            }
            case HASHED -> {
                for (int i = 0; i < n; i++) {
                    r1[i] = hash(ids[i], 17, size);
                }
                return a.gather(0, index(r1));
            }
            case QUOTIENT_REMAINDER -> {
                for (int i = 0; i < n; i++) {
                    r1[i] = ids[i] / size;
                    r2[i] = ids[i] % size;
                }
                return a.gather(0, index(r1)).mul(b.gather(0, index(r2)));
            }
            case FREQUENCY_THRESHOLD -> {
                for (int i = 0; i < n; i++) {
                    r1[i] = ids[i] < size ? ids[i] : size;
                }
                return a.gather(0, index(r1));
            }
            case MIXED_DIMENSION -> {
                float[] head = new float[n];
                for (int i = 0; i < n; i++) {
                    boolean frequent = ids[i] < size;
                    r1[i] = frequent ? ids[i] : 0;
                    r2[i] = frequent ? 0 : ids[i] - size;
                    head[i] = frequent ? 1 : 0;
                }
                Tensor isHead = Tensor.of(Shape.of(n, 1), head);
                Var wide = a.gather(0, index(r1)).mul(isHead);
                Var narrow = b.gather(0, index(r2)).matmul(tailProjection).mul(isHead.neg().add(1f));
                return wide.add(narrow);
            }
            default -> throw new IllegalStateException();
        }
    }

    @Override
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>();
        for (Var v : new Var[] {a, b, projection, tailProjection}) {
            if (v != null) {
                p.add(v);
            }
        }
        return p;
    }

    /** @return the number of stored floats */
    public long floats() {
        long s = 0;
        for (Var v : parameters()) {
            s += v.shape().size();
        }
        return s;
    }

    /** @return the table bytes at float32 */
    public long bytes() {
        return 4 * floats();
    }

    @Override
    public int outputSize(int inputSize) {
        return dim;
    }

    @Override
    public Shape outputShape(Shape inputShape) {
        return Shape.of(inputShape.dim(0), dim);
    }

    @Override
    public String toString() {
        return "CompactEmbedding(" + kind + ", " + vocabulary + " ids -> " + dim + ")";
    }
}
