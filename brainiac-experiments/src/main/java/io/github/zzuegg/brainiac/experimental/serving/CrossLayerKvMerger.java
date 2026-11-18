package io.github.zzuegg.brainiac.experimental.serving;

import io.github.zzuegg.brainiac.experimental.llm.KvBlockStore;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * MiniCache: training-free merging of the KV cache across adjacent layers (Liu et al., NeurIPS
 * 2024, <a href="https://arxiv.org/abs/2405.14366">arXiv:2405.14366</a>). In the middle and
 * deep layers of an LLM the key and value states of neighbouring layers point in nearly the
 * same direction. For a pair of layers MiniCache stores, per token and head, one shared unit
 * direction (the SLERP midpoint of the two layers' normalised states) plus each layer's norm,
 * and restores layer {@code l}'s state as {@code |x_l| * merged}. Tokens whose two states are
 * far apart in angle are <em>retention</em> tokens: they keep both originals. Unlike
 * cross-layer attention ({@code CrossLayerKvBlock}), nothing is trained.
 *
 * <p>{@link #merge} works on two arrays of per-head vectors; {@link #compress} applies it to
 * every layer pair {@code (l, l + 1)} from a start layer of a {@link KvBlockStore} sequence,
 * after prefill, writing the reconstructions back so later decode steps attend to them.
 */
public final class CrossLayerKvMerger {

    private final double retentionAngle;
    private final double t;

    private CrossLayerKvMerger(double retentionAngle, double t) {
        if (retentionAngle < 0 || t < 0 || t > 1) {
            throw new IllegalArgumentException("retentionAngle must be at least 0 and t in [0, 1]");
        }
        this.retentionAngle = retentionAngle;
        this.t = t;
    }

    /**
     * @param retentionAngle tokens whose two states differ by more than this angle (radians)
     *     keep both originals; {@code Math.PI} retains nothing
     * @return the merger with the paper's interpolation {@code t = 0.5}
     */
    public static CrossLayerKvMerger of(double retentionAngle) {
        return new CrossLayerKvMerger(retentionAngle, 0.5);
    }

    /**
     * A merged pair of layers.
     *
     * @param first the restored states of the first layer
     * @param second the restored states of the second layer
     * @param retained per vector whether it kept both originals
     * @param storedFloats numbers stored: a direction and two norms per merged vector, two
     *     whole vectors per retained one
     */
    public record Merged(float[][] first, float[][] second, boolean[] retained, long storedFloats) {

        /** @return the share of vectors that were retained */
        public double retentionShare() {
            int r = 0;
            for (boolean b : retained) {
                r += b ? 1 : 0;
            }
            return retained.length == 0 ? 0 : r / (double) retained.length;
        }
    }

    /**
     * Merges two layers' states.
     *
     * @param a the first layer's vectors {@code [n][d]}
     * @param b the second layer's vectors {@code [n][d]}
     * @return the merge
     */
    public Merged merge(float[][] a, float[][] b) {
        int n = a.length;
        float[][] ra = new float[n][];
        float[][] rb = new float[n][];
        boolean[] kept = new boolean[n];
        long stored = 0;
        for (int i = 0; i < n; i++) {
            double na = norm(a[i]);
            double nb = norm(b[i]);
            double cos = na == 0 || nb == 0 ? 1 : dot(a[i], b[i]) / (na * nb);
            double omega = Math.acos(Math.max(-1, Math.min(1, cos)));
            if (omega > retentionAngle) {
                kept[i] = true;
                ra[i] = a[i].clone();
                rb[i] = b[i].clone();
                stored += 2L * a[i].length;
                continue;
            }
            float[] m = slerp(a[i], b[i], na, nb, omega);
            ra[i] = scaled(m, na);
            rb[i] = scaled(m, nb);
            stored += a[i].length + 2;
        }
        return new Merged(ra, rb, kept, stored);
    }

    private float[] slerp(float[] a, float[] b, double na, double nb, double omega) {
        int d = a.length;
        float[] m = new float[d];
        double s = Math.sin(omega);
        double wa = s < 1e-6 ? 1 - t : Math.sin((1 - t) * omega) / s;
        double wb = s < 1e-6 ? t : Math.sin(t * omega) / s;
        for (int c = 0; c < d; c++) {
            m[c] = (float) (wa * (na == 0 ? 0 : a[c] / na) + wb * (nb == 0 ? 0 : b[c] / nb));
        }
        double nm = norm(m);
        for (int c = 0; c < d && nm > 0; c++) {
            m[c] /= (float) nm;
        }
        return m;
    }

    private static float[] scaled(float[] u, double s) {
        float[] o = new float[u.length];
        for (int c = 0; c < u.length; c++) {
            o[c] = (float) (u[c] * s);
        }
        return o;
    }

    /**
     * What {@link #compress} did.
     *
     * @param storedFloats numbers stored for every layer after merging
     * @param fullFloats numbers stored without merging
     * @param retentionShare the share of merged-pair vectors kept whole
     */
    public record Stats(long storedFloats, long fullFloats, double retentionShare) {

        /** @return stored over full */
        public double ratio() {
            return storedFloats / (double) fullFloats;
        }
    }

    /**
     * Merges the cache of one sequence for the layer pairs {@code (from, from + 1)},
     * {@code (from + 2, from + 3)} and so on, keys and values separately, and writes the
     * restored states back into the store.
     *
     * @param store the store
     * @param sequence the sequence
     * @param from the first merged layer
     * @return the memory accounting
     */
    public Stats compress(KvBlockStore store, int sequence, int from) {
        int layers = store.layers();
        long storedFloats = 0;
        long fullFloats = 0;
        long retained = 0;
        long mergedVectors = 0;
        Tensor dummy = Tensor.zeros(Shape.of(1, 1, 1));
        for (int l = 0; l < layers; l++) {
            KvBlockStore.View v = store.view(sequence, l, dummy);
            long floats = 2L * v.keys().shape().size();
            fullFloats += floats;
            if (l < from || l + 1 >= layers || (l - from) % 2 == 1) {
                if (l < from || (l - from) % 2 == 0) {
                    storedFloats += floats;   // unmerged layer (second layers of pairs are counted with the first)
                }
                continue;
            }
            KvBlockStore.View w = store.view(sequence, l + 1, dummy);
            Shape shape = v.keys().shape();
            Tensor[] restoredA = new Tensor[2];
            Tensor[] restoredB = new Tensor[2];
            Tensor[][] pair = {{v.keys(), w.keys()}, {v.values(), w.values()}};
            for (int kv = 0; kv < 2; kv++) {
                float[][] a = rows(pair[kv][0]);
                float[][] b = rows(pair[kv][1]);
                Merged m = merge(a, b);
                storedFloats += m.storedFloats();
                for (boolean r : m.retained()) {
                    retained += r ? 1 : 0;
                }
                mergedVectors += a.length;
                restoredA[kv] = Tensor.of(shape, flat(m.first()));
                restoredB[kv] = Tensor.of(shape, flat(m.second()));
            }
            store.write(sequence, l, 0, restoredA[0], restoredA[1]);
            store.write(sequence, l + 1, 0, restoredB[0], restoredB[1]);
        }
        return new Stats(storedFloats, fullFloats, mergedVectors == 0 ? 0 : retained / (double) mergedVectors);
    }

    /** {@code [slots, heads, dim]} as one row per (slot, head). */
    static float[][] rows(Tensor x) {
        float[] f = x.toArray();
        int d = x.shape().dim(x.shape().rank() - 1);
        float[][] out = new float[f.length / d][d];
        for (int i = 0; i < out.length; i++) {
            System.arraycopy(f, i * d, out[i], 0, d);
        }
        return out;
    }

    static float[] flat(float[][] rows) {
        int d = rows.length == 0 ? 0 : rows[0].length;
        float[] f = new float[rows.length * d];
        for (int i = 0; i < rows.length; i++) {
            System.arraycopy(rows[i], 0, f, i * d, d);
        }
        return f;
    }

    private static double dot(float[] a, float[] b) {
        double d = 0;
        for (int i = 0; i < a.length; i++) {
            d += a[i] * b[i];
        }
        return d;
    }

    private static double norm(float[] a) {
        return Math.sqrt(dot(a, a));
    }

    @Override
    public String toString() {
        return "MiniCache(retention angle " + retentionAngle + ")";
    }
}
