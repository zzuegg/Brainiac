package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.random.RandomGenerator;

/**
 * PEER (Parameter Efficient Expert Retrieval): a layer of {@code n^2} single-neuron
 * experts, of which each token uses only {@code topK} per head, found by product-key
 * retrieval. A query is split in two halves; each half scores {@code n} sub-keys, the
 * {@code topK} best of each are combined into {@code topK^2} candidate experts (the score of
 * expert {@code (i, j)} is the sum of its two sub-key scores), and the best {@code topK}
 * candidates are kept. So finding the best of a million experts costs two searches over a
 * thousand sub-keys. Expert {@code e} is one hidden neuron with a down vector {@code u_e}
 * and an up vector {@code v_e}:
 *
 * <pre>
 * y = sum_heads sum_{e in top} softmax(score)_e * act(u_e . x) * v_e
 * </pre>
 *
 * <p>Retrieval becomes the compute primitive: parameters grow with {@code n^2} while each
 * token touches only {@code heads * topK} neurons, and only those receive gradients.
 *
 * <p>Paper: He, "Mixture of A Million Experts", <a href="https://arxiv.org/abs/2407.04153">arXiv:2407.04153</a>;
 * product keys from Lample et al., <a href="https://arxiv.org/abs/1907.05242">arXiv:1907.05242</a>.
 */
public final class ProductKeyExperts implements Layer {

    private final int features;
    private final int subKeys;
    private final int heads;
    private final int topK;
    private final int keySize;
    private final Var query;
    private final Var keys1;
    private final Var keys2;
    private final Var down;
    private final Var up;

    /**
     * Creates the layer.
     *
     * @param features the width of input and output
     * @param subKeys {@code n}: the layer has {@code n * n} experts
     * @param heads independent retrievals per token
     * @param topK experts per head and token
     * @param keySize the query and key width, even
     * @param random the random source
     */
    public ProductKeyExperts(int features, int subKeys, int heads, int topK, int keySize, RandomGenerator random) {
        if (keySize % 2 != 0 || topK > subKeys) {
            throw new IllegalArgumentException("keySize must be even and topK <= subKeys");
        }
        this.features = features;
        this.subKeys = subKeys;
        this.heads = heads;
        this.topK = topK;
        this.keySize = keySize;
        this.query = MixerOps.weight(random, features, heads * keySize);
        float ks = (float) (1 / Math.sqrt(keySize / 2.0));
        this.keys1 = MixerOps.uniform(random, ks, subKeys, keySize / 2);
        this.keys2 = MixerOps.uniform(random, ks, subKeys, keySize / 2);
        int experts = subKeys * subKeys;
        this.down = MixerOps.uniform(random, (float) (1 / Math.sqrt(features)), experts, features);
        this.up = MixerOps.uniform(random, (float) (1 / Math.sqrt(heads * topK)), experts, features);
    }

    /**
     * Describes the layer for {@code Model.sequential().add(...)}.
     *
     * @param subKeys {@code n}, for {@code n * n} experts
     * @param heads retrieval heads
     * @param topK experts per head and token
     * @param keySize query width
     * @return the spec
     */
    public static LayerSpec of(int subKeys, int heads, int topK, int keySize) {
        return MixerOps.sameShape((in, r) -> new ProductKeyExperts(in.dim(-1), subKeys, heads, topK, keySize, r));
    }

    /** @return the number of experts, {@code n * n} */
    public int experts() {
        return subKeys * subKeys;
    }

    /**
     * Returns the retrieved experts.
     *
     * @param input {@code [..., features]}
     * @return {@code [rows * heads][topK]} expert indices
     */
    public int[][] retrieve(Var input) {
        int rows = Math.toIntExact(input.shape().size() / features);
        Var x = input.reshape(rows, features);
        Var[] s = subScores(x, rows);
        return select(s[0].value().toArray(), s[1].value().toArray(), rows * heads);
    }

    private Var[] subScores(Var x, int rows) {
        Var q = x.matmul(query).reshape(rows * heads, keySize);
        Var s1 = q.slice(1, 0, keySize / 2).matmul(keys1.transpose());   // [rows*heads, n]
        Var s2 = q.slice(1, keySize / 2, keySize).matmul(keys2.transpose());
        return new Var[] {s1, s2};
    }

    /** Product-key search: top-k of each half, then top-k of the k*k sums. */
    private int[][] select(float[] s1, float[] s2, int queries) {
        int[][] out = new int[queries][];
        float[] cand = new float[topK * topK];
        for (int qi = 0; qi < queries; qi++) {
            int[] a = MixerOps.topK(s1, qi * subKeys, subKeys, topK);
            int[] b = MixerOps.topK(s2, qi * subKeys, subKeys, topK);
            for (int i = 0; i < topK; i++) {
                for (int j = 0; j < topK; j++) {
                    cand[i * topK + j] = s1[qi * subKeys + a[i]] + s2[qi * subKeys + b[j]];
                }
            }
            int[] best = MixerOps.topK(cand, 0, topK * topK, topK);
            out[qi] = new int[topK];
            for (int k = 0; k < topK; k++) {
                out[qi][k] = a[best[k] / topK] * subKeys + b[best[k] % topK];
            }
        }
        return out;
    }

    @Override
    public Var forward(Var input) {
        Shape s = input.shape();
        if (s.dim(-1) != features) {
            throw new IllegalArgumentException("ProductKeyExperts expects [..., " + features + "], got " + s);
        }
        int rows = Math.toIntExact(s.size() / features);
        int queries = rows * heads;
        Var x = input.reshape(rows, features);
        Var[] sub = subScores(x, rows);
        int[][] chosen = select(sub[0].value().toArray(), sub[1].value().toArray(), queries);
        int[] e = new int[queries * topK];
        int[] i1 = new int[queries * topK];
        int[] i2 = new int[queries * topK];
        for (int q = 0; q < queries; q++) {
            for (int k = 0; k < topK; k++) {
                int id = chosen[q][k];
                e[q * topK + k] = id;
                i1[q * topK + k] = q * subKeys + id / subKeys;
                i2[q * topK + k] = q * subKeys + id % subKeys;
            }
        }
        // differentiable scores of the chosen experts, then the router softmax
        Var score = sub[0].reshape(queries * subKeys).gather(0, MixerOps.ids(i1))
                .add(sub[1].reshape(queries * subKeys).gather(0, MixerOps.ids(i2)))
                .reshape(rows, heads * topK);
        Var gate = MixerOps.softmax(score.reshape(rows, heads, topK)).reshape(rows, heads * topK, 1);
        Tensor ids = MixerOps.ids(e);
        Var u = down.gather(0, ids).reshape(rows, heads * topK, features);
        Var v = up.gather(0, ids).reshape(rows, heads * topK, features);
        Var h = Activation.GELU.forward(u.mul(x.reshape(rows, 1, features)).sum(-1, true));   // [rows, hk, 1]
        return h.mul(gate).mul(v).sum(1, false).reshape(s.toArray());
    }

    @Override
    public List<Var> parameters() {
        return new ArrayList<>(List.of(query, keys1, keys2, down, up));
    }

    @Override
    public String toString() {
        return "ProductKeyExperts(" + experts() + " experts, heads=" + heads + ", top-" + topK + ")";
    }
}
