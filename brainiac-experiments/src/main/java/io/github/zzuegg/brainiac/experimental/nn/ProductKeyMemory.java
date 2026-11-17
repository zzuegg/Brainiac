package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Activation;
import io.github.zzuegg.brainiac.nn.Init;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * A product-key memory layer (Lample et al. 2019, <a href="https://arxiv.org/abs/1907.05242">arXiv:1907.05242</a>;
 * scaled up in "Memory Layers at Scale", Berges et al. 2024,
 * <a href="https://arxiv.org/abs/2412.09764">arXiv:2412.09764</a>): a huge trainable
 * key-value table that replaces or sits next to a feed-forward layer, adding capacity
 * without adding compute. The {@code n^2} keys are the Cartesian product of two sets of
 * {@code n} half-keys, so finding the top {@code k} of {@code n^2} keys costs two searches
 * over {@code n} half-keys plus {@code k^2} candidates. Only the {@code k} selected value rows
 * are read and receive gradients: a sparse update of a table that can be far larger than the
 * dense layers.
 *
 * <p>Per head: query {@code q = x W_q}, split in halves; scores {@code s1 = q1 K1^T},
 * {@code s2 = q2 K2^T}; candidates {@code s1[i] + s2[j]} over the top-k of each; output
 * {@code softmax(top-k scores) . V[i * n + j]}, summed over heads. Input
 * {@code [batch, features]} or {@code [batch, steps, features]}; output the same with
 * {@code valueSize} features.
 */
public final class ProductKeyMemory implements Layer {

    private final int features;
    private final int heads;
    private final int halfKeys;
    private final int keySize;
    private final int valueSize;
    private final int topK;
    private final Var wq;
    private final Var subKeys1;
    private final Var subKeys2;
    private final Var values;

    /**
     * Creates the layer.
     *
     * @param features the input features
     * @param halfKeys {@code n}: the table has {@code n * n} rows
     * @param keySize the query size per head (even)
     * @param valueSize the value (output) size
     * @param heads the query heads, which share the table
     * @param topK the rows each head reads
     * @param random the random source for the weights
     */
    public ProductKeyMemory(int features, int halfKeys, int keySize, int valueSize, int heads, int topK,
            SplittableRandom random) {
        if (keySize % 2 != 0 || topK < 1 || topK > halfKeys) {
            throw new IllegalArgumentException("key size must be even and 1 <= topK <= halfKeys");
        }
        this.features = features;
        this.heads = heads;
        this.halfKeys = halfKeys;
        this.keySize = keySize;
        this.valueSize = valueSize;
        this.topK = topK;
        wq = Var.param(Init.XAVIER.create(Shape.of(features, heads * keySize), features, heads * keySize, random));
        float scale = (float) (1 / Math.sqrt(keySize / 2.0));
        subKeys1 = Var.param(Tensor.uniform(Shape.of(halfKeys, keySize / 2), -scale, scale, random));
        subKeys2 = Var.param(Tensor.uniform(Shape.of(halfKeys, keySize / 2), -scale, scale, random));
        values = Var.param(Tensor.normal(Shape.of(halfKeys * halfKeys, valueSize), 0,
                (float) (1 / Math.sqrt(valueSize)), random));
    }

    /**
     * Describes the layer for model builders.
     *
     * @param halfKeys {@code n}: the table has {@code n * n} rows
     * @param keySize the query size per head
     * @param valueSize the output size
     * @param heads the query heads
     * @param topK the rows each head reads
     * @return the spec
     */
    public static LayerSpec spec(int halfKeys, int keySize, int valueSize, int heads, int topK) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                int[] d = input.toArray();
                d[d.length - 1] = valueSize;
                return Shape.of(d);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                return new ProductKeyMemory(input.dim(-1), halfKeys, keySize, valueSize, heads, topK, random);
            }
        };
    }

    /** @return the value table {@code [n * n, valueSize]} */
    public Var values() {
        return values;
    }

    @Override
    public Var forward(Var input) {
        Shape shape = input.shape();
        int rows = Math.toIntExact(shape.size() / features);
        Var x = input.reshape(rows, features);
        int half = keySize / 2;
        int k = topK;
        int n = halfKeys;
        Var out = null;
        for (int h = 0; h < heads; h++) {
            Var q = x.matmul(wq.slice(1, h * keySize, (h + 1) * keySize));
            Var s1 = q.slice(1, 0, half).matmul(subKeys1.transpose());
            Var s2 = q.slice(1, half, keySize).matmul(subKeys2.transpose());
            float[] a1 = s1.value().toArray();
            float[] a2 = s2.value().toArray();
            float[] pick1 = new float[rows * k];
            float[] pick2 = new float[rows * k];
            float[] rowIdx = new float[rows * k];
            for (int r = 0; r < rows; r++) {
                int[] t1 = top(a1, r * n, n, k);
                int[] t2 = top(a2, r * n, n, k);
                // best k of the k x k candidate sums
                float[] cand = new float[k * k];
                for (int i = 0; i < k; i++) {
                    for (int j = 0; j < k; j++) {
                        cand[i * k + j] = a1[r * n + t1[i]] + a2[r * n + t2[j]];
                    }
                }
                int[] best = top(cand, 0, k * k, k);
                for (int c = 0; c < k; c++) {
                    int i = t1[best[c] / k];
                    int j = t2[best[c] % k];
                    pick1[r * k + c] = r * n + i;
                    pick2[r * k + c] = r * n + j;
                    rowIdx[r * k + c] = i * n + j;
                }
            }
            var device = x.value().device();
            Var scores = s1.reshape(rows * n).gather(0, Tensor.of(Shape.of(rows, k), pick1, device))
                    .add(s2.reshape(rows * n).gather(0, Tensor.of(Shape.of(rows, k), pick2, device)));
            Var w = Activation.SOFTMAX.forward(scores).reshape(rows, 1, k);
            Var picked = values.gather(0, Tensor.of(Shape.of(rows, k), rowIdx, device));
            Var head = w.matmul(picked).reshape(rows, valueSize);
            out = out == null ? head : out.add(head);
        }
        int[] dims = shape.toArray();
        dims[dims.length - 1] = valueSize;
        return out.reshape(dims);
    }

    private static int[] top(float[] a, int off, int n, int k) {
        Integer[] idx = new Integer[n];
        for (int i = 0; i < n; i++) {
            idx[i] = i;
        }
        Arrays.sort(idx, (x, y) -> Float.compare(a[off + y], a[off + x]));
        int[] out = new int[k];
        for (int i = 0; i < k; i++) {
            out[i] = idx[i];
        }
        return out;
    }

    @Override
    public List<Var> parameters() {
        return List.of(wq, subKeys1, subKeys2, values);
    }

    @Override
    public int outputSize(int inputSize) {
        return valueSize;
    }

    @Override
    public String toString() {
        return "ProductKeyMemory(" + halfKeys * halfKeys + " slots, top " + topK + ", " + heads + " heads)";
    }
}
