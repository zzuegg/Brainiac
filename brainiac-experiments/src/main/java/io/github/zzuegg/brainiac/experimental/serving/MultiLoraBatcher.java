package io.github.zzuegg.brainiac.experimental.serving;

import io.github.zzuegg.brainiac.experimental.nn.LoraAdapterDense;
import io.github.zzuegg.brainiac.nn.Dense;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Serving many LoRA adapters in one batch: S-LoRA (Sheng et al., MLSys 2024,
 * <a href="https://arxiv.org/abs/2311.03285">arXiv:2311.03285</a>) and Punica (Chen et al., MLSys
 * 2024, <a href="https://arxiv.org/abs/2310.18547">arXiv:2310.18547</a>). The frozen base weight
 * runs once for the whole batch, {@code X W0 + b}; each request's low-rank delta
 * {@code s x A_i B_i} is added by a <em>segmented gather matmul</em> (Punica's SGMV): the rows
 * using adapter {@code i} are gathered, multiplied by {@code A_i} then {@code B_i}, and
 * scattered back. Requests for different adapters share every base-model pass, so throughput
 * hardly depends on how many adapters are live.
 *
 * <p>The alternatives, also offered here for comparison: {@link #forwardMerged} folds one
 * adapter into the weight ({@code W0 + s A B}, built and cached on a switch) and serves only that
 * adapter's requests; {@link #forwardUnmerged} serves one adapter's rows unmerged. Adapters
 * are core {@link LoraAdapterDense} layers over the same base {@link Dense}.
 */
public final class MultiLoraBatcher {

    private final Tensor weight;
    private final Tensor bias;
    private final List<Tensor> a = new ArrayList<>();
    private final List<Tensor> b = new ArrayList<>();
    private int mergedAdapter = -1;
    private Tensor merged;
    private long merges;

    private MultiLoraBatcher(Dense base, List<LoraAdapterDense> adapters) {
        this.weight = base.weight().value();
        this.bias = base.bias().value();
        for (LoraAdapterDense ad : adapters) {
            Tensor ai = ad.a().value();
            Tensor bi = ad.b().value();
            Tensor ab = ai.matmul(bi);
            Tensor delta = ad.weight().value().sub(ad.frozen());
            float[] d = delta.toArray();
            float[] u = ab.toArray();
            double num = 0;
            double den = 0;
            for (int i = 0; i < d.length; i++) {
                num += d[i] * u[i];
                den += u[i] * u[i];
            }
            float s = den == 0 ? 1 : (float) (num / den);   // the adapter's alpha / r
            a.add(ai);
            b.add(bi.mul(s));
        }
    }

    /**
     * @param base the frozen base layer, the same for every adapter
     * @param adapters the adapters, by id
     * @return the batcher
     */
    public static MultiLoraBatcher of(Dense base, List<LoraAdapterDense> adapters) {
        return new MultiLoraBatcher(Objects.requireNonNull(base), List.copyOf(adapters));
    }

    /** @return the number of adapters */
    public int adapters() {
        return a.size();
    }

    /** @return how often {@link #forwardMerged} had to fold a new adapter into the weight */
    public long merges() {
        return merges;
    }

    /**
     * One batched pass, each row with its own adapter (SGMV).
     *
     * @param x the inputs {@code [n, in]}
     * @param adapter each row's adapter id, or -1 for the base model
     * @return {@code [n, out]}, before the base layer's activation
     */
    public Tensor forward(Tensor x, int[] adapter) {
        int n = x.shape().dim(0);
        int in = x.shape().dim(1);
        Tensor base = x.matmul(weight).add(bias);
        float[] out = base.toArray();
        int width = out.length / n;
        float[] rows = x.toArray();
        int[] count = new int[a.size()];
        for (int id : adapter) {
            if (id >= 0) {
                count[id]++;
            }
        }
        for (int id = 0; id < a.size(); id++) {
            if (count[id] == 0) {
                continue;
            }
            float[] seg = new float[count[id] * in];
            int[] where = new int[count[id]];
            int k = 0;
            for (int r = 0; r < n; r++) {
                if (adapter[r] == id) {
                    System.arraycopy(rows, r * in, seg, k * in, in);
                    where[k++] = r;
                }
            }
            float[] delta = Tensor.of(Shape.of(count[id], in), seg).matmul(a.get(id)).matmul(b.get(id)).toArray();
            for (int j = 0; j < where.length; j++) {
                for (int c = 0; c < width; c++) {
                    out[where[j] * width + c] += delta[j * width + c];
                }
            }
        }
        return Tensor.of(Shape.of(n, width), out);
    }

    /**
     * One pass of rows that all use one adapter, with the adapter folded into the weight
     * (rebuilt only when the adapter changes).
     *
     * @param x the inputs {@code [n, in]}
     * @param adapter the adapter of every row
     * @return {@code [n, out]}
     */
    public Tensor forwardMerged(Tensor x, int adapter) {
        if (adapter != mergedAdapter) {
            merged = weight.add(a.get(adapter).matmul(b.get(adapter)));
            mergedAdapter = adapter;
            merges++;
        }
        return x.matmul(merged).add(bias);
    }

    /**
     * One pass of rows that all use one adapter, unmerged.
     *
     * @param x the inputs {@code [n, in]}
     * @param adapter the adapter of every row
     * @return {@code [n, out]}
     */
    public Tensor forwardUnmerged(Tensor x, int adapter) {
        return x.matmul(weight).add(bias).add(x.matmul(a.get(adapter)).matmul(b.get(adapter)));
    }
}
