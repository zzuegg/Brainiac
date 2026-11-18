package io.github.zzuegg.brainiac.experimental.compress;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Network slimming (Liu et al. 2017, <a href="https://arxiv.org/abs/1708.06519">arXiv:1708.06519</a>):
 * train a CNN with an L1 penalty on every batch-norm scale {@code gamma} ({@link #l1Penalty});
 * channels whose {@code gamma} goes to zero contribute nothing and are removed together with the
 * matching filters of their convolution and the matching input channels of the next layer
 * ({@link #channelMasks} chooses them globally, the {@code select*} methods do the surgery),
 * giving a thinner dense network that needs no sparse kernels. Liu et al. 2019
 * (<a href="https://arxiv.org/abs/1810.05270">arXiv:1810.05270</a>) then showed that training the
 * pruned shape from scratch often matches, so slimming mainly finds a good width per layer.
 * {@link #filterNorms} gives the classic filter-magnitude baseline (Li et al. 2017).
 */
public final class NetworkSlimming {

    private NetworkSlimming() {
    }

    /**
     * @param gammas the batch-norm scales
     * @param lambda the penalty weight, for example 1e-4
     * @return {@code lambda * sum |gamma|}
     */
    public static Var l1Penalty(List<Var> gammas, float lambda) {
        Var s = null;
        for (Var g : gammas) {
            Var t = g.abs().sum();
            s = s == null ? t : s.add(t);
        }
        return s.mul(lambda);
    }

    /**
     * Chooses which channels to keep by a global threshold on {@code |gamma|}, keeping at least
     * {@code minKeep} of every layer (Liu et al. cap per-layer pruning the same way, as a global
     * threshold can otherwise empty a layer whose scales are smaller overall).
     *
     * @param gammas the batch-norm scales, one vector per layer
     * @param share the share of all channels to remove
     * @param minKeep the least share of each layer's channels to keep, for example 0.2
     * @return per layer, which channels survive
     */
    public static List<boolean[]> channelMasks(List<Var> gammas, double share, double minKeep) {
        List<float[]> scores = new ArrayList<>();
        for (Var g : gammas) {
            float[] v = g.value().toArray();
            for (int i = 0; i < v.length; i++) {
                v[i] = Math.abs(v[i]);
            }
            scores.add(v);
        }
        return keepGlobal(scores, share, minKeep);
    }

    /**
     * Per-layer pruning by a score, the same share in every layer.
     *
     * @param scores per layer, one score per channel
     * @param share the share to remove in each layer
     * @return per layer, which channels survive
     */
    public static List<boolean[]> perLayer(List<float[]> scores, double share) {
        List<boolean[]> keep = new ArrayList<>();
        for (float[] s : scores) {
            int drop = Math.min(s.length - 1, (int) Math.round(share * s.length));
            Integer[] idx = new Integer[s.length];
            for (int i = 0; i < idx.length; i++) {
                idx[i] = i;
            }
            Arrays.sort(idx, (a, b) -> Float.compare(s[a], s[b]));
            boolean[] k = new boolean[s.length];
            Arrays.fill(k, true);
            for (int i = 0; i < drop; i++) {
                k[idx[i]] = false;
            }
            keep.add(k);
        }
        return keep;
    }

    /**
     * @param filters convolution filters {@code [f, c, k...]}
     * @return the L1 norm of every filter
     */
    public static float[] filterNorms(Tensor filters) {
        int f = filters.shape().dim(0);
        float[] v = filters.toArray();
        int per = v.length / f;
        float[] n = new float[f];
        for (int i = 0; i < f; i++) {
            for (int j = 0; j < per; j++) {
                n[i] += Math.abs(v[i * per + j]);
            }
        }
        return n;
    }

    /**
     * Keeps a subset of a convolution's filters and input channels.
     *
     * @param filters {@code [f, c, k...]}
     * @param keepOut which filters survive, or null for all
     * @param keepIn which input channels survive, or null for all
     * @return the smaller filter tensor
     */
    public static Tensor selectConv(Tensor filters, boolean[] keepOut, boolean[] keepIn) {
        int[] d = filters.shape().toArray();
        int f = d[0];
        int c = d[1];
        int k = (int) (filters.shape().size() / ((long) f * c));
        boolean[] ko = keepOut == null ? all(f) : keepOut;
        boolean[] ki = keepIn == null ? all(c) : keepIn;
        int nf = count(ko);
        int nc = count(ki);
        float[] v = filters.toArray();
        float[] o = new float[nf * nc * k];
        int p = 0;
        for (int i = 0; i < f; i++) {
            if (!ko[i]) {
                continue;
            }
            for (int j = 0; j < c; j++) {
                if (ki[j]) {
                    System.arraycopy(v, (i * c + j) * k, o, p, k);
                    p += k;
                }
            }
        }
        d[0] = nf;
        d[1] = nc;
        return Tensor.of(Shape.of(d), o);
    }

    /**
     * @param vector a per-channel vector {@code [c]}
     * @param keep which entries survive
     * @return the kept entries
     */
    public static Tensor select(Tensor vector, boolean[] keep) {
        float[] v = vector.toArray();
        float[] o = new float[count(keep)];
        int p = 0;
        for (int i = 0; i < v.length; i++) {
            if (keep[i]) {
                o[p++] = v[i];
            }
        }
        return Tensor.of(Shape.of(o.length), o);
    }

    /**
     * @param matrix {@code [in, out]}
     * @param keep which rows survive
     * @return the kept rows
     */
    public static Tensor selectRows(Tensor matrix, boolean[] keep) {
        int cols = matrix.shape().dim(1);
        float[] v = matrix.toArray();
        float[] o = new float[count(keep) * cols];
        int p = 0;
        for (int i = 0; i < keep.length; i++) {
            if (keep[i]) {
                System.arraycopy(v, i * cols, o, p, cols);
                p += cols;
            }
        }
        return Tensor.of(Shape.of(count(keep), cols), o);
    }

    /**
     * @param keep a mask
     * @return how many are kept
     */
    public static int count(boolean[] keep) {
        int n = 0;
        for (boolean b : keep) {
            n += b ? 1 : 0;
        }
        return n;
    }

    private static List<boolean[]> keepGlobal(List<float[]> scores, double share, double minKeep) {
        int total = 0;
        for (float[] s : scores) {
            total += s.length;
        }
        float[] all = new float[total];
        int o = 0;
        for (float[] s : scores) {
            System.arraycopy(s, 0, all, o, s.length);
            o += s.length;
        }
        float[] sorted = all.clone();
        Arrays.sort(sorted);
        int drop = (int) Math.round(share * total);
        float cut = drop == 0 ? Float.NEGATIVE_INFINITY : sorted[drop - 1];
        int ties = 0;
        for (int i = 0; i < drop; i++) {
            ties += sorted[i] == cut ? 1 : 0;
        }
        List<boolean[]> keep = new ArrayList<>();
        int deficit = 0;
        for (float[] s : scores) {
            boolean[] k = new boolean[s.length];
            int best = 0;
            for (int i = 0; i < s.length; i++) {
                if (s[i] < cut) {
                    k[i] = false;
                } else if (s[i] == cut && ties > 0) {
                    k[i] = false;
                    ties--;
                } else {
                    k[i] = true;
                }
                best = s[i] > s[best] ? i : best;
            }
            k[best] = true;   // never remove a whole layer
            int floor = Math.max(1, (int) Math.ceil(minKeep * s.length));
            while (count(k) < floor) {   // restore the best-scoring removed channels
                int r = -1;
                for (int i = 0; i < s.length; i++) {
                    if (!k[i] && (r < 0 || s[i] > s[r])) {
                        r = i;
                    }
                }
                k[r] = true;
                deficit++;
            }
            keep.add(k);
        }
        // remove as many extra channels elsewhere as the floors restored, lowest scores first
        while (deficit > 0) {
            int layer = -1;
            int idx = -1;
            for (int l = 0; l < scores.size(); l++) {
                boolean[] k = keep.get(l);
                float[] s = scores.get(l);
                if (count(k) <= Math.max(1, (int) Math.ceil(minKeep * s.length))) {
                    continue;
                }
                for (int i = 0; i < s.length; i++) {
                    if (k[i] && (layer < 0 || s[i] < scores.get(layer)[idx])) {
                        layer = l;
                        idx = i;
                    }
                }
            }
            if (layer < 0) {
                break;
            }
            keep.get(layer)[idx] = false;
            deficit--;
        }
        return keep;
    }

    private static boolean[] all(int n) {
        boolean[] b = new boolean[n];
        Arrays.fill(b, true);
        return b;
    }
}
