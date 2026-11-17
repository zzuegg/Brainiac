package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Quest: query-aware page selection over a paged KV cache. Every page (block of slots) keeps
 * the per-channel minimum and maximum of its keys. For a decode query {@code q} the largest
 * logit any key of the page can give is bounded by {@code sum_c max(q_c min_c, q_c max_c)};
 * attention then reads only the {@code pages} pages with the highest bounds (plus the page
 * being written), so the bytes read per decode step stay constant however long the context,
 * while nothing is evicted: a later query can still reach every token. Eviction methods (H2O,
 * SnapKV) decide once what is gone for good.
 *
 * <p>This is a {@link KvBlockStore} decorator, so it drops in under {@link PagedLlama} over any
 * store (normally a {@link PagedKvCache} with the same block size): writes go through and update
 * the page bounds; single-token views return only the selected pages' slots; prompt passes (more
 * than one query) read everything. Each head ranks pages by its own bound and the heads' picks
 * are merged by their best rank, so one shared slot list serves all heads (the
 * {@code View} has one slot list per layer). {@link Mode#RECENT} keeps the same number of most
 * recent pages instead, the window baseline at equal bytes. {@link #bytesRead()} counts the key
 * and value bytes attention reads; on this CPU simulation the inner view is gathered whole and
 * then cut, which a real kernel would not do.
 *
 * <p>Tang et al., "Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference"
 * (ICML 2024), <a href="https://arxiv.org/abs/2406.10774">arXiv:2406.10774</a>.
 */
public final class QuestKvStore implements KvBlockStore {

    /** How decode steps pick pages. */
    public enum Mode {
        /** the pages with the best upper bounds for the query */
        QUEST,
        /** the most recent pages: a sliding window of the same size */
        RECENT,
        /** every page: the full-cache reference, counted the same way */
        ALL
    }

    private final KvBlockStore inner;
    private final int pageSize;
    private final int pages;
    private final int kvHeads;
    private final int headDim;
    private final Mode mode;
    private final Map<Integer, float[][][]> mins = new HashMap<>();   // seq -> [layer][page][kvHeads*headDim]
    private final Map<Integer, float[][][]> maxs = new HashMap<>();
    private long bytesRead;
    private long fullBytes;

    /**
     * @param inner the store holding the keys and values
     * @param pageSize slots per page
     * @param pages the pages each decode step reads (the current page included)
     * @param kvHeads key/value heads per layer
     * @param headDim the head size
     * @param mode how pages are picked
     */
    public QuestKvStore(KvBlockStore inner, int pageSize, int pages, int kvHeads, int headDim, Mode mode) {
        if (pageSize < 1 || pages < 1) {
            throw new IllegalArgumentException("page size and pages must be positive");
        }
        this.inner = Objects.requireNonNull(inner, "inner");
        this.pageSize = pageSize;
        this.pages = pages;
        this.kvHeads = kvHeads;
        this.headDim = headDim;
        this.mode = Objects.requireNonNull(mode, "mode");
    }

    /** @return the key and value bytes attention read so far */
    public long bytesRead() {
        return bytesRead;
    }

    /** @return the bytes full attention would have read for the same steps */
    public long fullBytesRead() {
        return fullBytes;
    }

    /**
     * Quest's upper bound of the logits a page can give a query.
     *
     * @param q the query (one head)
     * @param min the page's per-channel key minimum
     * @param max the page's per-channel key maximum
     * @param offset where the head's channels start in {@code min} and {@code max}
     * @return {@code sum_c max(q_c min_c, q_c max_c)}
     */
    public static double upperBound(float[] q, float[] min, float[] max, int offset) {
        double s = 0;
        for (int c = 0; c < q.length; c++) {
            s += Math.max(q[c] * min[offset + c], q[c] * max[offset + c]);
        }
        return s;
    }

    /**
     * Quest page selection for one head over plain arrays: the {@code pages} pages with the best
     * bounds, always including the last page.
     *
     * @param keys the head's keys {@code [token][dim]}
     * @param q the query
     * @param pageSize tokens per page
     * @param pages the pages read
     * @return per token whether attention reads it
     */
    public static boolean[] selectPages(float[][] keys, float[] q, int pageSize, int pages) {
        int n = keys.length;
        int count = (n + pageSize - 1) / pageSize;
        int dim = q.length;
        double[] bound = new double[count];
        float[] min = new float[dim];
        float[] max = new float[dim];
        for (int p = 0; p < count; p++) {
            Arrays.fill(min, Float.POSITIVE_INFINITY);
            Arrays.fill(max, Float.NEGATIVE_INFINITY);
            for (int t = p * pageSize; t < Math.min(n, (p + 1) * pageSize); t++) {
                for (int c = 0; c < dim; c++) {
                    min[c] = Math.min(min[c], keys[t][c]);
                    max[c] = Math.max(max[c], keys[t][c]);
                }
            }
            bound[p] = upperBound(q, min, max, 0);
        }
        boolean[] chosen = top(bound, pages);
        boolean[] keep = new boolean[n];
        for (int t = 0; t < n; t++) {
            keep[t] = chosen[t / pageSize];
        }
        return keep;
    }

    /** The {@code k} best pages, the last page always among them. */
    private static boolean[] top(double[] bound, int k) {
        int count = bound.length;
        boolean[] chosen = new boolean[count];
        if (count == 0) {
            return chosen;
        }
        chosen[count - 1] = true;
        Integer[] order = new Integer[count - 1];
        for (int i = 0; i < order.length; i++) {
            order[i] = i;
        }
        Arrays.sort(order, (a, b) -> Double.compare(bound[b], bound[a]));
        for (int i = 0; i < Math.min(k - 1, order.length); i++) {
            chosen[order[i]] = true;
        }
        return chosen;
    }

    @Override
    public int layers() {
        return inner.layers();
    }

    @Override
    public int newSequence() {
        int s = inner.newSequence();
        mins.put(s, new float[inner.layers()][0][]);
        maxs.put(s, new float[inner.layers()][0][]);
        return s;
    }

    @Override
    public int fork(int sequence) {
        int s = inner.fork(sequence);
        mins.put(s, copy(mins.get(sequence)));
        maxs.put(s, copy(maxs.get(sequence)));
        return s;
    }

    private static float[][][] copy(float[][][] a) {
        float[][][] c = new float[a.length][][];
        for (int l = 0; l < a.length; l++) {
            c[l] = new float[a[l].length][];
            for (int p = 0; p < a[l].length; p++) {
                c[l][p] = a[l][p].clone();
            }
        }
        return c;
    }

    @Override
    public void free(int sequence) {
        inner.free(sequence);
        mins.remove(sequence);
        maxs.remove(sequence);
    }

    @Override
    public int length(int sequence) {
        return inner.length(sequence);
    }

    @Override
    public void truncate(int sequence, int length) {
        inner.truncate(sequence, length);
        // bounds of a cut page stay loose (still valid upper bounds); drop whole pages beyond the end
        int keep = (length + pageSize - 1) / pageSize;
        float[][][] lo = mins.get(sequence);
        float[][][] hi = maxs.get(sequence);
        for (int l = 0; l < lo.length; l++) {
            if (lo[l].length > keep) {
                lo[l] = Arrays.copyOf(lo[l], keep);
                hi[l] = Arrays.copyOf(hi[l], keep);
            }
        }
    }

    @Override
    public boolean canAppend(int sequence, int steps) {
        return inner.canAppend(sequence, steps);
    }

    @Override
    public int append(int sequence, int steps) {
        return inner.append(sequence, steps);
    }

    @Override
    public void write(int sequence, int layer, int from, Tensor keys, Tensor values) {
        inner.write(sequence, layer, from, keys, values);
        float[] k = keys.toArray();
        int width = kvHeads * headDim;
        int steps = k.length / width;
        float[][][] lo = mins.get(sequence);
        float[][][] hi = maxs.get(sequence);
        for (int t = 0; t < steps; t++) {
            int page = (from + t) / pageSize;
            if (page >= lo[layer].length) {
                int old = lo[layer].length;
                lo[layer] = Arrays.copyOf(lo[layer], page + 1);
                hi[layer] = Arrays.copyOf(hi[layer], page + 1);
                for (int p = old; p <= page; p++) {
                    lo[layer][p] = new float[width];
                    hi[layer][p] = new float[width];
                    Arrays.fill(lo[layer][p], Float.POSITIVE_INFINITY);
                    Arrays.fill(hi[layer][p], Float.NEGATIVE_INFINITY);
                }
            }
            for (int c = 0; c < width; c++) {
                float v = k[t * width + c];
                lo[layer][page][c] = Math.min(lo[layer][page][c], v);
                hi[layer][page][c] = Math.max(hi[layer][page][c], v);
            }
        }
    }

    @Override
    public View view(int sequence, int layer, Tensor queries) {
        View all = inner.view(sequence, layer, queries);
        int n = all.slots().length;
        long perSlot = 2L * kvHeads * headDim * Float.BYTES;
        fullBytes += n * perSlot;
        int steps = queries.shape().dim(1);
        int count = (n + pageSize - 1) / pageSize;
        if (steps > 1 || mode == Mode.ALL || count <= pages) {
            bytesRead += n * perSlot;
            return all;
        }
        boolean[] chosen;
        if (mode == Mode.RECENT) {
            chosen = new boolean[count];
            for (int p = Math.max(0, count - pages); p < count; p++) {
                chosen[p] = true;
            }
        } else {
            chosen = questPages(sequence, layer, queries, count);
        }
        int kept = 0;
        for (int t = 0; t < n; t++) {
            kept += chosen[all.slots()[t] / pageSize] ? 1 : 0;
        }
        float[] idx = new float[kept];
        int[] slots = new int[kept];
        int i = 0;
        for (int t = 0; t < n; t++) {
            if (chosen[all.slots()[t] / pageSize]) {
                idx[i] = t;
                slots[i++] = all.slots()[t];
            }
        }
        bytesRead += kept * perSlot;
        Tensor index = Tensor.of(Shape.of(kept), idx, all.keys().device());
        return new View(all.keys().gather(0, index), all.values().gather(0, index), slots);
    }

    /** Each query head ranks the pages by its bound; pages go by their best rank over heads. */
    private boolean[] questPages(int sequence, int layer, Tensor queries, int count) {
        int heads = queries.shape().dim(0);
        float[] q = queries.toArray();
        float[][] lo = mins.get(sequence)[layer];
        float[][] hi = maxs.get(sequence)[layer];
        int group = heads / kvHeads;
        int[] bestRank = new int[count];
        Arrays.fill(bestRank, Integer.MAX_VALUE);
        float[] row = new float[headDim];
        for (int h = 0; h < heads; h++) {
            System.arraycopy(q, h * headDim, row, 0, headDim);
            int offset = (h / group) * headDim;
            double[] bound = new double[count - 1];
            Integer[] order = new Integer[count - 1];
            for (int p = 0; p < count - 1; p++) {
                bound[p] = upperBound(row, lo[p], hi[p], offset);
                order[p] = p;
            }
            Arrays.sort(order, (a, b) -> Double.compare(bound[b], bound[a]));
            for (int r = 0; r < order.length; r++) {
                bestRank[order[r]] = Math.min(bestRank[order[r]], r);
            }
        }
        Integer[] pagesByRank = new Integer[count - 1];
        for (int p = 0; p < count - 1; p++) {
            pagesByRank[p] = p;
        }
        Arrays.sort(pagesByRank, (a, b) -> Integer.compare(bestRank[a], bestRank[b]));
        boolean[] chosen = new boolean[count];
        chosen[count - 1] = true;
        for (int i = 0; i < Math.min(pages - 1, pagesByRank.length); i++) {
            chosen[pagesByRank[i]] = true;
        }
        return chosen;
    }

    @Override
    public long residentBytes() {
        return inner.residentBytes();
    }

    @Override
    public long peakResidentBytes() {
        return inner.peakResidentBytes();
    }

    /**
     * @param sequence the sequence
     * @return the pages it spans so far
     */
    public int pages(int sequence) {
        return mins.get(sequence)[0].length;
    }

    @Override
    public String toString() {
        return "QuestKvStore(" + mode + ", " + pages + " pages of " + pageSize + " over " + inner + ")";
    }
}
