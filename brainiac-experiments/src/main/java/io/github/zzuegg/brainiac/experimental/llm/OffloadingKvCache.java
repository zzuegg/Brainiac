package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A key/value cache that keeps only a few blocks per layer on the device and pages the rest
 * to host memory, choosing from the current queries which host blocks to bring back: the
 * activation-aware offloading of InfiniGen and ShadowKV, with Quest's query-aware block
 * selection. Every full block is written to the host tier as it fills. For each decode step
 * and layer the store keeps the last {@code recent} blocks resident, scores every other
 * block by an upper bound of its attention logits, {@code sum_c max(q_c * min_c, q_c *
 * max_c)} over each block's per-channel key minimum and maximum (summed over heads), and
 * makes the {@code hot - recent} best ones resident, fetching from the host those that are not
 * yet. Attention then reads only resident blocks, so device memory per sequence stays at
 * {@code hot} blocks per layer however long the context, at the price of the fetched bytes.
 *
 * <p>How blocks are scored is a {@link Selection}: Quest's per-channel bounds (tiny, but
 * loose for small heads), a 4-bit key sketch of every token kept on the device (ShadowKV and
 * InfiniGen keep such a cheap key proxy resident; it costs an eighth of the keys and values
 * and its bytes count as resident), or an oracle that reads the exact keys, to measure what the
 * proxies lose. Each head ranks the blocks by its own scores and the heads take turns picking.
 *
 * <p>Prompts (passes of more than one token) read every block, streamed from the host block
 * by block as FlexGen does, and count as transfers; {@link #resetPeak()} after the prompt
 * makes {@link #peakResidentBytes()} measure decoding alone.
 *
 * <p>This is a CPU simulation: both tiers are Java arrays and a "fetch" is a counted copy,
 * which is what the experiments measure (bytes kept hot, bytes moved, output error). On a GPU
 * backend the host tier would be pinned host memory and a fetch a device upload.
 *
 * <p>Tang et al., "Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference"
 * (2024), <a href="https://arxiv.org/abs/2406.10774">arXiv:2406.10774</a>; Lee et al.,
 * "InfiniGen" (OSDI 2024), <a href="https://arxiv.org/abs/2406.19707">arXiv:2406.19707</a>;
 * Sun et al., "ShadowKV" (2024), <a href="https://arxiv.org/abs/2410.21465">arXiv:2410.21465</a>.
 */
public final class OffloadingKvCache implements KvBlockStore {

    /** How a decode step scores the blocks it may fetch. */
    public enum Selection {
        /** Quest: the upper bound {@code sum_c max(q_c min_c, q_c max_c)} of each block's logits. */
        BOUNDS,
        /** The best logit over the block's tokens, from 4-bit keys kept on the device. */
        SKETCH,
        /** The best logit from the exact keys: a reference, not a deployable policy. */
        ORACLE
    }

    private final class Block {
        final float[] keys;
        final float[] values;
        final float[] min;
        final float[] max;
        final byte[] sketch;
        final float[] sketchScale;
        int filled;
        boolean resident = true;

        Block() {
            keys = new float[blockSize * width];
            values = new float[blockSize * width];
            min = new float[width];
            max = new float[width];
            sketch = new byte[blockSize * width];
            sketchScale = new float[blockSize * kvHeads];
            Arrays.fill(min, Float.POSITIVE_INFINITY);
            Arrays.fill(max, Float.NEGATIVE_INFINITY);
        }
    }

    private final class Sequence {
        final List<List<Block>> layers = new ArrayList<>();
        int length;

        Sequence() {
            for (int l = 0; l < layerCount; l++) {
                layers.add(new ArrayList<>());
            }
        }
    }

    private final int layerCount;
    private final int kvHeads;
    private final int headDim;
    private final int width;
    private final int blockSize;
    private final int hot;
    private final int recent;
    private final Device device;
    private final Map<Integer, Sequence> sequences = new HashMap<>();
    private int nextId;
    private long transferred;
    private long peak;
    private long fetches;
    private Selection selection = Selection.SKETCH;

    /**
     * @param layers the decoder layers
     * @param kvHeads the key/value heads
     * @param headDim the head size
     * @param blockSize slots per block
     * @param hot the blocks per layer and sequence kept on the device while decoding
     * @param recent of those, the latest blocks always kept
     * @param device where views are built
     */
    public OffloadingKvCache(int layers, int kvHeads, int headDim, int blockSize, int hot, int recent,
            Device device) {
        if (recent < 1 || hot < recent || blockSize < 1) {
            throw new IllegalArgumentException("need blockSize >= 1 and hot >= recent >= 1, got " + blockSize + ", "
                    + hot + ", " + recent);
        }
        this.layerCount = layers;
        this.kvHeads = kvHeads;
        this.headDim = headDim;
        this.width = kvHeads * headDim;
        this.blockSize = blockSize;
        this.hot = hot;
        this.recent = recent;
        this.device = device;
    }

    @Override
    public int layers() {
        return layerCount;
    }

    /** @return the bytes of one block of one layer, keys and values */
    public long bytesPerLayerBlock() {
        return 2L * blockSize * width * Float.BYTES;
    }

    /** @return the bytes fetched from the host tier so far */
    public long transferredBytes() {
        return transferred;
    }

    /** @return the blocks fetched from the host tier so far */
    public long fetches() {
        return fetches;
    }

    /** @return the bytes of every block, resident or not (what a device-only cache would hold) */
    public long totalBytes() {
        long blocks = 0;
        for (Sequence s : sequences.values()) {
            for (List<Block> layer : s.layers) {
                blocks += layer.size();
            }
        }
        return blocks * bytesPerLayerBlock();
    }

    /** Forgets the peak, so that it measures what follows (decoding after a prompt). */
    public void resetPeak() {
        peak = residentBytes();
    }

    @Override
    public int newSequence() {
        int id = nextId++;
        sequences.put(id, new Sequence());
        return id;
    }

    /**
     * Not supported: offloaded sequences are not shared.
     *
     * @param sequence the sequence
     * @return never
     */
    @Override
    public int fork(int sequence) {
        throw new UnsupportedOperationException("offloaded sequences cannot be forked");
    }

    @Override
    public void free(int sequence) {
        if (sequences.remove(sequence) == null) {
            throw new IllegalArgumentException("no sequence " + sequence);
        }
    }

    @Override
    public int length(int sequence) {
        return get(sequence).length;
    }

    @Override
    public void truncate(int sequence, int length) {
        Sequence s = get(sequence);
        if (length < 0 || length > s.length) {
            throw new IllegalArgumentException("cannot keep " + length + " of " + s.length);
        }
        int keep = (length + blockSize - 1) / blockSize;
        for (List<Block> layer : s.layers) {
            while (layer.size() > keep) {
                layer.removeLast();
            }
            if (keep > 0) {
                Block last = layer.getLast();
                last.filled = Math.min(last.filled, length - (keep - 1) * blockSize);
                Arrays.fill(last.min, Float.POSITIVE_INFINITY);
                Arrays.fill(last.max, Float.NEGATIVE_INFINITY);
                for (int t = 0; t < last.filled; t++) {
                    bound(last, last.keys, t * width);
                }
            }
        }
        s.length = length;
    }

    @Override
    public boolean canAppend(int sequence, int steps) {
        return sequences.containsKey(sequence);
    }

    @Override
    public int append(int sequence, int steps) {
        Sequence s = get(sequence);
        int from = s.length;
        int blocks = (s.length + steps + blockSize - 1) / blockSize;
        for (List<Block> layer : s.layers) {
            while (layer.size() < blocks) {
                layer.add(new Block());
            }
        }
        s.length += steps;
        return from;
    }

    @Override
    public void write(int sequence, int layer, int from, Tensor k, Tensor v) {
        Sequence s = get(sequence);
        float[] ka = k.toArray();
        float[] va = v.toArray();
        int steps = ka.length / width;
        List<Block> blocks = s.layers.get(layer);
        for (int t = 0; t < steps; t++) {
            int slot = from + t;
            Block b = blocks.get(slot / blockSize);
            int at = (slot % blockSize) * width;
            System.arraycopy(ka, t * width, b.keys, at, width);
            System.arraycopy(va, t * width, b.values, at, width);
            b.filled = Math.max(b.filled, slot % blockSize + 1);
            bound(b, ka, t * width);
            sketch(b, slot % blockSize, ka, t * width);
        }
    }

    /** Symmetric 4-bit levels of one token's keys, one scale per key/value head. */
    private void sketch(Block b, int row, float[] keys, int offset) {
        for (int h = 0; h < kvHeads; h++) {
            float amax = 0;
            for (int c = 0; c < headDim; c++) {
                amax = Math.max(amax, Math.abs(keys[offset + h * headDim + c]));
            }
            float scale = amax == 0 ? 1 : amax / 7;
            b.sketchScale[row * kvHeads + h] = scale;
            for (int c = 0; c < headDim; c++) {
                b.sketch[row * width + h * headDim + c] = (byte) Math.round(keys[offset + h * headDim + c] / scale);
            }
        }
    }

    private static void bound(Block b, float[] keys, int offset) {
        for (int c = 0; c < b.min.length; c++) {
            b.min[c] = Math.min(b.min[c], keys[offset + c]);
            b.max[c] = Math.max(b.max[c], keys[offset + c]);
        }
    }

    @Override
    public View view(int sequence, int layer, Tensor queries) {
        Sequence s = get(sequence);
        List<Block> blocks = s.layers.get(layer);
        int count = blocks.size();
        int heads = queries.shape().dim(0);
        int steps = queries.shape().dim(1);
        boolean[] keep = new boolean[count];
        if (steps > 1 || count <= hot) {
            Arrays.fill(keep, true);
        } else {
            float[] q = queries.toArray();
            int group = heads / kvHeads;
            int candidates = count - recent;
            // each head ranks the blocks by its own score; the heads then take turns
            Integer[][] order = new Integer[heads][candidates];
            for (int h = 0; h < heads; h++) {
                int kh = h / group;
                double[] score = new double[candidates];
                for (int i = 0; i < candidates; i++) {
                    score[i] = score(blocks.get(i), q, h * headDim, kh * headDim);
                    order[h][i] = i;
                }
                Arrays.sort(order[h], (a, b) -> Double.compare(score[b], score[a]));
            }
            int chosen = 0;
            for (int rank = 0; rank < candidates && chosen < hot - recent; rank++) {
                for (int h = 0; h < heads && chosen < hot - recent; h++) {
                    if (!keep[order[h][rank]]) {
                        keep[order[h][rank]] = true;
                        chosen++;
                    }
                }
            }
            for (int i = candidates; i < count; i++) {
                keep[i] = true;
            }
        }
        int slots = 0;
        for (int i = 0; i < count; i++) {
            Block b = blocks.get(i);
            if (keep[i]) {
                if (!b.resident) {
                    transferred += bytesPerLayerBlock();
                    fetches++;
                }
                slots += b.filled;
            }
            // full blocks outside the hot set live on the host; the tail stays on the device
            b.resident = keep[i] && (steps == 1 || i >= count - recent);
        }
        float[] k = new float[slots * width];
        float[] v = new float[slots * width];
        int[] index = new int[slots];
        int at = 0;
        for (int i = 0; i < count; i++) {
            if (keep[i]) {
                Block b = blocks.get(i);
                System.arraycopy(b.keys, 0, k, at * width, b.filled * width);
                System.arraycopy(b.values, 0, v, at * width, b.filled * width);
                for (int t = 0; t < b.filled; t++) {
                    index[at + t] = i * blockSize + t;
                }
                at += b.filled;
            }
        }
        peak = Math.max(peak, residentBytes());
        Shape shape = Shape.of(slots, kvHeads, headDim);
        return new View(Tensor.of(shape, k, device), Tensor.of(shape, v, device), index);
    }

    /** A head's score for a block, by the selection rule. */
    private double score(Block b, float[] q, int qOffset, int kOffset) {
        if (selection == Selection.BOUNDS) {
            double sum = 0;
            for (int c = 0; c < headDim; c++) {
                float qc = q[qOffset + c];
                sum += Math.max(qc * b.min[kOffset + c], qc * b.max[kOffset + c]);
            }
            return sum;
        }
        int kh = kOffset / headDim;
        double best = Double.NEGATIVE_INFINITY;
        for (int t = 0; t < b.filled; t++) {
            double dot = 0;
            for (int c = 0; c < headDim; c++) {
                dot += q[qOffset + c] * (selection == Selection.ORACLE ? b.keys[t * width + kOffset + c]
                        : b.sketch[t * width + kOffset + c] * b.sketchScale[t * kvHeads + kh]);
            }
            best = Math.max(best, dot);
        }
        return best;
    }

    /**
     * Sets how blocks are scored; {@link Selection#SKETCH} by default.
     *
     * @param rule the rule
     * @return this store
     */
    public OffloadingKvCache selection(Selection rule) {
        this.selection = java.util.Objects.requireNonNull(rule, "rule");
        return this;
    }

    @Override
    public long residentBytes() {
        long blocks = 0;
        long tokens = 0;
        for (Sequence s : sequences.values()) {
            for (List<Block> layer : s.layers) {
                for (Block b : layer) {
                    blocks += b.resident ? 1 : 0;
                    tokens += b.filled;
                }
            }
        }
        long sketches = selection == Selection.SKETCH ? tokens * (width / 2 + (long) kvHeads * Float.BYTES) : 0;
        return blocks * bytesPerLayerBlock() + sketches;
    }

    @Override
    public long peakResidentBytes() {
        return peak;
    }

    private Sequence get(int sequence) {
        Sequence s = sequences.get(sequence);
        if (s == null) {
            throw new IllegalArgumentException("no sequence " + sequence);
        }
        return s;
    }

    @Override
    public String toString() {
        return "OffloadingKvCache[hot " + hot + " blocks of " + blockSize + ", " + sequences.size() + " sequences]";
    }
}
