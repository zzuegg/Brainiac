package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.llm.LlamaConfig;
import io.github.zzuegg.brainiac.tensor.DType;
import io.github.zzuegg.brainiac.tensor.Device;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A paged key/value cache, as vLLM's PagedAttention keeps it: one fixed pool of blocks of
 * {@code blockSize} slots per layer, and per sequence a block table that maps its slots to
 * blocks. A sequence takes a new block only when its last one is full, so at most
 * {@code blockSize - 1} slots per sequence go unused, where a contiguous buffer that grows by
 * doubling (as the core {@code Llama} cache does) wastes up to half of it and must copy
 * everything when it grows. More sequences fit in the same memory, and nothing is copied.
 *
 * <p>Blocks are reference-counted: {@link #fork} shares every block of a sequence, and a
 * shared block is copied only when one of its owners writes into it (copy on write). That
 * makes beam search and {@link PrefixCache prefix caching} cost no memory for the shared part.
 *
 * <p>The pool is a float32 tensor {@code [blocks * blockSize, kvHeads, headDim]} per layer on
 * any device; writes go in with {@link Tensor#write} and reads come out with one
 * {@link Tensor#gather} per layer, so the same code runs on every backend.
 *
 * <p>Kwon et al., "Efficient Memory Management for Large Language Model Serving with
 * PagedAttention" (SOSP 2023), <a href="https://arxiv.org/abs/2309.06180">arXiv:2309.06180</a>.
 */
public final class PagedKvCache implements KvBlockStore {

    private static final class Sequence {
        int[] blocks = new int[4];
        int count;
        int length;

        void add(int block) {
            if (count == blocks.length) {
                blocks = Arrays.copyOf(blocks, 2 * count);
            }
            blocks[count++] = block;
        }
    }

    private final int layers;
    private final int kvHeads;
    private final int headDim;
    private final int blockSize;
    private final int blocks;
    private final Device device;
    private final Tensor[] keys;
    private final Tensor[] values;
    private final int[] refs;
    private final int[] freeList;
    private int free;
    private int peakUsed;
    private final Map<Integer, Sequence> sequences = new HashMap<>();
    private int nextId;

    /**
     * Creates the pool.
     *
     * @param layers the decoder layers
     * @param kvHeads the key/value heads per layer
     * @param headDim the size of each head
     * @param blockSize the slots per block, 16 in vLLM
     * @param blocks the blocks in the pool
     * @param device where the pool lives
     */
    public PagedKvCache(int layers, int kvHeads, int headDim, int blockSize, int blocks, Device device) {
        if (layers < 1 || kvHeads < 1 || headDim < 1 || blockSize < 1 || blocks < 1) {
            throw new IllegalArgumentException("every size must be positive");
        }
        this.layers = layers;
        this.kvHeads = kvHeads;
        this.headDim = headDim;
        this.blockSize = blockSize;
        this.blocks = blocks;
        this.device = device;
        this.keys = new Tensor[layers];
        this.values = new Tensor[layers];
        Shape shape = Shape.of(blocks * blockSize, kvHeads, headDim);
        for (int l = 0; l < layers; l++) {
            keys[l] = Tensor.zeros(shape, DType.FLOAT32, device);
            values[l] = Tensor.zeros(shape, DType.FLOAT32, device);
            Tensor.evaluate(List.of(keys[l], values[l]));
        }
        this.refs = new int[blocks];
        this.freeList = new int[blocks];
        for (int b = 0; b < blocks; b++) {
            freeList[b] = blocks - 1 - b;   // hand out block 0 first
        }
        this.free = blocks;
    }

    /**
     * A pool shaped for a model.
     *
     * @param config the model's shape
     * @param blockSize the slots per block
     * @param blocks the blocks in the pool
     * @param device where the pool lives
     * @return the pool
     */
    public static PagedKvCache forModel(LlamaConfig config, int blockSize, int blocks, Device device) {
        return new PagedKvCache(config.layers(), config.kvHeads(), config.headDim(), blockSize, blocks, device);
    }

    @Override
    public int layers() {
        return layers;
    }

    /** @return the slots per block */
    public int blockSize() {
        return blockSize;
    }

    /** @return the blocks in the pool */
    public int capacity() {
        return blocks;
    }

    /** @return the blocks not in use */
    public int freeBlocks() {
        return free;
    }

    /** @return the blocks in use */
    public int usedBlocks() {
        return blocks - free;
    }

    /** @return the most blocks ever in use at once */
    public int peakBlocks() {
        return peakUsed;
    }

    /** @return the bytes of one block over all layers, keys and values */
    public long bytesPerBlock() {
        return 2L * layers * blockSize * kvHeads * headDim * Float.BYTES;
    }

    /**
     * @param sequence the sequence
     * @return a copy of its block table
     */
    public int[] blockTable(int sequence) {
        Sequence s = get(sequence);
        return Arrays.copyOf(s.blocks, s.count);
    }

    /**
     * @param block a block
     * @return how many sequences share it
     */
    public int references(int block) {
        return refs[block];
    }

    @Override
    public int newSequence() {
        int id = nextId++;
        sequences.put(id, new Sequence());
        return id;
    }

    @Override
    public int fork(int sequence) {
        Sequence s = get(sequence);
        int id = nextId++;
        Sequence c = new Sequence();
        c.blocks = Arrays.copyOf(s.blocks, Math.max(4, s.blocks.length));
        c.count = s.count;
        c.length = s.length;
        for (int i = 0; i < s.count; i++) {
            refs[s.blocks[i]]++;
        }
        sequences.put(id, c);
        return id;
    }

    @Override
    public void free(int sequence) {
        Sequence s = sequences.remove(sequence);
        if (s == null) {
            throw new IllegalArgumentException("no sequence " + sequence);
        }
        for (int i = 0; i < s.count; i++) {
            release(s.blocks[i]);
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
            throw new IllegalArgumentException("cannot keep " + length + " of " + s.length + " slots");
        }
        int keep = (length + blockSize - 1) / blockSize;
        for (int i = keep; i < s.count; i++) {
            release(s.blocks[i]);
        }
        s.count = keep;
        s.length = length;
    }

    /**
     * @param sequence the sequence
     * @param steps new slots wanted
     * @return the free blocks {@link #append} would take, a copied shared block included
     */
    public int blocksNeeded(int sequence, int steps) {
        Sequence s = get(sequence);
        int need = (s.length + steps + blockSize - 1) / blockSize - s.count;
        if (steps > 0 && s.length % blockSize != 0 && refs[s.blocks[s.count - 1]] > 1) {
            need++;   // the shared partial block is copied before it is written
        }
        return need;
    }

    @Override
    public boolean canAppend(int sequence, int steps) {
        return blocksNeeded(sequence, steps) <= free;
    }

    @Override
    public int append(int sequence, int steps) {
        if (!canAppend(sequence, steps)) {
            throw new IllegalStateException("the pool has " + free + " free blocks, " + steps + " more slots need "
                    + blocksNeeded(sequence, steps));
        }
        Sequence s = get(sequence);
        int from = s.length;
        if (steps > 0 && s.length % blockSize != 0 && refs[s.blocks[s.count - 1]] > 1) {
            int shared = s.blocks[s.count - 1];
            int copy = take();
            for (int l = 0; l < layers; l++) {
                Tensor k = keys[l].slice(0, shared * blockSize, (shared + 1) * blockSize);
                Tensor v = values[l].slice(0, shared * blockSize, (shared + 1) * blockSize);
                Tensor.evaluate(List.of(k, v));   // read before the pool is written
                keys[l].write(0, copy * blockSize, k);
                values[l].write(0, copy * blockSize, v);
            }
            release(shared);
            s.blocks[s.count - 1] = copy;
        }
        while (s.count * blockSize < s.length + steps) {
            s.add(take());
        }
        s.length += steps;
        return from;
    }

    @Override
    public void write(int sequence, int layer, int from, Tensor k, Tensor v) {
        Sequence s = get(sequence);
        int steps = k.shape().dim(0);
        if (from < 0 || from + steps > s.length) {
            throw new IllegalArgumentException("slots " + from + ".." + (from + steps) + " are not appended");
        }
        int i = 0;
        while (i < steps) {
            int slot = from + i;
            int run = Math.min(steps - i, blockSize - slot % blockSize);
            int physical = s.blocks[slot / blockSize] * blockSize + slot % blockSize;
            keys[layer].write(0, physical, run == steps ? k : k.slice(0, i, i + run));
            values[layer].write(0, physical, run == steps ? v : v.slice(0, i, i + run));
            i += run;
        }
    }

    @Override
    public View view(int sequence, int layer, Tensor queries) {
        Sequence s = get(sequence);
        int n = s.length;
        float[] idx = new float[n];
        int[] slots = new int[n];
        for (int t = 0; t < n; t++) {
            idx[t] = s.blocks[t / blockSize] * blockSize + t % blockSize;
            slots[t] = t;
        }
        Tensor index = Tensor.of(Shape.of(n), idx, device);
        return new View(keys[layer].gather(0, index), values[layer].gather(0, index), slots);
    }

    @Override
    public long residentBytes() {
        return usedBlocks() * bytesPerBlock();
    }

    @Override
    public long peakResidentBytes() {
        return peakUsed * bytesPerBlock();
    }

    private Sequence get(int sequence) {
        Sequence s = sequences.get(sequence);
        if (s == null) {
            throw new IllegalArgumentException("no sequence " + sequence);
        }
        return s;
    }

    private int take() {
        int b = freeList[--free];
        refs[b] = 1;
        peakUsed = Math.max(peakUsed, blocks - free);
        return b;
    }

    private void release(int block) {
        if (--refs[block] == 0) {
            freeList[free++] = block;
        }
    }

    @Override
    public String toString() {
        return "PagedKvCache[" + usedBlocks() + "/" + blocks + " blocks of " + blockSize + ", " + sequences.size()
                + " sequences]";
    }
}
