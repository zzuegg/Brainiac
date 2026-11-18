package io.github.zzuegg.brainiac.experimental.kernels;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Block sparse row (BSR) storage, as in OpenAI's block-sparse kernels (Gray, Radford and
 * Kingma 2017): the weight {@code W [in, out]} is cut into {@code block x block} tiles and only
 * the tiles with a non-zero are stored, densely. {@code y = x W} runs a small dense
 * micro-kernel per stored tile, so the inner loops are contiguous and unit-stride, as in a
 * dense kernel. {@link #prune} makes such a weight by block L2 norm.
 */
public final class BlockSparseMatmul {

    private final int in;
    private final int out;
    private final int block;
    private final int[] rowStart;
    private final int[] blockColumns;
    private final float[] blocks;

    /**
     * @param weight {@code [in, out]}, both multiples of {@code block}; all-zero tiles dropped
     * @param block the tile side
     */
    public BlockSparseMatmul(Tensor weight, int block) {
        this.in = weight.shape().dim(0);
        this.out = weight.shape().dim(1);
        this.block = block;
        if (in % block != 0 || out % block != 0) {
            throw new IllegalArgumentException("block " + block + " must divide " + in + "x" + out);
        }
        float[] w = weight.toArray();
        int rows = in / block;
        int cols = out / block;
        boolean[] keep = new boolean[rows * cols];
        int count = 0;
        for (int br = 0; br < rows; br++) {
            for (int bc = 0; bc < cols; bc++) {
                boolean nz = false;
                for (int i = 0; i < block && !nz; i++) {
                    for (int j = 0; j < block; j++) {
                        if (w[(br * block + i) * out + bc * block + j] != 0) {
                            nz = true;
                            break;
                        }
                    }
                }
                keep[br * cols + bc] = nz;
                count += nz ? 1 : 0;
            }
        }
        rowStart = new int[rows + 1];
        blockColumns = new int[count];
        blocks = new float[count * block * block];
        int p = 0;
        for (int br = 0; br < rows; br++) {
            rowStart[br] = p;
            for (int bc = 0; bc < cols; bc++) {
                if (!keep[br * cols + bc]) {
                    continue;
                }
                blockColumns[p] = bc;
                for (int i = 0; i < block; i++) {
                    System.arraycopy(w, (br * block + i) * out + bc * block, blocks, (p * block + i) * block, block);
                }
                p++;
            }
        }
        rowStart[rows] = p;
    }

    /**
     * Zeros the blocks with the smallest L2 norm.
     *
     * @param weight {@code [in, out]}
     * @param block the tile side
     * @param sparsity the share of blocks to zero
     * @return the pruned weight
     */
    public static Tensor prune(Tensor weight, int block, double sparsity) {
        int in = weight.shape().dim(0);
        int out = weight.shape().dim(1);
        float[] w = weight.toArray();
        int rows = in / block;
        int cols = out / block;
        double[] norm = new double[rows * cols];
        for (int i = 0; i < in; i++) {
            for (int j = 0; j < out; j++) {
                norm[(i / block) * cols + j / block] += (double) w[i * out + j] * w[i * out + j];
            }
        }
        double[] sorted = norm.clone();
        java.util.Arrays.sort(sorted);
        int drop = (int) Math.round(sparsity * norm.length);
        double threshold = drop == 0 ? Double.NEGATIVE_INFINITY : sorted[drop - 1];
        float[] o = w.clone();
        int dropped = 0;
        for (int bIdx = 0; bIdx < norm.length && drop > 0; bIdx++) {
            if (norm[bIdx] <= threshold && dropped < drop) {
                dropped++;
                int br = bIdx / cols;
                int bc = bIdx % cols;
                for (int i = 0; i < block; i++) {
                    java.util.Arrays.fill(o, (br * block + i) * out + bc * block, (br * block + i) * out + bc * block + block,
                            0f);
                }
            }
        }
        return Tensor.of(Shape.of(in, out), o);
    }

    /**
     * @param x {@code [b, in]}
     * @return {@code x W}, {@code [b, out]}
     */
    public Tensor multiply(Tensor x) {
        int b = x.shape().dim(0);
        float[] xv = x.toArray();
        float[] y = new float[b * out];
        int rows = in / block;
        for (int r = 0; r < b; r++) {
            int xo = r * in;
            int yo = r * out;
            for (int br = 0; br < rows; br++) {
                for (int p = rowStart[br]; p < rowStart[br + 1]; p++) {
                    int yb = yo + blockColumns[p] * block;
                    int base = p * block * block;
                    for (int i = 0; i < block; i++) {
                        float a = xv[xo + br * block + i];
                        int wo = base + i * block;
                        for (int j = 0; j < block; j++) {
                            y[yb + j] += a * blocks[wo + j];
                        }
                    }
                }
            }
        }
        return Tensor.of(Shape.of(b, out), y);
    }

    /** @return the stored blocks */
    public int storedBlocks() {
        return blockColumns.length;
    }

    /** @return bytes stored: block values plus one column index per block and the row pointers */
    public long bytes() {
        return 4L * (blocks.length + blockColumns.length + rowStart.length);
    }

    @Override
    public String toString() {
        return "BlockSparseMatmul(" + in + "x" + out + ", " + block + "x" + block + ", " + blockColumns.length
                + " blocks)";
    }
}
