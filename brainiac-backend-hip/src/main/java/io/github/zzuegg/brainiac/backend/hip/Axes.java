package io.github.zzuegg.brainiac.backend.hip;

import java.util.Arrays;

/**
 * A shape with one stride array per operand, with size-1 axes dropped and adjacent axes merged
 * where every operand allows it. Ranks above {@link #MAX_AXES} run as several launches over the
 * outer axes.
 */
final class Axes {

    /** Axes one strided launch handles (the kernels' {@code MAX_AXES}). */
    static final int MAX_AXES = 8;

    final int[] shape;
    final long[][] strides;

    private Axes(int[] shape, long[][] strides) {
        this.shape = shape;
        this.strides = strides;
    }

    /** Merges axes for the operands' strides, all over {@code shape}. */
    static Axes merge(int[] shape, long[]... in) {
        int ops = in.length;
        int[] dims = new int[shape.length];
        long[][] st = new long[ops][shape.length];
        int r = 0;
        for (int d = 0; d < shape.length; d++) {
            if (shape[d] == 1) {
                continue;
            }
            boolean mergeable = r > 0;
            for (int o = 0; o < ops && mergeable; o++) {
                mergeable = st[o][r - 1] == in[o][d] * shape[d];
            }
            if (mergeable) {
                dims[r - 1] *= shape[d];
                for (int o = 0; o < ops; o++) {
                    st[o][r - 1] = in[o][d];
                }
            } else {
                dims[r] = shape[d];
                for (int o = 0; o < ops; o++) {
                    st[o][r] = in[o][d];
                }
                r++;
            }
        }
        if (r == 0) {
            dims = new int[] {1};
            for (int o = 0; o < ops; o++) {
                st[o] = new long[] {1};
            }
            r = 1;
        }
        long[][] out = new long[ops][];
        for (int o = 0; o < ops; o++) {
            out[o] = Arrays.copyOf(st[o], r);
        }
        return new Axes(Arrays.copyOf(dims, r), out);
    }

    int rank() {
        return shape.length;
    }

    /** Strides in row-major order for a contiguous array of this shape. */
    static long[] contiguous(int[] shape) {
        long[] s = new long[shape.length];
        long step = 1;
        for (int i = shape.length - 1; i >= 0; i--) {
            s[i] = step;
            step *= shape[i];
        }
        return s;
    }

    interface Block {
        /**
         * One launch over the inner axes.
         *
         * @param offsets each operand's offset for this index of the outer axes
         * @param shape the inner axes
         * @param strides each operand's strides over them
         * @param count the elements in the block
         */
        void run(long[] offsets, long[] shape, long[][] strides, long count);
    }

    /** Calls {@code block} once per index of the axes beyond the innermost {@link #MAX_AXES}. */
    void forEachBlock(Block block) {
        int rank = shape.length;
        int innerRank = Math.min(rank, MAX_AXES);
        int outer = rank - innerRank;
        long count = 1;
        long[] innerShape = new long[innerRank];
        for (int d = outer; d < rank; d++) {
            count *= shape[d];
            innerShape[d - outer] = shape[d];
        }
        long[][] innerStrides = new long[strides.length][];
        for (int o = 0; o < strides.length; o++) {
            innerStrides[o] = Arrays.copyOfRange(strides[o], outer, rank);
        }
        long blocks = 1;
        for (int d = 0; d < outer; d++) {
            blocks *= shape[d];
        }
        long[] offsets = new long[strides.length];
        for (long i = 0; i < blocks; i++) {
            long rest = i;
            Arrays.fill(offsets, 0);
            for (int d = outer - 1; d >= 0; d--) {
                long idx = rest % shape[d];
                rest /= shape[d];
                for (int o = 0; o < strides.length; o++) {
                    offsets[o] += idx * strides[o][d];
                }
            }
            block.run(offsets, innerShape, innerStrides, count);
        }
    }
}
