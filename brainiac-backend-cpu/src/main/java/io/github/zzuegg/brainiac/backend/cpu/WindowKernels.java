package io.github.zzuegg.brainiac.backend.cpu;

import io.github.zzuegg.brainiac.tensor.ConvGeometry;

/** Sliding-window, batched-matmul and indexing kernels for any number of dimensions. */
final class WindowKernels {

    private WindowKernels() {
    }

    /** Precomputed index tables for one geometry. */
    static final class Plan {
        final int channels;
        final int kernelSize;
        final int positions;
        final int imageSize;
        final int rank;
        final int[] image;
        /** per kernel cell and dimension: the offset {@code i * dilation} */
        final int[][] kernelOffset;
        /** per position and dimension: the first image cell {@code p * stride - padBefore} */
        final int[][] origin;
        final long[] imageStrides;

        Plan(ConvGeometry g) {
            channels = g.channels();
            kernelSize = g.kernelSize();
            positions = g.positionCount();
            imageSize = g.imageSize();
            image = g.image();
            rank = image.length;
            int[] kernel = g.kernel();
            int[] dilation = g.dilation();
            int[] stride = g.stride();
            int[] pad = g.padBefore();
            int[] pos = g.positions();
            kernelOffset = new int[kernelSize][rank];
            for (int k = 0; k < kernelSize; k++) {
                int rest = k;
                for (int d = rank - 1; d >= 0; d--) {
                    kernelOffset[k][d] = (rest % kernel[d]) * dilation[d];
                    rest /= kernel[d];
                }
            }
            origin = new int[positions][rank];
            for (int p = 0; p < positions; p++) {
                int rest = p;
                for (int d = rank - 1; d >= 0; d--) {
                    origin[p][d] = (rest % pos[d]) * stride[d] - pad[d];
                    rest /= pos[d];
                }
            }
            imageStrides = new long[rank];
            long step = 1;
            for (int d = rank - 1; d >= 0; d--) {
                imageStrides[d] = step;
                step *= image[d];
            }
        }

        /** @return the image cell index of kernel cell {@code k} at position {@code p}, or -1 for padding */
        int cell(int p, int k) {
            int[] o = origin[p];
            int[] off = kernelOffset[k];
            int idx = 0;
            for (int d = 0; d < rank; d++) {
                int c = o[d] + off[d];
                if (c < 0 || c >= image[d]) {
                    return -1;
                }
                idx += (int) (c * imageStrides[d]);
            }
            return idx;
        }

        /** @return the image cells of every kernel cell at every position, -1 for padding */
        int[] table() {
            int[] t = new int[positions * kernelSize];
            for (int p = 0; p < positions; p++) {
                for (int k = 0; k < kernelSize; k++) {
                    t[p * kernelSize + k] = cell(p, k);
                }
            }
            return t;
        }
    }

    static void im2col(float[] x, Plan plan, int[] table, float pad, float[] out, int rowFrom, int rowTo) {
        int c = plan.channels;
        int ks = plan.kernelSize;
        int width = c * ks;
        int img = plan.imageSize;
        for (int row = rowFrom; row < rowTo; row++) {
            int n = row / plan.positions;
            int p = row % plan.positions;
            int outBase = row * width;
            int tableBase = p * ks;
            for (int ch = 0; ch < c; ch++) {
                int inBase = (n * c + ch) * img;
                int o = outBase + ch * ks;
                for (int k = 0; k < ks; k++) {
                    int cell = table[tableBase + k];
                    out[o + k] = cell < 0 ? pad : x[inBase + cell];
                }
            }
        }
    }

    static void col2im(float[] cols, Plan plan, int[] table, float[] out, int planeFrom, int planeTo) {
        int c = plan.channels;
        int ks = plan.kernelSize;
        int width = c * ks;
        int img = plan.imageSize;
        for (int plane = planeFrom; plane < planeTo; plane++) {
            int n = plane / c;
            int ch = plane % c;
            int outBase = plane * img;
            java.util.Arrays.fill(out, outBase, outBase + img, 0f);
            for (int p = 0; p < plan.positions; p++) {
                int src = (n * plan.positions + p) * width + ch * ks;
                int tableBase = p * ks;
                for (int k = 0; k < ks; k++) {
                    int cell = table[tableBase + k];
                    if (cell >= 0) {
                        out[outBase + cell] += cols[src + k];
                    }
                }
            }
        }
    }

    static void matmulBatched(float[] a, float[] b, float[] z, int m, int k, int n, int rowFrom, int rowTo) {
        for (int row = rowFrom; row < rowTo; row++) {
            int batch = row / m;
            int i = row % m;
            int aBase = (batch * m + i) * k;
            int bBase = batch * k * n;
            int zBase = row * n;
            java.util.Arrays.fill(z, zBase, zBase + n, 0f);
            for (int kk = 0; kk < k; kk++) {
                float av = a[aBase + kk];
                if (av == 0f) {
                    continue;
                }
                int bRow = bBase + kk * n;
                for (int j = 0; j < n; j++) {
                    z[zBase + j] += av * b[bRow + j];
                }
            }
        }
    }

    static int index(float v, int length) {
        int i = (int) v;
        if (i != v || i < 0 || i >= length) {
            throw new IndexOutOfBoundsException("index " + v + " out of range [0, " + length + ")");
        }
        return i;
    }

    static void gather(float[] x, int length, int inner, int[] idx, float[] out, int from, int to) {
        int count = idx.length;
        for (int r = from; r < to; r++) {
            int o = r / count;
            int i = r % count;
            System.arraycopy(x, (o * length + idx[i]) * inner, out, r * inner, inner);
        }
    }

    /** Scatters columns {@code [colFrom, colTo)} of every inner block; no two calls touch the same output. */
    static void scatterAdd(float[] src, int outer, int length, int inner, int[] idx, float[] out, int colFrom,
            int colTo) {
        int count = idx.length;
        for (int o = 0; o < outer; o++) {
            for (int l = 0; l < length; l++) {
                java.util.Arrays.fill(out, (o * length + l) * inner + colFrom, (o * length + l) * inner + colTo, 0f);
            }
            for (int i = 0; i < count; i++) {
                int s = (o * count + i) * inner;
                int d = (o * length + idx[i]) * inner;
                for (int j = colFrom; j < colTo; j++) {
                    out[d + j] += src[s + j];
                }
            }
        }
    }
}
