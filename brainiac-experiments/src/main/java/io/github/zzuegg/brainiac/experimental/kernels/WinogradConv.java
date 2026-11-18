package io.github.zzuegg.brainiac.experimental.kernels;

import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

/**
 * Winograd minimal filtering for 3x3, stride-1, pad-1 convolutions (Lavin and Gray 2016, "Fast
 * Algorithms for Convolutional Neural Networks",
 * <a href="https://arxiv.org/abs/1509.09308">arXiv:1509.09308</a>).
 *
 * <p>{@code F(m x m, 3 x 3)} computes an {@code m x m} output tile from an
 * {@code (m + 2) x (m + 2)} input tile as {@code A^T [(G g G^T) . (B^T d B)] A}. Batched over
 * channels and tiles, the element-wise product becomes {@code (m + 2)^2} matmuls
 * {@code [filters, channels] x [channels, tiles]}, run by the core {@link Tensor#matmul}.
 * {@code F(2, 3)} uses 16 instead of 36 multiplications per tile and channel pair, and
 * {@code F(4, 3)} uses 36 instead of 144.
 *
 * <p>With {@code bits > 0}, both transformed operands are fake-quantised symmetrically per
 * transform position before the products, as an integer Winograd kernel would store them.
 * {@code F(4, 3)}'s larger transform coefficients widen the range, so it loses more.
 */
public final class WinogradConv {

    private static final float[][] BT2 = {{1, 0, -1, 0}, {0, 1, 1, 0}, {0, -1, 1, 0}, {0, 1, 0, -1}};
    private static final float[][] G2 = {{1, 0, 0}, {0.5f, 0.5f, 0.5f}, {0.5f, -0.5f, 0.5f}, {0, 0, 1}};
    private static final float[][] AT2 = {{1, 1, 1, 0}, {0, 1, -1, -1}};

    private static final float[][] BT4 = {
        {4, 0, -5, 0, 1, 0},
        {0, -4, -4, 1, 1, 0},
        {0, 4, -4, -1, 1, 0},
        {0, -2, -1, 2, 1, 0},
        {0, 2, -1, -2, 1, 0},
        {0, 4, 0, -5, 0, 1}};
    private static final float[][] G4 = {
        {1 / 4f, 0, 0},
        {-1 / 6f, -1 / 6f, -1 / 6f},
        {-1 / 6f, 1 / 6f, -1 / 6f},
        {1 / 24f, 1 / 12f, 1 / 6f},
        {1 / 24f, -1 / 12f, 1 / 6f},
        {0, 0, 1}};
    private static final float[][] AT4 = {{1, 1, 1, 1, 1, 0}, {0, 1, -1, 2, -2, 0}, {0, 1, 1, 4, 4, 0},
        {0, 1, -1, 8, -8, 1}};

    private final int m;
    private final int alpha;
    private final int bits;
    private final float[][] bt;
    private final float[][] g;
    private final float[][] at;

    /**
     * @param tile the output tile side, 2 or 4
     * @param bits the fake-quantisation bits of the transformed operands, 0 for float
     */
    public WinogradConv(int tile, int bits) {
        if (tile != 2 && tile != 4) {
            throw new IllegalArgumentException("tile must be 2 or 4: " + tile);
        }
        this.m = tile;
        this.alpha = tile + 2;
        this.bits = bits;
        this.bt = tile == 2 ? BT2 : BT4;
        this.g = tile == 2 ? G2 : G4;
        this.at = tile == 2 ? AT2 : AT4;
    }

    /**
     * Transforms filters once: {@code U[p][f][c] = (G g G^T)[p]}.
     *
     * @param w {@code [f, c, 3, 3]}
     * @return {@code [alpha^2][f * c]}
     */
    public float[][] transformFilters(Tensor w) {
        int f = w.shape().dim(0);
        int c = w.shape().dim(1);
        float[] wv = w.toArray();
        float[][] u = new float[alpha * alpha][f * c];
        float[] tmp = new float[alpha * 3];
        for (int fc = 0; fc < f * c; fc++) {
            int o = fc * 9;
            for (int i = 0; i < alpha; i++) {
                for (int j = 0; j < 3; j++) {
                    float s = 0;
                    for (int t = 0; t < 3; t++) {
                        s += g[i][t] * wv[o + t * 3 + j];
                    }
                    tmp[i * 3 + j] = s;
                }
            }
            for (int i = 0; i < alpha; i++) {
                for (int j = 0; j < alpha; j++) {
                    float s = 0;
                    for (int t = 0; t < 3; t++) {
                        s += tmp[i * 3 + t] * g[j][t];
                    }
                    u[i * alpha + j][fc] = s;
                }
            }
        }
        return u;
    }

    /**
     * Convolves.
     *
     * @param x {@code [b, c, h, w]}
     * @param w {@code [f, c, 3, 3]}
     * @return {@code [b, f, h, w]} (stride 1, pad 1, no bias)
     */
    public Tensor forward(Tensor x, Tensor w) {
        return forward(x, w.shape().dim(0), transformFilters(w));
    }

    /**
     * Convolves with pre-transformed filters.
     *
     * @param x {@code [b, c, h, w]}
     * @param f the filters
     * @param u {@link #transformFilters}
     * @return {@code [b, f, h, w]}
     */
    public Tensor forward(Tensor x, int f, float[][] u) {
        int b = x.shape().dim(0);
        int c = x.shape().dim(1);
        int h = x.shape().dim(2);
        int wd = x.shape().dim(3);
        int th = (h + m - 1) / m;
        int tw = (wd + m - 1) / m;
        int tiles = b * th * tw;
        float[] xv = x.toArray();
        int a2 = alpha * alpha;
        float[][] v = new float[a2][c * tiles];
        float[] d = new float[a2];
        float[] tmp = new float[a2];
        for (int bi = 0; bi < b; bi++) {
            for (int ci = 0; ci < c; ci++) {
                int plane = (bi * c + ci) * h * wd;
                for (int ty = 0; ty < th; ty++) {
                    for (int tx = 0; tx < tw; tx++) {
                        int y0 = ty * m - 1;
                        int x0 = tx * m - 1;
                        for (int i = 0; i < alpha; i++) {
                            int yy = y0 + i;
                            for (int j = 0; j < alpha; j++) {
                                int xx = x0 + j;
                                d[i * alpha + j] = yy >= 0 && yy < h && xx >= 0 && xx < wd ? xv[plane + yy * wd + xx] : 0;
                            }
                        }
                        // B^T d B
                        for (int i = 0; i < alpha; i++) {
                            for (int j = 0; j < alpha; j++) {
                                float s = 0;
                                for (int t = 0; t < alpha; t++) {
                                    s += bt[i][t] * d[t * alpha + j];
                                }
                                tmp[i * alpha + j] = s;
                            }
                        }
                        int tile = (bi * th + ty) * tw + tx;
                        for (int i = 0; i < alpha; i++) {
                            for (int j = 0; j < alpha; j++) {
                                float s = 0;
                                for (int t = 0; t < alpha; t++) {
                                    s += tmp[i * alpha + t] * bt[j][t];
                                }
                                v[i * alpha + j][ci * tiles + tile] = s;
                            }
                        }
                    }
                }
            }
        }
        float[][] prod = new float[a2][];
        for (int p = 0; p < a2; p++) {
            float[] up = bits > 0 ? fakeQuant(u[p]) : u[p];
            float[] vp = bits > 0 ? fakeQuant(v[p]) : v[p];
            prod[p] = Tensor.of(Shape.of(f, c), up).matmul(Tensor.of(Shape.of(c, tiles), vp)).toArray();
        }
        float[] out = new float[b * f * h * wd];
        float[] mt = new float[a2];
        float[] half = new float[m * alpha];
        for (int fi = 0; fi < f; fi++) {
            for (int bi = 0; bi < b; bi++) {
                for (int ty = 0; ty < th; ty++) {
                    for (int tx = 0; tx < tw; tx++) {
                        int tile = (bi * th + ty) * tw + tx;
                        for (int p = 0; p < a2; p++) {
                            mt[p] = prod[p][fi * tiles + tile];
                        }
                        for (int i = 0; i < m; i++) {
                            for (int j = 0; j < alpha; j++) {
                                float s = 0;
                                for (int t = 0; t < alpha; t++) {
                                    s += at[i][t] * mt[t * alpha + j];
                                }
                                half[i * alpha + j] = s;
                            }
                        }
                        int plane = (bi * f + fi) * h * wd;
                        for (int i = 0; i < m; i++) {
                            int yy = ty * m + i;
                            if (yy >= h) {
                                break;
                            }
                            for (int j = 0; j < m; j++) {
                                int xx = tx * m + j;
                                if (xx >= wd) {
                                    break;
                                }
                                float s = 0;
                                for (int t = 0; t < alpha; t++) {
                                    s += half[i * alpha + t] * at[j][t];
                                }
                                out[plane + yy * wd + xx] = s;
                            }
                        }
                    }
                }
            }
        }
        return Tensor.of(Shape.of(b, f, h, wd), out);
    }

    /**
     * Symmetric per-tensor fake quantisation.
     *
     * @param v values
     * @param bits bits
     * @return the values rounded to {@code 2^(bits-1) - 1} levels on each side
     */
    public static float[] fakeQuant(float[] v, int bits) {
        float max = 0;
        for (float x : v) {
            max = Math.max(max, Math.abs(x));
        }
        if (max == 0) {
            return v.clone();
        }
        int q = (1 << (bits - 1)) - 1;
        float scale = max / q;
        float[] out = new float[v.length];
        for (int i = 0; i < v.length; i++) {
            out[i] = Math.round(v[i] / scale) * scale;
        }
        return out;
    }

    private float[] fakeQuant(float[] v) {
        return fakeQuant(v, bits);
    }

    /** @return multiplications per output tile and channel pair in the transformed domain */
    public int multiplications() {
        return alpha * alpha;
    }

    @Override
    public String toString() {
        return "WinogradConv(F(" + m + ",3)" + (bits > 0 ? ", int" + bits : "") + ")";
    }
}
