package io.github.zzuegg.brainiac.experimental.vision;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.nn.Layer;
import io.github.zzuegg.brainiac.nn.LayerSpec;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.random.RandomGenerator;

/**
 * Multiresolution hash encoding (Instant-NGP, Muller et al., 2022). A coordinate in a box is
 * looked up in {@code levels} grids whose resolution grows geometrically from
 * {@code coarsest} to {@code finest} cells per side. Each grid's vertex features live in a
 * learned table of {@code 2^log2Table} rows: a coarse grid that fits is indexed densely, a
 * finer one by the spatial hash {@code xor_i(x_i * pi_i) mod T}, so distinct vertices may
 * share a row and the gradients sort out the collisions. The features of the
 * {@code 2^dims} corners around the point are interpolated (bi- or trilinearly) and the
 * levels are concatenated, giving {@code levels * features} values for a small MLP.
 *
 * <p>The tables hold most of the capacity, so the MLP behind them can be tiny and training is
 * fast. Gradients flow to the tables only, not to the coordinates.
 *
 * <p>Paper: <a href="https://arxiv.org/abs/2201.05989">arXiv:2201.05989</a>.
 */
public final class HashGridEncoding implements Layer {

    private static final long[] PRIMES = {1L, 2_654_435_761L, 805_459_861L};

    private final int dims;
    private final int features;
    private final int tableSize;
    private final float[] low;
    private final float[] high;
    private final int[] resolutions;
    private final List<Var> tables = new ArrayList<>();

    /**
     * Creates the encoding.
     *
     * @param dims the coordinate dimension, 2 or 3
     * @param levels the number of grids
     * @param features the features per vertex
     * @param log2Table the base-2 logarithm of the rows per table
     * @param coarsest the cells per side of the coarsest grid
     * @param finest the cells per side of the finest grid
     * @param low the lower corner of the box
     * @param high the upper corner
     * @param random the source of the initial table entries
     */
    public HashGridEncoding(int dims, int levels, int features, int log2Table, int coarsest, int finest,
            float[] low, float[] high, RandomGenerator random) {
        if (dims < 2 || dims > 3) {
            throw new IllegalArgumentException("dims must be 2 or 3, got " + dims);
        }
        this.dims = dims;
        this.features = features;
        this.tableSize = 1 << log2Table;
        this.low = low.clone();
        this.high = high.clone();
        this.resolutions = new int[levels];
        double growth = levels == 1 ? 1 : Math.exp((Math.log(finest) - Math.log(coarsest)) / (levels - 1));
        for (int l = 0; l < levels; l++) {
            resolutions[l] = (int) Math.floor(coarsest * Math.pow(growth, l) + 1e-6);
            float[] t = new float[tableSize * features];
            for (int i = 0; i < t.length; i++) {
                t[i] = (float) (1e-4 * (2 * random.nextDouble() - 1));
            }
            tables.add(Var.param(Tensor.of(Shape.of(tableSize, features), t)));
        }
    }

    /**
     * Describes the encoding over the unit box for model builders.
     *
     * @param levels the number of grids
     * @param features the features per vertex
     * @param log2Table the base-2 logarithm of the table size
     * @param coarsest the coarsest resolution
     * @param finest the finest resolution
     * @return the spec
     */
    public static LayerSpec spec(int levels, int features, int log2Table, int coarsest, int finest) {
        return new LayerSpec() {
            @Override
            public Shape outputShape(Shape input) {
                return Shape.of(input.dim(0), levels * features);
            }

            @Override
            public Layer create(Shape input, SplittableRandom random) {
                int d = input.dim(-1);
                float[] hi = new float[d];
                java.util.Arrays.fill(hi, 1f);
                return new HashGridEncoding(d, levels, features, log2Table, coarsest, finest, new float[d], hi,
                        random);
            }
        };
    }

    /** @return the table entries in all levels */
    public long tableParameters() {
        return (long) tables.size() * tableSize * features;
    }

    /**
     * The share of grid vertices, over all levels, beyond the table size: a lower bound on
     * the share that must collide with another vertex.
     *
     * @return the excess share
     */
    public double collisionShare() {
        long vertices = 0;
        long collided = 0;
        for (int res : resolutions) {
            long v = (long) Math.pow(res + 1, dims);
            vertices += v;
            collided += Math.max(0, v - tableSize);
        }
        return (double) collided / vertices;
    }

    @Override
    public Var forward(Var input) {
        int n = input.shape().dim(0);
        float[] x = input.value().toArray();
        int corners = 1 << dims;
        Var[] parts = new Var[resolutions.length];
        for (int l = 0; l < resolutions.length; l++) {
            int res = resolutions[l];
            boolean dense = Math.pow(res + 1, dims) <= tableSize;
            float[] index = new float[n * corners];
            float[] weight = new float[n * corners];
            int[] cell = new int[dims];
            float[] frac = new float[dims];
            for (int i = 0; i < n; i++) {
                for (int d = 0; d < dims; d++) {
                    float u = (x[i * dims + d] - low[d]) / (high[d] - low[d]) * res;
                    u = Math.max(0, Math.min(res - 1e-4f, u));
                    cell[d] = (int) u;
                    frac[d] = u - cell[d];
                }
                for (int c = 0; c < corners; c++) {
                    float w = 1;
                    long dense1 = 0;
                    long stride = 1;
                    long hash = 0;
                    for (int d = 0; d < dims; d++) {
                        int bit = (c >> d) & 1;
                        long v = cell[d] + bit;
                        w *= bit == 1 ? frac[d] : 1 - frac[d];
                        dense1 += v * stride;
                        stride *= res + 1;
                        hash ^= v * PRIMES[d];
                    }
                    index[i * corners + c] = dense ? dense1 : Math.floorMod(hash, tableSize);
                    weight[i * corners + c] = w;
                }
            }
            Var picked = tables.get(l).gather(0, Tensor.of(Shape.of(n * corners), index)).reshape(n, corners, features);
            parts[l] = picked.mul(Tensor.of(Shape.of(n, corners, 1), weight)).sum(1, false);
        }
        return Var.concat(1, parts);
    }

    @Override
    public int outputSize(int inputSize) {
        return resolutions.length * features;
    }

    @Override
    public List<Var> parameters() {
        return List.copyOf(tables);
    }

    @Override
    public String toString() {
        return "HashGridEncoding(" + resolutions.length + " levels x " + features + ", table " + tableSize + ")";
    }
}
