package io.github.zzuegg.brainiac.experimental.kernels;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;

/**
 * The schedule space of a tiled matmul {@code C [m, n] = A [m, k] B [k, n]} written in plain
 * Java, in the AutoTVM sense: a schedule fixes the loop order of the tile loops and of the
 * loops inside a tile, three tile sizes, the unrolling of the innermost loop and the number of
 * threads the i-tiles are split over. The kernel runs for real, and every schedule computes
 * the same product, up to float summation order. The class also provides the loop-structure
 * features that a {@link LearnedCostModel} learns from.
 */
public final class ScheduleSpace {

    /** The loop orders (outer to inner) of the tile loops and of the loops inside a tile. */
    public static final List<String> ORDERS = List.of("ijk", "ikj", "jik", "jki", "kij", "kji");
    /** Tile sizes. */
    public static final int[] TILES = {4, 8, 16, 32, 64, 128};
    /** Innermost unroll factors. */
    public static final int[] UNROLLS = {1, 4};
    /** Thread counts. */
    public static final int[] THREADS = {1, 2, 4};

    private final int m;
    private final int k;
    private final int n;

    /**
     * One point of the space.
     *
     * @param order the loop order, one of {@link #ORDERS}
     * @param ti the i tile
     * @param tj the j tile
     * @param tk the k tile
     * @param unroll the unroll factor of the innermost loop (only for j innermost)
     * @param threads threads over the i tiles
     */
    public record Schedule(String order, int ti, int tj, int tk, int unroll, int threads) {
        @Override
        public String toString() {
            return order + " " + ti + "x" + tj + "x" + tk + " u" + unroll + " t" + threads;
        }
    }

    /**
     * @param m rows of A
     * @param k columns of A
     * @param n columns of B
     */
    public ScheduleSpace(int m, int k, int n) {
        this.m = m;
        this.k = k;
        this.n = n;
    }

    /** @return the number of schedules */
    public int size() {
        return ORDERS.size() * TILES.length * TILES.length * TILES.length * UNROLLS.length * THREADS.length;
    }

    /**
     * @param index in {@code [0, size())}
     * @return the schedule with that index (threads vary fastest, then unroll, tk, tj, ti, order)
     */
    public Schedule get(int index) {
        int t = index % THREADS.length;
        index /= THREADS.length;
        int u = index % UNROLLS.length;
        index /= UNROLLS.length;
        int c = index % TILES.length;
        index /= TILES.length;
        int b = index % TILES.length;
        index /= TILES.length;
        int a = index % TILES.length;
        index /= TILES.length;
        return new Schedule(ORDERS.get(index), TILES[a], TILES[b], TILES[c], UNROLLS[u], THREADS[t]);
    }

    /**
     * @param r the random source
     * @return a uniformly random schedule
     */
    public Schedule random(SplittableRandom r) {
        return get(r.nextInt(size()));
    }

    /**
     * Changes one decision.
     *
     * @param s the schedule
     * @param r the random source
     * @return a neighbour
     */
    public Schedule mutate(Schedule s, SplittableRandom r) {
        return switch (r.nextInt(6)) {
            case 0 -> new Schedule(ORDERS.get(r.nextInt(ORDERS.size())), s.ti(), s.tj(), s.tk(), s.unroll(), s.threads());
            case 1 -> new Schedule(s.order(), step(s.ti(), r), s.tj(), s.tk(), s.unroll(), s.threads());
            case 2 -> new Schedule(s.order(), s.ti(), step(s.tj(), r), s.tk(), s.unroll(), s.threads());
            case 3 -> new Schedule(s.order(), s.ti(), s.tj(), step(s.tk(), r), s.unroll(), s.threads());
            case 4 -> new Schedule(s.order(), s.ti(), s.tj(), s.tk(), UNROLLS[r.nextInt(UNROLLS.length)], s.threads());
            default -> new Schedule(s.order(), s.ti(), s.tj(), s.tk(), s.unroll(), THREADS[r.nextInt(THREADS.length)]);
        };
    }

    private static int step(int tile, SplittableRandom r) {
        int i = Integer.numberOfTrailingZeros(tile) - 2 + (r.nextBoolean() ? 1 : -1);
        return TILES[Math.max(0, Math.min(TILES.length - 1, i))];
    }

    /** @return {@code 2 m k n}, the floating-point operations of one product */
    public double flops() {
        return 2.0 * m * k * n;
    }

    /**
     * Loop-structure features for a cost model: log tiles, the order (one-hot), unroll,
     * threads, the tile working set and whether it fits a 32 KiB L1 or a 1 MiB L2, the stride
     * of the innermost loop for each operand, and the number of tiles per thread.
     *
     * @param s the schedule
     * @return the features
     */
    public double[] features(Schedule s) {
        List<Double> f = new ArrayList<>();
        int ti = Math.min(s.ti(), m);
        int tj = Math.min(s.tj(), n);
        int tk = Math.min(s.tk(), k);
        f.add(log2(ti));
        f.add(log2(tj));
        f.add(log2(tk));
        for (String o : ORDERS) {
            f.add(o.equals(s.order()) ? 1.0 : 0.0);
        }
        char inner = s.order().charAt(2);
        f.add(inner == 'j' ? 1.0 : 0.0);
        f.add((double) s.unroll());
        f.add((double) s.threads());
        double bytes = 4.0 * ((double) ti * tk + (double) tk * tj + (double) ti * tj);
        f.add(Math.log(bytes) / Math.log(2));
        f.add(bytes <= 32 * 1024 ? 1.0 : 0.0);
        f.add(bytes <= 1024 * 1024 ? 1.0 : 0.0);
        // innermost strides (in floats) of A[i,k], B[k,j], C[i,j]
        f.add(inner == 'k' ? 1.0 : inner == 'i' ? k : 0.0);
        f.add(inner == 'j' ? 1.0 : inner == 'k' ? n : 0.0);
        f.add(inner == 'j' ? 1.0 : inner == 'i' ? n : 0.0);
        f.add(Math.ceil((double) m / ti) / s.threads());
        double[] out = new double[f.size()];
        for (int i = 0; i < out.length; i++) {
            out[i] = f.get(i);
        }
        return out;
    }

    private static double log2(int x) {
        return Math.log(x) / Math.log(2);
    }

    /**
     * Runs the product with a schedule.
     *
     * @param a {@code [m, k]} row-major
     * @param b {@code [k, n]} row-major
     * @param s the schedule
     * @return {@code [m, n]}
     */
    public float[] multiply(float[] a, float[] b, Schedule s) {
        float[] c = new float[m * n];
        int iTiles = (m + s.ti() - 1) / s.ti();
        if (s.threads() == 1) {
            for (int t = 0; t < iTiles; t++) {
                rowBand(a, b, c, s, t);
            }
            return c;
        }
        ForkJoinPool pool = new ForkJoinPool(s.threads());
        try {
            pool.submit(() -> IntStream.range(0, iTiles).parallel().forEach(t -> rowBand(a, b, c, s, t))).get();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException(e);
        } catch (ExecutionException e) {
            throw new IllegalStateException(e.getCause());
        } finally {
            pool.shutdown();
        }
        return c;
    }

    /** One i-tile (a band of rows): the remaining two tile loops and the tile body. */
    private void rowBand(float[] a, float[] b, float[] c, ScheduleSpace.Schedule s, int t) {
        int i0 = t * s.ti();
        int i1 = Math.min(m, i0 + s.ti());
        String order = s.order();
        // the tile loops over j and k, in the relative order the schedule gives them
        boolean jBeforeK = order.indexOf('j') < order.indexOf('k');
        if (jBeforeK) {
            for (int j0 = 0; j0 < n; j0 += s.tj()) {
                for (int k0 = 0; k0 < k; k0 += s.tk()) {
                    tile(a, b, c, order, s.unroll(), i0, i1, j0, Math.min(n, j0 + s.tj()), k0,
                            Math.min(k, k0 + s.tk()));
                }
            }
        } else {
            for (int k0 = 0; k0 < k; k0 += s.tk()) {
                for (int j0 = 0; j0 < n; j0 += s.tj()) {
                    tile(a, b, c, order, s.unroll(), i0, i1, j0, Math.min(n, j0 + s.tj()), k0,
                            Math.min(k, k0 + s.tk()));
                }
            }
        }
    }

    private void tile(float[] a, float[] b, float[] c, String order, int unroll, int i0, int i1, int j0, int j1,
            int k0, int k1) {
        switch (order) {
            case "ikj", "kij" -> {
                boolean iOuter = order.charAt(0) == 'i';
                int outerFrom = iOuter ? i0 : k0;
                int outerTo = iOuter ? i1 : k1;
                for (int o = outerFrom; o < outerTo; o++) {
                    int midFrom = iOuter ? k0 : i0;
                    int midTo = iOuter ? k1 : i1;
                    for (int p = midFrom; p < midTo; p++) {
                        int i = iOuter ? o : p;
                        int kk = iOuter ? p : o;
                        float av = a[i * k + kk];
                        int cb = i * n;
                        int bb = kk * n;
                        int j = j0;
                        if (unroll == 4) {
                            for (; j + 3 < j1; j += 4) {
                                c[cb + j] += av * b[bb + j];
                                c[cb + j + 1] += av * b[bb + j + 1];
                                c[cb + j + 2] += av * b[bb + j + 2];
                                c[cb + j + 3] += av * b[bb + j + 3];
                            }
                        }
                        for (; j < j1; j++) {
                            c[cb + j] += av * b[bb + j];
                        }
                    }
                }
            }
            case "ijk", "jik" -> {
                boolean iOuter = order.charAt(0) == 'i';
                for (int o = iOuter ? i0 : j0; o < (iOuter ? i1 : j1); o++) {
                    for (int p = iOuter ? j0 : i0; p < (iOuter ? j1 : i1); p++) {
                        int i = iOuter ? o : p;
                        int j = iOuter ? p : o;
                        float sum = 0;
                        for (int kk = k0; kk < k1; kk++) {
                            sum += a[i * k + kk] * b[kk * n + j];
                        }
                        c[i * n + j] += sum;
                    }
                }
            }
            default -> { // "jki", "kji": i innermost
                boolean jOuter = order.charAt(0) == 'j';
                for (int o = jOuter ? j0 : k0; o < (jOuter ? j1 : k1); o++) {
                    for (int p = jOuter ? k0 : j0; p < (jOuter ? k1 : j1); p++) {
                        int j = jOuter ? o : p;
                        int kk = jOuter ? p : o;
                        float bv = b[kk * n + j];
                        for (int i = i0; i < i1; i++) {
                            c[i * n + j] += a[i * k + kk] * bv;
                        }
                    }
                }
            }
        }
    }

    /**
     * Times a schedule.
     *
     * @param a {@code [m, k]}
     * @param b {@code [k, n]}
     * @param s the schedule
     * @param repeats timed runs (after one warm-up)
     * @return GFLOP/s of the fastest run
     */
    public double gflops(float[] a, float[] b, Schedule s, int repeats) {
        return KernelTimer.time(() -> multiply(a, b, s), 1, repeats).peakGflops(flops());
    }

    @Override
    public String toString() {
        return "ScheduleSpace(" + m + "x" + k + "x" + n + ", " + size() + " schedules)";
    }
}
