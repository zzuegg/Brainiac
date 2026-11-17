package io.github.zzuegg.brainiac.experimental.evolve;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.BiFunction;
import java.util.function.Function;

/**
 * MAP-Elites: illuminates a search space by keeping, for every cell of a grid over chosen
 * behaviour dimensions (say, a robot's height and speed), the best solution found with that
 * behaviour. New candidates are mutated (and optionally crossed) copies of random elites;
 * a candidate replaces the elite of its cell when it is better or the cell is empty. The
 * result is not one optimum but a map of diverse, high-performing solutions, which are
 * also stepping stones for each other.
 *
 * <pre>{@code
 * MapElites<double[]> map = MapElites.of(rng -> randomVector(rng), MapElites.gaussian(0.1),
 *         x -> new MapElites.Evaluation(-rastrigin(x), new double[] {x[0], x[1]}))
 *     .grid(new int[] {20, 20}, new double[] {-5, -5}, new double[] {5, 5});
 * map.run(200, 10_000);
 * System.out.println(map.coverage() + " " + map.qdScore());
 * }</pre>
 *
 * <p>Mouret and Clune 2015, "Illuminating search spaces by mapping elites",
 * <a href="https://arxiv.org/abs/1504.04909">arXiv:1504.04909</a>; the iso-line variation is
 * from Vassiliades and Mouret 2018, <a href="https://arxiv.org/abs/1804.03906">arXiv:1804.03906</a>.
 *
 * @param <G> the genome type
 */
public final class MapElites<G> {

    /**
     * What evaluating a genome gives.
     *
     * @param fitness the quality (higher is better)
     * @param descriptor the behaviour, one value per grid dimension
     */
    public record Evaluation(double fitness, double[] descriptor) {
    }

    /**
     * The occupant of a cell.
     *
     * @param genome the solution
     * @param fitness its quality
     * @param descriptor its behaviour
     * @param <G> the genome type
     */
    public record Elite<G>(G genome, double fitness, double[] descriptor) {
    }

    private final Function<SplittableRandom, G> create;
    private final BiFunction<G, SplittableRandom, G> mutate;
    private final Function<G, Evaluation> evaluate;
    private BiFunction<List<G>, SplittableRandom, G> crossover;
    private int[] bins = {10, 10};
    private double[] low = {0, 0};
    private double[] high = {1, 1};
    private long seed = 1;
    private Elite<G>[] cells;
    private SplittableRandom random;
    private final List<Integer> filled = new ArrayList<>();
    private int evaluations;

    private MapElites(Function<SplittableRandom, G> create, BiFunction<G, SplittableRandom, G> mutate,
            Function<G, Evaluation> evaluate) {
        this.create = Objects.requireNonNull(create);
        this.mutate = Objects.requireNonNull(mutate);
        this.evaluate = Objects.requireNonNull(evaluate);
    }

    /**
     * Sets up a search.
     *
     * @param create makes a random genome
     * @param mutate makes a mutated copy
     * @param evaluate scores a genome and gives its behaviour descriptor
     * @param <G> the genome type
     * @return the search, to configure
     */
    public static <G> MapElites<G> of(Function<SplittableRandom, G> create, BiFunction<G, SplittableRandom, G> mutate,
            Function<G, Evaluation> evaluate) {
        return new MapElites<>(create, mutate, evaluate);
    }

    /**
     * Gaussian mutation of a real vector.
     *
     * @param sigma the standard deviation
     * @return the mutation
     */
    public static BiFunction<double[], SplittableRandom, double[]> gaussian(double sigma) {
        return (x, r) -> {
            double[] y = x.clone();
            for (int i = 0; i < y.length; i++) {
                y[i] += sigma * r.nextGaussian();
            }
            return y;
        };
    }

    /**
     * Iso-line variation of two real vectors: {@code a + s1 N(0,1) + s2 N(0,1) (b - a)},
     * which spreads along the direction elites share.
     *
     * @param isoSigma the isotropic spread
     * @param lineSigma the spread along the line
     * @return the crossover, taking two parents
     */
    public static BiFunction<List<double[]>, SplittableRandom, double[]> isoLine(double isoSigma, double lineSigma) {
        return (p, r) -> {
            double[] a = p.get(0);
            double[] b = p.get(1);
            double line = lineSigma * r.nextGaussian();
            double[] y = new double[a.length];
            for (int i = 0; i < y.length; i++) {
                y[i] = a[i] + isoSigma * r.nextGaussian() + line * (b[i] - a[i]);
            }
            return y;
        };
    }

    /**
     * Sets the grid.
     *
     * @param binsPerDimension cells along each behaviour dimension
     * @param lowBounds the lowest descriptor value per dimension
     * @param highBounds the highest descriptor value per dimension
     * @return this
     */
    public MapElites<G> grid(int[] binsPerDimension, double[] lowBounds, double[] highBounds) {
        if (binsPerDimension.length != lowBounds.length || lowBounds.length != highBounds.length) {
            throw new IllegalArgumentException("bins and bounds need one entry per dimension");
        }
        this.bins = binsPerDimension.clone();
        this.low = lowBounds.clone();
        this.high = highBounds.clone();
        return this;
    }

    /**
     * Makes some children from two elites instead of one.
     *
     * @param variation makes a child from two parents
     * @return this
     */
    public MapElites<G> crossover(BiFunction<List<G>, SplittableRandom, G> variation) {
        this.crossover = variation;
        return this;
    }

    /**
     * Sets the seed.
     *
     * @param s the seed
     * @return this
     */
    public MapElites<G> seed(long s) {
        this.seed = s;
        return this;
    }

    /**
     * Fills the map: random genomes first, then variations of random elites.
     *
     * @param initial how many random genomes to start with
     * @param iterations how many variations to evaluate after that
     * @return this
     */
    public MapElites<G> run(int initial, int iterations) {
        if (cells == null) {
            ensureCells();
            for (int i = 0; i < initial; i++) {
                insert(create.apply(random));
            }
        }
        for (int i = 0; i < iterations; i++) {
            if (filled.isEmpty()) {
                insert(create.apply(random));
                continue;
            }
            G parent = cells[filled.get(random.nextInt(filled.size()))].genome();
            G child;
            if (crossover != null && random.nextBoolean()) {
                G other = cells[filled.get(random.nextInt(filled.size()))].genome();
                child = crossover.apply(List.of(parent, other), random);
            } else {
                child = mutate.apply(parent, random);
            }
            insert(child);
        }
        return this;
    }

    /**
     * Evaluates a genome and places it in the map if it beats its cell's elite.
     *
     * @param genome the candidate
     * @return whether it entered the map
     */
    public boolean insert(G genome) {
        ensureCells();
        Evaluation e = evaluate.apply(genome);
        evaluations++;
        int cell = cell(e.descriptor());
        if (cell < 0) {
            return false;
        }
        Elite<G> old = cells[cell];
        if (old == null || e.fitness() > old.fitness()) {
            if (old == null) {
                filled.add(cell);
            }
            cells[cell] = new Elite<>(genome, e.fitness(), e.descriptor().clone());
            return true;
        }
        return false;
    }

    @SuppressWarnings("unchecked")
    private void ensureCells() {
        if (cells == null) {
            int total = 1;
            for (int b : bins) {
                total = Math.multiplyExact(total, b);
            }
            cells = (Elite<G>[]) new Elite<?>[total];
            random = new SplittableRandom(seed);
        }
    }

    private int cell(double[] d) {
        if (d.length != bins.length) {
            throw new IllegalArgumentException("descriptor has " + d.length + " values, grid has " + bins.length);
        }
        int index = 0;
        for (int i = 0; i < d.length; i++) {
            if (!(d[i] >= low[i] && d[i] <= high[i])) {
                return -1;
            }
            int b = (int) Math.min(bins[i] - 1, Math.floor((d[i] - low[i]) / (high[i] - low[i]) * bins[i]));
            index = index * bins[i] + b;
        }
        return index;
    }

    /** @return the share of cells that hold an elite */
    public double coverage() {
        return cells == null ? 0 : filled.size() / (double) cells.length;
    }

    /**
     * The quality-diversity score: the sum over filled cells of {@code fitness - offset},
     * where the offset makes every fitness count positively.
     *
     * @param offset subtracted from each fitness (the lowest possible fitness)
     * @return the QD score
     */
    public double qdScore(double offset) {
        double s = 0;
        for (int c : filled) {
            s += cells[c].fitness() - offset;
        }
        return s;
    }

    /** @return the elites, in no particular order */
    public List<Elite<G>> elites() {
        List<Elite<G>> out = new ArrayList<>();
        for (int c : filled) {
            out.add(cells[c]);
        }
        return out;
    }

    /** @return the best elite, or null for an empty map */
    public Elite<G> best() {
        Elite<G> b = null;
        for (int c : filled) {
            if (b == null || cells[c].fitness() > b.fitness()) {
                b = cells[c];
            }
        }
        return b;
    }

    /** @return how many genomes were evaluated */
    public int evaluations() {
        return evaluations;
    }
}
