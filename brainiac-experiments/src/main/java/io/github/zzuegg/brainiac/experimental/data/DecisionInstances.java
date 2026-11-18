package io.github.zzuegg.brainiac.experimental.data;

import java.util.SplittableRandom;

/**
 * Small generators for decision problems whose inputs must be predicted: the benchmarks of
 * predict-then-optimise, solver-in-the-loop learning, differentiable sorting, QP layers and
 * finite-sum optimisation.
 * <ul>
 *   <li>{@link #shortestPaths}: the Elmachtoub and Grigas (2022) generator. A {@code g x g} grid
 *       whose edges point right or down; each edge cost is
 *       {@code [(B x / sqrt(p) + 3)^deg / 3^(deg-1) + 1] * eps} for a hidden random matrix {@code B},
 *       observed features {@code x} and multiplicative noise {@code eps}, so the degree controls
 *       how misspecified a linear cost model is;</li>
 *   <li>{@link #terrainMaps}: "Warcraft-lite" maps (Vlastelica et al. 2020): every cell of a grid
 *       is a small noisy pixel patch drawn from one of five terrain prototypes with a hidden
 *       cost, and the label is the cheapest 8-connected path between opposite corners;</li>
 *   <li>{@link #knapsacks}: multi-dimensional knapsacks with fixed weights and capacities whose
 *       item values are a polynomial of observed features;</li>
 *   <li>{@link #portfolios}: a budgeted portfolio LP (weights sum to one, each below a cap) whose
 *       returns are a nonlinear function of observed features;</li>
 *   <li>{@link #sudokus}: 4x4 Sudoku puzzles with unique-valued rows, columns and boxes;</li>
 *   <li>{@link #sortingSets}: sets of two-digit number images (each digit a fixed random 5x5
 *       pattern plus noise) to be ranked by their value, the MNIST-sort task at toy scale;</li>
 *   <li>{@link #leastSquares} and {@link #logistic}: finite sums with planted conditioning.</li>
 * </ul>
 */
public final class DecisionInstances {

    private DecisionInstances() {
    }

    /**
     * Shortest-path instances on a grid DAG.
     *
     * @param features observed features {@code [n][p]}
     * @param costs true edge costs {@code [n][edges]}
     * @param grid the number of nodes per side
     */
    public record PathInstances(double[][] features, double[][] costs, int grid) {

        /**
         * The number of edges: {@code 2 g (g - 1)}.
         *
         * @return the edge count
         */
        public int edges() {
            return 2 * grid * (grid - 1);
        }
    }

    /**
     * The edge index of the edge leaving node {@code (r, c)} to the right (horizontal edges
     * first, row by row, then vertical edges).
     *
     * @param grid nodes per side
     * @param r the row
     * @param c the column, below {@code grid - 1}
     * @return the edge index
     */
    public static int rightEdge(int grid, int r, int c) {
        return r * (grid - 1) + c;
    }

    /**
     * The edge index of the edge leaving node {@code (r, c)} downwards.
     *
     * @param grid nodes per side
     * @param r the row, below {@code grid - 1}
     * @param c the column
     * @return the edge index
     */
    public static int downEdge(int grid, int r, int c) {
        return grid * (grid - 1) + r * grid + c;
    }

    /**
     * Shortest-path instances with polynomial costs (Elmachtoub and Grigas 2022, section 6).
     *
     * @param n the number of instances
     * @param grid nodes per side, for example 5
     * @param p the feature dimension, for example 5
     * @param degree the polynomial degree (1 is a well-specified linear model)
     * @param noise the half-width of the multiplicative noise, for example 0.5
     * @param hiddenSeed the seed of the hidden matrix {@code B}, shared by train and test sets
     * @param seed the seed of the features and noise
     * @return the instances
     */
    public static PathInstances shortestPaths(int n, int grid, int p, int degree, double noise, long hiddenSeed,
            long seed) {
        int edges = 2 * grid * (grid - 1);
        double[][] b = bernoulliMatrix(edges, p, hiddenSeed);
        SplittableRandom r = new SplittableRandom(seed);
        double[][] x = new double[n][p];
        double[][] c = new double[n][edges];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                x[i][j] = r.nextGaussian();
            }
            for (int e = 0; e < edges; e++) {
                double s = 0;
                for (int j = 0; j < p; j++) {
                    s += b[e][j] * x[i][j];
                }
                double base = Math.pow(s / Math.sqrt(p) + 3, degree) / Math.pow(3, degree - 1) + 1;
                c[i][e] = base * r.nextDouble(1 - noise, 1 + noise);
            }
        }
        return new PathInstances(x, c, grid);
    }

    private static double[][] bernoulliMatrix(int rows, int cols, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] b = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                b[i][j] = r.nextBoolean() ? 1 : 0;
            }
        }
        return b;
    }

    /**
     * Terrain maps whose cells are pixel patches.
     *
     * @param patches the pixels of every cell {@code [n][cells][pixels]}
     * @param costs the hidden cell costs {@code [n][cells]}
     * @param terrain the terrain type of every cell {@code [n][cells]}
     * @param side the number of cells per side
     */
    public record TerrainMaps(double[][][] patches, double[][] costs, int[][] terrain, int side) {
    }

    /** The hidden cost of each of the five terrain types. */
    private static final double[] TERRAIN_COST = {0.8, 1.2, 2.5, 5.3, 7.7};

    /**
     * Warcraft-lite maps: terrain is laid out in smooth blobs, and every cell is a 3x3 patch of
     * its terrain's prototype plus Gaussian noise. The hidden costs are 0.8, 1.2, 2.5, 5.3 and 7.7.
     *
     * @param n the number of maps
     * @param side cells per side, for example 8
     * @param pixelNoise the pixel noise standard deviation, for example 0.3
     * @param seed the seed
     * @return the maps
     */
    public static TerrainMaps terrainMaps(int n, int side, double pixelNoise, long seed) {
        int pixels = 9;
        double[][] prototypes = new double[TERRAIN_COST.length][pixels];
        SplittableRandom pr = new SplittableRandom(7_771);
        for (double[] proto : prototypes) {
            for (int q = 0; q < pixels; q++) {
                proto[q] = pr.nextDouble(-1, 1);
            }
        }
        SplittableRandom r = new SplittableRandom(seed);
        int cells = side * side;
        double[][][] patches = new double[n][cells][pixels];
        double[][] costs = new double[n][cells];
        int[][] terrain = new int[n][cells];
        for (int m = 0; m < n; m++) {
            // a smooth random field thresholded into five terrain levels
            int bumps = 4;
            double[][] centre = new double[bumps][3];
            for (double[] b : centre) {
                b[0] = r.nextDouble(side);
                b[1] = r.nextDouble(side);
                b[2] = r.nextDouble(-1, 1);
            }
            for (int i = 0; i < side; i++) {
                for (int j = 0; j < side; j++) {
                    double f = 0;
                    for (double[] b : centre) {
                        double d2 = (i - b[0]) * (i - b[0]) + (j - b[1]) * (j - b[1]);
                        f += b[2] * Math.exp(-d2 / (0.15 * side * side));
                    }
                    f += r.nextGaussian() * 0.25;
                    int t = (int) Math.floor((Math.tanh(f) + 1) / 2 * TERRAIN_COST.length);
                    t = Math.max(0, Math.min(TERRAIN_COST.length - 1, t));
                    int cell = i * side + j;
                    terrain[m][cell] = t;
                    costs[m][cell] = TERRAIN_COST[t];
                    for (int q = 0; q < pixels; q++) {
                        patches[m][cell][q] = prototypes[t][q] + r.nextGaussian() * pixelNoise;
                    }
                }
            }
        }
        return new TerrainMaps(patches, costs, terrain, side);
    }

    /**
     * Multi-dimensional knapsack instances: maximise the value of chosen items subject to
     * {@code W x <= capacity}.
     *
     * @param features observed features {@code [n][items][p]}
     * @param values true item values {@code [n][items]}
     * @param weights item weights {@code [dims][items]}, shared by all instances
     * @param capacity capacities {@code [dims]}
     */
    public record Knapsacks(double[][][] features, double[][] values, double[][] weights, double[] capacity) {
    }

    /**
     * Knapsack instances whose item values are a degree-{@code d} polynomial of the features.
     *
     * @param n the number of instances
     * @param items items per instance, for example 10
     * @param dims the number of capacity constraints, for example 2
     * @param p features per item
     * @param degree the polynomial degree
     * @param hiddenSeed the seed of the weights, capacities and hidden map
     * @param seed the seed of the features
     * @return the instances
     */
    public static Knapsacks knapsacks(int n, int items, int dims, int p, int degree, long hiddenSeed, long seed) {
        SplittableRandom h = new SplittableRandom(hiddenSeed);
        double[][] w = new double[dims][items];
        double[] cap = new double[dims];
        for (int k = 0; k < dims; k++) {
            double total = 0;
            for (int i = 0; i < items; i++) {
                w[k][i] = h.nextDouble(1, 5);
                total += w[k][i];
            }
            cap[k] = total / 3;
        }
        double[] beta = new double[p];
        for (int j = 0; j < p; j++) {
            beta[j] = h.nextGaussian();
        }
        SplittableRandom r = new SplittableRandom(seed);
        double[][][] x = new double[n][items][p];
        double[][] v = new double[n][items];
        for (int m = 0; m < n; m++) {
            for (int i = 0; i < items; i++) {
                double s = 0;
                for (int j = 0; j < p; j++) {
                    x[m][i][j] = r.nextGaussian();
                    s += beta[j] * x[m][i][j];
                }
                v[m][i] = (Math.pow(s / Math.sqrt(p) + 3, degree) / Math.pow(3, degree - 1) + 1)
                        * r.nextDouble(0.8, 1.2);
            }
        }
        return new Knapsacks(x, v, w, cap);
    }

    /**
     * Portfolio instances: maximise {@code r . x} with {@code sum x = 1}, {@code 0 <= x <= cap}.
     *
     * @param features observed features {@code [n][p]}
     * @param returns true asset returns {@code [n][assets]}
     * @param cap the per-asset cap
     */
    public record Portfolios(double[][] features, double[][] returns, double cap) {
    }

    /**
     * Portfolio LPs whose returns are {@code (B x / sqrt(p) + tau)^deg} scaled, plus noise.
     *
     * @param n instances
     * @param assets the number of assets, for example 12
     * @param p features
     * @param degree the polynomial degree
     * @param hiddenSeed the seed of {@code B}
     * @param seed the seed of features and noise
     * @return the instances, with a cap of 0.25
     */
    public static Portfolios portfolios(int n, int assets, int p, int degree, long hiddenSeed, long seed) {
        double[][] b = bernoulliMatrix(assets, p, hiddenSeed);
        SplittableRandom r = new SplittableRandom(seed);
        double[][] x = new double[n][p];
        double[][] ret = new double[n][assets];
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < p; j++) {
                x[i][j] = r.nextGaussian();
            }
            for (int a = 0; a < assets; a++) {
                double s = 0;
                for (int j = 0; j < p; j++) {
                    s += b[a][j] * x[i][j];
                }
                ret[i][a] = 0.05 * Math.pow(s / Math.sqrt(p) + 2, degree) / Math.pow(2, degree - 1)
                        + 0.02 * r.nextGaussian();
            }
        }
        return new Portfolios(x, ret, 0.25);
    }

    /**
     * 4x4 Sudoku puzzles.
     *
     * @param puzzles the clues, values 1..4 and 0 for a blank {@code [n][16]}
     * @param solutions the completed grids {@code [n][16]}
     */
    public record Sudokus(int[][] puzzles, int[][] solutions) {
    }

    /**
     * Random 4x4 Sudokus: a valid base grid shuffled by digit relabelling, row swaps within
     * bands, column swaps within stacks, band and stack swaps and transposition, with
     * {@code blanks} cells removed at random.
     *
     * @param n the number of puzzles
     * @param blanks the number of blank cells, for example 6
     * @param seed the seed
     * @return the puzzles
     */
    public static Sudokus sudokus(int n, int blanks, long seed) {
        int[][] base = {{1, 2, 3, 4}, {3, 4, 1, 2}, {2, 1, 4, 3}, {4, 3, 2, 1}};
        SplittableRandom r = new SplittableRandom(seed);
        int[][] puzzles = new int[n][16];
        int[][] solutions = new int[n][16];
        for (int m = 0; m < n; m++) {
            int[] digits = permutation(4, r);
            int[] rows = blockPermutation(r);
            int[] cols = blockPermutation(r);
            boolean transpose = r.nextBoolean();
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    int v = transpose ? base[cols[j]][rows[i]] : base[rows[i]][cols[j]];
                    solutions[m][i * 4 + j] = digits[v - 1] + 1;
                }
            }
            puzzles[m] = solutions[m].clone();
            int[] order = permutation(16, r);
            for (int k = 0; k < blanks; k++) {
                puzzles[m][order[k]] = 0;
            }
        }
        return new Sudokus(puzzles, solutions);
    }

    private static int[] blockPermutation(SplittableRandom r) {
        int[] bands = r.nextBoolean() ? new int[] {0, 1} : new int[] {1, 0};
        int[] out = new int[4];
        for (int b = 0; b < 2; b++) {
            boolean swap = r.nextBoolean();
            out[2 * b] = 2 * bands[b] + (swap ? 1 : 0);
            out[2 * b + 1] = 2 * bands[b] + (swap ? 0 : 1);
        }
        return out;
    }

    private static int[] permutation(int n, SplittableRandom r) {
        int[] p = new int[n];
        for (int i = 0; i < n; i++) {
            p[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = p[i];
            p[i] = p[j];
            p[j] = t;
        }
        return p;
    }

    /**
     * Sets of number images to sort.
     *
     * @param images the pixels {@code [sets][items][50]}: two 5x5 digit glyphs side by side
     * @param values the number shown by each image, 0..99 {@code [sets][items]}
     */
    public record SortingSets(double[][][] images, int[][] values) {
    }

    /**
     * Random sets of distinct two-digit numbers rendered as noisy glyph images.
     *
     * @param sets the number of sets
     * @param items numbers per set, for example 5 or 9
     * @param noise the pixel noise standard deviation, for example 0.3
     * @param seed the seed
     * @return the sets
     */
    public static SortingSets sortingSets(int sets, int items, double noise, long seed) {
        double[][] glyph = new double[10][25];
        SplittableRandom g = new SplittableRandom(4_242);
        for (double[] d : glyph) {
            for (int q = 0; q < 25; q++) {
                d[q] = g.nextDouble() < 0.4 ? 1 : 0;
            }
        }
        SplittableRandom r = new SplittableRandom(seed);
        double[][][] images = new double[sets][items][50];
        int[][] values = new int[sets][items];
        for (int s = 0; s < sets; s++) {
            int[] pick = permutation(100, r);
            for (int i = 0; i < items; i++) {
                int v = pick[i];
                values[s][i] = v;
                for (int q = 0; q < 25; q++) {
                    images[s][i][q] = glyph[v / 10][q] + r.nextGaussian() * noise;
                    images[s][i][25 + q] = glyph[v % 10][q] + r.nextGaussian() * noise;
                }
            }
        }
        return new SortingSets(images, values);
    }

    /**
     * A finite-sum problem {@code (1/n) sum_i loss(a_i . w, b_i)}.
     *
     * @param a the design {@code [n][d]}
     * @param b the targets {@code [n]} (0/1 labels for logistic regression)
     * @param truth the planted weights
     */
    public record FiniteSum(double[][] a, double[] b, double[] truth) {
    }

    /**
     * Least squares with a design whose covariance has condition number about {@code kappa}:
     * Gaussian rows scaled per column by singular values log-spaced from 1 to
     * {@code 1 / sqrt(kappa)}, then rotated by a random orthogonal-ish mixing.
     *
     * @param n rows
     * @param d columns
     * @param kappa the planted condition number of {@code A^T A}
     * @param noise the target noise standard deviation
     * @param seed the seed
     * @return the problem
     */
    public static FiniteSum leastSquares(int n, int d, double kappa, double noise, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] a = conditionedDesign(n, d, kappa, r);
        double[] w = new double[d];
        for (int j = 0; j < d; j++) {
            w[j] = r.nextGaussian();
        }
        double[] b = new double[n];
        for (int i = 0; i < n; i++) {
            double s = 0;
            for (int j = 0; j < d; j++) {
                s += a[i][j] * w[j];
            }
            b[i] = s + noise * r.nextGaussian();
        }
        return new FiniteSum(a, b, w);
    }

    /**
     * Logistic regression data with planted conditioning and labels drawn from the model.
     *
     * @param n rows
     * @param d columns
     * @param kappa the planted condition number of the design covariance
     * @param seed the seed
     * @return the problem, with 0/1 labels
     */
    public static FiniteSum logistic(int n, int d, double kappa, long seed) {
        SplittableRandom r = new SplittableRandom(seed);
        double[][] a = conditionedDesign(n, d, kappa, r);
        double[] w = new double[d];
        for (int j = 0; j < d; j++) {
            w[j] = r.nextGaussian() * 2;
        }
        double[] b = new double[n];
        for (int i = 0; i < n; i++) {
            double s = 0;
            for (int j = 0; j < d; j++) {
                s += a[i][j] * w[j];
            }
            b[i] = r.nextDouble() < 1 / (1 + Math.exp(-s)) ? 1 : 0;
        }
        return new FiniteSum(a, b, w);
    }

    private static double[][] conditionedDesign(int n, int d, double kappa, SplittableRandom r) {
        double[] scale = new double[d];
        for (int j = 0; j < d; j++) {
            scale[j] = Math.pow(kappa, -0.5 * j / Math.max(1, d - 1));
        }
        // a random rotation from Gram-Schmidt on a Gaussian matrix
        double[][] q = new double[d][d];
        for (int i = 0; i < d; i++) {
            for (int j = 0; j < d; j++) {
                q[i][j] = r.nextGaussian();
            }
            for (int k = 0; k < i; k++) {
                double dot = 0;
                for (int j = 0; j < d; j++) {
                    dot += q[i][j] * q[k][j];
                }
                for (int j = 0; j < d; j++) {
                    q[i][j] -= dot * q[k][j];
                }
            }
            double norm = 0;
            for (int j = 0; j < d; j++) {
                norm += q[i][j] * q[i][j];
            }
            norm = Math.sqrt(norm);
            for (int j = 0; j < d; j++) {
                q[i][j] /= norm;
            }
        }
        double[][] a = new double[n][d];
        double[] z = new double[d];
        for (int i = 0; i < n; i++) {
            for (int k = 0; k < d; k++) {
                z[k] = r.nextGaussian() * scale[k];
            }
            for (int j = 0; j < d; j++) {
                double s = 0;
                for (int k = 0; k < d; k++) {
                    s += z[k] * q[k][j];
                }
                a[i][j] = s;
            }
        }
        return a;
    }
}
