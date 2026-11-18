package io.github.zzuegg.brainiac.experimental.data;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Planning problems with known answers: occupancy grids (random obstacles, mazes and rooms),
 * sliding-tile and Lights Out puzzles with exact optimal costs from a breadth-first search
 * over the whole state space, and continuous 2-D obstacle fields (discs and a wall with a
 * narrow passage) for sampling-based planners.
 */
public final class PlanningMaps {

    private PlanningMaps() {
    }

    /**
     * An occupancy grid, row-major, cell {@code y * width + x}.
     *
     * @param width the width
     * @param height the height
     * @param blocked per cell whether it is an obstacle
     */
    public record Grid(int width, int height, boolean[] blocked) {

        /**
         * @param x a column
         * @param y a row
         * @return whether the cell is inside and free
         */
        public boolean free(int x, int y) {
            return x >= 0 && y >= 0 && x < width && y < height && !blocked[y * width + x];
        }

        /** @return a copy */
        public Grid copy() {
            return new Grid(width, height, blocked.clone());
        }

        /**
         * Breadth-first 4-connected distances from a cell.
         *
         * @param from the start cell
         * @return per cell the number of steps, -1 if unreachable
         */
        public int[] distances(int from) {
            int[] d = new int[width * height];
            Arrays.fill(d, -1);
            ArrayDeque<Integer> q = new ArrayDeque<>();
            d[from] = 0;
            q.add(from);
            int[][] moves = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
            while (!q.isEmpty()) {
                int c = q.poll();
                int x = c % width;
                int y = c / width;
                for (int[] m : moves) {
                    int nx = x + m[0];
                    int ny = y + m[1];
                    if (free(nx, ny) && d[ny * width + nx] < 0) {
                        d[ny * width + nx] = d[c] + 1;
                        q.add(ny * width + nx);
                    }
                }
            }
            return d;
        }

        /**
         * A random connected start and goal at least a given 4-connected distance apart.
         *
         * @param minDistance the minimum distance
         * @param random the random source
         * @return {@code {start, goal}} cells, or {@code null} after 200 failed attempts
         */
        public int[] randomPair(int minDistance, SplittableRandom random) {
            for (int attempt = 0; attempt < 200; attempt++) {
                int s = random.nextInt(width * height);
                if (blocked[s]) {
                    continue;
                }
                int[] d = distances(s);
                List<Integer> far = new ArrayList<>();
                for (int i = 0; i < d.length; i++) {
                    if (d[i] >= minDistance) {
                        far.add(i);
                    }
                }
                if (!far.isEmpty()) {
                    return new int[] {s, far.get(random.nextInt(far.size()))};
                }
            }
            return null;
        }
    }

    /**
     * Uniform random obstacles.
     *
     * @param width the width
     * @param height the height
     * @param density the obstacle probability per cell
     * @param random the random source
     * @return the grid
     */
    public static Grid randomObstacles(int width, int height, double density, SplittableRandom random) {
        boolean[] b = new boolean[width * height];
        for (int i = 0; i < b.length; i++) {
            b[i] = random.nextDouble() < density;
        }
        return new Grid(width, height, b);
    }

    /**
     * Rectangular blocks of random size (more structured than {@link #randomObstacles}).
     *
     * @param width the width
     * @param height the height
     * @param blocks the number of blocks
     * @param maxSide the largest block side
     * @param random the random source
     * @return the grid
     */
    public static Grid blocks(int width, int height, int blocks, int maxSide, SplittableRandom random) {
        boolean[] b = new boolean[width * height];
        for (int k = 0; k < blocks; k++) {
            int w = 1 + random.nextInt(maxSide);
            int h = 1 + random.nextInt(maxSide);
            int x0 = random.nextInt(Math.max(1, width - w));
            int y0 = random.nextInt(Math.max(1, height - h));
            for (int y = y0; y < y0 + h; y++) {
                for (int x = x0; x < x0 + w; x++) {
                    b[y * width + x] = true;
                }
            }
        }
        return new Grid(width, height, b);
    }

    /**
     * A perfect maze by randomised depth-first search on the odd cells, with corridors one cell
     * wide; some extra walls are knocked out so the maze has loops.
     *
     * @param width the width (odd works best)
     * @param height the height
     * @param loops the share of extra walls removed
     * @param random the random source
     * @return the grid
     */
    public static Grid maze(int width, int height, double loops, SplittableRandom random) {
        boolean[] b = new boolean[width * height];
        Arrays.fill(b, true);
        int cw = (width - 1) / 2;
        int ch = (height - 1) / 2;
        boolean[] seen = new boolean[cw * ch];
        ArrayDeque<int[]> stack = new ArrayDeque<>();
        stack.push(new int[] {0, 0});
        seen[0] = true;
        b[width + 1] = false;
        int[][] dirs = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        while (!stack.isEmpty()) {
            int[] c = stack.peek();
            List<int[]> options = new ArrayList<>();
            for (int[] d : dirs) {
                int nx = c[0] + d[0];
                int ny = c[1] + d[1];
                if (nx >= 0 && ny >= 0 && nx < cw && ny < ch && !seen[ny * cw + nx]) {
                    options.add(new int[] {nx, ny});
                }
            }
            if (options.isEmpty()) {
                stack.pop();
                continue;
            }
            int[] n = options.get(random.nextInt(options.size()));
            seen[n[1] * cw + n[0]] = true;
            b[(2 * n[1] + 1) * width + 2 * n[0] + 1] = false;
            b[(c[1] + n[1] + 1) * width + c[0] + n[0] + 1] = false;
            stack.push(n);
        }
        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                if (b[y * width + x] && random.nextDouble() < loops) {
                    boolean horizontal = !b[y * width + x - 1] && !b[y * width + x + 1];
                    boolean vertical = !b[(y - 1) * width + x] && !b[(y + 1) * width + x];
                    if (horizontal ^ vertical) {
                        b[y * width + x] = false;
                    }
                }
            }
        }
        return new Grid(width, height, b);
    }

    /**
     * Rooms: an outer wall and a grid of rooms separated by walls with one or two doors each.
     *
     * @param width the width
     * @param height the height
     * @param roomsX rooms per row
     * @param roomsY rooms per column
     * @param random the random source
     * @return the grid
     */
    public static Grid rooms(int width, int height, int roomsX, int roomsY, SplittableRandom random) {
        boolean[] b = new boolean[width * height];
        for (int x = 0; x < width; x++) {
            b[x] = true;
            b[(height - 1) * width + x] = true;
        }
        for (int y = 0; y < height; y++) {
            b[y * width] = true;
            b[y * width + width - 1] = true;
        }
        int[] xs = new int[roomsX + 1];
        int[] ys = new int[roomsY + 1];
        for (int i = 0; i <= roomsX; i++) {
            xs[i] = Math.min(width - 1, i * (width - 1) / roomsX);
        }
        for (int i = 0; i <= roomsY; i++) {
            ys[i] = Math.min(height - 1, i * (height - 1) / roomsY);
        }
        for (int i = 1; i < roomsX; i++) {
            for (int y = 0; y < height; y++) {
                b[y * width + xs[i]] = true;
            }
            for (int j = 0; j < roomsY; j++) {
                int span = ys[j + 1] - ys[j] - 1;
                if (span < 2) {
                    continue;
                }
                int door = ys[j] + 1 + random.nextInt(span - 1);
                b[door * width + xs[i]] = false;
                b[(door + 1) * width + xs[i]] = false;
            }
        }
        for (int j = 1; j < roomsY; j++) {
            for (int x = 0; x < width; x++) {
                b[ys[j] * width + x] = true;
            }
            for (int i = 0; i < roomsX; i++) {
                int span = xs[i + 1] - xs[i] - 1;
                if (span < 2) {
                    continue;
                }
                int door = xs[i] + 1 + random.nextInt(span - 1);
                b[ys[j] * width + door] = false;
                b[ys[j] * width + door + 1] = false;
            }
        }
        for (int k = 0; k < roomsX * roomsY; k++) {
            int x = 1 + random.nextInt(width - 2);
            int y = 1 + random.nextInt(height - 2);
            if (!b[y * width + x] && random.nextBoolean()) {
                b[y * width + x] = true;
            }
        }
        return new Grid(width, height, b);
    }

    // ---- sliding-tile puzzles --------------------------------------------------------

    /**
     * The solved sliding-tile board: tiles {@code 1 .. n^2-1} then the blank (0).
     *
     * @param n the side
     * @return the board
     */
    public static int[] solvedTiles(int n) {
        int[] s = new int[n * n];
        for (int i = 0; i < s.length - 1; i++) {
            s[i] = i + 1;
        }
        return s;
    }

    /**
     * The boards one blank move away.
     *
     * @param board a board
     * @param n the side
     * @return the neighbours
     */
    public static List<int[]> tileMoves(int[] board, int n) {
        int blank = 0;
        while (board[blank] != 0) {
            blank++;
        }
        int bx = blank % n;
        int by = blank / n;
        List<int[]> out = new ArrayList<>(4);
        int[][] d = {{1, 0}, {-1, 0}, {0, 1}, {0, -1}};
        for (int[] m : d) {
            int x = bx + m[0];
            int y = by + m[1];
            if (x < 0 || y < 0 || x >= n || y >= n) {
                continue;
            }
            int[] c = board.clone();
            c[blank] = c[y * n + x];
            c[y * n + x] = 0;
            out.add(c);
        }
        return out;
    }

    /**
     * A board scrambled by a random walk of blank moves that never undoes the previous move.
     *
     * @param n the side
     * @param depth the number of moves
     * @param random the random source
     * @return the board
     */
    public static int[] scrambleTiles(int n, int depth, SplittableRandom random) {
        int[] b = solvedTiles(n);
        int[] prev = null;
        for (int k = 0; k < depth; k++) {
            List<int[]> m = tileMoves(b, n);
            int[] next;
            do {
                next = m.get(random.nextInt(m.size()));
            } while (prev != null && m.size() > 1 && Arrays.equals(next, prev));
            prev = b;
            b = next;
        }
        return b;
    }

    /**
     * Ranks an 8-puzzle board (a permutation of 0..8) to {@code [0, 9!)}.
     *
     * @param board the board
     * @return the Lehmer rank
     */
    public static int rank9(int[] board) {
        int r = 0;
        boolean[] used = new boolean[9];
        for (int i = 0; i < 9; i++) {
            int smaller = 0;
            for (int v = 0; v < board[i]; v++) {
                if (!used[v]) {
                    smaller++;
                }
            }
            used[board[i]] = true;
            r = r * (9 - i) + smaller;
        }
        return r;
    }

    /**
     * Exact optimal costs of every solvable 8-puzzle board, by breadth-first search from the
     * goal (181,440 states).
     *
     * @return per {@link #rank9} the optimal number of moves, -1 if unsolvable
     */
    public static byte[] eightPuzzleTable() {
        byte[] d = new byte[362880];
        Arrays.fill(d, (byte) -1);
        int[] goal = solvedTiles(3);
        ArrayDeque<int[]> q = new ArrayDeque<>();
        d[rank9(goal)] = 0;
        q.add(goal);
        while (!q.isEmpty()) {
            int[] b = q.poll();
            int db = d[rank9(b)];
            for (int[] c : tileMoves(b, 3)) {
                int rc = rank9(c);
                if (d[rc] < 0) {
                    d[rc] = (byte) (db + 1);
                    q.add(c);
                }
            }
        }
        return d;
    }

    /**
     * The Manhattan distance of a sliding-tile board.
     *
     * @param board the board
     * @param n the side
     * @return the sum of tile distances to their goal cells
     */
    public static int manhattan(int[] board, int n) {
        int s = 0;
        for (int i = 0; i < board.length; i++) {
            int v = board[i];
            if (v == 0) {
                continue;
            }
            int g = v - 1;
            s += Math.abs(i % n - g % n) + Math.abs(i / n - g / n);
        }
        return s;
    }

    /**
     * Manhattan distance plus linear conflicts (still admissible).
     *
     * @param board the board
     * @param n the side
     * @return the heuristic
     */
    public static int manhattanLinearConflict(int[] board, int n) {
        int s = manhattan(board, n);
        for (int row = 0; row < n; row++) {
            for (int i = 0; i < n; i++) {
                int a = board[row * n + i];
                if (a == 0 || (a - 1) / n != row) {
                    continue;
                }
                for (int j = i + 1; j < n; j++) {
                    int b = board[row * n + j];
                    if (b != 0 && (b - 1) / n == row && b < a) {
                        s += 2;
                    }
                }
            }
        }
        for (int col = 0; col < n; col++) {
            for (int i = 0; i < n; i++) {
                int a = board[i * n + col];
                if (a == 0 || (a - 1) % n != col) {
                    continue;
                }
                for (int j = i + 1; j < n; j++) {
                    int b = board[j * n + col];
                    if (b != 0 && (b - 1) % n == col && b < a) {
                        s += 2;
                    }
                }
            }
        }
        return s;
    }

    // ---- Lights Out ------------------------------------------------------------------

    /**
     * Presses one cell of a Lights Out board (toggles it and its 4-neighbours).
     *
     * @param state the board as a bit mask
     * @param n the side
     * @param cell the cell pressed
     * @return the new board
     */
    public static int pressLight(int state, int n, int cell) {
        int x = cell % n;
        int y = cell / n;
        int s = state ^ (1 << cell);
        if (x > 0) {
            s ^= 1 << (cell - 1);
        }
        if (x < n - 1) {
            s ^= 1 << (cell + 1);
        }
        if (y > 0) {
            s ^= 1 << (cell - n);
        }
        if (y < n - 1) {
            s ^= 1 << (cell + n);
        }
        return s;
    }

    /**
     * Exact optimal press counts of every Lights Out board, by breadth-first search from the
     * dark board.
     *
     * @param n the side, 3 or 4
     * @return per board mask the fewest presses to turn every light off, -1 if unsolvable
     */
    public static byte[] lightsOutTable(int n) {
        int states = 1 << (n * n);
        byte[] d = new byte[states];
        Arrays.fill(d, (byte) -1);
        int[] q = new int[states];
        int head = 0;
        int tail = 0;
        d[0] = 0;
        q[tail++] = 0;
        while (head < tail) {
            int s = q[head++];
            for (int c = 0; c < n * n; c++) {
                int t = pressLight(s, n, c);
                if (d[t] < 0) {
                    d[t] = (byte) (d[s] + 1);
                    q[tail++] = t;
                }
            }
        }
        return d;
    }

    // ---- continuous fields -----------------------------------------------------------

    /**
     * A continuous 2-D obstacle field in the unit square.
     *
     * @param discs obstacle discs {@code [k][3]} as {@code (x, y, radius)}
     * @param boxes obstacle rectangles {@code [k][4]} as {@code (x1, y1, x2, y2)}
     * @param start a start point
     * @param goal a goal point
     */
    public record Field(double[][] discs, double[][] boxes, double[] start, double[] goal) {

        /**
         * @param x a point
         * @param y a point
         * @return whether the point is inside the unit square and outside every obstacle
         */
        public boolean free(double x, double y) {
            if (x < 0 || y < 0 || x > 1 || y > 1) {
                return false;
            }
            for (double[] d : discs) {
                double dx = x - d[0];
                double dy = y - d[1];
                if (dx * dx + dy * dy < d[2] * d[2]) {
                    return false;
                }
            }
            for (double[] b : boxes) {
                if (x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3]) {
                    return false;
                }
            }
            return true;
        }

        /**
         * Rasterises the field.
         *
         * @param cells the grid side
         * @return the grid, a cell blocked if its centre is not free
         */
        public Grid raster(int cells) {
            boolean[] b = new boolean[cells * cells];
            for (int y = 0; y < cells; y++) {
                for (int x = 0; x < cells; x++) {
                    b[y * cells + x] = !free((x + 0.5) / cells, (y + 0.5) / cells);
                }
            }
            return new Grid(cells, cells, b);
        }
    }

    /**
     * A field with a vertical wall at {@code x = 0.5} pierced by one narrow gap, random discs
     * on both sides, and start and goal on opposite sides.
     *
     * @param gap the gap width, such as 0.04
     * @param discs the number of random discs
     * @param random the random source
     * @return the field
     */
    public static Field narrowPassage(double gap, int discs, SplittableRandom random) {
        double gy = 0.15 + 0.7 * random.nextDouble();
        double[][] boxes = {{0.48, 0, 0.52, gy - gap / 2}, {0.48, gy + gap / 2, 0.52, 1}};
        double[] start = {0.05 + 0.1 * random.nextDouble(), 0.1 + 0.8 * random.nextDouble()};
        double[] goal = {0.85 + 0.1 * random.nextDouble(), 0.1 + 0.8 * random.nextDouble()};
        List<double[]> ds = new ArrayList<>();
        while (ds.size() < discs) {
            double[] d = {0.1 + 0.8 * random.nextDouble(), random.nextDouble(), 0.03 + 0.05 * random.nextDouble()};
            double ds1 = Math.hypot(d[0] - start[0], d[1] - start[1]);
            double ds2 = Math.hypot(d[0] - goal[0], d[1] - goal[1]);
            double dg = Math.hypot(d[0] - 0.5, d[1] - gy);
            if (ds1 > d[2] + 0.03 && ds2 > d[2] + 0.03 && dg > d[2] + 0.08) {
                ds.add(d);
            }
        }
        return new Field(ds.toArray(new double[0][]), boxes, start, goal);
    }

    /**
     * A field of random discs only.
     *
     * @param discs the number of discs
     * @param random the random source
     * @return the field, with start bottom-left and goal top-right
     */
    public static Field clutter(int discs, SplittableRandom random) {
        double[] start = {0.05, 0.05};
        double[] goal = {0.95, 0.95};
        List<double[]> ds = new ArrayList<>();
        while (ds.size() < discs) {
            double[] d = {random.nextDouble(), random.nextDouble(), 0.03 + 0.07 * random.nextDouble()};
            if (Math.hypot(d[0] - start[0], d[1] - start[1]) > d[2] + 0.03
                    && Math.hypot(d[0] - goal[0], d[1] - goal[1]) > d[2] + 0.03) {
                ds.add(d);
            }
        }
        return new Field(ds.toArray(new double[0][]), new double[0][], start, goal);
    }
}
