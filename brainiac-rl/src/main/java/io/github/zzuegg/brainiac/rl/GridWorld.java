package io.github.zzuegg.brainiac.rl;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Walk a grid from the start to the goal. The map is given as rows of characters:
 * <ul>
 *   <li>{@code S} start, {@code G} goal (reward +1, episode ends), {@code .} open floor;</li>
 *   <li>{@code #} wall (bumping into it keeps the agent in place);</li>
 *   <li>{@code H} hole (reward -1, episode ends);</li>
 *   <li>{@code C} cliff (reward -100, back to the start, as in Sutton and Barto's cliff
 *       walk).</li>
 * </ul>
 * Actions: 0 up, 1 right, 2 down, 3 left. Every step also gives {@link #stepReward}. With
 * {@link #slippery(double)} the agent sometimes slides to the side instead. The observation
 * is a one-hot vector over the cells, or the normalized row and column with
 * {@link #coordinates(boolean)}.
 *
 * <pre>{@code
 * GridWorld maze = GridWorld.of(
 *         "S..#....",
 *         ".#.#.##.",
 *         ".#...#..",
 *         ".####.#.",
 *         "......#G");
 * }</pre>
 */
public final class GridWorld implements Environment {

    private static final int[][] MOVES = {{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
    private static final char[] ARROWS = {'^', '>', 'v', '<'};

    private final char[][] map;
    private final int rows;
    private final int cols;
    private final int startRow;
    private final int startCol;
    private float stepReward = -0.01f;
    private double slip;
    private int maxSteps = 200;
    private boolean coordinates;
    private int row;
    private int col;
    private int steps;
    private boolean over = true;
    private SplittableRandom random = new SplittableRandom(0);

    private GridWorld(String... rows) {
        if (rows.length == 0) {
            throw new IllegalArgumentException("a grid needs at least one row");
        }
        this.rows = rows.length;
        this.cols = rows[0].length();
        this.map = new char[this.rows][];
        int sr = -1;
        int sc = -1;
        boolean goal = false;
        for (int r = 0; r < this.rows; r++) {
            if (rows[r].length() != cols) {
                throw new IllegalArgumentException("every row must have " + cols + " cells, row " + r + " has "
                        + rows[r].length());
            }
            map[r] = rows[r].toCharArray();
            for (int c = 0; c < cols; c++) {
                switch (map[r][c]) {
                    case 'S' -> {
                        if (sr >= 0) {
                            throw new IllegalArgumentException("more than one start");
                        }
                        sr = r;
                        sc = c;
                    }
                    case 'G' -> goal = true;
                    case '.', '#', 'H', 'C' -> { }
                    default -> throw new IllegalArgumentException("unknown cell '" + map[r][c] + "' at " + r + ","
                            + c);
                }
            }
        }
        if (sr < 0 || !goal) {
            throw new IllegalArgumentException("a grid needs a start S and at least one goal G");
        }
        startRow = sr;
        startCol = sc;
    }

    /**
     * Builds a grid from rows of characters.
     *
     * @param rows the map, all rows the same length
     * @return the environment
     */
    public static GridWorld of(String... rows) {
        return new GridWorld(rows);
    }

    /**
     * Sutton and Barto's cliff walk: a 4x12 grid with a cliff along the bottom between start
     * and goal; -1 per step, -100 for falling. Q-learning learns the risky path along the
     * edge, SARSA the safer one further up.
     *
     * @return the environment
     */
    public static GridWorld cliff() {
        return of("............",
                  "............",
                  "............",
                  "SCCCCCCCCCCG").stepReward(-1);
    }

    /**
     * A 4x4 frozen lake: holes end the episode; slippery ice makes moves uncertain.
     *
     * @return the environment
     */
    public static GridWorld lake() {
        return of("S...",
                  ".H.H",
                  "...H",
                  "H..G").stepReward(0).slippery(1.0 / 3);
    }

    /**
     * @param reward the reward added on every step, usually small and negative
     * @return this environment
     */
    public GridWorld stepReward(float reward) {
        this.stepReward = reward;
        return this;
    }

    /**
     * @param probability how often a move slides to one side instead (split evenly)
     * @return this environment
     */
    public GridWorld slippery(double probability) {
        if (probability < 0 || probability > 1) {
            throw new IllegalArgumentException("probability must be in [0, 1]");
        }
        this.slip = probability;
        return this;
    }

    /**
     * @param steps the most steps per episode
     * @return this environment
     */
    public GridWorld maxSteps(int steps) {
        this.maxSteps = steps;
        return this;
    }

    /**
     * @param on observe the normalized row and column instead of a one-hot cell
     * @return this environment
     */
    public GridWorld coordinates(boolean on) {
        this.coordinates = on;
        return this;
    }

    /** @return the number of cells */
    public int cells() {
        return rows * cols;
    }

    @Override
    public Space observationSpace() {
        return coordinates ? Space.box(2, 0, 1) : Space.box(rows * cols, 0, 1);
    }

    @Override
    public Space actionSpace() {
        return Space.discrete(4);
    }

    @Override
    public float[] reset(SplittableRandom random) {
        this.random = random;
        row = startRow;
        col = startCol;
        steps = 0;
        over = false;
        return observation();
    }

    private float[] observation() {
        if (coordinates) {
            return new float[] {rows > 1 ? row / (rows - 1f) : 0, cols > 1 ? col / (cols - 1f) : 0};
        }
        float[] x = new float[rows * cols];
        x[row * cols + col] = 1;
        return x;
    }

    @Override
    public Step step(float[] action) {
        if (over) {
            throw new IllegalStateException("the episode is over; call reset");
        }
        int a = Math.round(action[0]);
        if (a < 0 || a > 3) {
            throw new IllegalArgumentException("grid actions are 0 up, 1 right, 2 down, 3 left; got " + action[0]);
        }
        if (slip > 0 && random.nextDouble() < slip) {
            a = (a + (random.nextBoolean() ? 1 : 3)) % 4;
        }
        int r = row + MOVES[a][0];
        int c = col + MOVES[a][1];
        if (r >= 0 && r < rows && c >= 0 && c < cols && map[r][c] != '#') {
            row = r;
            col = c;
        }
        steps++;
        float reward = stepReward;
        boolean terminated = false;
        switch (map[row][col]) {
            case 'G' -> {
                reward += 1;
                terminated = true;
            }
            case 'H' -> {
                reward -= 1;
                terminated = true;
            }
            case 'C' -> {
                reward = -100;
                row = startRow;
                col = startCol;
            }
            default -> { }
        }
        boolean cut = !terminated && steps >= maxSteps;
        over = terminated || cut;
        return new Step(observation(), reward, terminated, cut);
    }

    @Override
    public String render() {
        StringBuilder s = new StringBuilder();
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                s.append(r == row && c == col ? '@' : map[r][c]);
            }
            s.append('\n');
        }
        return s.toString();
    }

    /**
     * Draws the map with an arrow in every open cell for the action a policy picks there.
     *
     * @param agent the agent
     * @return the map with arrows
     */
    public String renderPolicy(Agent agent) {
        StringBuilder s = new StringBuilder();
        boolean wasCoordinates = coordinates;
        int r0 = row;
        int c0 = col;
        try {
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    if (map[r][c] == '.' || map[r][c] == 'S') {
                        row = r;
                        col = c;
                        s.append(ARROWS[Math.round(agent.act(observation(), false)[0])]);
                    } else {
                        s.append(map[r][c]);
                    }
                }
                s.append('\n');
            }
        } finally {
            coordinates = wasCoordinates;
            row = r0;
            col = c0;
        }
        return s.toString();
    }

    /** @return the cells, row by row, as given */
    public List<String> map() {
        List<String> m = new ArrayList<>();
        for (char[] r : map) {
            m.add(new String(r));
        }
        return m;
    }

    @Override
    public String toString() {
        return "GridWorld(" + rows + "x" + cols + (slip > 0 ? ", slippery" : "") + ")";
    }
}
