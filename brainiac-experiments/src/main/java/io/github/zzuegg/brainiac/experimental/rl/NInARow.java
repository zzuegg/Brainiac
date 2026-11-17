package io.github.zzuegg.brainiac.experimental.rl;

import java.util.Arrays;

/**
 * The family of "get {@code k} in a row" games on a {@code rows x cols} board: tic-tac-toe
 * (3x3, 3 in a row, place anywhere), gomoku-like variants, and with gravity Connect Four
 * style games where a move picks a column and the piece drops to the lowest free cell
 * (for example Connect-3 on a 4x5 board). Positions are immutable.
 */
public final class NInARow implements TwoPlayerGame {

    private final int rows;
    private final int cols;
    private final int k;
    private final boolean gravity;
    private final byte[] cells;
    private final int toMove;
    private final int winner;
    private final int filled;

    private NInARow(int rows, int cols, int k, boolean gravity, byte[] cells, int toMove, int winner, int filled) {
        this.rows = rows;
        this.cols = cols;
        this.k = k;
        this.gravity = gravity;
        this.cells = cells;
        this.toMove = toMove;
        this.winner = winner;
        this.filled = filled;
    }

    /**
     * An empty board.
     *
     * @param rows board rows
     * @param cols board columns
     * @param k how many in a row win
     * @param gravity whether moves pick a column and pieces drop
     * @return the opening position, player {@code +1} to move
     */
    public static NInARow of(int rows, int cols, int k, boolean gravity) {
        if (k > Math.max(rows, cols) || k < 2) {
            throw new IllegalArgumentException("k must fit on the board");
        }
        return new NInARow(rows, cols, k, gravity, new byte[rows * cols], 1, 0, 0);
    }

    /** @return an empty tic-tac-toe board */
    public static NInARow ticTacToe() {
        return of(3, 3, 3, false);
    }

    /** @return an empty Connect-3 board: 4 rows, 5 columns, gravity */
    public static NInARow connectThree() {
        return of(4, 5, 3, true);
    }

    @Override
    public int actions() {
        return gravity ? cols : rows * cols;
    }

    @Override
    public boolean[] legal() {
        boolean[] l = new boolean[actions()];
        if (over()) {
            return l;
        }
        for (int a = 0; a < l.length; a++) {
            l[a] = cells[a] == 0; // with gravity, a column is open while its top cell is empty
        }
        return l;
    }

    @Override
    public NInARow play(int action) {
        if (over() || action < 0 || action >= actions()) {
            throw new IllegalArgumentException("illegal move " + action);
        }
        int cell = action;
        if (gravity) {
            cell = -1;
            for (int r = rows - 1; r >= 0; r--) {
                if (cells[r * cols + action] == 0) {
                    cell = r * cols + action;
                    break;
                }
            }
        }
        if (cell < 0 || cells[cell] != 0) {
            throw new IllegalArgumentException("illegal move " + action);
        }
        byte[] next = cells.clone();
        next[cell] = (byte) toMove;
        int w = wins(next, cell) ? toMove : 0;
        return new NInARow(rows, cols, k, gravity, next, -toMove, w, filled + 1);
    }

    private boolean wins(byte[] b, int cell) {
        int r0 = cell / cols;
        int c0 = cell % cols;
        int p = b[cell];
        int[][] dirs = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};
        for (int[] d : dirs) {
            int count = 1;
            for (int s = -1; s <= 1; s += 2) {
                int r = r0 + s * d[0];
                int c = c0 + s * d[1];
                while (r >= 0 && r < rows && c >= 0 && c < cols && b[r * cols + c] == p) {
                    count++;
                    r += s * d[0];
                    c += s * d[1];
                }
            }
            if (count >= k) {
                return true;
            }
        }
        return false;
    }

    @Override
    public boolean over() {
        return winner != 0 || filled == cells.length;
    }

    @Override
    public int winner() {
        return winner;
    }

    @Override
    public int toMove() {
        return toMove;
    }

    /** Two planes, the mover's pieces then the opponent's. */
    @Override
    public float[] encode() {
        float[] x = new float[2 * cells.length];
        for (int i = 0; i < cells.length; i++) {
            if (cells[i] == toMove) {
                x[i] = 1;
            } else if (cells[i] == -toMove) {
                x[cells.length + i] = 1;
            }
        }
        return x;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                int v = cells[r * cols + c];
                sb.append(v == 1 ? 'X' : v == -1 ? 'O' : '.');
            }
            sb.append('\n');
        }
        return sb.toString();
    }

    @Override
    public boolean equals(Object o) {
        return o instanceof NInARow n && n.rows == rows && n.cols == cols && n.k == k && n.gravity == gravity
                && Arrays.equals(n.cells, cells);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(cells) * 31 + rows * 7 + cols;
    }
}
