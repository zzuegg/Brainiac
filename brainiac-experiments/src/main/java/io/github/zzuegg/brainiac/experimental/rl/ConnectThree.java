package io.github.zzuegg.brainiac.experimental.rl;

import java.util.ArrayList;
import java.util.List;

/**
 * Connect-Three: a 4-column, 4-row board with gravity (a move picks a column, the piece falls
 * to the lowest empty cell); three in a row horizontally, vertically or diagonally wins.
 */
public final class ConnectThree implements BoardGame {

    private static final int COLS = 4;
    private static final int ROWS = 4;

    private final int[] board;   // row 0 is the bottom
    private final int toMove;

    private ConnectThree(int[] board, int toMove) {
        this.board = board;
        this.toMove = toMove;
    }

    /** @return the empty board, player 1 to move */
    public static ConnectThree start() {
        return new ConnectThree(new int[COLS * ROWS], 1);
    }

    @Override
    public int toMove() {
        return toMove;
    }

    @Override
    public List<Integer> legalMoves() {
        List<Integer> m = new ArrayList<>();
        if (winner() == 0) {
            for (int c = 0; c < COLS; c++) {
                if (board[(ROWS - 1) * COLS + c] == 0) {
                    m.add(c);
                }
            }
        }
        return m;
    }

    @Override
    public ConnectThree play(int column) {
        int[] b = board.clone();
        for (int r = 0; r < ROWS; r++) {
            if (b[r * COLS + column] == 0) {
                b[r * COLS + column] = toMove;
                return new ConnectThree(b, 3 - toMove);
            }
        }
        throw new IllegalArgumentException("column " + column + " is full");
    }

    private int at(int r, int c) {
        return r < 0 || r >= ROWS || c < 0 || c >= COLS ? 0 : board[r * COLS + c];
    }

    @Override
    public int winner() {
        int[][] dirs = {{0, 1}, {1, 0}, {1, 1}, {1, -1}};
        for (int r = 0; r < ROWS; r++) {
            for (int c = 0; c < COLS; c++) {
                int p = at(r, c);
                if (p == 0) {
                    continue;
                }
                for (int[] d : dirs) {
                    if (at(r + d[0], c + d[1]) == p && at(r + 2 * d[0], c + 2 * d[1]) == p) {
                        return p;
                    }
                }
            }
        }
        for (int v : board) {
            if (v == 0) {
                return 0;
            }
        }
        return -1;
    }

    @Override
    public int moveCount() {
        return COLS;
    }

    @Override
    public float[] features(int player) {
        float[] f = new float[2 * board.length];
        for (int i = 0; i < board.length; i++) {
            if (board[i] == player) {
                f[2 * i] = 1;
            } else if (board[i] != 0) {
                f[2 * i + 1] = 1;
            }
        }
        return f;
    }

    @Override
    public String key() {
        StringBuilder b = new StringBuilder(board.length + 1);
        for (int v : board) {
            b.append(v);
        }
        return b.append(toMove).toString();
    }
}
