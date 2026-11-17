package io.github.zzuegg.brainiac.experimental.rl;

import java.util.ArrayList;
import java.util.List;

/** Tic-tac-toe on a 3x3 board; moves are cell indices 0-8, row by row. */
public final class TicTacToe implements BoardGame {

    private static final int[][] LINES = {
        {0, 1, 2}, {3, 4, 5}, {6, 7, 8}, {0, 3, 6}, {1, 4, 7}, {2, 5, 8}, {0, 4, 8}, {2, 4, 6}};

    private final int[] board;
    private final int toMove;

    private TicTacToe(int[] board, int toMove) {
        this.board = board;
        this.toMove = toMove;
    }

    /** @return the empty board, player 1 to move */
    public static TicTacToe start() {
        return new TicTacToe(new int[9], 1);
    }

    @Override
    public int toMove() {
        return toMove;
    }

    @Override
    public List<Integer> legalMoves() {
        List<Integer> m = new ArrayList<>();
        if (winner() == 0) {
            for (int i = 0; i < 9; i++) {
                if (board[i] == 0) {
                    m.add(i);
                }
            }
        }
        return m;
    }

    @Override
    public TicTacToe play(int move) {
        if (board[move] != 0 || winner() != 0) {
            throw new IllegalArgumentException("illegal move " + move);
        }
        int[] b = board.clone();
        b[move] = toMove;
        return new TicTacToe(b, 3 - toMove);
    }

    @Override
    public int winner() {
        for (int[] l : LINES) {
            if (board[l[0]] != 0 && board[l[0]] == board[l[1]] && board[l[1]] == board[l[2]]) {
                return board[l[0]];
            }
        }
        for (int c : board) {
            if (c == 0) {
                return 0;
            }
        }
        return -1;
    }

    @Override
    public int moveCount() {
        return 9;
    }

    @Override
    public float[] features(int player) {
        float[] f = new float[18];
        for (int i = 0; i < 9; i++) {
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
        StringBuilder b = new StringBuilder(10);
        for (int c : board) {
            b.append(c);
        }
        return b.append(toMove).toString();
    }

    @Override
    public String toString() {
        StringBuilder b = new StringBuilder();
        for (int i = 0; i < 9; i++) {
            b.append(".XO".charAt(board[i]));
            if (i % 3 == 2) {
                b.append('\n');
            }
        }
        return b.toString();
    }
}
