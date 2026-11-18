package io.github.zzuegg.brainiac.experimental.data;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Small tables of names and numbers with questions that need a lookup, a comparison, a count
 * or a sum, for table question answering. A table has a {@code NAME} column and 2 to 4
 * numeric columns of values 0 to 9; questions come from four templates:
 *
 * <pre>
 * LOOKUP   WHAT AGE OF ANN          -> the cell            (select one cell, no aggregation)
 * COMPARE  WHO MOST PTS             -> the name of the max (select one name cell)
 * COUNT    HOW MANY PTS ABOVE 4     -> a count             (select the matching cells, COUNT)
 * SUM      TOTAL WINS               -> a sum               (select the column, SUM)
 * </pre>
 *
 * <p>Every question carries its answer and the gold cell selection and aggregation, so a
 * weakly supervised model (trained on the answer only) can still be diagnosed. Names are
 * distinct within a table, and the maximum of a compared column is unique.
 */
public final class ToyTables {

    /** The largest cell value. */
    public static final int MAX_VALUE = 9;
    /** The entity names. */
    public static final String[] NAMES = {"ANN", "BOB", "EVE", "TOM", "SUE", "MAX", "IDA", "LEO", "KIM", "RAY"};
    /** The numeric column headers. */
    public static final String[] COLUMNS = {"AGE", "PTS", "WINS", "GOALS"};
    /** The question words. */
    public static final String[] WORDS = {"WHAT", "OF", "WHO", "MOST", "HOW", "MANY", "ABOVE", "TOTAL", "NAME"};

    private ToyTables() {
    }

    /** The question templates. */
    public enum Kind {
        /** The value of a column for a named row. */
        LOOKUP,
        /** The name of the row with the largest value of a column. */
        COMPARE,
        /** The number of rows whose value of a column exceeds a threshold. */
        COUNT,
        /** The sum of a column. */
        SUM
    }

    /** How selected cells are combined into the answer. */
    public enum Aggregation {
        /** The selected cell itself. */
        NONE,
        /** The number of selected cells. */
        COUNT,
        /** The sum of the selected cells. */
        SUM,
        /** The mean of the selected cells. */
        AVERAGE
    }

    /**
     * A table.
     *
     * @param header the column names; column 0 is {@code NAME}
     * @param cells {@code [rows][columns]} cell texts
     */
    public record Table(String[] header, String[][] cells) {

        /** @return the rows */
        public int rows() {
            return cells.length;
        }

        /** @return the columns */
        public int columns() {
            return header.length;
        }

        /**
         * The numeric value of a cell.
         *
         * @param row the row
         * @param column a numeric column (not 0)
         * @return its value
         */
        public int value(int row, int column) {
            return Integer.parseInt(cells[row][column]);
        }

        /**
         * The rank of a cell's value within its column (0 = smallest; ties share a rank), as
         * the rank embedding of TAPAS uses.
         *
         * @param row the row
         * @param column the column; 0 for names gives 0
         * @return the dense rank
         */
        public int rank(int row, int column) {
            if (column == 0) {
                return 0;
            }
            int v = value(row, column);
            boolean[] seen = new boolean[MAX_VALUE + 1];
            for (int r = 0; r < rows(); r++) {
                seen[value(r, column)] = true;
            }
            int rank = 0;
            for (int x = 0; x < v; x++) {
                rank += seen[x] ? 1 : 0;
            }
            return rank;
        }
    }

    /**
     * A question.
     *
     * @param kind its template
     * @param words its words
     * @param answer the answer as text (a name or a number)
     * @param numeric whether the answer is a number
     * @param gold {@code [rows][columns]} the cells to select
     * @param aggregation the gold aggregation
     */
    public record Question(Kind kind, List<String> words, String answer, boolean numeric, boolean[][] gold,
            Aggregation aggregation) {

        /** @return the answer as a number (only for numeric answers) */
        public double value() {
            return Double.parseDouble(answer);
        }
    }

    /**
     * A random table.
     *
     * @param rows rows, 2 to {@link #NAMES} length
     * @param columns columns including the name column, 2 to 5
     * @param random the source
     * @return the table
     */
    public static Table table(int rows, int columns, SplittableRandom random) {
        if (rows < 2 || rows > NAMES.length || columns < 2 || columns > COLUMNS.length + 1) {
            throw new IllegalArgumentException("bad table size");
        }
        int[] names = random.ints(0, NAMES.length).distinct().limit(rows).toArray();
        int[] cols = random.ints(0, COLUMNS.length).distinct().limit(columns - 1).toArray();
        String[] header = new String[columns];
        header[0] = "NAME";
        for (int c = 1; c < columns; c++) {
            header[c] = COLUMNS[cols[c - 1]];
        }
        String[][] cells = new String[rows][columns];
        for (int r = 0; r < rows; r++) {
            cells[r][0] = NAMES[names[r]];
            for (int c = 1; c < columns; c++) {
                cells[r][c] = Integer.toString(random.nextInt(MAX_VALUE + 1));
            }
        }
        return new Table(header, cells);
    }

    /**
     * A random question of a kind about a table. For {@link Kind#COMPARE} a column with a
     * unique maximum is chosen, and if none exists one cell is raised to make it unique.
     *
     * @param table the table (modified only to break a tie for COMPARE)
     * @param kind the template
     * @param random the source
     * @return the question
     */
    public static Question question(Table table, Kind kind, SplittableRandom random) {
        int rows = table.rows();
        int cols = table.columns();
        int col = 1 + random.nextInt(cols - 1);
        boolean[][] gold = new boolean[rows][cols];
        switch (kind) {
            case LOOKUP -> {
                int r = random.nextInt(rows);
                gold[r][col] = true;
                return new Question(kind, List.of("WHAT", table.header()[col], "OF", table.cells()[r][0]),
                        table.cells()[r][col], true, gold, Aggregation.NONE);
            }
            case COMPARE -> {
                int best = 0;
                for (int r = 1; r < rows; r++) {
                    best = table.value(r, col) > table.value(best, col) ? r : best;
                }
                int max = table.value(best, col);
                for (int r = 0; r < rows; r++) {
                    if (r != best && table.value(r, col) == max) {
                        if (max < MAX_VALUE) {
                            table.cells()[best][col] = Integer.toString(max + 1);
                        } else {
                            table.cells()[r][col] = Integer.toString(max - 1);
                        }
                    }
                }
                gold[best][0] = true;
                return new Question(kind, List.of("WHO", "MOST", table.header()[col]), table.cells()[best][0], false,
                        gold, Aggregation.NONE);
            }
            case COUNT -> {
                int threshold = 2 + random.nextInt(6);
                int count = 0;
                for (int r = 0; r < rows; r++) {
                    if (table.value(r, col) > threshold) {
                        gold[r][col] = true;
                        count++;
                    }
                }
                return new Question(kind, List.of("HOW", "MANY", table.header()[col], "ABOVE",
                        Integer.toString(threshold)), Integer.toString(count), true, gold, Aggregation.COUNT);
            }
            default -> {
                int sum = 0;
                for (int r = 0; r < rows; r++) {
                    gold[r][col] = true;
                    sum += table.value(r, col);
                }
                return new Question(kind, List.of("TOTAL", table.header()[col]), Integer.toString(sum), true, gold,
                        Aggregation.SUM);
            }
        }
    }

    /**
     * Every word that can appear in a table, a question or an answer: the question words,
     * {@code NAME}, the column headers, the entity names and the numbers 0 to
     * {@code maxNumber}.
     *
     * @param maxNumber the largest number word (sums of 8 rows reach 72)
     * @return the words, in a fixed order
     */
    public static List<String> vocabulary(int maxNumber) {
        List<String> v = new ArrayList<>(Arrays.asList(WORDS));
        v.addAll(Arrays.asList(COLUMNS));
        v.addAll(Arrays.asList(NAMES));
        for (int i = 0; i <= maxNumber; i++) {
            v.add(Integer.toString(i));
        }
        return v;
    }
}
