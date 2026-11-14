package io.github.zzuegg.brainiac.data;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Pattern;

/**
 * Loads a {@link Dataset} from a CSV file whose first row names the columns. Created by
 * {@link Dataset#csv(Path)}.
 *
 * <pre>{@code
 * Dataset iris = Dataset.csv(Path.of("iris.csv"))
 *         .features("sepal_length", "sepal_width", "petal_length", "petal_width")
 *         .label("species").oneHot()
 *         .load();
 * }</pre>
 *
 * <p>Values are separated by commas unless {@link #delimiter(char)} says otherwise.
 * Surrounding whitespace and double quotes are removed; quoted values may not contain the
 * delimiter. Empty lines are skipped.
 */
public final class CsvLoader {

    private final Path file;
    private char delimiter = ',';
    private List<String> featureColumns = List.of();
    private List<String> targetColumns = List.of();
    private String labelColumn;
    private boolean oneHot;
    private boolean normalize;

    CsvLoader(Path file) {
        this.file = Objects.requireNonNull(file, "file");
    }

    /**
     * Sets the column separator.
     *
     * @param delimiter for example {@code ';'} or {@code '\t'}
     * @return this loader
     */
    public CsvLoader delimiter(char delimiter) {
        this.delimiter = delimiter;
        return this;
    }

    /**
     * Chooses the numeric input columns.
     *
     * @param columns the column names, in the order the model sees them
     * @return this loader
     */
    public CsvLoader features(String... columns) {
        this.featureColumns = List.of(columns);
        return this;
    }

    /**
     * Chooses numeric target columns, for predicting numbers.
     *
     * @param columns the column names
     * @return this loader
     */
    public CsvLoader targets(String... columns) {
        this.targetColumns = List.of(columns);
        return this;
    }

    /**
     * Chooses a single target column. Numeric unless followed by {@link #oneHot()}.
     *
     * @param column the column name
     * @return this loader
     */
    public CsvLoader label(String column) {
        this.labelColumn = Objects.requireNonNull(column, "column");
        return this;
    }

    /**
     * Treats the label column as class names: each distinct value becomes one target
     * column holding 1 for that class and 0 otherwise, in order of first appearance. The
     * names are available from {@link Dataset#classes()}.
     *
     * @return this loader
     */
    public CsvLoader oneHot() {
        if (labelColumn == null) {
            throw new IllegalStateException("call label(column) before oneHot()");
        }
        this.oneHot = true;
        return this;
    }

    /**
     * Standardizes features to mean 0 and standard deviation 1 over the whole file. To
     * keep test data unseen, use {@link Standardizer} on a split instead.
     *
     * @return this loader
     */
    public CsvLoader normalize() {
        this.normalize = true;
        return this;
    }

    /**
     * Reads the file.
     *
     * @return the dataset
     * @throws UncheckedIOException if the file cannot be read
     * @throws IllegalArgumentException if a column is missing or a value is not a number
     */
    public Dataset load() {
        if (featureColumns.isEmpty()) {
            throw new IllegalStateException("choose feature columns with features(...)");
        }
        if (labelColumn == null && targetColumns.isEmpty()) {
            throw new IllegalStateException("choose a target with label(column) or targets(...)");
        }
        if (labelColumn != null && !targetColumns.isEmpty()) {
            throw new IllegalStateException("use either label(column) or targets(...), not both");
        }
        List<String> lines;
        try {
            lines = Files.readAllLines(file, StandardCharsets.UTF_8);
        } catch (IOException e) {
            throw new UncheckedIOException("cannot read " + file, e);
        }
        lines.removeIf(String::isBlank);
        if (lines.size() < 2) {
            throw new IllegalArgumentException(file + " needs a header row and at least one data row");
        }
        List<String> header = Arrays.asList(split(lines.get(0)));
        int[] featureIdx = indices(header, featureColumns);
        List<String> targetNames = labelColumn != null ? List.of(labelColumn) : targetColumns;
        int[] targetIdx = indices(header, targetNames);

        int n = lines.size() - 1;
        int f = featureIdx.length;
        float[] features = new float[n * f];
        String[][] rawTargets = new String[n][];
        for (int r = 0; r < n; r++) {
            int lineNo = r + 2;
            String[] cells = split(lines.get(r + 1));
            if (cells.length != header.size()) {
                throw new IllegalArgumentException(file + " line " + lineNo + " has " + cells.length
                        + " values, header has " + header.size());
            }
            for (int j = 0; j < f; j++) {
                features[r * f + j] = number(cells[featureIdx[j]], lineNo, featureColumns.get(j));
            }
            rawTargets[r] = new String[targetIdx.length];
            for (int j = 0; j < targetIdx.length; j++) {
                rawTargets[r][j] = cells[targetIdx[j]];
            }
        }

        float[] targets;
        int targetCount;
        List<String> classes = List.of();
        if (oneHot) {
            Map<String, Integer> classIndex = new LinkedHashMap<>();
            for (String[] row : rawTargets) {
                classIndex.putIfAbsent(row[0], classIndex.size());
            }
            targetCount = classIndex.size();
            targets = new float[n * targetCount];
            for (int r = 0; r < n; r++) {
                targets[r * targetCount + classIndex.get(rawTargets[r][0])] = 1f;
            }
            classes = new ArrayList<>(classIndex.keySet());
        } else {
            targetCount = targetIdx.length;
            targets = new float[n * targetCount];
            for (int r = 0; r < n; r++) {
                for (int j = 0; j < targetCount; j++) {
                    targets[r * targetCount + j] = number(rawTargets[r][j], r + 2, targetNames.get(j));
                }
            }
        }
        Dataset data = Dataset.create(features, targets, n, f, targetCount, classes);
        return normalize ? data.standardize(Standardizer.fit(data)) : data;
    }

    private String[] split(String line) {
        String[] cells = line.split(Pattern.quote(String.valueOf(delimiter)), -1);
        for (int i = 0; i < cells.length; i++) {
            String c = cells[i].strip();
            if (c.length() >= 2 && c.startsWith("\"") && c.endsWith("\"")) {
                c = c.substring(1, c.length() - 1);
            }
            cells[i] = c;
        }
        return cells;
    }

    private int[] indices(List<String> header, List<String> columns) {
        int[] idx = new int[columns.size()];
        for (int i = 0; i < idx.length; i++) {
            idx[i] = header.indexOf(columns.get(i));
            if (idx[i] < 0) {
                throw new IllegalArgumentException("column '" + columns.get(i) + "' not found in " + file
                        + "; columns are " + header);
            }
        }
        return idx;
    }

    private float number(String cell, int line, String column) {
        try {
            return Float.parseFloat(cell);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(file + " line " + line + ", column '" + column
                    + "': '" + cell + "' is not a number", e);
        }
    }
}
