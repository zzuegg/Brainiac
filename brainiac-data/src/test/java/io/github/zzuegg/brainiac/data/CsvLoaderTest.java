package io.github.zzuegg.brainiac.data;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class CsvLoaderTest {

    @TempDir
    Path dir;

    private Path write(String content) throws IOException {
        Path f = dir.resolve("data.csv");
        Files.writeString(f, content);
        return f;
    }

    @Test
    void loadsFeaturesAndOneHotLabels() throws IOException {
        Path f = write("""
                a,b,kind
                1,2,cat
                3,4,dog

                5,6,cat
                """);
        Dataset d = Dataset.csv(f).features("b", "a").label("kind").oneHot().load();
        assertEquals(3, d.size());
        assertEquals(List.of("cat", "dog"), d.classes());
        assertArrayEquals(new float[] {2, 1, 4, 3, 6, 5}, d.features().toArray());
        assertArrayEquals(new float[] {1, 0, 0, 1, 1, 0}, d.targets().toArray());
    }

    @Test
    void loadsNumericTargetsWithOtherDelimiterAndQuotes() throws IOException {
        Path f = write("x;\"y\";z\n1;2;3\n4;\"5\";6\n");
        Dataset d = Dataset.csv(f).delimiter(';').features("x").targets("y", "z").load();
        assertArrayEquals(new float[] {2, 3, 5, 6}, d.targets().toArray());
        assertTrue(d.classes().isEmpty());
    }

    @Test
    void normalizesFeatures() throws IOException {
        Path f = write("x,y\n1,0\n3,1\n");
        Dataset d = Dataset.csv(f).features("x").label("y").normalize().load();
        assertArrayEquals(new float[] {-1, 1}, d.features().toArray());
        assertArrayEquals(new float[] {0, 1}, d.targets().toArray());
    }

    @Test
    void explainsProblems() throws IOException {
        Path f = write("x,y\n1,0\nhello,1\n");
        IllegalArgumentException notNumber = assertThrows(IllegalArgumentException.class,
                () -> Dataset.csv(f).features("x").label("y").load());
        assertTrue(notNumber.getMessage().endsWith("line 3, column 'x': 'hello' is not a number"), notNumber.getMessage());
        IllegalArgumentException missing = assertThrows(IllegalArgumentException.class,
                () -> Dataset.csv(f).features("nope").label("y").load());
        assertTrue(missing.getMessage().contains("column 'nope' not found"), missing.getMessage());
        assertThrows(IllegalStateException.class, () -> Dataset.csv(f).label("y").load());
        assertThrows(IllegalStateException.class, () -> Dataset.csv(f).features("x").load());
        assertThrows(IllegalStateException.class, () -> Dataset.csv(f).oneHot());
        assertThrows(UncheckedIOException.class,
                () -> Dataset.csv(dir.resolve("missing.csv")).features("x").label("y").load());
    }
}
