package io.github.zzuegg.brainiac.data;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** Tokenizers defined outside the module: pretokenizing, saving and loading. */
class TokenizerExtensionTest {

    /** Maps each lower-case letter to its position, everything else to 26. */
    record Letters(int offset) implements Tokenizer {
        @Override
        public int[] encode(String text) {
            return text.chars().map(c -> c >= 'a' && c <= 'z' ? (c - 'a' + offset) % 26 : 26).toArray();
        }

        @Override
        public String decode(int... ids) {
            StringBuilder b = new StringBuilder();
            for (int id : ids) {
                b.append(id == 26 ? '?' : (char) ('a' + (id - offset + 26) % 26));
            }
            return b.toString();
        }

        @Override
        public int size() {
            return 27;
        }

        @Override
        public String kind() {
            return "letters";
        }

        @Override
        public void save(Path path) throws IOException {
            Tokenizer.write(path, kind(), List.of(Integer.toString(offset)));
        }
    }

    @Test
    void pretokenizeSplitsWordsAndJoinsBack() {
        List<String> chunks = Tokenizer.pretokenize("Hello, world 42!");
        assertEquals(List.of("Hello", ",", " world", " 42", "!"), chunks);
        assertEquals("Hello, world 42!", String.join("", chunks));
    }

    @Test
    void registeredKindsLoad(@TempDir Path dir) throws IOException {
        Tokenizer.register("letters", lines -> new Letters(Integer.parseInt(lines.get(0))));
        Path file = dir.resolve("t.txt");
        new Letters(3).save(file);
        Tokenizer back = Tokenizer.load(file);
        assertEquals(new Letters(3), back);
        assertArrayEquals(new Letters(3).encode("abc z"), back.encode("abc z"));
        assertThrows(IllegalArgumentException.class, () -> Tokenizer.register("bpe", lines -> null));
        Files.writeString(file, "brainiac-tokenizer nothing 0\n");
        assertThrows(IOException.class, () -> Tokenizer.load(file));
    }
}
