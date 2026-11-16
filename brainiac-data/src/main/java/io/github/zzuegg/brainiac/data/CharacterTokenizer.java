package io.github.zzuegg.brainiac.data;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.TreeSet;

/**
 * One token per character. The vocabulary is the characters of the training text, sorted;
 * a character it has not seen becomes the last token, which decodes to {@code U+FFFD}.
 */
public final class CharacterTokenizer implements Tokenizer {

    private final String characters;
    private final Map<Character, Integer> ids = new HashMap<>();

    /**
     * Creates a tokenizer for given characters.
     *
     * @param characters the vocabulary; a character's id is its index here
     */
    public CharacterTokenizer(String characters) {
        this.characters = characters;
        for (int i = 0; i < characters.length(); i++) {
            if (ids.put(characters.charAt(i), i) != null) {
                throw new IllegalArgumentException("character '" + characters.charAt(i) + "' appears twice");
            }
        }
    }

    /**
     * Collects the characters of a text.
     *
     * @param text the text
     * @return the tokenizer
     */
    public static CharacterTokenizer train(String text) {
        TreeSet<Character> chars = new TreeSet<>();
        for (int i = 0; i < text.length(); i++) {
            chars.add(text.charAt(i));
        }
        StringBuilder b = new StringBuilder(chars.size());
        chars.forEach(b::append);
        return new CharacterTokenizer(b.toString());
    }

    /** @return the known characters, in id order */
    public String characters() {
        return characters;
    }

    @Override
    public int[] encode(String text) {
        int[] out = new int[text.length()];
        for (int i = 0; i < out.length; i++) {
            out[i] = ids.getOrDefault(text.charAt(i), characters.length());
        }
        return out;
    }

    @Override
    public String decode(int... tokens) {
        StringBuilder b = new StringBuilder(tokens.length);
        for (int id : tokens) {
            if (id < 0 || id > characters.length()) {
                throw new IllegalArgumentException("token " + id + " is outside [0, " + size() + ")");
            }
            b.append(id == characters.length() ? '�' : characters.charAt(id));
        }
        return b.toString();
    }

    /** @return the known characters plus one token for unknown ones */
    @Override
    public int size() {
        return characters.length() + 1;
    }

    @Override
    public String kind() {
        return "char";
    }

    @Override
    public void save(Path path) throws IOException {
        List<String> lines = new ArrayList<>(characters.length());
        for (int i = 0; i < characters.length(); i++) {
            lines.add(Integer.toString(characters.charAt(i)));
        }
        Pretokenizer.write(path, kind(), lines);
    }

    static CharacterTokenizer read(List<String> lines) {
        StringBuilder b = new StringBuilder(lines.size());
        for (String line : lines) {
            b.append((char) Integer.parseInt(line.trim()));
        }
        return new CharacterTokenizer(b.toString());
    }

    @Override
    public String toString() {
        return "CharacterTokenizer(" + size() + " tokens)";
    }
}
