package io.github.zzuegg.brainiac.io;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * A small JSON reader and writer, enough for weight headers, tokenizer files, model
 * configs and chat templates.
 *
 * <p>Objects become {@code Map<String, Object>} (keys in file order), arrays {@code
 * List<Object>}, strings {@code String}, {@code true}/{@code false} {@code Boolean},
 * {@code null} {@code null}, and numbers {@code Long} when they are whole and fit, else
 * {@code Double}.
 *
 * <pre>{@code
 * Map<String, Object> config = Json.object(Json.parse(Files.readString(path)));
 * int layers = Json.integer(config, "num_hidden_layers");
 * }</pre>
 */
public final class Json {

    private final String text;
    private int at;

    private Json(String text) {
        this.text = text;
    }

    /**
     * Parses one JSON value.
     *
     * @param text the JSON text
     * @return the value
     * @throws IllegalArgumentException if the text is not valid JSON
     */
    public static Object parse(String text) {
        Json p = new Json(text);
        p.space();
        Object v = p.value();
        p.space();
        if (p.at != text.length()) {
            throw p.error("trailing characters");
        }
        return v;
    }

    /**
     * Reads and parses a UTF-8 JSON file.
     *
     * @param file the file
     * @return the value
     * @throws IOException if the file cannot be read
     */
    public static Object read(Path file) throws IOException {
        return parse(Files.readString(file, StandardCharsets.UTF_8));
    }

    /**
     * Casts a parsed value to an object.
     *
     * @param value a parsed value
     * @return it as a map
     * @throws IllegalArgumentException if it is not an object
     */
    @SuppressWarnings("unchecked")
    public static Map<String, Object> object(Object value) {
        if (value instanceof Map<?, ?> m) {
            return (Map<String, Object>) m;
        }
        throw new IllegalArgumentException("expected a JSON object, got " + describe(value));
    }

    /**
     * Casts a parsed value to an array.
     *
     * @param value a parsed value
     * @return it as a list
     * @throws IllegalArgumentException if it is not an array
     */
    @SuppressWarnings("unchecked")
    public static List<Object> array(Object value) {
        if (value instanceof List<?> l) {
            return (List<Object>) l;
        }
        throw new IllegalArgumentException("expected a JSON array, got " + describe(value));
    }

    /**
     * Reads a whole number from an object.
     *
     * @param object the object
     * @param key the key
     * @return the number
     * @throws IllegalArgumentException if the key is missing or not a whole number
     */
    public static long integer(Map<String, Object> object, String key) {
        if (object.get(key) instanceof Number n && n.doubleValue() == Math.rint(n.doubleValue())) {
            return n.longValue();
        }
        throw new IllegalArgumentException("expected a whole number at \"" + key + "\", got "
                + describe(object.get(key)));
    }

    /**
     * Reads a whole number from an object, or a default when the key is missing or null.
     *
     * @param object the object
     * @param key the key
     * @param otherwise the default
     * @return the number
     */
    public static long integer(Map<String, Object> object, String key, long otherwise) {
        return object.get(key) == null ? otherwise : integer(object, key);
    }

    /**
     * Reads a number from an object, or a default when the key is missing or null.
     *
     * @param object the object
     * @param key the key
     * @param otherwise the default
     * @return the number
     */
    public static double number(Map<String, Object> object, String key, double otherwise) {
        Object v = object.get(key);
        if (v == null) {
            return otherwise;
        }
        if (v instanceof Number n) {
            return n.doubleValue();
        }
        throw new IllegalArgumentException("expected a number at \"" + key + "\", got " + describe(v));
    }

    /**
     * Reads a string from an object, or a default when the key is missing or null.
     *
     * @param object the object
     * @param key the key
     * @param otherwise the default
     * @return the string
     */
    public static String string(Map<String, Object> object, String key, String otherwise) {
        Object v = object.get(key);
        if (v == null) {
            return otherwise;
        }
        if (v instanceof String s) {
            return s;
        }
        throw new IllegalArgumentException("expected a string at \"" + key + "\", got " + describe(v));
    }

    private static String describe(Object v) {
        return v == null ? "null" : v instanceof String s ? "\"" + s + "\"" : v.getClass().getSimpleName();
    }

    // ---------------------------------------------------------------- writing

    /**
     * Writes a value as compact JSON. Accepts maps with string keys, lists, arrays of
     * primitives, strings, numbers, booleans and null.
     *
     * @param value the value
     * @return the JSON text
     */
    public static String write(Object value) {
        StringBuilder out = new StringBuilder();
        write(value, out);
        return out.toString();
    }

    private static void write(Object v, StringBuilder out) {
        switch (v) {
            case null -> out.append("null");
            case String s -> quote(s, out);
            case Boolean b -> out.append(b);
            case Double d -> number(d, out);
            case Float f -> number(f, out);
            case Number n -> out.append(n.longValue());
            case Map<?, ?> m -> {
                out.append('{');
                boolean first = true;
                for (Map.Entry<?, ?> e : m.entrySet()) {
                    if (!first) {
                        out.append(',');
                    }
                    first = false;
                    quote(String.valueOf(e.getKey()), out);
                    out.append(':');
                    write(e.getValue(), out);
                }
                out.append('}');
            }
            case Iterable<?> list -> {
                out.append('[');
                boolean first = true;
                for (Object e : list) {
                    if (!first) {
                        out.append(',');
                    }
                    first = false;
                    write(e, out);
                }
                out.append(']');
            }
            case int[] a -> {
                List<Object> l = new ArrayList<>();
                for (int x : a) {
                    l.add((long) x);
                }
                write(l, out);
            }
            case long[] a -> {
                List<Object> l = new ArrayList<>();
                for (long x : a) {
                    l.add(x);
                }
                write(l, out);
            }
            case float[] a -> {
                List<Object> l = new ArrayList<>();
                for (float x : a) {
                    l.add((double) x);
                }
                write(l, out);
            }
            default -> throw new IllegalArgumentException("cannot write " + v.getClass().getName() + " as JSON");
        }
    }

    private static void number(double d, StringBuilder out) {
        if (!Double.isFinite(d)) {
            throw new IllegalArgumentException("JSON has no " + d);
        }
        if (d == Math.rint(d) && Math.abs(d) < 1e15) {
            out.append((long) d);
        } else {
            out.append(d);
        }
    }

    private static void quote(String s, StringBuilder out) {
        out.append('"');
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"' -> out.append("\\\"");
                case '\\' -> out.append("\\\\");
                case '\n' -> out.append("\\n");
                case '\r' -> out.append("\\r");
                case '\t' -> out.append("\\t");
                case '\b' -> out.append("\\b");
                case '\f' -> out.append("\\f");
                default -> {
                    if (c < 0x20) {
                        out.append(String.format("\\u%04x", (int) c));
                    } else {
                        out.append(c);
                    }
                }
            }
        }
        out.append('"');
    }

    // ---------------------------------------------------------------- parsing

    private Object value() {
        if (at >= text.length()) {
            throw error("unexpected end");
        }
        char c = text.charAt(at);
        return switch (c) {
            case '{' -> parseObject();
            case '[' -> parseArray();
            case '"' -> parseString();
            case 't' -> literal("true", Boolean.TRUE);
            case 'f' -> literal("false", Boolean.FALSE);
            case 'n' -> literal("null", null);
            default -> {
                if (c == '-' || (c >= '0' && c <= '9')) {
                    yield parseNumber();
                }
                throw error("unexpected '" + c + "'");
            }
        };
    }

    private Map<String, Object> parseObject() {
        at++;
        Map<String, Object> map = new LinkedHashMap<>();
        space();
        if (peek() == '}') {
            at++;
            return map;
        }
        while (true) {
            space();
            if (peek() != '"') {
                throw error("expected a key");
            }
            String key = parseString();
            space();
            expect(':');
            space();
            map.put(key, value());
            space();
            char c = next();
            if (c == '}') {
                return map;
            }
            if (c != ',') {
                throw error("expected ',' or '}'");
            }
        }
    }

    private List<Object> parseArray() {
        at++;
        List<Object> list = new ArrayList<>();
        space();
        if (peek() == ']') {
            at++;
            return list;
        }
        while (true) {
            space();
            list.add(value());
            space();
            char c = next();
            if (c == ']') {
                return list;
            }
            if (c != ',') {
                throw error("expected ',' or ']'");
            }
        }
    }

    private String parseString() {
        at++;
        StringBuilder sb = null;
        int start = at;
        while (true) {
            if (at >= text.length()) {
                throw error("unterminated string");
            }
            char c = text.charAt(at);
            if (c == '"') {
                String s = sb == null ? text.substring(start, at) : sb.append(text, start, at).toString();
                at++;
                return s;
            }
            if (c == '\\') {
                if (sb == null) {
                    sb = new StringBuilder();
                }
                sb.append(text, start, at);
                at++;
                char e = next();
                switch (e) {
                    case '"' -> sb.append('"');
                    case '\\' -> sb.append('\\');
                    case '/' -> sb.append('/');
                    case 'b' -> sb.append('\b');
                    case 'f' -> sb.append('\f');
                    case 'n' -> sb.append('\n');
                    case 'r' -> sb.append('\r');
                    case 't' -> sb.append('\t');
                    case 'u' -> {
                        if (at + 4 > text.length()) {
                            throw error("bad \\u escape");
                        }
                        sb.append((char) Integer.parseInt(text, at, at + 4, 16));
                        at += 4;
                    }
                    default -> throw error("bad escape \\" + e);
                }
                start = at;
            } else {
                at++;
            }
        }
    }

    private Object parseNumber() {
        int start = at;
        boolean whole = true;
        if (peek() == '-') {
            at++;
        }
        while (at < text.length()) {
            char c = text.charAt(at);
            if (c >= '0' && c <= '9') {
                at++;
            } else if (c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') {
                whole = false;
                at++;
            } else {
                break;
            }
        }
        String s = text.substring(start, at);
        try {
            if (whole) {
                try {
                    return Long.parseLong(s);
                } catch (NumberFormatException tooBig) {
                    return Double.parseDouble(s);
                }
            }
            return Double.parseDouble(s);
        } catch (NumberFormatException e) {
            throw error("bad number " + s);
        }
    }

    private Object literal(String word, Object value) {
        if (!text.startsWith(word, at)) {
            throw error("unexpected '" + text.charAt(at) + "'");
        }
        at += word.length();
        return value;
    }

    private void space() {
        while (at < text.length()) {
            char c = text.charAt(at);
            if (c == ' ' || c == '\n' || c == '\r' || c == '\t') {
                at++;
            } else {
                break;
            }
        }
    }

    private char peek() {
        if (at >= text.length()) {
            throw error("unexpected end");
        }
        return text.charAt(at);
    }

    private char next() {
        char c = peek();
        at++;
        return c;
    }

    private void expect(char c) {
        if (next() != c) {
            throw error("expected '" + c + "'");
        }
    }

    private IllegalArgumentException error(String what) {
        return new IllegalArgumentException("bad JSON at offset " + at + ": " + what);
    }
}
