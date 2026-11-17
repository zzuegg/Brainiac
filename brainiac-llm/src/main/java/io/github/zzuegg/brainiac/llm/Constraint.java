package io.github.zzuegg.brainiac.llm;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.io.PretrainedTokenizer;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;

/**
 * A language the generated text must belong to, for constrained decoding: at each step the
 * {@link Generator} allows only the tokens whose text keeps the output a prefix of some
 * accepted string, and a stop token only once the output is accepted. Built from a regular
 * expression, a list of choices or a JSON schema; all compile to the same automaton. Any
 * other language plugs in through {@link #of(Supplier)} with its own {@link Matcher}.
 */
public final class Constraint {

    /** Where a partial output stands: immutable, so a search can branch from it. */
    public interface Matcher {

        /**
         * @param c the next character
         * @return the matcher after it, or {@code null} when no accepted string continues so
         */
        Matcher feed(char c);

        /** @return whether the text so far is accepted as a whole */
        boolean accepting();

        /** @return whether some longer text is accepted too */
        boolean canContinue();

        /**
         * @param text the next characters
         * @return the matcher after them, or {@code null}
         */
        default Matcher feed(CharSequence text) {
            Matcher m = this;
            for (int i = 0; i < text.length() && m != null; i++) {
                m = m.feed(text.charAt(i));
            }
            return m;
        }
    }

    // the automaton: CHAR states move on a character set, SPLIT states on nothing
    private static final int CHAR = 0;
    private static final int SPLIT = 1;
    private static final int MATCH = 2;

    private final String pattern;
    private int[] kind = new int[16];
    private int[] out = new int[16];
    private int[] out1 = new int[16];
    private CharSet[] sets = new CharSet[16];
    private int count;
    private final int start;
    /** The start of a custom language, or {@code null} for a compiled pattern. */
    private final Supplier<? extends Matcher> custom;

    private Constraint(String pattern) {
        this.pattern = pattern;
        this.custom = null;
        Node ast = new Parser(pattern).parse();
        int match = add(MATCH, -1, -1, null);
        this.start = compile(ast, match);
    }

    private Constraint(Supplier<? extends Matcher> custom) {
        this.pattern = null;
        this.custom = custom;
        this.start = -1;
    }

    /**
     * A language of one's own, such as a grammar a regular expression cannot express: the
     * generator asks the matcher which characters may follow. Matchers should be immutable,
     * since a search may branch from one.
     *
     * @param start makes the matcher before any text
     * @return the constraint
     */
    public static Constraint of(Supplier<? extends Matcher> start) {
        return new Constraint(java.util.Objects.requireNonNull(start, "start"));
    }

    /**
     * Text in the language of a GBNF grammar, as llama.cpp's {@code --grammar}; see {@link Grammar}.
     *
     * @param gbnf the rules, starting at {@code root}
     * @return the constraint
     */
    public static Constraint grammar(String gbnf) {
        return Grammar.parse(gbnf).constraint();
    }

    /**
     * Text matching a regular expression as a whole. Supported: literals, {@code .}, classes
     * such as {@code [a-z_]} and {@code [^"]}, {@code \d \w \s \D \W \S}, escapes, groups
     * {@code ( )} and {@code (?: )}, {@code |}, and {@code * + ? {m} {m,} {m,n}}.
     *
     * @param pattern the expression
     * @return the constraint
     */
    public static Constraint regex(String pattern) {
        return new Constraint(pattern);
    }

    /**
     * Exactly one of some strings.
     *
     * @param choices the allowed outputs
     * @return the constraint
     */
    public static Constraint choice(String... choices) {
        if (choices.length == 0) {
            throw new IllegalArgumentException("no choices");
        }
        StringBuilder b = new StringBuilder();
        for (String c : choices) {
            b.append(b.isEmpty() ? "" : "|").append(quote(c));
        }
        return new Constraint(b.toString());
    }

    /**
     * JSON that a schema accepts. Supported: {@code type} string, integer, number, boolean,
     * null, array ({@code items}, {@code minItems}, {@code maxItems}) and object
     * ({@code properties} in their order, {@code required}); {@code enum}, {@code const},
     * {@code anyOf}/{@code oneOf}, and a string's {@code minLength}, {@code maxLength} and
     * {@code pattern}. Objects take only their listed properties; a space may follow
     * {@code :} and {@code ,}.
     *
     * @param schema the schema, as JSON text
     * @return the constraint
     */
    public static Constraint jsonSchema(String schema) {
        return jsonSchema(Json.object(Json.parse(schema)));
    }

    /**
     * {@link #jsonSchema(String)} from a parsed schema.
     *
     * @param schema the schema
     * @return the constraint
     */
    public static Constraint jsonSchema(Map<String, Object> schema) {
        return new Constraint(schemaRegex(schema));
    }

    /** @return the regular expression this constraint compiled, or {@code null} for a custom one */
    public String pattern() {
        return pattern;
    }

    /** @return a matcher before any text */
    public Matcher start() {
        if (custom != null) {
            return java.util.Objects.requireNonNull(custom.get(), "the custom constraint's start matcher");
        }
        return new State(this, closure(new int[] {start}));
    }

    /**
     * @param text a whole output
     * @return whether it is accepted
     */
    public boolean matches(String text) {
        Matcher m = start().feed(text);
        return m != null && m.accepting();
    }

    /**
     * Escapes a string for use as a literal inside {@link #regex}.
     *
     * @param literal the text
     * @return the escaped text
     */
    public static String quote(String literal) {
        StringBuilder b = new StringBuilder();
        for (char c : literal.toCharArray()) {
            if ("\\.[]{}()*+?|^$-".indexOf(c) >= 0) {
                b.append('\\');
            }
            b.append(c);
        }
        return b.toString();
    }

    /**
     * The text of every token, sorted so that tokens sharing a prefix are matched once: finds
     * which tokens a {@link Matcher} allows.
     */
    public static final class Vocabulary {

        private final String[] texts;
        private final int[] order;
        private final int[] shared;

        /**
         * @param texts each token's text by id; {@code null} or empty for tokens never allowed
         */
        public Vocabulary(String[] texts) {
            this.texts = texts.clone();
            Integer[] ids = new Integer[texts.length];
            int n = 0;
            for (int t = 0; t < texts.length; t++) {
                if (texts[t] != null && !texts[t].isEmpty()) {
                    ids[n++] = t;
                }
            }
            Integer[] kept = Arrays.copyOf(ids, n);
            Arrays.sort(kept, (a, b) -> texts[a].compareTo(texts[b]));
            order = new int[n];
            shared = new int[n];
            for (int k = 0; k < n; k++) {
                order[k] = kept[k];
                if (k > 0) {
                    String a = texts[order[k - 1]];
                    String b = texts[order[k]];
                    int l = 0;
                    while (l < a.length() && l < b.length() && a.charAt(l) == b.charAt(l)) {
                        l++;
                    }
                    shared[k] = l;
                }
            }
        }

        /**
         * The text of each token as it continues other text: decoded after a reference token
         * so that tokenizers which drop a leading space at the start keep it.
         *
         * @param tokenizer the tokenizer
         * @param size the vocabulary size, at least the tokenizer's
         * @return the vocabulary
         */
        public static Vocabulary of(io.github.zzuegg.brainiac.data.Tokenizer tokenizer, int size) {
            // special tokens have no text here: they end a sequence, they do not write it
            java.util.function.Function<int[], String> decode = tokenizer instanceof PretrainedTokenizer p
                    ? ids -> p.decode(ids, true) : tokenizer::decode;
            int[] ref = tokenizer.encode("a");
            int r = ref[ref.length - 1];
            String base = decode.apply(new int[] {r});
            String[] texts = new String[size];
            for (int t = 0; t < Math.min(size, tokenizer.size()); t++) {
                String both = decode.apply(new int[] {r, t});
                texts[t] = both.startsWith(base) ? both.substring(base.length()) : decode.apply(new int[] {t});
            }
            return new Vocabulary(texts);
        }

        /** @return the number of token ids */
        public int size() {
            return texts.length;
        }

        /**
         * @param id a token
         * @return its text, or {@code null}
         */
        public String text(int id) {
            return texts[id];
        }

        /**
         * @param m where the output stands
         * @return for each token, whether its whole text keeps the output acceptable
         */
        public boolean[] allowed(Matcher m) {
            boolean[] ok = new boolean[texts.length];
            Matcher[] at = new Matcher[16];
            at[0] = m;
            int dead = Integer.MAX_VALUE;   // the shortest prefix known to fail
            for (int k = 0; k < order.length; k++) {
                String t = texts[order[k]];
                int from = shared[k];
                if (from >= dead) {
                    continue;   // shares the failed prefix
                }
                dead = Integer.MAX_VALUE;
                if (at.length <= t.length()) {
                    at = Arrays.copyOf(at, t.length() * 2);
                }
                Matcher cur = at[from];
                int i = from;
                for (; i < t.length() && cur != null; i++) {
                    cur = cur.feed(t.charAt(i));
                    at[i + 1] = cur;
                }
                if (cur == null) {
                    dead = i;   // at[i] is the failure: prefixes up to i - 1 matched
                } else {
                    ok[order[k]] = true;
                }
            }
            return ok;
        }

        /**
         * @param m where the output stands
         * @param token the chosen token
         * @return the matcher after its text, or {@code null}
         */
        public Matcher advance(Matcher m, int token) {
            String t = texts[token];
            return t == null ? null : m.feed(t);
        }
    }

    // ---- matching

    private record State(Constraint c, int[] states) implements Matcher {

        @Override
        public Matcher feed(char ch) {
            int[] next = new int[states.length];
            int n = 0;
            for (int s : states) {
                if (c.kind[s] == CHAR && c.sets[s].contains(ch)) {
                    next[n++] = c.out[s];
                }
            }
            if (n == 0) {
                return null;
            }
            return new State(c, c.closure(Arrays.copyOf(next, n)));
        }

        @Override
        public boolean accepting() {
            for (int s : states) {
                if (c.kind[s] == MATCH) {
                    return true;
                }
            }
            return false;
        }

        @Override
        public boolean canContinue() {
            for (int s : states) {
                if (c.kind[s] == CHAR) {
                    return true;
                }
            }
            return false;
        }

        @Override
        public boolean equals(Object o) {
            return o instanceof State s && s.c == c && Arrays.equals(s.states, states);
        }

        @Override
        public int hashCode() {
            return Arrays.hashCode(states);
        }
    }

    /** The states reachable without a character, sorted, holding only CHAR and MATCH ones. */
    private int[] closure(int[] from) {
        BitSet seen = new BitSet(count);
        BitSet kept = new BitSet(count);
        int[] stack = new int[count + from.length];
        int top = 0;
        for (int s : from) {
            stack[top++] = s;
        }
        while (top > 0) {
            int s = stack[--top];
            if (seen.get(s)) {
                continue;
            }
            seen.set(s);
            if (kind[s] == SPLIT) {
                if (top + 2 > stack.length) {
                    stack = Arrays.copyOf(stack, stack.length * 2);
                }
                stack[top++] = out1[s];
                stack[top++] = out[s];
            } else {
                kept.set(s);
            }
        }
        return kept.stream().toArray();
    }

    // ---- compiling, back to front: each node is built knowing the state that follows it

    private int add(int k, int o, int o1, CharSet set) {
        if (count == kind.length) {
            int n = count * 2;
            kind = Arrays.copyOf(kind, n);
            out = Arrays.copyOf(out, n);
            out1 = Arrays.copyOf(out1, n);
            sets = Arrays.copyOf(sets, n);
        }
        kind[count] = k;
        out[count] = o;
        out1[count] = o1;
        sets[count] = set;
        return count++;
    }

    private int compile(Node n, int next) {
        return switch (n) {
            case Chars c -> add(CHAR, next, -1, c.set());
            case Seq s -> {
                int at = next;
                for (int i = s.parts().size() - 1; i >= 0; i--) {
                    at = compile(s.parts().get(i), at);
                }
                yield at;
            }
            case Alt a -> {
                int at = compile(a.options().getLast(), next);
                for (int i = a.options().size() - 2; i >= 0; i--) {
                    at = add(SPLIT, compile(a.options().get(i), next), at, null);
                }
                yield at;
            }
            case Repeat r -> {
                int at = next;
                if (r.max() < 0) {
                    int loop = add(SPLIT, -1, next, null);
                    int body = compile(r.body(), loop);   // may grow the arrays: assign after
                    out[loop] = body;
                    at = loop;
                } else {
                    for (int i = r.min(); i < r.max(); i++) {
                        at = add(SPLIT, compile(r.body(), at), next, null);
                    }
                }
                for (int i = 0; i < r.min(); i++) {
                    at = compile(r.body(), at);
                }
                yield at;
            }
        };
    }

    // ---- parsing

    private sealed interface Node permits Chars, Seq, Alt, Repeat {
    }

    private record Chars(CharSet set) implements Node {
    }

    private record Seq(List<Node> parts) implements Node {
    }

    private record Alt(List<Node> options) implements Node {
    }

    /** {@code max} -1 for no upper bound. */
    private record Repeat(Node body, int min, int max) implements Node {
    }

    /** Sorted inclusive ranges, possibly negated. */
    private record CharSet(char[] ranges, boolean negated) {

        boolean contains(char c) {
            boolean in = false;
            for (int i = 0; i < ranges.length && !in; i += 2) {
                in = c >= ranges[i] && c <= ranges[i + 1];
            }
            return in != negated;
        }

        static CharSet of(char c) {
            return new CharSet(new char[] {c, c}, false);
        }
    }

    private static final char[] DIGIT = {'0', '9'};
    private static final char[] WORD = {'0', '9', 'A', 'Z', '_', '_', 'a', 'z'};
    private static final char[] SPACE = {'\t', '\r', ' ', ' '};

    private static final class Parser {
        private final String p;
        private int i;

        Parser(String p) {
            this.p = p;
        }

        Node parse() {
            Node n = alternation();
            if (i < p.length()) {
                throw error("unexpected " + p.charAt(i));
            }
            return n;
        }

        private IllegalArgumentException error(String what) {
            return new IllegalArgumentException("regex " + p + " at " + i + ": " + what);
        }

        private Node alternation() {
            List<Node> options = new ArrayList<>();
            options.add(sequence());
            while (i < p.length() && p.charAt(i) == '|') {
                i++;
                options.add(sequence());
            }
            return options.size() == 1 ? options.getFirst() : new Alt(options);
        }

        private Node sequence() {
            List<Node> parts = new ArrayList<>();
            while (i < p.length() && p.charAt(i) != '|' && p.charAt(i) != ')') {
                parts.add(repeat(atom()));
            }
            return parts.size() == 1 ? parts.getFirst() : new Seq(parts);
        }

        private Node repeat(Node atom) {
            Node n = atom;
            while (i < p.length()) {
                char c = p.charAt(i);
                if (c == '*') {
                    n = new Repeat(n, 0, -1);
                } else if (c == '+') {
                    n = new Repeat(n, 1, -1);
                } else if (c == '?') {
                    n = new Repeat(n, 0, 1);
                } else if (c == '{' && i + 1 < p.length() && Character.isDigit(p.charAt(i + 1))) {
                    int close = p.indexOf('}', i);
                    if (close < 0) {
                        throw error("unclosed {");
                    }
                    String[] b = p.substring(i + 1, close).split(",", -1);
                    int min = Integer.parseInt(b[0].trim());
                    int max = b.length == 1 ? min : b[1].isBlank() ? -1 : Integer.parseInt(b[1].trim());
                    if (max >= 0 && max < min) {
                        throw error("{" + min + "," + max + "}");
                    }
                    n = new Repeat(n, min, max);
                    i = close;
                } else {
                    break;
                }
                i++;
            }
            return n;
        }

        private Node atom() {
            char c = p.charAt(i++);
            switch (c) {
                case '(' -> {
                    if (p.startsWith("?:", i)) {
                        i += 2;
                    }
                    Node inner = i < p.length() && p.charAt(i) == ')' ? new Seq(List.of()) : alternation();
                    if (i >= p.length() || p.charAt(i) != ')') {
                        throw error("unclosed (");
                    }
                    i++;
                    return inner;
                }
                case '[' -> {
                    return new Chars(charClass());
                }
                case '.' -> {
                    return new Chars(new CharSet(new char[] {'\n', '\n'}, true));
                }
                case '\\' -> {
                    return new Chars(escape());
                }
                case '*', '+', '?' -> throw error("nothing to repeat");
                default -> {
                    return new Chars(CharSet.of(c));
                }
            }
        }

        private CharSet escape() {
            if (i >= p.length()) {
                throw error("trailing \\");
            }
            char c = p.charAt(i++);
            return switch (c) {
                case 'd' -> new CharSet(DIGIT, false);
                case 'D' -> new CharSet(DIGIT, true);
                case 'w' -> new CharSet(WORD, false);
                case 'W' -> new CharSet(WORD, true);
                case 's' -> new CharSet(SPACE, false);
                case 'S' -> new CharSet(SPACE, true);
                default -> CharSet.of(literal(c));
            };
        }

        private char literal(char c) {
            return switch (c) {
                case 'n' -> '\n';
                case 't' -> '\t';
                case 'r' -> '\r';
                case 'f' -> '\f';
                case 'u' -> {
                    char v = (char) Integer.parseInt(p.substring(i, i + 4), 16);
                    i += 4;
                    yield v;
                }
                case 'x' -> {
                    char v = (char) Integer.parseInt(p.substring(i, i + 2), 16);
                    i += 2;
                    yield v;
                }
                default -> c;
            };
        }

        private CharSet charClass() {
            boolean negated = i < p.length() && p.charAt(i) == '^';
            if (negated) {
                i++;
            }
            List<char[]> ranges = new ArrayList<>();
            boolean first = true;
            while (i < p.length() && (p.charAt(i) != ']' || first)) {
                first = false;
                char c = p.charAt(i++);
                char lo;
                if (c == '\\') {
                    char e = p.charAt(i);
                    if ("dwsDWS".indexOf(e) >= 0) {
                        i++;
                        CharSet s = switch (e) {
                            case 'd' -> new CharSet(DIGIT, false);
                            case 'w' -> new CharSet(WORD, false);
                            case 's' -> new CharSet(SPACE, false);
                            default -> throw error("negated class \\" + e + " inside [ ]");
                        };
                        for (int k = 0; k < s.ranges().length; k += 2) {
                            ranges.add(new char[] {s.ranges()[k], s.ranges()[k + 1]});
                        }
                        continue;
                    }
                    i++;
                    lo = literal(e);
                } else {
                    lo = c;
                }
                char hi = lo;
                if (i + 1 < p.length() && p.charAt(i) == '-' && p.charAt(i + 1) != ']') {
                    i++;
                    char h = p.charAt(i++);
                    hi = h == '\\' ? literal(p.charAt(i++)) : h;
                }
                ranges.add(new char[] {lo, hi});
            }
            if (i >= p.length()) {
                throw error("unclosed [");
            }
            i++;
            char[] flat = new char[ranges.size() * 2];
            for (int k = 0; k < ranges.size(); k++) {
                flat[2 * k] = ranges.get(k)[0];
                flat[2 * k + 1] = ranges.get(k)[1];
            }
            return new CharSet(flat, negated);
        }
    }

    // ---- JSON schema to a regular expression

    private static final String WS = "[ ]?";
    private static final String STRING_CHAR = "(?:[^\"\\\\\\x00-\\x1f]|\\\\[\"\\\\/bfnrt])";
    private static final String INTEGER = "-?(?:0|[1-9][0-9]*)";
    private static final String NUMBER = INTEGER + "(?:\\.[0-9]+)?(?:[eE][+\\-]?[0-9]+)?";

    private static String schemaRegex(Map<String, Object> s) {
        if (s.get("enum") instanceof List<?> values) {
            List<String> options = new ArrayList<>();
            for (Object v : values) {
                options.add(quote(Json.write(v)));
            }
            return "(?:" + String.join("|", options) + ")";
        }
        if (s.containsKey("const")) {
            return quote(Json.write(s.get("const")));
        }
        Object any = s.containsKey("anyOf") ? s.get("anyOf") : s.get("oneOf");
        if (any instanceof List<?> options) {
            List<String> parts = new ArrayList<>();
            for (Object o : options) {
                parts.add(schemaRegex(Json.object(o)));
            }
            return "(?:" + String.join("|", parts) + ")";
        }
        if (s.containsKey("$ref")) {
            throw new IllegalArgumentException("JSON schema $ref is not supported");
        }
        Object type = s.get("type");
        if (type instanceof List<?> types) {
            List<String> parts = new ArrayList<>();
            for (Object t : types) {
                Map<String, Object> one = new java.util.LinkedHashMap<>(s);
                one.put("type", t);
                parts.add(schemaRegex(one));
            }
            return "(?:" + String.join("|", parts) + ")";
        }
        if (type == null) {
            type = s.containsKey("properties") ? "object" : s.containsKey("items") ? "array" : null;
        }
        if (!(type instanceof String t)) {
            throw new IllegalArgumentException("JSON schema needs a type, enum or const: " + s);
        }
        return switch (t) {
            case "string" -> {
                if (s.get("pattern") instanceof String pattern) {
                    String inner = pattern.startsWith("^") ? pattern.substring(1) : pattern;
                    inner = inner.endsWith("$") ? inner.substring(0, inner.length() - 1) : inner;
                    yield "\"(?:" + inner + ")\"";
                }
                long min = Json.integer(s, "minLength", 0);
                long max = Json.integer(s, "maxLength", -1);
                yield "\"" + STRING_CHAR + (min == 0 && max < 0 ? "*" : "{" + min + "," + (max < 0 ? "" : max) + "}")
                        + "\"";
            }
            case "integer" -> INTEGER;
            case "number" -> NUMBER;
            case "boolean" -> "(?:true|false)";
            case "null" -> "null";
            case "array" -> {
                String item = s.get("items") instanceof Map<?, ?> items ? schemaRegex(Json.object(items))
                        : "(?:" + NUMBER + "|\"" + STRING_CHAR + "*\"|true|false|null)";
                long min = Json.integer(s, "minItems", 0);
                long max = Json.integer(s, "maxItems", -1);
                if (max == 0) {
                    yield "\\[\\]";
                }
                String more = "(?:," + WS + item + ")";
                String rest = more + "{" + Math.max(0, min - 1) + "," + (max < 0 ? "" : max - 1) + "}";
                String some = item + rest;
                yield "\\[" + (min == 0 ? "(?:" + some + ")?" : some) + "\\]";
            }
            case "object" -> objectRegex(s);
            default -> throw new IllegalArgumentException("JSON schema type " + t + " is not supported");
        };
    }

    private static String objectRegex(Map<String, Object> s) {
        Map<String, Object> props = s.get("properties") instanceof Map<?, ?> m ? Json.object(m) : Map.of();
        List<?> required = s.get("required") instanceof List<?> r ? r : List.of();
        List<String> keys = new ArrayList<>(props.keySet());
        int n = keys.size();
        String[] member = new String[n];
        boolean[] needed = new boolean[n];
        for (int k = 0; k < n; k++) {
            member[k] = quote(Json.write(keys.get(k))) + WS + ":" + WS
                    + schemaRegex(Json.object(props.get(keys.get(k))));
            needed[k] = required.contains(keys.get(k));
        }
        // the first member present is any optional one up to the first required one
        List<String> starts = new ArrayList<>();
        boolean allOptional = true;
        for (int first = 0; first < n; first++) {
            StringBuilder b = new StringBuilder(member[first]);
            for (int k = first + 1; k < n; k++) {
                String part = "," + WS + member[k];
                b.append(needed[k] ? part : "(?:" + part + ")?");
            }
            starts.add(b.toString());
            if (needed[first]) {
                allOptional = false;
                break;
            }
        }
        String body = starts.isEmpty() ? "" : "(?:" + String.join("|", starts) + ")";
        if (allOptional && !body.isEmpty()) {
            body += "?";
        }
        return "\\{" + WS + body + WS + "\\}";
    }
}
