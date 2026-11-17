package io.github.zzuegg.brainiac.llm;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * A context-free grammar in llama.cpp's GBNF notation, for constrained decoding of languages
 * a regular expression cannot describe, such as nested brackets, arithmetic or a small DSL.
 *
 * <pre>
 * root   ::= expr
 * expr   ::= term ([+-] term)*
 * term   ::= num | "(" expr ")"
 * num    ::= [0-9]+
 * </pre>
 *
 * <p>Supported: rules {@code name ::= ...} (the start is {@code root}), string literals with
 * the escapes {@code \n \r \t \\ \" \xHH \\uHHHH}, character classes {@code [a-z]} and
 * {@code [^"]}, {@code .}, groups, alternatives {@code |}, the repetitions {@code * + ?
 * {m} {m,} {m,n}}, and {@code #} comments. Left-recursive rules are refused, as in llama.cpp.
 *
 * <p>The matcher is a pushdown automaton: it keeps every stack of rule positions the text so
 * far could be in, so ambiguous grammars work too.
 */
public final class Grammar {

    /** An element of an alternative: characters, or a reference to a rule. */
    private sealed interface Element permits Chars, Ref {
    }

    /** Pairs of inclusive ranges; {@code negated} matches every character outside them. */
    private record Chars(char[] ranges, boolean negated) implements Element {

        boolean contains(char c) {
            boolean in = false;
            for (int i = 0; i < ranges.length && !in; i += 2) {
                in = c >= ranges[i] && c <= ranges[i + 1];
            }
            return in != negated;
        }
    }

    private record Ref(int rule) implements Element {
    }

    /** One position in an alternative, and the positions to return to after it. */
    private record Frame(int rule, int alt, int index, Frame next) {
    }

    private final List<String> names = new ArrayList<>();
    private final List<List<Element[]>> rules = new ArrayList<>();
    private final Map<String, Integer> byName = new HashMap<>();
    private final Set<String> defined = new LinkedHashSet<>();
    private final int root;

    private Grammar(String text) {
        new Parser(text).parse();
        for (String n : names) {
            if (!n.contains("~") && !defined.contains(n)) {
                throw new IllegalArgumentException("rule " + n + " is used but not defined");
            }
        }
        Integer r = byName.get("root");
        if (r == null || !defined.contains("root")) {
            throw new IllegalArgumentException("the grammar has no root rule");
        }
        root = r;
        refuseLeftRecursion();
    }

    /**
     * Parses a grammar.
     *
     * @param gbnf the rules
     * @return the grammar
     * @throws IllegalArgumentException if the text is not valid GBNF, a rule is missing or left-recursive
     */
    public static Grammar parse(String gbnf) {
        return new Grammar(gbnf);
    }

    /** @return the constraint that keeps generated text inside the grammar's language */
    public Constraint constraint() {
        return Constraint.of(this::start);
    }

    /**
     * @param text a whole text
     * @return whether the grammar accepts it
     */
    public boolean matches(String text) {
        Constraint.Matcher m = start().feed(text);
        return m != null && m.accepting();
    }

    /** @return the matcher before any text */
    public Constraint.Matcher start() {
        Set<Frame> stacks = new LinkedHashSet<>();
        boolean[] accept = new boolean[1];
        for (int a = 0; a < rules.get(root).size(); a++) {
            expand(new Frame(root, a, 0, null), stacks, accept);
        }
        return new State(this, stacks.toArray(Frame[]::new), accept[0]);
    }

    /** Resolves references until each stack's top is characters; an empty stack accepts. */
    private void expand(Frame f, Set<Frame> out, boolean[] accept) {
        if (f == null) {
            accept[0] = true;
            return;
        }
        Element[] alt = rules.get(f.rule).get(f.alt);
        if (f.index == alt.length) {
            expand(f.next, out, accept);
        } else if (alt[f.index] instanceof Ref ref) {
            // a reference in last place returns straight to the caller, so loops do not grow the stack
            Frame back = f.index + 1 == alt.length ? f.next : new Frame(f.rule, f.alt, f.index + 1, f.next);
            for (int a = 0; a < rules.get(ref.rule).size(); a++) {
                expand(new Frame(ref.rule, a, 0, back), out, accept);
            }
        } else {
            out.add(f);
        }
    }

    private record State(Grammar g, Frame[] stacks, boolean accepting) implements Constraint.Matcher {

        @Override
        public Constraint.Matcher feed(char c) {
            Set<Frame> next = new LinkedHashSet<>();
            boolean[] accept = new boolean[1];
            for (Frame f : stacks) {
                Chars set = (Chars) g.rules.get(f.rule).get(f.alt)[f.index];
                if (set.contains(c)) {
                    g.expand(new Frame(f.rule, f.alt, f.index + 1, f.next), next, accept);
                }
            }
            if (next.isEmpty() && !accept[0]) {
                return null;
            }
            return new State(g, next.toArray(Frame[]::new), accept[0]);
        }

        @Override
        public boolean canContinue() {
            return stacks.length > 0;
        }

        @Override
        public boolean equals(Object o) {
            return o instanceof State s && s.g == g && s.accepting == accepting
                    && Set.of(s.stacks).equals(Set.of(stacks));
        }

        @Override
        public int hashCode() {
            return Set.of(stacks).hashCode() * 31 + Boolean.hashCode(accepting);
        }
    }

    // ---- checks

    private void refuseLeftRecursion() {
        int n = rules.size();
        boolean[] nullable = new boolean[n];
        for (boolean changed = true; changed; ) {
            changed = false;
            for (int r = 0; r < n; r++) {
                if (!nullable[r] && rules.get(r).stream().anyMatch(alt -> nullablePrefix(alt, nullable) == alt.length)) {
                    nullable[r] = true;
                    changed = true;
                }
            }
        }
        // left[r] lists the rules r can start with
        List<Set<Integer>> left = new ArrayList<>();
        for (int r = 0; r < n; r++) {
            Set<Integer> s = new LinkedHashSet<>();
            for (Element[] alt : rules.get(r)) {
                for (Element e : alt) {
                    if (!(e instanceof Ref ref)) {
                        break;
                    }
                    s.add(ref.rule);
                    if (!nullable[ref.rule]) {
                        break;
                    }
                }
            }
            left.add(s);
        }
        for (int r = 0; r < n; r++) {
            boolean[] seen = new boolean[n];
            List<Integer> todo = new ArrayList<>(left.get(r));
            while (!todo.isEmpty()) {
                int x = todo.removeLast();
                if (x == r) {
                    throw new IllegalArgumentException("rule " + display(r) + " is left-recursive");
                }
                if (!seen[x]) {
                    seen[x] = true;
                    todo.addAll(left.get(x));
                }
            }
        }
    }

    private static int nullablePrefix(Element[] alt, boolean[] nullable) {
        int i = 0;
        while (i < alt.length && alt[i] instanceof Ref ref && nullable[ref.rule]) {
            i++;
        }
        return i;
    }

    private String display(int r) {
        String n = names.get(r);
        int t = n.indexOf('~');
        return t < 0 ? n : n.substring(0, t);
    }

    // ---- building

    private int rule(String name) {
        Integer r = byName.get(name);
        if (r == null) {
            r = names.size();
            names.add(name);
            rules.add(new ArrayList<>());
            byName.put(name, r);
        }
        return r;
    }

    /** A new rule named after the one it helps, so errors can name the original. */
    private int helper(String parent, List<Element[]> alts) {
        int r = rule(parent + "~" + names.size());
        rules.get(r).addAll(alts);
        return r;
    }

    private final class Parser {
        private final String s;
        private int pos;
        private String current;

        Parser(String s) {
            this.s = s;
        }

        void parse() {
            skip();
            while (pos < s.length()) {
                String name = name();
                skip();
                if (!s.startsWith("::=", pos)) {
                    throw error("expected ::= after " + name);
                }
                pos += 3;
                if (!defined.add(name)) {
                    throw error("rule " + name + " is defined twice");
                }
                current = name;
                int r = rule(name);
                rules.get(r).addAll(alternatives(false));
                skip();
            }
        }

        /** Alternatives up to the end of the rule, or up to {@code )} when nested. */
        List<Element[]> alternatives(boolean nested) {
            List<Element[]> alts = new ArrayList<>();
            alts.add(sequence(nested));
            while (pos < s.length() && s.charAt(pos) == '|') {
                pos++;
                alts.add(sequence(nested));
            }
            return alts;
        }

        Element[] sequence(boolean nested) {
            List<Element> seq = new ArrayList<>();
            while (true) {
                skip();
                if (pos >= s.length() || s.charAt(pos) == '|' || s.charAt(pos) == ')' || startsRule()) {
                    break;
                }
                int from = seq.size();
                atom(seq);
                skip();
                if (pos < s.length()) {
                    repeat(seq, from);
                }
            }
            return seq.toArray(Element[]::new);
        }

        /** Whether a new rule {@code name ::=} starts at the cursor, which ends the current one. */
        private boolean startsRule() {
            int p = pos;
            while (p < s.length() && isNameChar(s.charAt(p))) {
                p++;
            }
            if (p == pos) {
                return false;
            }
            while (p < s.length() && (s.charAt(p) == ' ' || s.charAt(p) == '\t')) {
                p++;
            }
            return s.startsWith("::=", p);
        }

        void atom(List<Element> seq) {
            char c = s.charAt(pos);
            if (c == '"') {
                pos++;
                while (true) {
                    if (pos >= s.length()) {
                        throw error("unterminated string");
                    }
                    if (s.charAt(pos) == '"') {
                        pos++;
                        break;
                    }
                    char ch = character();
                    seq.add(new Chars(new char[] {ch, ch}, false));
                }
            } else if (c == '[') {
                pos++;
                boolean negated = pos < s.length() && s.charAt(pos) == '^';
                if (negated) {
                    pos++;
                }
                List<Character> ranges = new ArrayList<>();
                while (true) {
                    if (pos >= s.length()) {
                        throw error("unterminated character class");
                    }
                    if (s.charAt(pos) == ']') {
                        pos++;
                        break;
                    }
                    char lo = character();
                    char hi = lo;
                    if (pos + 1 < s.length() && s.charAt(pos) == '-' && s.charAt(pos + 1) != ']') {
                        pos++;
                        hi = character();
                        if (hi < lo) {
                            throw error("range " + lo + "-" + hi + " is backwards");
                        }
                    }
                    ranges.add(lo);
                    ranges.add(hi);
                }
                char[] r = new char[ranges.size()];
                for (int i = 0; i < r.length; i++) {
                    r[i] = ranges.get(i);
                }
                seq.add(new Chars(r, negated));
            } else if (c == '.') {
                pos++;
                seq.add(new Chars(new char[0], true));
            } else if (c == '(') {
                pos++;
                List<Element[]> alts = alternatives(true);
                skip();
                if (pos >= s.length() || s.charAt(pos) != ')') {
                    throw error("expected )");
                }
                pos++;
                seq.add(new Ref(helper(current, alts)));
            } else if (isNameChar(c)) {
                seq.add(new Ref(rule(name())));
            } else {
                throw error("unexpected " + c);
            }
        }

        /** Applies a postfix repetition to the last atom, which starts at {@code from}. */
        void repeat(List<Element> seq, int from) {
            char c = s.charAt(pos);
            int min;
            int max;
            if (c == '*' || c == '+' || c == '?') {
                pos++;
                min = c == '+' ? 1 : 0;
                max = c == '?' ? 1 : -1;
            } else if (c == '{') {
                pos++;
                min = number();
                max = min;
                if (s.charAt(pos) == ',') {
                    pos++;
                    skip();
                    max = s.charAt(pos) == '}' ? -1 : number();
                }
                skip();
                if (s.charAt(pos) != '}') {
                    throw error("expected }");
                }
                pos++;
                if (max >= 0 && max < min) {
                    throw error("repetition {" + min + "," + max + "} is backwards");
                }
            } else {
                return;
            }
            Element[] body = seq.subList(from, seq.size()).toArray(Element[]::new);
            seq.subList(from, seq.size()).clear();
            for (int i = 0; i < min; i++) {
                seq.addAll(Arrays.asList(body));
            }
            if (max < 0) {
                // star ::= body star | ""
                int star = helper(current, new ArrayList<>());
                Element[] loop = Arrays.copyOf(body, body.length + 1);
                loop[body.length] = new Ref(star);
                rules.get(star).add(loop);
                rules.get(star).add(new Element[0]);
                seq.add(new Ref(star));
            } else if (max > min) {
                // nested optionals: opt_k ::= body opt_(k-1) | ""
                int opt = -1;
                for (int k = 0; k < max - min; k++) {
                    Element[] more = opt < 0 ? body : Arrays.copyOf(body, body.length + 1);
                    if (opt >= 0) {
                        more[body.length] = new Ref(opt);
                    }
                    List<Element[]> alts = new ArrayList<>();
                    alts.add(more);
                    alts.add(new Element[0]);
                    opt = helper(current, alts);
                }
                seq.add(new Ref(opt));
            }
        }

        char character() {
            char c = s.charAt(pos++);
            if (c != '\\') {
                return c;
            }
            if (pos >= s.length()) {
                throw error("dangling escape");
            }
            char e = s.charAt(pos++);
            return switch (e) {
                case 'n' -> '\n';
                case 'r' -> '\r';
                case 't' -> '\t';
                case '\\', '"', '[', ']', '-', '^' -> e;
                case 'x' -> hex(2);
                case 'u' -> hex(4);
                case 'U' -> {
                    int cp = Integer.parseInt(s.substring(pos, pos + 8), 16);
                    pos += 8;
                    if (cp > 0xFFFF) {
                        throw error("characters beyond U+FFFF are not supported in escapes");
                    }
                    yield (char) cp;
                }
                default -> throw error("unknown escape \\" + e);
            };
        }

        private char hex(int digits) {
            if (pos + digits > s.length()) {
                throw error("short escape");
            }
            char c = (char) Integer.parseInt(s.substring(pos, pos + digits), 16);
            pos += digits;
            return c;
        }

        int number() {
            skip();
            int p = pos;
            while (pos < s.length() && Character.isDigit(s.charAt(pos))) {
                pos++;
            }
            if (p == pos) {
                throw error("expected a number");
            }
            int n = Integer.parseInt(s.substring(p, pos));
            skip();
            return n;
        }

        String name() {
            int p = pos;
            while (pos < s.length() && isNameChar(s.charAt(pos))) {
                pos++;
            }
            if (p == pos) {
                throw error("expected a rule name");
            }
            return s.substring(p, pos);
        }

        /** Skips blanks, newlines and comments; a rule ends where the next {@code name ::=} starts. */
        void skip() {
            while (pos < s.length()) {
                char c = s.charAt(pos);
                if (c == '#') {
                    while (pos < s.length() && s.charAt(pos) != '\n') {
                        pos++;
                    }
                } else if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
                    pos++;
                } else {
                    break;
                }
            }
        }

        private static boolean isNameChar(char c) {
            return c < 128 && (Character.isLetterOrDigit(c) || c == '-' || c == '_');
        }

        IllegalArgumentException error(String what) {
            int line = 1;
            for (int i = 0; i < Math.min(pos, s.length()); i++) {
                if (s.charAt(i) == '\n') {
                    line++;
                }
            }
            return new IllegalArgumentException("grammar line " + line + ": " + what);
        }
    }
}
