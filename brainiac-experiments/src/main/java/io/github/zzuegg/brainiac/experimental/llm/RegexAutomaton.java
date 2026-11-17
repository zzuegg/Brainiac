package io.github.zzuegg.brainiac.experimental.llm;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A regular expression compiled to a deterministic automaton over characters, for
 * constrained decoding: walk a token's text from the current state, and the token is
 * allowed if the walk does not die (Willard and Louf 2023, "Efficient Guided Generation",
 * <a href="https://arxiv.org/abs/2307.09702">arXiv:2307.09702</a>, the idea behind
 * Outlines and llama.cpp grammars).
 *
 * <p>The syntax is a practical subset: literals, {@code .}, escapes ({@code \d \w \s \D \W
 * \S} and escaped metacharacters), classes {@code [a-z_]} and {@code [^"]}, groups
 * {@code ( )}, alternation {@code |} and the quantifiers {@code * + ? {m} {m,} {m,n}}. The
 * whole text must match (as {@code String.matches}). It is compiled with Thompson's
 * construction, and the DFA states are built lazily from sets of NFA states.
 */
public final class RegexAutomaton {

    /** The state of a walk that can no longer match. */
    public static final int DEAD = -1;

    private interface CharTest {
        boolean test(char c);
    }

    private sealed interface Node permits Leaf, Cat, Alt, Rep {
    }

    private record Leaf(CharTest test) implements Node {
    }

    private record Cat(List<Node> parts) implements Node {
    }

    private record Alt(List<Node> options) implements Node {
    }

    /** {@code max < 0} for unbounded. */
    private record Rep(Node node, int min, int max) implements Node {
    }

    private final String pattern;
    private final List<CharTest> tests = new ArrayList<>();
    private final List<int[]> edges = new ArrayList<>();
    private int finalState;
    private final List<BitSet> dfaStates = new ArrayList<>();
    private final Map<BitSet, Integer> dfaIds = new HashMap<>();
    private final Map<Long, Integer> transitions = new HashMap<>();

    private RegexAutomaton(String pattern) {
        this.pattern = pattern;
        int[] pos = {0};
        Node root = parseAlt(pattern, pos);
        if (pos[0] != pattern.length()) {
            throw new IllegalArgumentException("unexpected '" + pattern.charAt(pos[0]) + "' at " + pos[0] + " in " + pattern);
        }
        int[] frag = compile(root);
        finalState = newState(null);
        connect(frag[1], finalState);
        BitSet start = new BitSet();
        start.set(frag[0]);
        dfaId(closure(start));
    }

    /**
     * @param pattern the regular expression
     * @return the automaton
     * @throws IllegalArgumentException for bad syntax
     */
    public static RegexAutomaton compile(String pattern) {
        return new RegexAutomaton(pattern);
    }

    /** @return the start state */
    public int start() {
        return 0;
    }

    /**
     * @param state a state
     * @param c the next character
     * @return the state after it, or {@link #DEAD}
     */
    public int step(int state, char c) {
        if (state == DEAD) {
            return DEAD;
        }
        long key = ((long) state << 16) | c;
        Integer cached = transitions.get(key);
        if (cached != null) {
            return cached;
        }
        BitSet next = new BitSet();
        BitSet cur = dfaStates.get(state);
        for (int s = cur.nextSetBit(0); s >= 0; s = cur.nextSetBit(s + 1)) {
            CharTest t = tests.get(s);
            if (t != null && t.test(c)) {
                for (int e : edges.get(s)) {
                    next.set(e);
                }
            }
        }
        int id = next.isEmpty() ? DEAD : dfaId(closure(next));
        transitions.put(key, id);
        return id;
    }

    /**
     * @param state a state
     * @param text characters to read
     * @return the state after them, or {@link #DEAD}
     */
    public int walk(int state, CharSequence text) {
        for (int i = 0; i < text.length() && state != DEAD; i++) {
            state = step(state, text.charAt(i));
        }
        return state;
    }

    /**
     * @param state a state
     * @return whether the text read so far matches
     */
    public boolean accepting(int state) {
        return state != DEAD && dfaStates.get(state).get(finalState);
    }

    /**
     * @param state a state
     * @return whether some continuation can still reach an accepting state (every live
     *     state can, by construction, except where only an empty class follows)
     */
    public boolean alive(int state) {
        return state != DEAD;
    }

    /**
     * @param text a text
     * @return whether the whole text matches
     */
    public boolean matches(CharSequence text) {
        return accepting(walk(start(), text));
    }

    /** @return the pattern */
    public String pattern() {
        return pattern;
    }

    // ---------------------------------------------------------------- DFA

    private int dfaId(BitSet set) {
        Integer id = dfaIds.get(set);
        if (id == null) {
            id = dfaStates.size();
            dfaStates.add(set);
            dfaIds.put(set, id);
        }
        return id;
    }

    private BitSet closure(BitSet set) {
        BitSet out = (BitSet) set.clone();
        ArrayDeque<Integer> todo = new ArrayDeque<>();
        for (int s = set.nextSetBit(0); s >= 0; s = set.nextSetBit(s + 1)) {
            todo.push(s);
        }
        while (!todo.isEmpty()) {
            int s = todo.pop();
            if (tests.get(s) == null) {
                for (int e : edges.get(s)) {
                    if (!out.get(e)) {
                        out.set(e);
                        todo.push(e);
                    }
                }
            }
        }
        return out;
    }

    // ---------------------------------------------------------------- NFA

    private int newState(CharTest test) {
        tests.add(test);
        edges.add(new int[0]);
        return tests.size() - 1;
    }

    private void connect(int from, int to) {
        int[] e = edges.get(from);
        int[] n = java.util.Arrays.copyOf(e, e.length + 1);
        n[e.length] = to;
        edges.set(from, n);
    }

    /** @return {start, end}: end is an epsilon state with no edges yet */
    private int[] compile(Node node) {
        return switch (node) {
            case Leaf l -> {
                int s = newState(l.test());
                int e = newState(null);
                connect(s, e);
                yield new int[] {s, e};
            }
            case Cat c -> {
                int s = newState(null);
                int end = s;
                for (Node p : c.parts()) {
                    int[] f = compile(p);
                    connect(end, f[0]);
                    end = f[1];
                }
                yield new int[] {s, end};
            }
            case Alt a -> {
                int s = newState(null);
                int e = newState(null);
                for (Node o : a.options()) {
                    int[] f = compile(o);
                    connect(s, f[0]);
                    connect(f[1], e);
                }
                yield new int[] {s, e};
            }
            case Rep r -> {
                int s = newState(null);
                int end = s;
                for (int i = 0; i < r.min(); i++) {
                    int[] f = compile(r.node());
                    connect(end, f[0]);
                    end = f[1];
                }
                if (r.max() < 0) {
                    int[] f = compile(r.node());
                    int loop = newState(null);
                    connect(end, loop);
                    connect(loop, f[0]);
                    connect(f[1], loop);
                    end = loop;
                } else {
                    int e = newState(null);
                    for (int i = r.min(); i < r.max(); i++) {
                        connect(end, e);
                        int[] f = compile(r.node());
                        connect(end, f[0]);
                        end = f[1];
                    }
                    connect(end, e);
                    end = e;
                }
                yield new int[] {s, end};
            }
        };
    }

    // ---------------------------------------------------------------- parser

    private static Node parseAlt(String p, int[] pos) {
        List<Node> options = new ArrayList<>();
        options.add(parseCat(p, pos));
        while (pos[0] < p.length() && p.charAt(pos[0]) == '|') {
            pos[0]++;
            options.add(parseCat(p, pos));
        }
        return options.size() == 1 ? options.get(0) : new Alt(options);
    }

    private static Node parseCat(String p, int[] pos) {
        List<Node> parts = new ArrayList<>();
        while (pos[0] < p.length() && p.charAt(pos[0]) != '|' && p.charAt(pos[0]) != ')') {
            parts.add(parseRepeat(p, pos));
        }
        return parts.size() == 1 ? parts.get(0) : new Cat(parts);
    }

    private static Node parseRepeat(String p, int[] pos) {
        Node atom = parseAtom(p, pos);
        while (pos[0] < p.length()) {
            char c = p.charAt(pos[0]);
            if (c == '*') {
                atom = new Rep(atom, 0, -1);
            } else if (c == '+') {
                atom = new Rep(atom, 1, -1);
            } else if (c == '?') {
                atom = new Rep(atom, 0, 1);
            } else if (c == '{') {
                int close = p.indexOf('}', pos[0]);
                if (close < 0) {
                    throw new IllegalArgumentException("unclosed { in " + p);
                }
                String[] mn = p.substring(pos[0] + 1, close).split(",", -1);
                int min = Integer.parseInt(mn[0].trim());
                int max = mn.length == 1 ? min : mn[1].isBlank() ? -1 : Integer.parseInt(mn[1].trim());
                atom = new Rep(atom, min, max);
                pos[0] = close;
            } else {
                break;
            }
            pos[0]++;
        }
        return atom;
    }

    private static Node parseAtom(String p, int[] pos) {
        char c = p.charAt(pos[0]++);
        switch (c) {
            case '(' -> {
                Node inner = parseAlt(p, pos);
                if (pos[0] >= p.length() || p.charAt(pos[0]) != ')') {
                    throw new IllegalArgumentException("unclosed ( in " + p);
                }
                pos[0]++;
                return inner;
            }
            case '[' -> {
                return new Leaf(parseClass(p, pos));
            }
            case '.' -> {
                return new Leaf(ch -> ch != '\n');
            }
            case '\\' -> {
                return new Leaf(escape(p.charAt(pos[0]++)));
            }
            case '*', '+', '?', '{', ')' -> throw new IllegalArgumentException("nothing to repeat at " + (pos[0] - 1) + " in " + p);
            default -> {
                return new Leaf(ch -> ch == c);
            }
        }
    }

    private static CharTest escape(char e) {
        return switch (e) {
            case 'd' -> Character::isDigit;
            case 'D' -> ch -> !Character.isDigit(ch);
            case 'w' -> ch -> Character.isLetterOrDigit(ch) || ch == '_';
            case 'W' -> ch -> !(Character.isLetterOrDigit(ch) || ch == '_');
            case 's' -> Character::isWhitespace;
            case 'S' -> ch -> !Character.isWhitespace(ch);
            case 'n' -> ch -> ch == '\n';
            case 't' -> ch -> ch == '\t';
            default -> ch -> ch == e;
        };
    }

    private static CharTest parseClass(String p, int[] pos) {
        boolean negate = pos[0] < p.length() && p.charAt(pos[0]) == '^';
        if (negate) {
            pos[0]++;
        }
        List<CharTest> parts = new ArrayList<>();
        boolean first = true;
        while (pos[0] < p.length() && (first || p.charAt(pos[0]) != ']')) {
            first = false;
            char c = p.charAt(pos[0]++);
            if (c == '\\') {
                parts.add(escape(p.charAt(pos[0]++)));
            } else if (pos[0] + 1 < p.length() && p.charAt(pos[0]) == '-' && p.charAt(pos[0] + 1) != ']') {
                char hi = p.charAt(pos[0] + 1);
                pos[0] += 2;
                parts.add(ch -> ch >= c && ch <= hi);
            } else {
                parts.add(ch -> ch == c);
            }
        }
        if (pos[0] >= p.length()) {
            throw new IllegalArgumentException("unclosed [ in " + p);
        }
        pos[0]++;
        return ch -> {
            boolean any = false;
            for (CharTest t : parts) {
                if (t.test(ch)) {
                    any = true;
                    break;
                }
            }
            return any != negate;
        };
    }

    @Override
    public String toString() {
        return "RegexAutomaton(" + pattern + ", " + dfaStates.size() + " states built)";
    }
}
