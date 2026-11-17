package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.llm.Constraint;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * A regular expression as a deterministic automaton over characters with integer states,
 * for constrained decoding: walk a token's text from the current state, and the token is
 * allowed if the walk does not die (Willard and Louf 2023, "Efficient Guided Generation",
 * <a href="https://arxiv.org/abs/2307.09702">arXiv:2307.09702</a>, the idea behind
 * Outlines and llama.cpp grammars).
 *
 * <p>The regular expression itself is a core {@link Constraint}: its syntax (literals,
 * {@code .}, escapes, classes, groups, {@code |} and the quantifiers {@code * + ? {m} {m,}
 * {m,n}}), and its automaton. This class only numbers the core's
 * {@link Constraint.Matcher} states as they are reached and caches the transitions between
 * them, which turns the NFA into a lazily built DFA whose states fit in an {@code int}, as
 * {@link TokenConstraint} needs. The whole text must match (as {@code String.matches}).
 */
public final class RegexAutomaton {

    /** The state of a walk that can no longer match. */
    public static final int DEAD = -1;

    private final Constraint constraint;
    private final List<Constraint.Matcher> states = new ArrayList<>();
    private final Map<Constraint.Matcher, Integer> ids = new HashMap<>();
    private final Map<Long, Integer> transitions = new HashMap<>();

    private RegexAutomaton(Constraint constraint) {
        this.constraint = Objects.requireNonNull(constraint);
        id(constraint.start());
    }

    /**
     * @param pattern the regular expression
     * @return the automaton
     * @throws IllegalArgumentException for bad syntax
     */
    public static RegexAutomaton compile(String pattern) {
        return new RegexAutomaton(Constraint.regex(pattern));
    }

    /**
     * Numbers the states of a core constraint (a regular expression, a choice or a JSON
     * schema).
     *
     * @param constraint the constraint
     * @return the automaton
     */
    public static RegexAutomaton of(Constraint constraint) {
        return new RegexAutomaton(constraint);
    }

    /** @return the core constraint, for {@code Generator.Options.withConstraint} */
    public Constraint constraint() {
        return constraint;
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
        Constraint.Matcher next = states.get(state).feed(c);
        int id = next == null ? DEAD : id(next);
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
        return state != DEAD && states.get(state).accepting();
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
        return constraint.pattern();
    }

    private int id(Constraint.Matcher m) {
        Integer id = ids.get(m);
        if (id == null) {
            id = states.size();
            states.add(m);
            ids.put(m, id);
        }
        return id;
    }

    @Override
    public String toString() {
        return "RegexAutomaton(" + pattern() + ", " + states.size() + " states built)";
    }
}
