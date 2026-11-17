package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.data.Tokenizer;
import io.github.zzuegg.brainiac.llm.Constraint;

import java.util.Objects;

/**
 * A {@link TokenConstraint} from a {@link RegexAutomaton} and the text of every token: a
 * token is allowed if reading its text keeps the automaton alive, and the end-of-sequence
 * token only where the text so far matches. Tokens span several characters, so the mask is
 * computed per token by walking its text (cached by the automaton's transitions), which is
 * how Outlines and llama.cpp grammars constrain real BPE vocabularies.
 *
 * <p>The language is a core {@link Constraint}, so the same grammar drives the core
 * {@code Generator}: pass {@link #constraint()} to {@code Generator.Options.withConstraint}.
 * This class is the token-level adapter for decoders that work on ids and logits rather
 * than a tokenizer, such as {@link DecodingPipeline}.
 */
public final class GrammarConstraint implements TokenConstraint {

    private final RegexAutomaton automaton;
    private final String[] pieces;
    private final int eos;

    /**
     * @param automaton the grammar
     * @param pieces the text of each token; {@code null} or empty for tokens that are never
     *     allowed (other than {@code eos})
     * @param eos the end-of-sequence token, or -1 for none
     */
    public GrammarConstraint(RegexAutomaton automaton, String[] pieces, int eos) {
        this.automaton = Objects.requireNonNull(automaton);
        this.pieces = pieces.clone();
        this.eos = eos;
    }

    /**
     * Reads each token's text from a tokenizer.
     *
     * @param pattern the regular expression
     * @param tokenizer the tokenizer
     * @param eos the end-of-sequence token, or -1
     * @return the constraint
     */
    public static GrammarConstraint of(String pattern, Tokenizer tokenizer, int eos) {
        return of(Constraint.regex(pattern), tokenizer, eos);
    }

    /**
     * Reads each token's text from a tokenizer, for any core constraint (a regular
     * expression, a choice or a JSON schema).
     *
     * @param constraint the language
     * @param tokenizer the tokenizer
     * @param eos the end-of-sequence token, or -1
     * @return the constraint
     */
    public static GrammarConstraint of(Constraint constraint, Tokenizer tokenizer, int eos) {
        String[] pieces = new String[tokenizer.size()];
        for (int t = 0; t < pieces.length; t++) {
            pieces[t] = t == eos ? null : tokenizer.decode(t);
        }
        return new GrammarConstraint(RegexAutomaton.of(constraint), pieces, eos);
    }

    /** @return the core constraint, for {@code Generator.Options.withConstraint} */
    public Constraint constraint() {
        return automaton.constraint();
    }

    /** @return the automaton */
    public RegexAutomaton automaton() {
        return automaton;
    }

    @Override
    public int start() {
        return automaton.start();
    }

    @Override
    public int advance(int state, int token) {
        if (state < 0) {
            return -1;
        }
        if (token == eos) {
            return automaton.accepting(state) ? state : -1;
        }
        if (token < 0 || token >= pieces.length || pieces[token] == null || pieces[token].isEmpty()) {
            return -1;
        }
        int next = automaton.walk(state, pieces[token]);
        return next == RegexAutomaton.DEAD ? -1 : next;
    }

    @Override
    public boolean complete(int state) {
        return automaton.accepting(state);
    }
}
