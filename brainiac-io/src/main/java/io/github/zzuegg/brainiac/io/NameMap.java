package io.github.zzuegg.brainiac.io;

import java.util.ArrayList;
import java.util.List;
import java.util.function.UnaryOperator;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Renames weights with regular-expression rules, tried in order; the first rule whose
 * pattern matches the whole name wins. Names no rule matches are kept, or dropped with
 * {@link #dropUnmatched()}.
 *
 * <pre>{@code
 * NameMap llama = NameMap.create()
 *         .rule("model\\.embed_tokens\\.weight", "embed")
 *         .rule("model\\.layers\\.(\\d+)\\.self_attn\\.q_proj\\.weight", "block$1.query")
 *         .drop(".*rotary_emb\\.inv_freq")
 *         .dropUnmatched();
 * WeightSource w = WeightSource.open(dir).renamed(llama);
 * }</pre>
 */
public final class NameMap implements UnaryOperator<String> {

    private record Rule(Pattern pattern, String replacement) {
    }

    private final List<Rule> rules = new ArrayList<>();
    private boolean dropUnmatched;

    private NameMap() {
    }

    /**
     * Starts an empty mapping, which keeps every name.
     *
     * @return the mapping
     */
    public static NameMap create() {
        return new NameMap();
    }

    /**
     * Adds a rule.
     *
     * @param regex a pattern for the whole name
     * @param replacement the new name; {@code $1} and so on insert groups
     * @return this mapping
     */
    public NameMap rule(String regex, String replacement) {
        rules.add(new Rule(Pattern.compile(regex), replacement));
        return this;
    }

    /**
     * Adds a rule that drops the names it matches.
     *
     * @param regex a pattern for the whole name
     * @return this mapping
     */
    public NameMap drop(String regex) {
        rules.add(new Rule(Pattern.compile(regex), null));
        return this;
    }

    /**
     * Makes names no rule matches dropped instead of kept.
     *
     * @return this mapping
     */
    public NameMap dropUnmatched() {
        dropUnmatched = true;
        return this;
    }

    @Override
    public String apply(String name) {
        for (Rule r : rules) {
            Matcher m = r.pattern.matcher(name);
            if (m.matches()) {
                return r.replacement == null ? null : m.replaceAll(r.replacement);
            }
        }
        return dropUnmatched ? null : name;
    }
}
