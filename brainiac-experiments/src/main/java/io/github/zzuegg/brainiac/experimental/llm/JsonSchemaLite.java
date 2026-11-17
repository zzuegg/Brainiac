package io.github.zzuegg.brainiac.experimental.llm;

import io.github.zzuegg.brainiac.llm.Constraint;

import java.util.ArrayList;
import java.util.List;

/**
 * A small JSON-schema subset compiled to a regular expression, for structured output
 * ("JSON mode" with a schema): a flat object with required properties in a fixed order,
 * each a string, an integer, a number, a boolean, an enumeration of strings, or an array
 * of one of those. The regex accepts exactly the compact JSON texts (no spaces) that
 * satisfy the schema, and {@link GrammarConstraint} turns it into a token mask.
 *
 * <pre>{@code
 * String regex = JsonSchemaLite.object()
 *         .string("name", 1, 8).integer("age").bool("member").enumeration("tier", "gold", "free")
 *         .toRegex();
 * }</pre>
 */
public final class JsonSchemaLite {

    private final List<String> parts = new ArrayList<>();

    private JsonSchemaLite() {
    }

    /** @return an empty object schema */
    public static JsonSchemaLite object() {
        return new JsonSchemaLite();
    }

    private JsonSchemaLite property(String name, String valueRegex) {
        parts.add("\"" + Constraint.quote(name) + "\":" + valueRegex);
        return this;
    }

    /**
     * @param name the property
     * @param minLength the fewest characters
     * @param maxLength the most characters
     * @return this schema
     */
    public JsonSchemaLite string(String name, int minLength, int maxLength) {
        return property(name, stringValue(minLength, maxLength));
    }

    /**
     * @param name the property
     * @return this schema, with an integer property (no leading zeros)
     */
    public JsonSchemaLite integer(String name) {
        return property(name, INTEGER);
    }

    /**
     * @param name the property
     * @return this schema, with a decimal number property
     */
    public JsonSchemaLite number(String name) {
        return property(name, INTEGER + "(\\.[0-9]+)?");
    }

    /**
     * @param name the property
     * @return this schema, with a boolean property
     */
    public JsonSchemaLite bool(String name) {
        return property(name, "(true|false)");
    }

    /**
     * @param name the property
     * @param values the allowed strings
     * @return this schema
     */
    public JsonSchemaLite enumeration(String name, String... values) {
        StringBuilder b = new StringBuilder("(");
        for (int i = 0; i < values.length; i++) {
            b.append(i > 0 ? "|" : "").append('"').append(Constraint.quote(values[i])).append('"');
        }
        return property(name, b.append(')').toString());
    }

    /**
     * @param name the property
     * @param maxItems the most items
     * @return this schema, with an array of integers
     */
    public JsonSchemaLite integerArray(String name, int maxItems) {
        String item = INTEGER;
        String body = maxItems < 1 ? "" : "(" + item + "(," + item + "){0," + (maxItems - 1) + "})?";
        return property(name, "\\[" + body + "\\]");
    }

    /** @return the regular expression of the whole object */
    public String toRegex() {
        return "\\{" + String.join(",", parts) + "\\}";
    }

    /**
     * The schema as a core constraint, for {@code Generator.Options.withConstraint} or
     * {@link GrammarConstraint#of(Constraint, io.github.zzuegg.brainiac.data.Tokenizer, int)}.
     * Unlike {@code Constraint.jsonSchema}, it accepts only compact JSON (no spaces) and
     * strings of {@code [a-zA-Z0-9 _.-]} with length bounds.
     *
     * @return {@code Constraint.regex(toRegex())}
     */
    public Constraint toConstraint() {
        return Constraint.regex(toRegex());
    }

    private static final String INTEGER = "(-?(0|[1-9][0-9]{0,8}))";

    private static String stringValue(int min, int max) {
        return "\"[a-zA-Z0-9 _.-]{" + min + "," + max + "}\"";
    }
}
