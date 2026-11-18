package io.github.zzuegg.brainiac.studio.chat;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.llm.Constraint;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * The form an answer must have: free text, a JSON object, JSON that fits a schema, or text
 * that fits a grammar in llama.cpp's GBNF notation. Models on this PC and on worker hosts
 * are held to it word by word; providers are asked for it through their own structured output
 * setting, and none of them takes a grammar.
 *
 * @param kind {@code text}, {@code json}, {@code schema} or {@code grammar}
 * @param schema the JSON schema of a {@code schema} format, else null
 * @param grammar the GBNF grammar of a {@code grammar} format, else null
 */
public record Format(String kind, Map<String, Object> schema, String grammar) {

    /** Free text. */
    public static final Format NONE = new Format("text", null, null);

    /** Any JSON object. */
    public static final Format JSON = new Format("json", null, null);

    /** Added to the instructions of a JSON answer, for the providers that can only be asked. */
    public static final String JSON_HINT = "Answer with one JSON object and nothing else.";

    /** Checks the format. */
    public Format {
        switch (String.valueOf(kind)) {
            case "text", "json" -> {
                schema = null;
                grammar = null;
            }
            case "schema" -> {
                if (schema == null || schema.isEmpty()) {
                    throw new IllegalArgumentException("a schema format needs a JSON schema");
                }
                grammar = null;
            }
            case "grammar" -> {
                if (grammar == null || grammar.isBlank()) {
                    throw new IllegalArgumentException("a grammar format needs a grammar");
                }
                schema = null;
            }
            default -> throw new IllegalArgumentException("an answer's format is text, json, schema or grammar");
        }
    }

    /**
     * Returns whether the answer is free text.
     *
     * @return whether it is
     */
    public boolean none() {
        return kind.equals("text");
    }

    /**
     * Returns whether the answer is JSON, with or without a schema.
     *
     * @return whether it is
     */
    public boolean json() {
        return kind.equals("json") || kind.equals("schema");
    }

    /**
     * Reads a request's {@code format}: {@code {"type": "json"}}, {@code {"type": "schema",
     * "schema": {...}}} (the schema as an object or as JSON text) or {@code {"type": "grammar",
     * "grammar": "root ::= ..."}}.
     *
     * @param body the request
     * @return the format; free text when the request has none
     * @throws IllegalArgumentException if it is not one of these
     */
    public static Format of(Map<String, Object> body) {
        if (!(body.get("format") instanceof Map<?, ?> f)) {
            return NONE;
        }
        Map<String, Object> format = Json.object(f);
        String type = Json.string(format, "type", "text");
        Object schema = format.get("schema");
        if (schema instanceof String text && !text.isBlank()) {
            try {
                schema = Json.parse(text);
            } catch (RuntimeException e) {
                throw new IllegalArgumentException("the schema is not JSON: " + e.getMessage());
            }
        }
        if (type.equals("schema") && !(schema instanceof Map<?, ?>)) {
            throw new IllegalArgumentException("a schema format needs a JSON schema, an object such as {\"type\": \"object\"}");
        }
        return new Format(type, schema instanceof Map<?, ?> m ? Json.object(m) : null, Json.string(format, "grammar", null));
    }

    /**
     * Returns the format as {@link #of} reads it.
     *
     * @return the format
     */
    public Map<String, Object> toJson() {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("type", kind);
        if (schema != null) {
            m.put("schema", schema);
        }
        if (grammar != null) {
            m.put("grammar", grammar);
        }
        return m;
    }

    /**
     * Returns the language the answer of a model run here must belong to.
     *
     * @return the constraint, or null for free text
     * @throws IllegalArgumentException if the schema or grammar cannot be used
     */
    public Constraint constraint() {
        try {
            return switch (kind) {
                case "json" -> Constraint.jsonObject();
                case "schema" -> Constraint.jsonSchema(schema);
                case "grammar" -> Constraint.grammar(grammar);
                default -> null;
            };
        } catch (RuntimeException e) {
            throw new IllegalArgumentException((kind.equals("grammar") ? "the grammar" : "the schema")
                    + " cannot be used: " + e.getMessage(), e);
        }
    }

    /**
     * Returns the schema with {@code additionalProperties: false} on every object that does not
     * say otherwise, as the providers with strict structured output want it.
     *
     * @return the closed schema
     */
    public Map<String, Object> closedSchema() {
        return Json.object(close(schema));
    }

    private static Object close(Object node) {
        if (node instanceof Map<?, ?> m) {
            Map<String, Object> out = new LinkedHashMap<>();
            m.forEach((k, v) -> out.put(String.valueOf(k), close(v)));
            if (("object".equals(out.get("type")) || out.containsKey("properties")) && !out.containsKey("additionalProperties")) {
                out.put("additionalProperties", false);
            }
            return out;
        }
        if (node instanceof List<?> l) {
            return l.stream().map(Format::close).toList();
        }
        return node;
    }
}
