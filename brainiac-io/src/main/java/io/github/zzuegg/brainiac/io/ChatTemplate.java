package io.github.zzuegg.brainiac.io;

import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * Renders a model's chat template: the Jinja template published with it that turns a
 * list of messages into the exact text the model was fine-tuned on.
 *
 * <p>This is a small Jinja interpreter covering what chat templates use: {@code if},
 * {@code for} (with {@code loop}), {@code set} (with {@code namespace}), {@code macro},
 * whitespace control, filters such as {@code trim}, {@code tojson}, {@code length} and
 * {@code join}, tests such as {@code is defined}, string and dict methods, slicing, and
 * {@code raise_exception}. Like Hugging Face's {@code apply_chat_template}, blocks trim the
 * newline after them and the indentation before them.
 *
 * <pre>{@code
 * ChatTemplate t = ChatTemplate.of(tokenizer.chatTemplate());
 * String prompt = t.render(List.of(
 *         ChatTemplate.message("system", "You are terse."),
 *         ChatTemplate.message("user", "Hi!")), true, Map.of("bos_token", "<s>"));
 * }</pre>
 */
public final class ChatTemplate {

    private final List<Node> body;

    private ChatTemplate(List<Node> body) {
        this.body = body;
    }

    /**
     * Parses a template.
     *
     * @param source the Jinja source
     * @return the template
     * @throws IllegalArgumentException if it cannot be parsed
     */
    public static ChatTemplate of(String source) {
        Objects.requireNonNull(source, "source");
        if (source.endsWith("\n")) {
            source = source.substring(0, source.length() - 1);   // as Jinja, without keep_trailing_newline
        }
        Parser p = new Parser(lex(source));
        List<Node> body = p.block();
        if (p.at < p.tokens.size()) {
            throw new IllegalArgumentException("unexpected {% " + p.tokens.get(p.at).text + " %}");
        }
        return new ChatTemplate(body);
    }

    /**
     * Makes a message.
     *
     * @param role {@code "system"}, {@code "user"}, {@code "assistant"} or {@code "tool"}
     * @param content its text
     * @return the message, as templates expect it
     */
    public static Map<String, Object> message(String role, String content) {
        Map<String, Object> m = new LinkedHashMap<>();
        m.put("role", role);
        m.put("content", content);
        return m;
    }

    /**
     * Renders messages.
     *
     * @param messages the conversation, each a map with at least {@code role} and {@code content}
     * @param addGenerationPrompt whether to end with the start of an assistant turn
     * @param extra more variables, such as {@code bos_token}, {@code eos_token} or {@code tools}
     * @return the prompt text
     */
    public String render(List<? extends Map<String, ?>> messages, boolean addGenerationPrompt, Map<String, ?> extra) {
        Map<String, Object> vars = new HashMap<>(extra);
        vars.put("messages", messages);
        vars.put("add_generation_prompt", addGenerationPrompt);
        return render(vars);
    }

    /**
     * Renders with any variables.
     *
     * @param variables the variables
     * @return the text
     */
    public String render(Map<String, ?> variables) {
        Scope scope = new Scope(null);
        scope.vars.putAll(variables);
        StringBuilder out = new StringBuilder();
        exec(body, scope, out);
        return out.toString();
    }

    // ================================================================ lexer

    private enum Kind { TEXT, OUTPUT, STATEMENT }

    private record Tok(Kind kind, String text) {
    }

    private static List<Tok> lex(String src) {
        List<Tok> out = new ArrayList<>();
        int at = 0;
        boolean trimNext = false;   // a -%} or -}} before, or a block (trim_blocks)
        boolean afterBlock = false;
        while (at < src.length()) {
            int open = nextOpen(src, at);
            String text = src.substring(at, open < 0 ? src.length() : open);
            if (afterBlock && text.startsWith("\n")) {
                text = text.substring(1);   // trim_blocks
            } else if (afterBlock && text.startsWith("\r\n")) {
                text = text.substring(2);
            }
            if (trimNext) {
                text = text.stripLeading();
            }
            trimNext = false;
            afterBlock = false;
            if (open < 0) {
                out.add(new Tok(Kind.TEXT, text));
                break;
            }
            char type = src.charAt(open + 1);
            boolean trimBefore = open + 2 < src.length() && src.charAt(open + 2) == '-';
            String close = type == '{' ? "}}" : type == '%' ? "%}" : "#}";
            int end = src.indexOf(close, open + 2);
            if (end < 0) {
                throw new IllegalArgumentException("unclosed " + src.substring(open, open + 2));
            }
            boolean trimAfter = src.charAt(end - 1) == '-' && end - 1 > open + 2;
            String inner = src.substring(open + 2 + (trimBefore ? 1 : 0), end - (trimAfter ? 1 : 0)).strip();
            if (trimBefore) {
                text = text.stripTrailing();
            } else if (type != '{') {
                // lstrip_blocks: spaces and tabs from the start of the line up to a block
                int line = text.lastIndexOf('\n') + 1;
                if (text.substring(line).chars().allMatch(c -> c == ' ' || c == '\t')) {
                    text = text.substring(0, line);
                }
            }
            if (!text.isEmpty()) {
                out.add(new Tok(Kind.TEXT, text));
            }
            if (type == '{') {
                out.add(new Tok(Kind.OUTPUT, inner));
            } else if (type == '%') {
                out.add(new Tok(Kind.STATEMENT, inner));
                afterBlock = true;
            } else {
                afterBlock = true;
            }
            trimNext = trimAfter;
            at = end + 2;
        }
        return out;
    }

    private static int nextOpen(String s, int from) {
        for (int i = s.indexOf('{', from); i >= 0 && i + 1 < s.length(); i = s.indexOf('{', i + 1)) {
            char c = s.charAt(i + 1);
            if (c == '{' || c == '%' || c == '#') {
                return i;
            }
        }
        return -1;
    }

    // ================================================================ statements

    private sealed interface Node permits Text, Output, If, For, Set, Macro, CallBlock {
    }

    private record Text(String text) implements Node {
    }

    private record Output(Expr expr) implements Node {
    }

    private record If(List<Expr> conditions, List<List<Node>> branches) implements Node {
    }

    private record For(List<String> targets, Expr iterable, Expr filter, List<Node> body, List<Node> orElse)
            implements Node {
    }

    private record Set(Expr target, String attr, Expr value, List<Node> body) implements Node {
    }

    private record Macro(String name, List<String> params, List<Expr> defaults, List<Node> body) implements Node {
    }

    private record CallBlock(Expr call, List<Node> body) implements Node {
    }

    private static final class Parser {
        final List<Tok> tokens;
        int at;

        Parser(List<Tok> tokens) {
            this.tokens = tokens;
        }

        /** Parses nodes up to (not including) a statement starting with one of {@code ends}. */
        List<Node> block(String... ends) {
            List<Node> nodes = new ArrayList<>();
            while (at < tokens.size()) {
                Tok t = tokens.get(at);
                if (t.kind == Kind.TEXT) {
                    nodes.add(new Text(t.text));
                    at++;
                } else if (t.kind == Kind.OUTPUT) {
                    nodes.add(new Output(new Exprs(t.text).whole()));
                    at++;
                } else {
                    String word = firstWord(t.text);
                    for (String e : ends) {
                        if (word.equals(e)) {
                            return nodes;
                        }
                    }
                    at++;
                    nodes.add(statement(word, t.text.substring(word.length()).strip()));
                }
            }
            if (ends.length > 0) {
                throw new IllegalArgumentException("missing {% " + ends[ends.length - 1] + " %}");
            }
            return nodes;
        }

        private static String firstWord(String s) {
            int i = 0;
            while (i < s.length() && (Character.isLetterOrDigit(s.charAt(i)) || s.charAt(i) == '_')) {
                i++;
            }
            return s.substring(0, i);
        }

        private String take() {
            return tokens.get(at++).text;
        }

        Node statement(String word, String rest) {
            switch (word) {
                case "if" -> {
                    List<Expr> conds = new ArrayList<>();
                    List<List<Node>> branches = new ArrayList<>();
                    conds.add(new Exprs(rest).whole());
                    while (true) {
                        branches.add(block("elif", "else", "endif"));
                        String t = take();
                        String w = firstWord(t);
                        if (w.equals("elif")) {
                            conds.add(new Exprs(t.substring(4)).whole());
                        } else if (w.equals("else")) {
                            conds.add(null);
                            branches.add(block("endif"));
                            take();
                            break;
                        } else {
                            break;
                        }
                    }
                    return new If(conds, branches);
                }
                case "for" -> {
                    Exprs e = new Exprs(rest);
                    List<String> targets = new ArrayList<>();
                    do {
                        targets.add(e.name());
                    } while (e.accept(","));
                    e.expectWord("in");
                    Expr iterable = e.ternaryNoIf();
                    Expr filter = null;
                    if (e.acceptWord("if")) {
                        filter = e.or();
                    }
                    e.end();
                    List<Node> body = block("else", "endfor");
                    List<Node> orElse = List.of();
                    if (firstWord(take()).equals("else")) {
                        orElse = block("endfor");
                        take();
                    }
                    return new For(targets, iterable, filter, body, orElse);
                }
                case "set" -> {
                    Exprs e = new Exprs(rest);
                    String name = e.name();
                    String attr = null;
                    if (e.accept(".")) {
                        attr = e.name();
                    }
                    if (!e.accept("=")) {
                        e.end();
                        List<Node> body = block("endset");
                        take();
                        return new Set(new Var(name), attr, null, body);
                    }
                    Expr value = e.whole();
                    return new Set(new Var(name), attr, value, null);
                }
                case "macro" -> {
                    Exprs e = new Exprs(rest);
                    String name = e.name();
                    e.expect("(");
                    List<String> params = new ArrayList<>();
                    List<Expr> defaults = new ArrayList<>();
                    if (!e.accept(")")) {
                        do {
                            params.add(e.name());
                            defaults.add(e.accept("=") ? e.ternary() : null);
                        } while (e.accept(","));
                        e.expect(")");
                    }
                    e.end();
                    List<Node> body = block("endmacro");
                    take();
                    return new Macro(name, params, defaults, body);
                }
                case "call" -> {
                    Expr call = new Exprs(rest).whole();
                    List<Node> body = block("endcall");
                    take();
                    return new CallBlock(call, body);
                }
                case "raw" -> {
                    StringBuilder sb = new StringBuilder();
                    while (!(tokens.get(at).kind == Kind.STATEMENT && firstWord(tokens.get(at).text).equals("endraw"))) {
                        Tok t = tokens.get(at++);
                        sb.append(t.kind == Kind.TEXT ? t.text : t.kind == Kind.OUTPUT ? "{{ " + t.text + " }}"
                                : "{% " + t.text + " %}");
                    }
                    at++;
                    return new Text(sb.toString());
                }
                case "generation", "endgeneration" -> {
                    return new Text("");
                }
                default -> throw new IllegalArgumentException("unsupported statement {% " + word + " %}");
            }
        }
    }

    // ================================================================ expressions

    private sealed interface Expr permits Lit, Var, Attr, Index, Slice, Call, Filter, Test, Unary, Binary, Cond,
            ListLit, DictLit {
    }

    private record Lit(Object value) implements Expr {
    }

    private record Var(String name) implements Expr {
    }

    private record Attr(Expr target, String name) implements Expr {
    }

    private record Index(Expr target, Expr index) implements Expr {
    }

    private record Slice(Expr target, Expr start, Expr stop, Expr step) implements Expr {
    }

    private record Call(Expr target, List<Expr> args, Map<String, Expr> kwargs) implements Expr {
    }

    private record Filter(Expr target, String name, List<Expr> args, Map<String, Expr> kwargs) implements Expr {
    }

    private record Test(Expr target, String name, List<Expr> args, boolean negate) implements Expr {
    }

    private record Unary(String op, Expr operand) implements Expr {
    }

    private record Binary(String op, Expr left, Expr right) implements Expr {
    }

    private record Cond(Expr condition, Expr then, Expr otherwise) implements Expr {
    }

    private record ListLit(List<Expr> items) implements Expr {
    }

    private record DictLit(List<Expr> keys, List<Expr> values) implements Expr {
    }

    /** A recursive-descent parser for one expression. */
    private static final class Exprs {
        final List<String> t = new ArrayList<>();
        int at;

        Exprs(String src) {
            int i = 0;
            while (i < src.length()) {
                char c = src.charAt(i);
                if (Character.isWhitespace(c)) {
                    i++;
                } else if (c == '\'' || c == '"') {
                    StringBuilder sb = new StringBuilder("\u0001");   // marks a string literal
                    int j = i + 1;
                    while (j < src.length() && src.charAt(j) != c) {
                        char d = src.charAt(j);
                        if (d == '\\' && j + 1 < src.length()) {
                            char e = src.charAt(++j);
                            sb.append(switch (e) {
                                case 'n' -> '\n';
                                case 't' -> '\t';
                                case 'r' -> '\r';
                                default -> e;
                            });
                        } else {
                            sb.append(d);
                        }
                        j++;
                    }
                    t.add(sb.toString());
                    i = j + 1;
                } else if (Character.isDigit(c)) {
                    int j = i;
                    while (j < src.length() && (Character.isDigit(src.charAt(j)) || src.charAt(j) == '.'
                            && j + 1 < src.length() && Character.isDigit(src.charAt(j + 1)))) {
                        j++;
                    }
                    t.add(src.substring(i, j));
                    i = j;
                } else if (Character.isLetter(c) || c == '_') {
                    int j = i;
                    while (j < src.length() && (Character.isLetterOrDigit(src.charAt(j)) || src.charAt(j) == '_')) {
                        j++;
                    }
                    t.add(src.substring(i, j));
                    i = j;
                } else {
                    String two = i + 1 < src.length() ? src.substring(i, i + 2) : "";
                    if (List.of("==", "!=", "<=", ">=", "//", "**").contains(two)) {
                        t.add(two);
                        i += 2;
                    } else {
                        t.add(String.valueOf(c));
                        i++;
                    }
                }
            }
        }

        String peek() {
            return at < t.size() ? t.get(at) : "";
        }

        boolean accept(String s) {
            if (peek().equals(s)) {
                at++;
                return true;
            }
            return false;
        }

        boolean acceptWord(String s) {
            return accept(s);
        }

        void expect(String s) {
            if (!accept(s)) {
                throw new IllegalArgumentException("expected '" + s + "' but found '" + peek() + "' in " + t);
            }
        }

        void expectWord(String s) {
            expect(s);
        }

        void end() {
            if (at != t.size()) {
                throw new IllegalArgumentException("unexpected '" + peek() + "' in " + t);
            }
        }

        String name() {
            String n = peek();
            if (n.isEmpty() || !(Character.isLetter(n.charAt(0)) || n.charAt(0) == '_')) {
                throw new IllegalArgumentException("expected a name but found '" + n + "' in " + t);
            }
            at++;
            return n;
        }

        Expr whole() {
            Expr e = ternary();
            end();
            return e;
        }

        Expr ternary() {
            Expr e = or();
            if (accept("if")) {
                Expr cond = or();
                Expr otherwise = accept("else") ? ternary() : new Lit(Undefined.INSTANCE);
                return new Cond(cond, e, otherwise);
            }
            return e;
        }

        Expr ternaryNoIf() {
            return or();
        }

        Expr or() {
            Expr e = and();
            while (accept("or")) {
                e = new Binary("or", e, and());
            }
            return e;
        }

        Expr and() {
            Expr e = not();
            while (accept("and")) {
                e = new Binary("and", e, not());
            }
            return e;
        }

        Expr not() {
            if (accept("not")) {
                return new Unary("not", not());
            }
            return compare();
        }

        Expr compare() {
            Expr e = concat();
            while (true) {
                String op = peek();
                if (List.of("==", "!=", "<", ">", "<=", ">=").contains(op)) {
                    at++;
                    e = new Binary(op, e, concat());
                } else if (op.equals("in")) {
                    at++;
                    e = new Binary("in", e, concat());
                } else if (op.equals("not") && at + 1 < t.size() && t.get(at + 1).equals("in")) {
                    at += 2;
                    e = new Unary("not", new Binary("in", e, concat()));
                } else if (op.equals("is")) {
                    at++;
                    boolean negate = accept("not");
                    String name = name();
                    List<Expr> args = new ArrayList<>();
                    if (accept("(")) {
                        if (!accept(")")) {
                            do {
                                args.add(ternary());
                            } while (accept(","));
                            expect(")");
                        }
                    } else if (!peek().isEmpty() && (peek().charAt(0) == '\u0001' || Character.isDigit(peek().charAt(0))
                            || peek().equals("none"))) {
                        args.add(primary());
                    }
                    e = new Test(e, name, args, negate);
                } else {
                    return e;
                }
            }
        }

        Expr concat() {
            Expr e = additive();
            while (accept("~")) {
                e = new Binary("~", e, additive());
            }
            return e;
        }

        Expr additive() {
            Expr e = multiplicative();
            while (peek().equals("+") || peek().equals("-")) {
                String op = t.get(at++);
                e = new Binary(op, e, multiplicative());
            }
            return e;
        }

        Expr multiplicative() {
            Expr e = unary();
            while (List.of("*", "/", "//", "%").contains(peek())) {
                String op = t.get(at++);
                e = new Binary(op, e, unary());
            }
            return e;
        }

        Expr unary() {
            if (accept("-")) {
                return new Unary("-", unary());
            }
            if (accept("+")) {
                return unary();
            }
            return postfix(primary());
        }

        Expr primary() {
            String s = peek();
            at++;
            if (s.isEmpty()) {
                throw new IllegalArgumentException("unexpected end of expression " + t);
            }
            if (s.charAt(0) == '\u0001') {
                StringBuilder sb = new StringBuilder(s.substring(1));
                while (!peek().isEmpty() && peek().charAt(0) == '\u0001') {   // adjacent literals join
                    sb.append(t.get(at++).substring(1));
                }
                return new Lit(sb.toString());
            }
            if (Character.isDigit(s.charAt(0))) {
                return new Lit(s.contains(".") ? (Object) Double.parseDouble(s) : (Object) Long.parseLong(s));
            }
            switch (s) {
                case "true", "True" -> {
                    return new Lit(true);
                }
                case "false", "False" -> {
                    return new Lit(false);
                }
                case "none", "None" -> {
                    return new Lit(null);
                }
                case "(" -> {
                    Expr e = ternary();
                    if (accept(",")) {
                        List<Expr> items = new ArrayList<>(List.of(e));
                        if (!peek().equals(")")) {
                            do {
                                items.add(ternary());
                            } while (accept(",") && !peek().equals(")"));
                        }
                        expect(")");
                        return new ListLit(items);
                    }
                    expect(")");
                    return e;
                }
                case "[" -> {
                    List<Expr> items = new ArrayList<>();
                    if (!accept("]")) {
                        do {
                            if (peek().equals("]")) {
                                break;
                            }
                            items.add(ternary());
                        } while (accept(","));
                        expect("]");
                    }
                    return new ListLit(items);
                }
                case "{" -> {
                    List<Expr> keys = new ArrayList<>();
                    List<Expr> values = new ArrayList<>();
                    if (!accept("}")) {
                        do {
                            if (peek().equals("}")) {
                                break;
                            }
                            keys.add(ternary());
                            expect(":");
                            values.add(ternary());
                        } while (accept(","));
                        expect("}");
                    }
                    return new DictLit(keys, values);
                }
                default -> {
                    if (Character.isLetter(s.charAt(0)) || s.charAt(0) == '_') {
                        return new Var(s);
                    }
                    throw new IllegalArgumentException("unexpected '" + s + "' in " + t);
                }
            }
        }

        Expr postfix(Expr e) {
            while (true) {
                if (accept(".")) {
                    e = new Attr(e, name());
                } else if (accept("[")) {
                    Expr start = null;
                    if (!peek().equals(":")) {
                        start = ternary();
                    }
                    if (accept(":")) {
                        Expr stop = peek().equals(":") || peek().equals("]") ? null : ternary();
                        Expr step = null;
                        if (accept(":") && !peek().equals("]")) {
                            step = ternary();
                        }
                        expect("]");
                        e = new Slice(e, start, stop, step);
                    } else {
                        expect("]");
                        e = new Index(e, start);
                    }
                } else if (accept("(")) {
                    List<Expr> args = new ArrayList<>();
                    Map<String, Expr> kwargs = new LinkedHashMap<>();
                    arguments(args, kwargs);
                    e = new Call(e, args, kwargs);
                } else if (accept("|")) {
                    String name = name();
                    List<Expr> args = new ArrayList<>();
                    Map<String, Expr> kwargs = new LinkedHashMap<>();
                    if (accept("(")) {
                        arguments(args, kwargs);
                    }
                    e = new Filter(e, name, args, kwargs);
                } else {
                    return e;
                }
            }
        }

        void arguments(List<Expr> args, Map<String, Expr> kwargs) {
            if (accept(")")) {
                return;
            }
            do {
                if (peek().equals(")")) {
                    break;
                }
                if (at + 1 < t.size() && t.get(at + 1).equals("=") && Character.isLetter(peek().charAt(0))) {
                    String k = name();
                    expect("=");
                    kwargs.put(k, ternary());
                } else {
                    args.add(ternary());
                }
            } while (accept(","));
            expect(")");
        }
    }

    // ================================================================ evaluation

    /** Jinja's undefined value: falsy, prints as nothing. */
    private enum Undefined { INSTANCE }

    private static final class Scope {
        final Scope parent;
        final Map<String, Object> vars = new HashMap<>();

        Scope(Scope parent) {
            this.parent = parent;
        }

        Object get(String name) {
            for (Scope s = this; s != null; s = s.parent) {
                if (s.vars.containsKey(name)) {
                    return s.vars.get(name);
                }
            }
            return Undefined.INSTANCE;
        }

        void set(String name, Object value) {
            vars.put(name, value);
        }
    }

    /** A mutable object made by {@code namespace()}. */
    private static final class Namespace extends LinkedHashMap<String, Object> {
        private static final long serialVersionUID = 1L;
    }

    private record MacroValue(Macro macro, Scope closure) {
    }

    private static final class Raise extends RuntimeException {
        private static final long serialVersionUID = 1L;

        Raise(String message) {
            super(message);
        }
    }

    private void exec(List<Node> nodes, Scope scope, StringBuilder out) {
        for (Node n : nodes) {
            switch (n) {
                case Text t -> out.append(t.text);
                case Output o -> out.append(str(eval(o.expr, scope)));
                case If i -> {
                    for (int k = 0; k < i.conditions.size(); k++) {
                        Expr c = i.conditions.get(k);
                        if (c == null || truthy(eval(c, scope))) {
                            exec(i.branches.get(k), scope, out);
                            break;
                        }
                    }
                }
                case For f -> loop(f, scope, out);
                case Set s -> {
                    Object value;
                    if (s.value == null) {
                        StringBuilder sb = new StringBuilder();
                        exec(s.body, scope, sb);
                        value = sb.toString();
                    } else {
                        value = eval(s.value, scope);
                    }
                    String name = ((Var) s.target).name;
                    if (s.attr != null) {
                        if (scope.get(name) instanceof Namespace ns) {
                            ns.put(s.attr, value);
                        } else {
                            throw new IllegalArgumentException("cannot set " + s.attr + " on " + name);
                        }
                    } else {
                        scope.set(name, value);
                    }
                }
                case Macro m -> scope.set(m.name, new MacroValue(m, scope));
                case CallBlock c -> out.append(str(eval(c.call, scope)));
            }
        }
    }

    private void loop(For f, Scope scope, StringBuilder out) {
        Object it = eval(f.iterable, scope);
        List<Object> items = new ArrayList<>(iterate(it));
        if (f.filter != null) {
            List<Object> kept = new ArrayList<>();
            for (Object item : items) {
                Scope s = new Scope(scope);
                bind(f.targets, item, s);
                if (truthy(eval(f.filter, s))) {
                    kept.add(item);
                }
            }
            items = kept;
        }
        if (items.isEmpty()) {
            exec(f.orElse, scope, out);
            return;
        }
        for (int i = 0; i < items.size(); i++) {
            // loop bodies see and change the enclosing scope's variables only through namespaces
            Scope s = new Scope(scope);
            bind(f.targets, items.get(i), s);
            Map<String, Object> loop = new HashMap<>();
            loop.put("index", (long) i + 1);
            loop.put("index0", (long) i);
            loop.put("revindex", (long) (items.size() - i));
            loop.put("revindex0", (long) (items.size() - i - 1));
            loop.put("first", i == 0);
            loop.put("last", i == items.size() - 1);
            loop.put("length", (long) items.size());
            loop.put("previtem", i > 0 ? items.get(i - 1) : Undefined.INSTANCE);
            loop.put("nextitem", i + 1 < items.size() ? items.get(i + 1) : Undefined.INSTANCE);
            s.set("loop", loop);
            exec(f.body, s, out);
        }
    }

    private static void bind(List<String> targets, Object item, Scope s) {
        if (targets.size() == 1) {
            s.set(targets.getFirst(), item);
            return;
        }
        List<Object> parts = iterate(item);
        for (int k = 0; k < targets.size(); k++) {
            s.set(targets.get(k), k < parts.size() ? parts.get(k) : Undefined.INSTANCE);
        }
    }

    @SuppressWarnings("unchecked")
    private static List<Object> iterate(Object v) {
        return switch (v) {
            case null -> List.of();
            case Undefined u -> List.of();
            case List<?> l -> (List<Object>) l;
            case Map<?, ?> m -> new ArrayList<>(m.keySet());
            case String s -> s.codePoints().mapToObj(c -> (Object) new String(Character.toChars(c))).toList();
            default -> throw new IllegalArgumentException("cannot iterate over " + v);
        };
    }

    private Object eval(Expr e, Scope scope) {
        return switch (e) {
            case Lit l -> l.value;
            case Var v -> scope.get(v.name);
            case Attr a -> attribute(eval(a.target, scope), a.name);
            case Index i -> index(eval(i.target, scope), eval(i.index, scope));
            case Slice s -> slice(eval(s.target, scope), s.start == null ? null : eval(s.start, scope),
                    s.stop == null ? null : eval(s.stop, scope), s.step == null ? null : eval(s.step, scope));
            case Call c -> call(c, scope);
            case Filter f -> filter(f, scope);
            case Test t -> {
                boolean r = test(t, scope);
                yield t.negate != r;
            }
            case Unary u -> switch (u.op) {
                case "not" -> !truthy(eval(u.operand, scope));
                default -> {
                    Object v = eval(u.operand, scope);
                    yield v instanceof Long l ? (Object) (-l) : (Object) (-num(v));
                }
            };
            case Binary b -> binary(b, scope);
            case Cond c -> truthy(eval(c.condition, scope)) ? eval(c.then, scope) : eval(c.otherwise, scope);
            case ListLit l -> {
                List<Object> list = new ArrayList<>();
                for (Expr x : l.items) {
                    list.add(eval(x, scope));
                }
                yield list;
            }
            case DictLit d -> {
                Map<String, Object> map = new LinkedHashMap<>();
                for (int k = 0; k < d.keys.size(); k++) {
                    map.put(str(eval(d.keys.get(k), scope)), eval(d.values.get(k), scope));
                }
                yield map;
            }
        };
    }

    private static Object attribute(Object target, String name) {
        if (target instanceof Map<?, ?> m) {
            if (m.containsKey(name)) {
                return m.get(name);
            }
            return List.of("items", "keys", "values", "get").contains(name) ? new BoundMethod(target, name)
                    : Undefined.INSTANCE;
        }
        return new BoundMethod(target, name);
    }

    /** A method looked up on a value, called later: {@code s.strip()}, {@code d.items()}. */
    private record BoundMethod(Object self, String name) {
    }

    private static Object index(Object target, Object key) {
        if (target instanceof Map<?, ?> m) {
            Object k = key instanceof Long || key instanceof Double ? key : str(key);
            return m.containsKey(k) ? m.get(k) : m.containsKey(str(key)) ? m.get(str(key)) : Undefined.INSTANCE;
        }
        if (target instanceof List<?> l) {
            int i = (int) (long) toLong(key);
            if (i < 0) {
                i += l.size();
            }
            return i >= 0 && i < l.size() ? l.get(i) : Undefined.INSTANCE;
        }
        if (target instanceof String s) {
            int i = (int) (long) toLong(key);
            if (i < 0) {
                i += s.length();
            }
            return i >= 0 && i < s.length() ? String.valueOf(s.charAt(i)) : Undefined.INSTANCE;
        }
        return Undefined.INSTANCE;
    }

    private static Object slice(Object target, Object start, Object stop, Object step) {
        List<Object> items = target instanceof String s
                ? s.chars().mapToObj(c -> (Object) String.valueOf((char) c)).toList() : iterate(target);
        int n = items.size();
        int st = step == null ? 1 : (int) (long) toLong(step);
        List<Object> out = new ArrayList<>();
        if (st > 0) {
            int a = start == null ? 0 : clamp((int) (long) toLong(start), n);
            int b = stop == null ? n : clamp((int) (long) toLong(stop), n);
            for (int i = a; i < b; i += st) {
                out.add(items.get(i));
            }
        } else {
            int a = start == null ? n - 1 : Math.min(clamp((int) (long) toLong(start), n), n - 1);
            int b = stop == null ? -1 : clamp((int) (long) toLong(stop), n);
            for (int i = a; i > b; i += st) {
                out.add(items.get(i));
            }
        }
        if (target instanceof String) {
            return String.join("", out.stream().map(Object::toString).toList());
        }
        return out;
    }

    private static int clamp(int i, int n) {
        if (i < 0) {
            i += n;
        }
        return Math.max(0, Math.min(n, i));
    }

    private Object call(Call c, Scope scope) {
        List<Object> args = new ArrayList<>();
        for (Expr a : c.args) {
            args.add(eval(a, scope));
        }
        Map<String, Object> kwargs = new LinkedHashMap<>();
        for (Map.Entry<String, Expr> k : c.kwargs.entrySet()) {
            kwargs.put(k.getKey(), eval(k.getValue(), scope));
        }
        if (c.target instanceof Var v && scope.get(v.name) == Undefined.INSTANCE) {
            switch (v.name) {
                case "raise_exception" -> throw new Raise(str(args.isEmpty() ? "" : args.getFirst()));
                case "range" -> {
                    long a = args.size() > 1 ? toLong(args.get(0)) : 0;
                    long b = toLong(args.get(args.size() > 1 ? 1 : 0));
                    long st = args.size() > 2 ? toLong(args.get(2)) : 1;
                    List<Object> r = new ArrayList<>();
                    for (long i = a; st > 0 ? i < b : i > b; i += st) {
                        r.add(i);
                    }
                    return r;
                }
                case "namespace" -> {
                    Namespace ns = new Namespace();
                    ns.putAll(kwargs);
                    return ns;
                }
                case "dict" -> {
                    return new LinkedHashMap<>(kwargs);
                }
                case "strftime_now" -> {
                    String fmt = str(args.getFirst()).replace("%d", "dd").replace("%b", "MMM").replace("%Y", "yyyy")
                            .replace("%m", "MM").replace("%B", "MMMM");
                    return LocalDate.now().format(DateTimeFormatter.ofPattern(fmt, Locale.ROOT));
                }
                default -> throw new IllegalArgumentException(v.name + " is not defined");
            }
        }
        Object target = eval(c.target, scope);
        if (target instanceof MacroValue m) {
            Scope s = new Scope(m.closure);
            for (int i = 0; i < m.macro.params.size(); i++) {
                String p = m.macro.params.get(i);
                Object value = i < args.size() ? args.get(i) : kwargs.containsKey(p) ? kwargs.get(p)
                        : m.macro.defaults.get(i) != null ? eval(m.macro.defaults.get(i), scope) : Undefined.INSTANCE;
                s.set(p, value);
            }
            StringBuilder sb = new StringBuilder();
            exec(m.macro.body, s, sb);
            return sb.toString();
        }
        if (target instanceof BoundMethod b) {
            return method(b.self, b.name, args);
        }
        throw new IllegalArgumentException("cannot call " + target);
    }

    @SuppressWarnings("unchecked")
    private static Object method(Object self, String name, List<Object> args) {
        if (self instanceof String s) {
            String a0 = args.isEmpty() ? null : str(args.getFirst());
            return switch (name) {
                case "strip" -> a0 == null ? s.strip() : stripChars(stripChars(s, a0, true), a0, false);
                case "lstrip" -> a0 == null ? s.stripLeading() : stripChars(s, a0, true);
                case "rstrip" -> a0 == null ? s.stripTrailing() : stripChars(s, a0, false);
                case "upper" -> s.toUpperCase(Locale.ROOT);
                case "lower" -> s.toLowerCase(Locale.ROOT);
                case "title" -> title(s);
                case "capitalize" -> capitalize(s);
                case "startswith" -> args.getFirst() instanceof List<?> l ? l.stream().anyMatch(p -> s.startsWith(str(p)))
                        : s.startsWith(a0);
                case "endswith" -> args.getFirst() instanceof List<?> l ? l.stream().anyMatch(p -> s.endsWith(str(p)))
                        : s.endsWith(a0);
                case "replace" -> s.replace(a0, str(args.get(1)));
                case "split" -> {
                    List<Object> parts = new ArrayList<>();
                    if (a0 == null) {
                        for (String p : s.strip().split("\\s+")) {
                            if (!p.isEmpty()) {
                                parts.add(p);
                            }
                        }
                    } else {
                        int limit = args.size() > 1 ? (int) (long) toLong(args.get(1)) : -1;
                        int from = 0;
                        while (true) {
                            int i = s.indexOf(a0, from);
                            if (i < 0 || (limit >= 0 && parts.size() == limit)) {
                                break;
                            }
                            parts.add(s.substring(from, i));
                            from = i + a0.length();
                        }
                        parts.add(s.substring(from));
                    }
                    yield parts;
                }
                case "find" -> (long) s.indexOf(a0);
                case "count" -> (long) (s.length() - s.replace(a0, "").length()) / Math.max(1, a0.length());
                case "join" -> String.join(s, iterate(args.getFirst()).stream().map(ChatTemplate::str).toList());
                case "format" -> {
                    String r = s;
                    for (Object a : args) {
                        r = r.replaceFirst("\\{\\}", java.util.regex.Matcher.quoteReplacement(str(a)));
                    }
                    yield r;
                }
                default -> throw new IllegalArgumentException("unsupported string method " + name);
            };
        }
        if (self instanceof Map<?, ?> m) {
            Map<Object, Object> map = (Map<Object, Object>) m;
            return switch (name) {
                case "items" -> {
                    List<Object> items = new ArrayList<>();
                    for (Map.Entry<?, ?> e : map.entrySet()) {
                        items.add(java.util.Arrays.asList(e.getKey(), e.getValue()));
                    }
                    yield items;
                }
                case "keys" -> new ArrayList<>(map.keySet());
                case "values" -> new ArrayList<>(map.values());
                case "get" -> map.containsKey(args.getFirst()) ? map.get(args.getFirst())
                        : args.size() > 1 ? args.get(1) : null;
                default -> throw new IllegalArgumentException("unsupported dict method " + name);
            };
        }
        if (self instanceof List<?> l && name.equals("index")) {
            return (long) l.indexOf(args.getFirst());
        }
        throw new IllegalArgumentException("unsupported method " + name + " on " + self);
    }

    private static String stripChars(String s, String chars, boolean left) {
        int a = 0;
        int b = s.length();
        if (left) {
            while (a < b && chars.indexOf(s.charAt(a)) >= 0) {
                a++;
            }
        } else {
            while (b > a && chars.indexOf(s.charAt(b - 1)) >= 0) {
                b--;
            }
        }
        return s.substring(a, b);
    }

    private static String title(String s) {
        StringBuilder sb = new StringBuilder();
        boolean start = true;
        for (char c : s.toCharArray()) {
            sb.append(start ? Character.toUpperCase(c) : Character.toLowerCase(c));
            start = !Character.isLetterOrDigit(c);
        }
        return sb.toString();
    }

    private static String capitalize(String s) {
        return s.isEmpty() ? s : Character.toUpperCase(s.charAt(0)) + s.substring(1).toLowerCase(Locale.ROOT);
    }

    @SuppressWarnings("unchecked")
    private Object filter(Filter f, Scope scope) {
        Object v = eval(f.target, scope);
        List<Object> args = new ArrayList<>();
        for (Expr a : f.args) {
            args.add(eval(a, scope));
        }
        Map<String, Object> kwargs = new LinkedHashMap<>();
        for (Map.Entry<String, Expr> k : f.kwargs.entrySet()) {
            kwargs.put(k.getKey(), eval(k.getValue(), scope));
        }
        return switch (f.name) {
            case "trim" -> str(v).strip();
            case "upper" -> str(v).toUpperCase(Locale.ROOT);
            case "lower" -> str(v).toLowerCase(Locale.ROOT);
            case "title" -> title(str(v));
            case "capitalize" -> capitalize(str(v));
            case "string" -> str(v);
            case "safe", "e", "escape" -> v;
            case "int" -> v instanceof String s ? (Object) Long.parseLong(s.strip()) : (Object) toLong(v);
            case "float" -> num(v);
            case "length", "count" -> (long) (v instanceof String s ? s.length() : v instanceof Map<?, ?> m ? m.size()
                    : iterate(v).size());
            case "first" -> iterate(v).isEmpty() ? Undefined.INSTANCE : iterate(v).getFirst();
            case "last" -> iterate(v).isEmpty() ? Undefined.INSTANCE : iterate(v).getLast();
            case "list" -> new ArrayList<>(iterate(v));
            case "reverse" -> {
                if (v instanceof String s) {
                    yield new StringBuilder(s).reverse().toString();
                }
                List<Object> l = new ArrayList<>(iterate(v));
                Collections.reverse(l);
                yield l;
            }
            case "join" -> String.join(args.isEmpty() ? "" : str(args.getFirst()),
                    iterate(v).stream().map(x -> kwargs.containsKey("attribute")
                            ? str(attribute(x, str(kwargs.get("attribute")))) : str(x)).toList());
            case "default", "d" -> v == Undefined.INSTANCE || (args.size() > 1 && truthy(args.get(1)) && !truthy(v))
                    ? (args.isEmpty() ? "" : args.getFirst()) : v;
            case "tojson" -> toJson(v, kwargs.containsKey("indent") ? (int) (long) toLong(kwargs.get("indent"))
                    : args.isEmpty() ? -1 : (int) (long) toLong(args.getFirst()), 0);
            case "items" -> method(v, "items", List.of());
            case "indent" -> {
                int width = args.isEmpty() ? 4 : (int) (long) toLong(args.getFirst());
                boolean first = args.size() > 1 && truthy(args.get(1)) || truthy(kwargs.getOrDefault("first", false));
                String pad = " ".repeat(width);
                String[] lines = str(v).split("\n", -1);
                StringBuilder sb = new StringBuilder();
                for (int i = 0; i < lines.length; i++) {
                    if (i > 0) {
                        sb.append('\n');
                    }
                    sb.append((i > 0 || first) && !lines[i].isEmpty() ? pad + lines[i] : lines[i]);
                }
                yield sb.toString();
            }
            case "replace" -> str(v).replace(str(args.get(0)), str(args.get(1)));
            case "map" -> {
                List<Object> out = new ArrayList<>();
                for (Object x : iterate(v)) {
                    if (kwargs.containsKey("attribute")) {
                        out.add(attribute(x, str(kwargs.get("attribute"))));
                    } else {
                        out.add(applySimpleFilter(str(args.getFirst()), x));
                    }
                }
                yield out;
            }
            case "selectattr", "rejectattr" -> {
                boolean select = f.name.equals("selectattr");
                List<Object> out = new ArrayList<>();
                for (Object x : iterate(v)) {
                    Object a = attribute(x, str(args.getFirst()));
                    boolean ok = args.size() < 2 ? truthy(a)
                            : simpleTest(str(args.get(1)), a, args.size() > 2 ? args.get(2) : null);
                    if (ok == select) {
                        out.add(x);
                    }
                }
                yield out;
            }
            case "select", "reject" -> {
                boolean select = f.name.equals("select");
                List<Object> out = new ArrayList<>();
                for (Object x : iterate(v)) {
                    boolean ok = args.isEmpty() ? truthy(x)
                            : simpleTest(str(args.getFirst()), x, args.size() > 1 ? args.get(1) : null);
                    if (ok == select) {
                        out.add(x);
                    }
                }
                yield out;
            }
            case "unique" -> new ArrayList<>(new java.util.LinkedHashSet<>(iterate(v)));
            case "dictsort" -> {
                List<Object> items = (List<Object>) method(v, "items", List.of());
                items.sort((a, b) -> str(((List<Object>) a).getFirst()).compareTo(str(((List<Object>) b).getFirst())));
                yield items;
            }
            case "abs" -> v instanceof Long l ? (Object) Math.abs(l) : (Object) Math.abs(num(v));
            case "round" -> (double) Math.round(num(v));
            default -> throw new IllegalArgumentException("unsupported filter " + f.name);
        };
    }

    private static Object applySimpleFilter(String name, Object v) {
        return switch (name) {
            case "trim" -> str(v).strip();
            case "lower" -> str(v).toLowerCase(Locale.ROOT);
            case "upper" -> str(v).toUpperCase(Locale.ROOT);
            case "string" -> str(v);
            case "int" -> toLong(v);
            case "length" -> (long) iterate(v).size();
            default -> throw new IllegalArgumentException("unsupported filter in map: " + name);
        };
    }

    private boolean test(Test t, Scope scope) {
        Object v = eval(t.target, scope);
        Object arg = t.args.isEmpty() ? null : eval(t.args.getFirst(), scope);
        return simpleTest(t.name, v, arg);
    }

    private static boolean simpleTest(String name, Object v, Object arg) {
        return switch (name) {
            case "defined" -> v != Undefined.INSTANCE;
            case "undefined" -> v == Undefined.INSTANCE;
            case "none" -> v == null;
            case "string" -> v instanceof String;
            case "number" -> v instanceof Number;
            case "integer" -> v instanceof Long;
            case "float" -> v instanceof Double;
            case "boolean" -> v instanceof Boolean;
            case "true" -> Boolean.TRUE.equals(v);
            case "false" -> Boolean.FALSE.equals(v);
            case "mapping" -> v instanceof Map;
            case "sequence", "iterable" -> v instanceof List || v instanceof String || v instanceof Map;
            case "callable" -> v instanceof MacroValue || v instanceof BoundMethod;
            case "equalto", "eq", "==", "sameas" -> equal(v, arg);
            case "ne", "!=" -> !equal(v, arg);
            case "in" -> contains(arg, v);
            case "odd" -> toLong(v) % 2 != 0;
            case "even" -> toLong(v) % 2 == 0;
            case "divisibleby" -> toLong(v) % toLong(arg) == 0;
            case "lower" -> str(v).equals(str(v).toLowerCase(Locale.ROOT));
            case "upper" -> str(v).equals(str(v).toUpperCase(Locale.ROOT));
            default -> throw new IllegalArgumentException("unsupported test " + name);
        };
    }

    private Object binary(Binary b, Scope scope) {
        switch (b.op) {
            case "and" -> {
                Object l = eval(b.left, scope);
                return truthy(l) ? eval(b.right, scope) : l;
            }
            case "or" -> {
                Object l = eval(b.left, scope);
                return truthy(l) ? l : eval(b.right, scope);
            }
            default -> {
            }
        }
        Object l = eval(b.left, scope);
        Object r = eval(b.right, scope);
        return switch (b.op) {
            case "==" -> equal(l, r);
            case "!=" -> !equal(l, r);
            case "<" -> compare(l, r) < 0;
            case ">" -> compare(l, r) > 0;
            case "<=" -> compare(l, r) <= 0;
            case ">=" -> compare(l, r) >= 0;
            case "in" -> contains(r, l);
            case "~" -> str(l) + str(r);
            case "+" -> {
                if (l instanceof String || r instanceof String) {
                    yield str(l) + str(r);
                }
                if (l instanceof List<?> a && r instanceof List<?> c) {
                    List<Object> out = new ArrayList<>(a);
                    out.addAll(c);
                    yield out;
                }
                yield arith(l, r, '+');
            }
            case "-" -> arith(l, r, '-');
            case "*" -> {
                if (l instanceof String s) {
                    yield s.repeat((int) (long) toLong(r));
                }
                yield arith(l, r, '*');
            }
            case "/" -> num(l) / num(r);
            case "//" -> l instanceof Long && r instanceof Long ? (Object) Math.floorDiv((Long) l, (Long) r)
                    : (Object) Math.floor(num(l) / num(r));
            case "%" -> l instanceof Long && r instanceof Long ? (Object) Math.floorMod((Long) l, (Long) r)
                    : (Object) (num(l) % num(r));
            default -> throw new IllegalArgumentException("unsupported operator " + b.op);
        };
    }

    private static Object arith(Object l, Object r, char op) {
        if (l instanceof Long a && r instanceof Long c) {
            return switch (op) {
                case '+' -> a + c;
                case '-' -> a - c;
                default -> a * c;
            };
        }
        double a = num(l);
        double c = num(r);
        return switch (op) {
            case '+' -> a + c;
            case '-' -> a - c;
            default -> a * c;
        };
    }

    private static boolean contains(Object container, Object item) {
        return switch (container) {
            case String s -> s.contains(str(item));
            case Map<?, ?> m -> m.containsKey(item) || m.containsKey(str(item));
            case List<?> l -> l.stream().anyMatch(x -> equal(x, item));
            default -> false;
        };
    }

    private static boolean equal(Object a, Object b) {
        if (a instanceof Number x && b instanceof Number y) {
            return x.doubleValue() == y.doubleValue();
        }
        if (a == Undefined.INSTANCE || b == Undefined.INSTANCE) {
            return a == b;
        }
        return Objects.equals(a, b);
    }

    private static int compare(Object a, Object b) {
        if (a instanceof Number || b instanceof Number) {
            return Double.compare(num(a), num(b));
        }
        return str(a).compareTo(str(b));
    }

    private static boolean truthy(Object v) {
        return switch (v) {
            case null -> false;
            case Undefined u -> false;
            case Boolean b -> b;
            case String s -> !s.isEmpty();
            case Long l -> l != 0;
            case Double d -> d != 0;
            case List<?> l -> !l.isEmpty();
            case Map<?, ?> m -> !m.isEmpty() || m instanceof Namespace;
            default -> true;
        };
    }

    private static double num(Object v) {
        if (v instanceof Number n) {
            return n.doubleValue();
        }
        if (v instanceof Boolean b) {
            return b ? 1 : 0;
        }
        if (v instanceof String s) {
            return Double.parseDouble(s.strip());
        }
        throw new IllegalArgumentException("not a number: " + v);
    }

    private static Long toLong(Object v) {
        if (v instanceof Long l) {
            return l;
        }
        if (v instanceof Integer i) {
            return (long) i;
        }
        return (long) num(v);
    }

    private static String str(Object v) {
        return switch (v) {
            case null -> "None";
            case Undefined u -> "";
            case String s -> s;
            case Boolean b -> b ? "True" : "False";
            case List<?> l -> {
                StringBuilder sb = new StringBuilder("[");
                for (int i = 0; i < l.size(); i++) {
                    if (i > 0) {
                        sb.append(", ");
                    }
                    sb.append(repr(l.get(i)));
                }
                yield sb.append(']').toString();
            }
            case Map<?, ?> m -> {
                StringBuilder sb = new StringBuilder("{");
                boolean first = true;
                for (Map.Entry<?, ?> e : m.entrySet()) {
                    if (!first) {
                        sb.append(", ");
                    }
                    first = false;
                    sb.append(repr(e.getKey())).append(": ").append(repr(e.getValue()));
                }
                yield sb.append('}').toString();
            }
            default -> String.valueOf(v);
        };
    }

    private static String repr(Object v) {
        return v instanceof String s ? "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'" : str(v);
    }

    /** Python's {@code json.dumps(v, ensure_ascii=False, indent=indent)}, as Hugging Face's {@code tojson}. */
    private static String toJson(Object v, int indent, int depth) {
        String nl = indent < 0 ? "" : "\n" + " ".repeat(indent * (depth + 1));
        String close = indent < 0 ? "" : "\n" + " ".repeat(indent * depth);
        String sep = indent < 0 ? ", " : ",";
        return switch (v) {
            case null -> "null";
            case Undefined u -> "null";
            case Boolean b -> b.toString();
            case Long l -> l.toString();
            case Integer i -> i.toString();
            case Double d -> String.valueOf(d);
            case String s -> {
                StringBuilder sb = new StringBuilder("\"");
                for (char c : s.toCharArray()) {
                    switch (c) {
                        case '"' -> sb.append("\\\"");
                        case '\\' -> sb.append("\\\\");
                        case '\n' -> sb.append("\\n");
                        case '\r' -> sb.append("\\r");
                        case '\t' -> sb.append("\\t");
                        case '\b' -> sb.append("\\b");
                        case '\f' -> sb.append("\\f");
                        default -> {
                            if (c < 0x20) {
                                sb.append(String.format("\\u%04x", (int) c));
                            } else {
                                sb.append(c);
                            }
                        }
                    }
                }
                yield sb.append('"').toString();
            }
            case List<?> l -> {
                if (l.isEmpty()) {
                    yield "[]";
                }
                StringBuilder sb = new StringBuilder("[");
                for (int i = 0; i < l.size(); i++) {
                    sb.append(i > 0 ? sep : "").append(nl).append(toJson(l.get(i), indent, depth + 1));
                }
                yield sb.append(close).append(']').toString();
            }
            case Map<?, ?> m -> {
                if (m.isEmpty()) {
                    yield "{}";
                }
                StringBuilder sb = new StringBuilder("{");
                boolean first = true;
                for (Map.Entry<?, ?> e : m.entrySet()) {
                    sb.append(first ? "" : sep).append(nl).append(toJson(String.valueOf(e.getKey()), indent, depth + 1))
                            .append(": ").append(toJson(e.getValue(), indent, depth + 1));
                    first = false;
                }
                yield sb.append(close).append('}').toString();
            }
            default -> toJson(String.valueOf(v), indent, depth);
        };
    }
}
