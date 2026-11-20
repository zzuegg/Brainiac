package io.github.zzuegg.brainiac.studio.agent.js;

import io.github.zzuegg.brainiac.studio.agent.js.Lexer.Kind;
import io.github.zzuegg.brainiac.studio.agent.js.Lexer.Token;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** Reads tokens into a {@link Node.Program}. */
final class Parser {

    private static final Map<String, Integer> BINARY = Map.ofEntries(
            Map.entry("??", 1), Map.entry("||", 2), Map.entry("&&", 3), Map.entry("|", 4), Map.entry("^", 5),
            Map.entry("&", 6), Map.entry("==", 7), Map.entry("!=", 7), Map.entry("===", 7), Map.entry("!==", 7),
            Map.entry("<", 8), Map.entry(">", 8), Map.entry("<=", 8), Map.entry(">=", 8), Map.entry("instanceof", 8),
            Map.entry("in", 8), Map.entry("<<", 9), Map.entry(">>", 9), Map.entry(">>>", 9), Map.entry("+", 10),
            Map.entry("-", 10), Map.entry("*", 11), Map.entry("/", 11), Map.entry("%", 11), Map.entry("**", 12));

    private static final List<String> ASSIGN = List.of("=", "+=", "-=", "*=", "/=", "%=", "**=", "&&=", "||=", "??=",
            "<<=", ">>=", ">>>=", "&=", "|=", "^=");

    private final List<Token> tokens;
    private int at;
    private boolean noIn;   // in a for head, 'in' ends the expression

    private Parser(List<Token> tokens) {
        this.tokens = tokens;
    }

    static Node.Program parse(String src) {
        Parser p = new Parser(Lexer.tokens(src));
        List<Node> body = new ArrayList<>();
        while (p.peek().kind() != Kind.EOF) {
            body.add(p.statement());
        }
        return new Node.Program(body, 1);
    }

    // ------------------------------------------------------------------------ tokens

    private Token peek() {
        return tokens.get(at);
    }

    private Token peek(int ahead) {
        return tokens.get(Math.min(at + ahead, tokens.size() - 1));
    }

    private Token next() {
        Token t = tokens.get(at);
        if (t.kind() != Kind.EOF) {
            at++;
        }
        return t;
    }

    private boolean eat(String p) {
        if (peek().is(p)) {
            at++;
            return true;
        }
        return false;
    }

    private Token expect(String p) {
        Token t = peek();
        if (!t.is(p)) {
            throw error(t, "expected '" + p + "'");
        }
        return next();
    }

    private ScriptError error(Token t, String what) {
        String got = switch (t.kind()) {
            case EOF -> "the end of the script";
            case STRING -> "a string";
            case NUMBER -> "a number";
            case TEMPLATE -> "a template";
            default -> "'" + t.text() + "'";
        };
        return new ScriptError("line " + t.line() + ": " + what + " but found " + got);
    }

    private boolean isName(Token t) {
        return t.kind() == Kind.NAME || t.kind() == Kind.KEYWORD && Lexer.SOFT.contains(t.text())
                && !t.text().equals("undefined");
    }

    private String name() {
        Token t = peek();
        if (!isName(t)) {
            throw error(t, "expected a name");
        }
        return next().text();
    }

    // a property name may be any word, keywords too
    private String propertyName() {
        Token t = next();
        if (t.kind() == Kind.NAME || t.kind() == Kind.KEYWORD) {
            return t.text();
        }
        throw error(t, "expected a property name");
    }

    private void endStatement() {
        if (eat(";")) {
            return;
        }
        Token t = peek();
        if (t.is("}") || t.kind() == Kind.EOF || t.newlineBefore()) {
            return;
        }
        throw error(t, "expected ';' or a new line");
    }

    // -------------------------------------------------------------------- statements

    private Node statement() {
        Token t = peek();
        int line = t.line();
        if (t.kind() == Kind.PUNCT) {
            if (t.is("{")) {
                return block();
            }
            if (t.is(";")) {
                next();
                return new Node.Empty(line);
            }
        }
        if (t.kind() == Kind.KEYWORD) {
            switch (t.text()) {
                case "var", "let", "const" -> {
                    Node d = declaration();
                    endStatement();
                    return d;
                }
                case "function" -> {
                    return functionDeclaration();
                }
                case "async" -> {
                    if (peek(1).is("function") && !peek(1).newlineBefore()) {
                        next();
                        return functionDeclaration();
                    }
                }
                case "class" -> throw new ScriptError("line " + line + ": classes are not supported; use functions "
                        + "and plain objects");
                case "if" -> {
                    next();
                    expect("(");
                    Node test = expression();
                    expect(")");
                    Node then = statement();
                    Node otherwise = eat("else") ? statement() : null;
                    return new Node.If(test, then, otherwise, line);
                }
                case "for" -> {
                    return forStatement();
                }
                case "while" -> {
                    next();
                    expect("(");
                    Node test = expression();
                    expect(")");
                    return new Node.While(test, statement(), false, line);
                }
                case "do" -> {
                    next();
                    Node body = statement();
                    expect("while");
                    expect("(");
                    Node test = expression();
                    expect(")");
                    eat(";");
                    return new Node.While(test, body, true, line);
                }
                case "return" -> {
                    next();
                    Node value = null;
                    if (!peek().is(";") && !peek().is("}") && peek().kind() != Kind.EOF && !peek().newlineBefore()) {
                        value = expression();
                    }
                    endStatement();
                    return new Node.Return(value, line);
                }
                case "break" -> {
                    next();
                    if (isName(peek()) && !peek().newlineBefore()) {
                        throw new ScriptError("line " + line + ": labels are not supported");
                    }
                    endStatement();
                    return new Node.Break(line);
                }
                case "continue" -> {
                    next();
                    endStatement();
                    return new Node.Continue(line);
                }
                case "throw" -> {
                    next();
                    Node value = expression();
                    endStatement();
                    return new Node.Throw(value, line);
                }
                case "try" -> {
                    return tryStatement();
                }
                case "switch" -> {
                    return switchStatement();
                }
                default -> {
                    // an expression statement
                }
            }
        }
        Node e = expression();
        endStatement();
        return new Node.ExprStatement(e, line);
    }

    private Node.Block block() {
        int line = expect("{").line();
        List<Node> body = new ArrayList<>();
        while (!peek().is("}")) {
            if (peek().kind() == Kind.EOF) {
                throw error(peek(), "expected '}'");
            }
            body.add(statement());
        }
        next();
        return new Node.Block(body, line);
    }

    private Node.Declare declaration() {
        Token kw = next();
        List<Node.Pattern> targets = new ArrayList<>();
        List<Node> values = new ArrayList<>();
        do {
            targets.add(pattern());
            values.add(eat("=") ? assignment() : null);
        } while (eat(","));
        return new Node.Declare(kw.text(), targets, values, kw.line());
    }

    private Node functionDeclaration() {
        int line = expect("function").line();
        eat("*");
        String n = name();
        Node.Func f = functionRest(n, line);
        return new Node.FunctionDecl(n, f, line);
    }

    // after the name: the parameters and the body
    private Node.Func functionRest(String n, int line) {
        expect("(");
        List<Node.Pattern> params = new ArrayList<>();
        Node.Pattern rest = null;
        while (!peek().is(")")) {
            if (eat("...")) {
                rest = pattern();
                break;
            }
            params.add(parameter());
            if (!eat(",")) {
                break;
            }
        }
        expect(")");
        boolean saved = noIn;
        noIn = false;
        Node.Block body = block();
        noIn = saved;
        return new Node.Func(n, params, rest, body.body(), null, false, line);
    }

    private Node.Pattern parameter() {
        int line = peek().line();
        Node.Pattern p = pattern();
        return eat("=") ? new Node.Defaulted(p, assignment(), line) : p;
    }

    private Node.Pattern pattern() {
        Token t = peek();
        if (t.is("{")) {
            next();
            List<Node.PropPattern> props = new ArrayList<>();
            while (!peek().is("}")) {
                if (eat("...")) {
                    props.add(new Node.PropPattern(null, false, new Node.Name(name(), t.line()), null, true));
                } else {
                    Node key;
                    boolean computed = false;
                    Token k = peek();
                    if (eat("[")) {
                        key = assignment();
                        expect("]");
                        computed = true;
                    } else if (k.kind() == Kind.STRING) {
                        key = new Node.Literal(next().text(), k.line());
                    } else {
                        key = new Node.Literal(propertyName(), k.line());
                    }
                    Node.Pattern target = eat(":") ? pattern()
                            : new Node.Name((String) ((Node.Literal) key).value(), k.line());
                    Node fallback = eat("=") ? assignment() : null;
                    props.add(new Node.PropPattern(key, computed, target, fallback, false));
                }
                if (!eat(",")) {
                    break;
                }
            }
            expect("}");
            return new Node.ObjectPattern(props, t.line());
        }
        if (t.is("[")) {
            next();
            List<Node.Pattern> items = new ArrayList<>();
            List<Node> fallbacks = new ArrayList<>();
            Node.Pattern rest = null;
            while (!peek().is("]")) {
                if (peek().is(",")) {
                    next();
                    items.add(null);
                    fallbacks.add(null);
                    continue;
                }
                if (eat("...")) {
                    rest = pattern();
                    break;
                }
                items.add(pattern());
                fallbacks.add(eat("=") ? assignment() : null);
                if (!eat(",")) {
                    break;
                }
            }
            expect("]");
            return new Node.ArrayPattern(items, fallbacks, rest, t.line());
        }
        return new Node.Name(name(), t.line());
    }

    private Node forStatement() {
        int line = expect("for").line();
        if (peek().is("await")) {
            next();
        }
        expect("(");
        Node init = null;
        Token t = peek();
        if (t.is("var") || t.is("let") || t.is("const")) {
            // for (const x of y), for (const k in o), or for (let i = 0; ...)
            int save = at;
            next();
            Node.Pattern target = pattern();
            if (peek().is("of") || peek().is("in")) {
                boolean in = next().text().equals("in");
                Node iterable = in ? expression() : assignment();
                expect(")");
                return new Node.ForOf(t.text(), target, iterable, statement(), in, line);
            }
            at = save;
            noIn = true;
            init = declaration();
            noIn = false;
        } else if (!t.is(";")) {
            int save = at;
            if (isName(t) && (peek(1).is("of") || peek(1).is("in"))) {
                next();
                boolean in = next().text().equals("in");
                Node iterable = in ? expression() : assignment();
                expect(")");
                return new Node.ForOf("", new Node.Name(t.text(), t.line()), iterable, statement(), in, line);
            }
            at = save;
            noIn = true;
            init = new Node.ExprStatement(expression(), t.line());
            noIn = false;
        }
        expect(";");
        Node test = peek().is(";") ? null : expression();
        expect(";");
        Node update = peek().is(")") ? null : expression();
        expect(")");
        return new Node.For(init, test, update, statement(), line);
    }

    private Node tryStatement() {
        int line = expect("try").line();
        Node.Block body = block();
        Node.Pattern param = null;
        Node.Block handler = null;
        Node.Block finalizer = null;
        if (eat("catch")) {
            if (eat("(")) {
                param = pattern();
                expect(")");
            }
            handler = block();
        }
        if (eat("finally")) {
            finalizer = block();
        }
        if (handler == null && finalizer == null) {
            throw error(peek(), "expected 'catch' or 'finally'");
        }
        return new Node.Try(body, param, handler, finalizer, line);
    }

    private Node switchStatement() {
        int line = expect("switch").line();
        expect("(");
        Node subject = expression();
        expect(")");
        expect("{");
        List<Node> tests = new ArrayList<>();
        List<List<Node>> bodies = new ArrayList<>();
        while (!eat("}")) {
            if (eat("default")) {
                tests.add(null);
            } else {
                expect("case");
                tests.add(expression());
            }
            expect(":");
            List<Node> body = new ArrayList<>();
            while (!peek().is("case") && !peek().is("default") && !peek().is("}")) {
                if (peek().kind() == Kind.EOF) {
                    throw error(peek(), "expected '}'");
                }
                body.add(statement());
            }
            bodies.add(body);
        }
        return new Node.Switch(subject, tests, bodies, line);
    }

    // ------------------------------------------------------------------- expressions

    private Node expression() {
        Node first = assignment();
        if (!peek().is(",")) {
            return first;
        }
        List<Node> items = new ArrayList<>(List.of(first));
        while (eat(",")) {
            items.add(assignment());
        }
        return new Node.Sequence(items, first.line());
    }

    private Node assignment() {
        Node arrow = arrowFunction();
        if (arrow != null) {
            return arrow;
        }
        Token start = peek();
        Node left = conditional();
        Token op = peek();
        if (op.kind() == Kind.PUNCT && ASSIGN.contains(op.text())) {
            if (!(left instanceof Node.Name || left instanceof Node.Member)) {
                if (op.text().equals("=") && (left instanceof Node.ArrayLit || left instanceof Node.ObjectLit)) {
                    throw new ScriptError("line " + start.line() + ": assigning to a pattern is not supported; "
                            + "declare it with const or let");
                }
                throw new ScriptError("line " + op.line() + ": cannot assign to this");
            }
            next();
            return new Node.Assign(op.text(), left, assignment(), op.line());
        }
        return left;
    }

    // x => ..., (a, b) => ..., async x => ..., async (a) => ...
    private Node arrowFunction() {
        int save = at;
        int line = peek().line();
        if (peek().is("async") && !peek(1).newlineBefore() && (isName(peek(1)) || peek(1).is("("))) {
            next();
        }
        Token t = peek();
        List<Node.Pattern> params = new ArrayList<>();
        Node.Pattern rest = null;
        if (isName(t) && peek(1).is("=>")) {
            next();
            params.add(new Node.Name(t.text(), t.line()));
        } else if (t.is("(") && closesBeforeArrow()) {
            next();
            while (!peek().is(")")) {
                if (eat("...")) {
                    rest = pattern();
                    break;
                }
                params.add(parameter());
                if (!eat(",")) {
                    break;
                }
            }
            expect(")");
        } else {
            at = save;
            return null;
        }
        expect("=>");
        if (peek().is("{")) {
            boolean saved = noIn;
            noIn = false;
            Node.Block body = block();
            noIn = saved;
            return new Node.Func(null, params, rest, body.body(), null, true, line);
        }
        return new Node.Func(null, params, rest, null, assignment(), true, line);
    }

    // whether the bracket here closes right before =>
    private boolean closesBeforeArrow() {
        int depth = 0;
        for (int i = at; i < tokens.size(); i++) {
            Token t = tokens.get(i);
            if (t.kind() == Kind.EOF) {
                return false;
            }
            if (t.is("(") || t.is("[") || t.is("{")) {
                depth++;
            } else if (t.is(")") || t.is("]") || t.is("}")) {
                depth--;
                if (depth == 0) {
                    return i + 1 < tokens.size() && tokens.get(i + 1).is("=>");
                }
            }
        }
        return false;
    }

    private Node conditional() {
        Node test = binary(1);
        if (peek().is("?")) {
            int line = next().line();
            boolean saved = noIn;
            noIn = false;
            Node then = assignment();
            noIn = saved;
            expect(":");
            Node otherwise = assignment();
            return new Node.Conditional(test, then, otherwise, line);
        }
        return test;
    }

    // precedence climbing: only operators at least as strong as min join here
    private Node binary(int min) {
        Node left = unary();
        while (true) {
            Token op = peek();
            Integer prec = (op.kind() == Kind.PUNCT || op.kind() == Kind.KEYWORD) ? BINARY.get(op.text()) : null;
            if (prec == null || prec < min || noIn && op.is("in")) {
                return left;
            }
            next();
            Node right = op.text().equals("**") ? binary(prec) : binary(prec + 1);   // ** binds to the right
            left = switch (op.text()) {
                case "&&", "||", "??" -> new Node.Logical(op.text(), left, right, op.line());
                default -> new Node.Binary(op.text(), left, right, op.line());
            };
        }
    }

    private Node unary() {
        Token t = peek();
        if (t.kind() == Kind.PUNCT && (t.is("!") || t.is("-") || t.is("+") || t.is("~"))
                || t.is("typeof") || t.is("void") || t.is("delete")) {
            next();
            return new Node.Unary(t.text(), unary(), t.line());
        }
        if (t.is("await")) {
            next();
            return unary();   // tools answer at once, so await just takes the value
        }
        if (t.is("++") || t.is("--")) {
            next();
            Node target = unary();
            return new Node.Update(t.text(), true, target, t.line());
        }
        Node e = postfix();
        Token p = peek();
        if ((p.is("++") || p.is("--")) && !p.newlineBefore()) {
            next();
            return new Node.Update(p.text(), false, e, p.line());
        }
        return e;
    }

    private Node postfix() {
        Node e;
        Token t = peek();
        if (t.is("new")) {
            next();
            Node callee = primary();
            while (peek().is(".")) {
                next();
                callee = new Node.Member(callee, new Node.Literal(propertyName(), t.line()), false, false, t.line());
            }
            List<Node> args = peek().is("(") ? arguments() : List.of();
            e = new Node.New(callee, args, t.line());
        } else {
            e = primary();
        }
        while (true) {
            Token p = peek();
            if (p.is(".")) {
                next();
                e = new Node.Member(e, new Node.Literal(propertyName(), p.line()), false, false, p.line());
            } else if (p.is("?.")) {
                next();
                if (peek().is("(")) {
                    e = new Node.Call(e, arguments(), true, p.line());
                } else if (eat("[")) {
                    Node key = expression();
                    expect("]");
                    e = new Node.Member(e, key, true, true, p.line());
                } else {
                    e = new Node.Member(e, new Node.Literal(propertyName(), p.line()), false, true, p.line());
                }
            } else if (p.is("[")) {
                next();
                boolean saved = noIn;
                noIn = false;
                Node key = expression();
                noIn = saved;
                expect("]");
                e = new Node.Member(e, key, true, false, p.line());
            } else if (p.is("(")) {
                e = new Node.Call(e, arguments(), false, p.line());
            } else if (p.kind() == Kind.TEMPLATE && !p.newlineBefore()) {
                throw new ScriptError("line " + p.line() + ": tagged templates are not supported");
            } else {
                return e;
            }
        }
    }

    private List<Node> arguments() {
        expect("(");
        boolean saved = noIn;
        noIn = false;
        List<Node> args = new ArrayList<>();
        while (!peek().is(")")) {
            int line = peek().line();
            args.add(eat("...") ? new Node.Spread(assignment(), line) : assignment());
            if (!eat(",")) {
                break;
            }
        }
        expect(")");
        noIn = saved;
        return args;
    }

    private Node primary() {
        Token t = next();
        int line = t.line();
        switch (t.kind()) {
            case NUMBER -> {
                return new Node.Literal(t.number(), line);
            }
            case STRING -> {
                return new Node.Literal(t.text(), line);
            }
            case TEMPLATE -> {
                List<String> strings = new ArrayList<>();
                List<Node> values = new ArrayList<>();
                List<String> parts = t.parts();
                for (int i = 0; i < parts.size(); i++) {
                    if (i % 2 == 0) {
                        strings.add(parts.get(i));
                    } else {
                        Node.Program p = Parser.parse(parts.get(i));
                        if (p.body().size() != 1 || !(p.body().getFirst() instanceof Node.ExprStatement es)) {
                            throw new ScriptError("line " + line + ": a ${...} in a template must hold one expression");
                        }
                        values.add(es.expr());
                    }
                }
                return new Node.Template(strings, values, line);
            }
            case REGEX -> {
                return new Node.RegexLit(t.text(), t.flags(), line);
            }
            case NAME -> {
                return new Node.Name(t.text(), line);
            }
            case KEYWORD -> {
                switch (t.text()) {
                    case "true" -> {
                        return new Node.Literal(Boolean.TRUE, line);
                    }
                    case "false" -> {
                        return new Node.Literal(Boolean.FALSE, line);
                    }
                    case "null" -> {
                        return new Node.Literal(null, line);
                    }
                    case "undefined" -> {
                        return new Node.Literal(Interpreter.UNDEFINED, line);
                    }
                    case "this" -> {
                        return new Node.This(line);
                    }
                    case "function" -> {
                        eat("*");
                        String n = isName(peek()) ? next().text() : null;
                        return functionRest(n, line);
                    }
                    case "async" -> {
                        if (peek().is("function")) {
                            next();
                            String n = isName(peek()) ? next().text() : null;
                            return functionRest(n, line);
                        }
                        return new Node.Name("async", line);
                    }
                    case "of" -> {
                        return new Node.Name("of", line);
                    }
                    default -> throw error(t, "expected a value");
                }
            }
            case PUNCT -> {
                if (t.is("(")) {
                    boolean saved = noIn;
                    noIn = false;
                    Node e = expression();
                    noIn = saved;
                    expect(")");
                    return e;
                }
                if (t.is("[")) {
                    List<Node> items = new ArrayList<>();
                    while (!peek().is("]")) {
                        if (peek().is(",")) {
                            next();
                            items.add(new Node.Literal(Interpreter.UNDEFINED, line));
                            continue;
                        }
                        int l = peek().line();
                        items.add(eat("...") ? new Node.Spread(assignment(), l) : assignment());
                        if (!eat(",")) {
                            break;
                        }
                    }
                    expect("]");
                    return new Node.ArrayLit(items, line);
                }
                if (t.is("{")) {
                    return objectLiteral(line);
                }
                throw error(t, "expected a value");
            }
            default -> throw error(t, "expected a value");
        }
    }

    private Node objectLiteral(int line) {
        List<Node.Prop> props = new ArrayList<>();
        boolean saved = noIn;
        noIn = false;
        while (!peek().is("}")) {
            if (eat("...")) {
                props.add(new Node.Prop(null, false, assignment()));
            } else {
                Token k = peek();
                if ((k.is("async") || k.is("get") || k.is("set")) && !peek(1).is(":") && !peek(1).is(",")
                        && !peek(1).is("(") && !peek(1).is("}")) {
                    if (!k.is("async")) {
                        throw new ScriptError("line " + k.line() + ": getters and setters are not supported");
                    }
                    next();
                    k = peek();
                }
                Node key;
                boolean computed = false;
                if (eat("[")) {
                    key = assignment();
                    expect("]");
                    computed = true;
                } else if (k.kind() == Kind.STRING) {
                    key = new Node.Literal(next().text(), k.line());
                } else if (k.kind() == Kind.NUMBER) {
                    key = new Node.Literal(Interpreter.str(next().number()), k.line());
                } else {
                    key = new Node.Literal(propertyName(), k.line());
                }
                if (peek().is("(")) {
                    Node.Func f = functionRest(computed ? null : (String) ((Node.Literal) key).value(), k.line());
                    props.add(new Node.Prop(key, computed, f));
                } else if (eat(":")) {
                    props.add(new Node.Prop(key, computed, assignment()));
                } else if (!computed && (k.kind() == Kind.NAME || isName(k))) {
                    props.add(new Node.Prop(key, false, new Node.Name(k.text(), k.line())));
                } else {
                    throw error(peek(), "expected ':'");
                }
            }
            if (!eat(",")) {
                break;
            }
        }
        expect("}");
        noIn = saved;
        return new Node.ObjectLit(props, line);
    }
}
