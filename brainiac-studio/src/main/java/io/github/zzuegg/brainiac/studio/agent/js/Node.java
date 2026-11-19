package io.github.zzuegg.brainiac.studio.agent.js;

import java.util.List;

/** The parsed script: statements and expressions. */
sealed interface Node {

    /** @return the line it starts on */
    int line();

    // ------------------------------------------------------------- patterns (what names bind)

    /** What a declaration or parameter binds: a name, or a destructuring pattern. */
    sealed interface Pattern extends Node {
    }

    record Name(String name, int line) implements Pattern, Node {
    }

    /** One entry of an object pattern: {@code key: target = fallback}, or {@code ...rest}. */
    record PropPattern(Node key, boolean computed, Pattern target, Node fallback, boolean rest) {
    }

    record ObjectPattern(List<PropPattern> props, int line) implements Pattern {
    }

    /** An array pattern's entries; null for a hole. */
    record ArrayPattern(List<Pattern> items, List<Node> fallbacks, Pattern rest, int line) implements Pattern {
    }

    /** A pattern with a fallback, in a parameter list. */
    record Defaulted(Pattern target, Node fallback, int line) implements Pattern {
    }

    // ------------------------------------------------------------------------ statements

    record Program(List<Node> body, int line) implements Node {
    }

    record Block(List<Node> body, int line) implements Node {
    }

    record Declare(String kind, List<Pattern> targets, List<Node> values, int line) implements Node {
    }

    record FunctionDecl(String name, Func func, int line) implements Node {
    }

    record If(Node test, Node then, Node otherwise, int line) implements Node {
    }

    record For(Node init, Node test, Node update, Node body, int line) implements Node {
    }

    record ForOf(String kind, Pattern target, Node iterable, Node body, boolean in, int line) implements Node {
    }

    record While(Node test, Node body, boolean doWhile, int line) implements Node {
    }

    record Return(Node value, int line) implements Node {
    }

    record Break(int line) implements Node {
    }

    record Continue(int line) implements Node {
    }

    record Throw(Node value, int line) implements Node {
    }

    record Try(Block body, Pattern param, Block handler, Block finalizer, int line) implements Node {
    }

    record Switch(Node subject, List<Node> tests, List<List<Node>> bodies, int line) implements Node {
    }

    record ExprStatement(Node expr, int line) implements Node {
    }

    record Empty(int line) implements Node {
    }

    // ----------------------------------------------------------------------- expressions

    record Literal(Object value, int line) implements Node {
    }

    record Template(List<String> strings, List<Node> values, int line) implements Node {
    }

    record RegexLit(String source, String flags, int line) implements Node {
    }

    record ArrayLit(List<Node> items, int line) implements Node {
    }

    /** One entry of an object literal: {@code key: value}, or {@code ...value} when key is null. */
    record Prop(Node key, boolean computed, Node value) {
    }

    record ObjectLit(List<Prop> props, int line) implements Node {
    }

    record Spread(Node value, int line) implements Node {
    }

    /** A function's parameters and body; an arrow with an expression body has {@code expr}. */
    record Func(String name, List<Pattern> params, Pattern rest, List<Node> body, Node expr, boolean arrow, int line)
            implements Node {
    }

    record Member(Node object, Node key, boolean computed, boolean optional, int line) implements Node {
    }

    record Call(Node callee, List<Node> args, boolean optional, int line) implements Node {
    }

    record New(Node callee, List<Node> args, int line) implements Node {
    }

    record Unary(String op, Node value, int line) implements Node {
    }

    record Update(String op, boolean prefix, Node target, int line) implements Node {
    }

    record Binary(String op, Node left, Node right, int line) implements Node {
    }

    record Logical(String op, Node left, Node right, int line) implements Node {
    }

    record Conditional(Node test, Node then, Node otherwise, int line) implements Node {
    }

    record Assign(String op, Node target, Node value, int line) implements Node {
    }

    record Sequence(List<Node> items, int line) implements Node {
    }

    record This(int line) implements Node {
    }
}
