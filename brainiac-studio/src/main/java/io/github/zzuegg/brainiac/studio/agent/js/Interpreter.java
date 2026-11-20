package io.github.zzuegg.brainiac.studio.agent.js;

import io.github.zzuegg.brainiac.studio.agent.js.Values.Break;
import io.github.zzuegg.brainiac.studio.agent.js.Values.Closure;
import io.github.zzuegg.brainiac.studio.agent.js.Values.Continue;
import io.github.zzuegg.brainiac.studio.agent.js.Values.Fn;
import io.github.zzuegg.brainiac.studio.agent.js.Values.JsDate;
import io.github.zzuegg.brainiac.studio.agent.js.Values.JsError;
import io.github.zzuegg.brainiac.studio.agent.js.Values.JsMap;
import io.github.zzuegg.brainiac.studio.agent.js.Values.JsRegex;
import io.github.zzuegg.brainiac.studio.agent.js.Values.JsSet;
import io.github.zzuegg.brainiac.studio.agent.js.Values.Native;
import io.github.zzuegg.brainiac.studio.agent.js.Values.Return;
import io.github.zzuegg.brainiac.studio.agent.js.Values.Scope;
import io.github.zzuegg.brainiac.studio.agent.js.Values.Thrown;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Runs scripts written in a large part of JavaScript: variables, functions and arrows, objects,
 * arrays, destructuring, spreads, template strings, regular expressions, loops, exceptions, and
 * the common methods of strings, arrays, objects, Math, JSON, Map and Set. There are no classes,
 * generators, getters or modules, and {@code await} simply takes the value, as the host's
 * functions answer at once. A script reaches nothing but what the host gives it.
 */
public final class Interpreter {

    /** JavaScript's {@code undefined}; {@code null} is Java's null. */
    public static final Object UNDEFINED = new Object() {
        @Override
        public String toString() {
            return "undefined";
        }
    };

    /** What a script may use from outside, and how far it may go. */
    public interface Host {

        /**
         * The functions the script can call, under a global name, such as {@code tools}.
         *
         * @return name to function, each taking and returning plain Java values (maps, lists,
         *         strings, numbers, booleans, null)
         */
        Map<String, Map<String, HostFunction>> namespaces();

        /**
         * Takes a line the script logged.
         *
         * @param line the line
         */
        void log(String line);
    }

    /** A host function. */
    @FunctionalInterface
    public interface HostFunction {

        /**
         * Runs it.
         *
         * @param args the script's arguments, as plain Java values
         * @return the result, as plain Java values; a {@link HostError} becomes an error the script can catch
         */
        Object call(List<Object> args);
    }

    /** An error from a host function, which the script sees as a thrown error. */
    public static final class HostError extends RuntimeException {

        private static final long serialVersionUID = 1L;

        /** The error's name in the script. */
        public final String kind;

        /**
         * Creates it.
         *
         * @param kind its name, such as {@code ToolError}
         * @param message what went wrong
         */
        public HostError(String kind, String message) {
            super(message, null, false, false);
            this.kind = kind;
        }
    }

    /**
     * Limits.
     *
     * @param steps the most evaluation steps
     * @param millis the most time spent in the script itself, not counting host calls
     * @param depth the most nested function calls
     * @param chars the longest string or the most array items
     */
    public record Limits(long steps, long millis, int depth, int chars) {

        /** Limits that suit agent scripts. */
        public static final Limits DEFAULT = new Limits(20_000_000, 30_000, 200, 5_000_000);
    }

    private final Host host;
    private final Limits limits;
    private final Scope globals = new Scope(null);
    private long steps;
    private int depth;
    private long started;
    private long hostNanos;

    /**
     * Creates an interpreter.
     *
     * @param host what the script may use
     * @param limits how far it may go
     */
    public Interpreter(Host host, Limits limits) {
        this.host = host;
        this.limits = limits;
        Builtins.install(this, globals);
        for (Map.Entry<String, Map<String, HostFunction>> ns : host.namespaces().entrySet()) {
            Map<String, Object> o = new LinkedHashMap<>();
            for (Map.Entry<String, HostFunction> f : ns.getValue().entrySet()) {
                HostFunction hf = f.getValue();
                o.put(f.getKey(), new Native(f.getKey(), (in, self, args) -> in.callHost(f.getKey(), hf, args)));
            }
            globals.declare(ns.getKey(), o, true);
        }
    }

    /**
     * Runs a script. A top-level {@code return} gives its result; otherwise the value of its last
     * expression statement does.
     *
     * @param source the script
     * @return the result, as plain Java values (numbers that are whole become longs)
     * @throws ScriptError if it cannot be read, throws an error it does not catch, or goes past a limit
     */
    public Object run(String source) {
        Node.Program program = Parser.parse(source);
        started = System.nanoTime();
        depth = 0;
        try {
            Object last = UNDEFINED;
            Scope scope = new Scope(globals);
            hoist(program.body(), scope);
            for (Node n : program.body()) {
                Object v = exec(n, scope);
                if (completes(n)) {
                    last = v;
                }
            }
            return toJava(last);
        } catch (Return r) {
            return toJava(r.value);
        } catch (Thrown t) {
            throw new ScriptError("uncaught " + Builtins.describeError(this, t.value));
        } catch (Break | Continue e) {
            throw new ScriptError("break or continue outside a loop");
        } catch (StackOverflowError e) {
            throw new ScriptError("the script nests calls too deeply");
        }
    }

    // ------------------------------------------------------------------------- limits

    private void tick() {
        if (++steps > limits.steps()) {
            throw new ScriptError("the script ran " + limits.steps() + " steps, its limit; it may loop forever");
        }
        if ((steps & 0x3FF) == 0 && (System.nanoTime() - started - hostNanos) / 1_000_000 > limits.millis()) {
            throw new ScriptError("the script ran " + limits.millis() / 1000 + " seconds, its limit");
        }
    }

    Object callHost(String name, HostFunction f, List<Object> args) {
        List<Object> plain = new ArrayList<>();
        for (Object a : args) {
            plain.add(toJava(a));
        }
        long t = System.nanoTime();
        try {
            return fromJava(f.call(plain));
        } catch (HostError e) {
            JsError err = new JsError(e.kind, e.getMessage());
            err.put("tool", name);
            throw new Thrown(err);
        } finally {
            hostNanos += System.nanoTime() - t;
        }
    }

    String checkLength(String s) {
        if (s.length() > limits.chars()) {
            throw new ScriptError("a string grew past " + limits.chars() + " characters, the limit");
        }
        return s;
    }

    void checkSize(List<?> l) {
        if (l.size() > limits.chars()) {
            throw new ScriptError("an array grew past " + limits.chars() + " items, the limit");
        }
    }

    // ------------------------------------------------------------------------- errors

    Thrown typeError(int line, String message) {
        return new Thrown(new JsError("TypeError", message + " (line " + line + ")"));
    }

    Thrown error(String kind, String message) {
        return new Thrown(new JsError(kind, message));
    }

    // --------------------------------------------------------------------- statements

    private void hoist(List<Node> body, Scope scope) {
        for (Node n : body) {
            if (n instanceof Node.FunctionDecl f) {
                scope.declare(f.name(), new Closure(f.func(), scope, UNDEFINED), false);
            }
        }
    }

    private Object execBlock(List<Node> body, Scope scope) {
        hoist(body, scope);
        Object last = UNDEFINED;
        for (Node n : body) {
            Object v = exec(n, scope);
            if (completes(n)) {
                last = v;
            }
        }
        return last;
    }

    /** Whether a statement gives the value a script ends with, as its last expression does. */
    private static boolean completes(Node n) {
        return n instanceof Node.ExprStatement || n instanceof Node.Block || n instanceof Node.If || n instanceof Node.Try;
    }

    Object exec(Node n, Scope scope) {
        tick();
        switch (n) {
            case Node.ExprStatement e -> {
                return eval(e.expr(), scope);
            }
            case Node.Declare d -> {
                for (int i = 0; i < d.targets().size(); i++) {
                    Node v = d.values().get(i);
                    Object value = v == null ? UNDEFINED : eval(v, scope);
                    if (v instanceof Node.Func f && f.name() == null && d.targets().get(i) instanceof Node.Name nm) {
                        value = named((Closure) value, nm.name());
                    }
                    bind(d.targets().get(i), value, scope, d.kind().equals("const"));
                }
                return UNDEFINED;
            }
            case Node.FunctionDecl f -> {
                return UNDEFINED;   // hoisted
            }
            case Node.Block b -> {
                return execBlock(b.body(), new Scope(scope));
            }
            case Node.If i -> {
                if (truthy(eval(i.test(), scope))) {
                    return exec(i.then(), scope);
                } else if (i.otherwise() != null) {
                    return exec(i.otherwise(), scope);
                }
                return UNDEFINED;
            }
            case Node.For f -> {
                forLoop(f, scope);
                return UNDEFINED;
            }
            case Node.ForOf f -> {
                forOf(f, scope);
                return UNDEFINED;
            }
            case Node.While w -> {
                boolean first = true;
                while ((w.doWhile() && first) || truthy(eval(w.test(), scope))) {
                    first = false;
                    tick();
                    try {
                        exec(w.body(), scope);
                    } catch (Break b) {
                        break;
                    } catch (Continue c) {
                        // the next round
                    }
                }
                return UNDEFINED;
            }
            case Node.Return r -> throw new Return(r.value() == null ? UNDEFINED : eval(r.value(), scope));
            case Node.Break b -> throw Break.INSTANCE;
            case Node.Continue c -> throw Continue.INSTANCE;
            case Node.Throw t -> throw new Thrown(eval(t.value(), scope));
            case Node.Try t -> {
                return tryStatement(t, scope);
            }
            case Node.Switch s -> {
                switchStatement(s, scope);
                return UNDEFINED;
            }
            case Node.Empty e -> {
                return UNDEFINED;
            }
            default -> {
                return eval(n, scope);
            }
        }
    }

    private void forLoop(Node.For f, Scope outer) {
        Scope loop = new Scope(outer);
        if (f.init() != null) {
            exec(f.init(), loop);
        }
        boolean perIteration = f.init() instanceof Node.Declare d && !d.kind().equals("var");
        Scope iter = loop;
        while (true) {
            if (perIteration) {   // each round gets its own copy, so closures keep that round's values
                Scope next = new Scope(outer);
                next.vars.putAll(iter.vars);
                iter = next;
            }
            if (f.test() != null && !truthy(eval(f.test(), iter))) {
                break;
            }
            tick();
            try {
                exec(f.body(), iter);
            } catch (Break b) {
                break;
            } catch (Continue c) {
                // the update follows
            }
            if (f.update() != null) {
                eval(f.update(), iter);
            }
        }
    }

    private void forOf(Node.ForOf f, Scope outer) {
        Object subject = eval(f.iterable(), outer);
        List<Object> items;
        if (f.in()) {
            items = new ArrayList<>(Builtins.keys(this, subject, f.line()));
        } else {
            items = iterate(subject, f.line());
        }
        for (Object item : items) {
            tick();
            Scope s = new Scope(outer);
            if (f.kind().isEmpty()) {
                assignName(((Node.Name) f.target()).name(), item, outer, f.line());
            } else {
                bind(f.target(), item, s, f.kind().equals("const"));
            }
            try {
                exec(f.body(), s);
            } catch (Break b) {
                break;
            } catch (Continue c) {
                // the next item
            }
        }
    }

    private Object tryStatement(Node.Try t, Scope scope) {
        try {
            return exec(t.body(), scope);
        } catch (Thrown e) {
            if (t.handler() == null) {
                throw e;
            }
            Scope s = new Scope(scope);
            if (t.param() != null) {
                bind(t.param(), e.value, s, false);
            }
            return execBlock(t.handler().body(), s);
        } finally {
            if (t.finalizer() != null) {
                exec(t.finalizer(), scope);
            }
        }
    }

    private void switchStatement(Node.Switch s, Scope outer) {
        Object v = eval(s.subject(), outer);
        Scope scope = new Scope(outer);
        int start = -1;
        for (int i = 0; i < s.tests().size(); i++) {
            Node test = s.tests().get(i);
            if (test != null && strictEquals(v, eval(test, scope))) {
                start = i;
                break;
            }
        }
        if (start < 0) {
            start = s.tests().indexOf(null);
        }
        if (start < 0) {
            return;
        }
        try {
            for (int i = start; i < s.bodies().size(); i++) {
                for (Node n : s.bodies().get(i)) {
                    exec(n, scope);
                }
            }
        } catch (Break b) {
            // the end of the switch
        }
    }

    // ------------------------------------------------------------------------ binding

    void bind(Node.Pattern p, Object value, Scope scope, boolean constant) {
        switch (p) {
            case Node.Name n -> scope.declare(n.name(), value, constant);
            case Node.Defaulted d -> bind(d.target(), value == UNDEFINED ? eval(d.fallback(), scope) : value, scope,
                    constant);
            case Node.ObjectPattern o -> {
                if (value == null || value == UNDEFINED) {
                    throw typeError(o.line(), "cannot destructure " + str(value));
                }
                List<String> used = new ArrayList<>();
                for (Node.PropPattern pp : o.props()) {
                    if (pp.rest()) {
                        Map<String, Object> rest = new LinkedHashMap<>();
                        if (value instanceof Map<?, ?> m) {
                            for (Map.Entry<?, ?> e : m.entrySet()) {
                                if (!used.contains(String.valueOf(e.getKey()))) {
                                    rest.put(String.valueOf(e.getKey()), e.getValue());
                                }
                            }
                        }
                        bind(pp.target(), rest, scope, constant);
                        continue;
                    }
                    String key = pp.computed() ? key(eval(pp.key(), scope)) : (String) ((Node.Literal) pp.key()).value();
                    used.add(key);
                    Object v = get(value, key, o.line());
                    if (v == UNDEFINED && pp.fallback() != null) {
                        v = eval(pp.fallback(), scope);
                    }
                    bind(pp.target(), v, scope, constant);
                }
            }
            case Node.ArrayPattern a -> {
                List<Object> items = iterate(value, a.line());
                for (int i = 0; i < a.items().size(); i++) {
                    Node.Pattern t = a.items().get(i);
                    if (t == null) {
                        continue;
                    }
                    Object v = i < items.size() ? items.get(i) : UNDEFINED;
                    if (v == UNDEFINED && a.fallbacks().get(i) != null) {
                        v = eval(a.fallbacks().get(i), scope);
                    }
                    bind(t, v, scope, constant);
                }
                if (a.rest() != null) {
                    bind(a.rest(), new ArrayList<>(items.subList(Math.min(items.size(), a.items().size()), items.size())),
                            scope, constant);
                }
            }
        }
    }

    private void assignName(String name, Object value, Scope scope, int line) {
        Scope s = scope.find(name);
        if (s == null) {
            throw new Thrown(new JsError("ReferenceError", name + " is not defined (line " + line + ")"));
        }
        if (s.constants != null && s.constants.contains(name)) {
            throw typeError(line, "cannot assign to the constant " + name);
        }
        s.vars.put(name, value);
    }

    // -------------------------------------------------------------------- expressions

    Object eval(Node n, Scope scope) {
        tick();
        switch (n) {
            case Node.Literal l -> {
                return l.value();
            }
            case Node.Name nm -> {
                Scope s = scope.find(nm.name());
                if (s == null) {
                    throw new Thrown(new JsError("ReferenceError", nm.name() + " is not defined (line " + nm.line() + ")"));
                }
                return s.vars.get(nm.name());
            }
            case Node.Template t -> {
                StringBuilder b = new StringBuilder(t.strings().getFirst());
                for (int i = 0; i < t.values().size(); i++) {
                    b.append(str(eval(t.values().get(i), scope))).append(t.strings().get(i + 1));
                }
                return checkLength(b.toString());
            }
            case Node.RegexLit r -> {
                try {
                    return new JsRegex(r.source(), r.flags());
                } catch (RuntimeException e) {
                    throw new Thrown(new JsError("SyntaxError", "bad regular expression /" + r.source() + "/: "
                            + e.getMessage() + " (line " + r.line() + ")"));
                }
            }
            case Node.ArrayLit a -> {
                List<Object> out = new ArrayList<>();
                for (Node item : a.items()) {
                    if (item instanceof Node.Spread sp) {
                        out.addAll(iterate(eval(sp.value(), scope), sp.line()));
                    } else {
                        out.add(eval(item, scope));
                    }
                }
                checkSize(out);
                return out;
            }
            case Node.ObjectLit o -> {
                Map<String, Object> out = new LinkedHashMap<>();
                for (Node.Prop p : o.props()) {
                    if (p.key() == null) {
                        Object v = eval(p.value(), scope);
                        if (v instanceof Map<?, ?> m) {
                            for (Map.Entry<?, ?> e : m.entrySet()) {
                                out.put(String.valueOf(e.getKey()), e.getValue());
                            }
                        } else if (v instanceof List<?> l) {
                            for (int i = 0; i < l.size(); i++) {
                                out.put(String.valueOf(i), l.get(i));
                            }
                        } else if (v instanceof String s) {
                            for (int i = 0; i < s.length(); i++) {
                                out.put(String.valueOf(i), String.valueOf(s.charAt(i)));
                            }
                        }
                        continue;
                    }
                    String key = p.computed() ? key(eval(p.key(), scope)) : (String) ((Node.Literal) p.key()).value();
                    Object v = eval(p.value(), scope);
                    if (p.value() instanceof Node.Func f && f.name() == null && v instanceof Closure c) {
                        v = named(c, key);
                    }
                    out.put(key, v);
                }
                return out;
            }
            case Node.Func f -> {
                return new Closure(f, scope, f.arrow() ? thisOf(scope) : UNDEFINED);
            }
            case Node.This t -> {
                return thisOf(scope);
            }
            case Node.Member m -> {
                Object o = eval(m.object(), scope);
                if (m.optional() && (o == null || o == UNDEFINED)) {
                    return UNDEFINED;
                }
                Object key = m.computed() ? eval(m.key(), scope) : ((Node.Literal) m.key()).value();
                return get(o, key, m.line());
            }
            case Node.Call c -> {
                return call(c, scope);
            }
            case Node.New nw -> {
                Object f = eval(nw.callee(), scope);
                List<Object> args = args(nw.args(), scope);
                if (f instanceof Native nat && nat.construct != null) {
                    return nat.construct.run(this, UNDEFINED, args);
                }
                if (f instanceof Closure c) {   // a constructor function: this is a fresh object
                    Map<String, Object> self = new LinkedHashMap<>();
                    Object r = c.call(this, self, args);
                    return r instanceof Map<?, ?> || r instanceof List<?> ? r : self;
                }
                throw typeError(nw.line(), str(f) + " cannot be used with new");
            }
            case Node.Unary u -> {
                return unary(u, scope);
            }
            case Node.Update u -> {
                double old = num(eval(u.target(), scope));
                double v = u.op().equals("++") ? old + 1 : old - 1;
                assign(u.target(), v, scope, u.line());
                return u.prefix() ? v : old;
            }
            case Node.Binary b -> {
                return binary(b.op(), eval(b.left(), scope), eval(b.right(), scope), b.line());
            }
            case Node.Logical l -> {
                Object left = eval(l.left(), scope);
                return switch (l.op()) {
                    case "&&" -> truthy(left) ? eval(l.right(), scope) : left;
                    case "||" -> truthy(left) ? left : eval(l.right(), scope);
                    default -> left == null || left == UNDEFINED ? eval(l.right(), scope) : left;
                };
            }
            case Node.Conditional c -> {
                return truthy(eval(c.test(), scope)) ? eval(c.then(), scope) : eval(c.otherwise(), scope);
            }
            case Node.Assign a -> {
                return assignment(a, scope);
            }
            case Node.Sequence s -> {
                Object last = UNDEFINED;
                for (Node x : s.items()) {
                    last = eval(x, scope);
                }
                return last;
            }
            case Node.Spread s -> throw new ScriptError("line " + s.line() + ": '...' is not allowed here");
            default -> throw new ScriptError("line " + n.line() + ": cannot evaluate this");
        }
    }

    private Object thisOf(Scope scope) {
        Scope s = scope.find("this");
        return s == null ? UNDEFINED : s.vars.get("this");
    }

    private static Closure named(Closure c, String name) {
        if (c.func.name() != null) {
            return c;
        }
        Node.Func f = c.func;
        return new Closure(new Node.Func(name, f.params(), f.rest(), f.body(), f.expr(), f.arrow(), f.line()), c.scope,
                c.self);
    }

    private List<Object> args(List<Node> nodes, Scope scope) {
        List<Object> out = new ArrayList<>();
        for (Node a : nodes) {
            if (a instanceof Node.Spread s) {
                out.addAll(iterate(eval(s.value(), scope), s.line()));
            } else {
                out.add(eval(a, scope));
            }
        }
        return out;
    }

    private Object call(Node.Call c, Scope scope) {
        Object self = UNDEFINED;
        Object f;
        String what;
        if (c.callee() instanceof Node.Member m) {
            self = eval(m.object(), scope);
            if (m.optional() && (self == null || self == UNDEFINED)) {
                return UNDEFINED;
            }
            Object key = m.computed() ? eval(m.key(), scope) : ((Node.Literal) m.key()).value();
            f = get(self, key, m.line());
            what = describe(m) ;
        } else {
            f = eval(c.callee(), scope);
            what = c.callee() instanceof Node.Name nm ? nm.name() : "this expression";
        }
        if (c.optional() && (f == null || f == UNDEFINED)) {
            return UNDEFINED;
        }
        if (!(f instanceof Fn fn)) {
            throw typeError(c.line(), what + " is not a function");
        }
        return fn.call(this, self, args(c.args(), scope));
    }

    private static String describe(Node.Member m) {
        String obj = m.object() instanceof Node.Name n ? n.name()
                : m.object() instanceof Node.Member inner ? describe(inner) : "(…)";
        return m.computed() ? obj + "[…]" : obj + "." + ((Node.Literal) m.key()).value();
    }

    /** Calls a function the script wrote. */
    Object invoke(Closure c, Object self, List<Object> args) {
        if (++depth > limits.depth()) {
            depth = 0;
            throw new ScriptError("functions call each other more than " + limits.depth() + " deep");
        }
        try {
            Scope s = new Scope(c.scope);
            if (!c.func.arrow()) {
                s.declare("this", self, false);
                s.declare("arguments", new ArrayList<>(args), false);
            }
            List<Node.Pattern> params = c.func.params();
            for (int i = 0; i < params.size(); i++) {
                bind(params.get(i), i < args.size() ? args.get(i) : UNDEFINED, s, false);
            }
            if (c.func.rest() != null) {
                bind(c.func.rest(), new ArrayList<>(args.subList(Math.min(args.size(), params.size()), args.size())), s,
                        false);
            }
            if (c.func.expr() != null) {
                return eval(c.func.expr(), s);
            }
            try {
                execBlock(c.func.body(), s);
            } catch (Return r) {
                return r.value;
            }
            return UNDEFINED;
        } finally {
            depth--;
        }
    }

    /** Calls any function value. */
    Object callFn(Object f, Object self, List<Object> args, int line) {
        if (!(f instanceof Fn fn)) {
            throw typeError(line, str(f) + " is not a function");
        }
        return fn.call(this, self, args);
    }

    private Object unary(Node.Unary u, Scope scope) {
        switch (u.op()) {
            case "typeof" -> {
                if (u.value() instanceof Node.Name n && scope.find(n.name()) == null) {
                    return "undefined";
                }
                return typeOf(eval(u.value(), scope));
            }
            case "delete" -> {
                if (u.value() instanceof Node.Member m) {
                    Object o = eval(m.object(), scope);
                    Object key = m.computed() ? eval(m.key(), scope) : ((Node.Literal) m.key()).value();
                    if (o instanceof Map<?, ?> map) {
                        map.remove(key(key));
                    }
                }
                return Boolean.TRUE;
            }
            default -> {
                // the others take the value
            }
        }
        Object v = eval(u.value(), scope);
        return switch (u.op()) {
            case "!" -> !truthy(v);
            case "-" -> -num(v);
            case "+" -> num(v);
            case "~" -> (double) ~toInt32(v);
            case "void" -> UNDEFINED;
            default -> throw new ScriptError("line " + u.line() + ": unknown operator " + u.op());
        };
    }

    private Object assignment(Node.Assign a, Scope scope) {
        String op = a.op();
        if (op.equals("=")) {
            Object v = eval(a.value(), scope);
            if (a.value() instanceof Node.Func f && f.name() == null && a.target() instanceof Node.Name n
                    && v instanceof Closure c) {
                v = named(c, n.name());
            }
            assign(a.target(), v, scope, a.line());
            return v;
        }
        Object old = eval(a.target(), scope);
        Object v;
        switch (op) {
            case "&&=" -> {
                if (!truthy(old)) {
                    return old;
                }
                v = eval(a.value(), scope);
            }
            case "||=" -> {
                if (truthy(old)) {
                    return old;
                }
                v = eval(a.value(), scope);
            }
            case "??=" -> {
                if (old != null && old != UNDEFINED) {
                    return old;
                }
                v = eval(a.value(), scope);
            }
            default -> v = binary(op.substring(0, op.length() - 1), old, eval(a.value(), scope), a.line());
        }
        assign(a.target(), v, scope, a.line());
        return v;
    }

    private void assign(Node target, Object value, Scope scope, int line) {
        if (target instanceof Node.Name n) {
            assignName(n.name(), value, scope, line);
        } else if (target instanceof Node.Member m) {
            Object o = eval(m.object(), scope);
            Object key = m.computed() ? eval(m.key(), scope) : ((Node.Literal) m.key()).value();
            put(o, key, value, line);
        } else {
            throw new ScriptError("line " + line + ": cannot assign to this");
        }
    }

    // ------------------------------------------------------------------- properties

    Object get(Object o, Object key, int line) {
        if (o == null || o == UNDEFINED) {
            throw typeError(line, "cannot read '" + str(key) + "' of " + str(o));
        }
        if (o instanceof List<?> l) {
            if (key instanceof Double d) {
                int i = d.intValue();
                return i == d && i >= 0 && i < l.size() ? l.get(i) : UNDEFINED;
            }
            String k = key(key);
            if (k.equals("length")) {
                return (double) l.size();
            }
            Integer idx = index(k);
            if (idx != null) {
                return idx < l.size() ? l.get(idx) : UNDEFINED;
            }
            return Builtins.arrayMethod(this, k);
        }
        if (o instanceof String s) {
            if (key instanceof Double d) {
                int i = d.intValue();
                return i == d && i >= 0 && i < s.length() ? String.valueOf(s.charAt(i)) : UNDEFINED;
            }
            String k = key(key);
            if (k.equals("length")) {
                return (double) s.length();
            }
            Integer idx = index(k);
            if (idx != null) {
                return idx < s.length() ? String.valueOf(s.charAt(idx)) : UNDEFINED;
            }
            return Builtins.stringMethod(this, k);
        }
        String k = key(key);
        if (o instanceof Map<?, ?> m) {
            if (m.containsKey(k)) {
                return m.get(k);
            }
            return Builtins.objectMethod(this, k);
        }
        if (o instanceof Double) {
            return Builtins.numberMethod(this, k);
        }
        if (o instanceof Native n) {
            if (n.props.containsKey(k)) {
                return n.props.get(k);
            }
            return k.equals("name") ? n.name : Builtins.functionMethod(this, k);
        }
        if (o instanceof Fn f) {
            return k.equals("name") ? f.name() : Builtins.functionMethod(this, k);
        }
        if (o instanceof JsMap || o instanceof JsSet) {
            return Builtins.collectionMember(this, o, k);
        }
        if (o instanceof JsRegex r) {
            return switch (k) {
                case "source" -> r.source;
                case "flags" -> r.flags;
                case "global" -> r.global();
                case "lastIndex" -> (double) r.lastIndex;
                default -> Builtins.regexMethod(this, k);
            };
        }
        if (o instanceof JsDate) {
            return Builtins.dateMethod(this, k);
        }
        if (o instanceof Boolean) {
            return k.equals("toString") ? Builtins.objectMethod(this, k) : UNDEFINED;
        }
        return UNDEFINED;
    }

    @SuppressWarnings("unchecked")
    void put(Object o, Object key, Object value, int line) {
        if (o instanceof Map<?, ?> m) {
            ((Map<String, Object>) m).put(key(key), value);
            return;
        }
        if (o instanceof List<?> l) {
            List<Object> list = (List<Object>) l;
            String k = key(key);
            if (k.equals("length")) {
                int n = (int) num(value);
                while (list.size() > n) {
                    list.removeLast();
                }
                while (list.size() < n) {
                    list.add(UNDEFINED);
                }
                checkSize(list);
                return;
            }
            Integer idx = index(k);
            if (idx == null) {
                throw typeError(line, "arrays take numbered items, not '" + k + "'");
            }
            if (idx > list.size() + limits.chars()) {
                throw new ScriptError("line " + line + ": that index is far past the array's end");
            }
            while (list.size() <= idx) {
                list.add(UNDEFINED);
            }
            checkSize(list);
            list.set(idx, value);
            return;
        }
        if (o instanceof JsRegex r && key(key).equals("lastIndex")) {
            r.lastIndex = (int) num(value);
            return;
        }
        if (o instanceof Native n) {
            n.props.put(key(key), value);
            return;
        }
        throw typeError(line, "cannot set '" + str(key) + "' on " + typeOf(o) + " " + abbreviate(str(o)));
    }

    private static String abbreviate(String s) {
        return s.length() > 40 ? s.substring(0, 40) + "…" : s;
    }

    private static Integer index(String k) {
        if (k.isEmpty() || k.length() > 9) {
            return null;
        }
        for (int i = 0; i < k.length(); i++) {
            if (!Character.isDigit(k.charAt(i))) {
                return null;
            }
        }
        if (k.length() > 1 && k.charAt(0) == '0') {
            return null;
        }
        return Integer.parseInt(k);
    }

    /** A value as a property key. */
    String key(Object k) {
        return str(k);
    }

    /** What for-of, spreads and Array.from walk over. */
    List<Object> iterate(Object v, int line) {
        if (v instanceof List<?> l) {
            return new ArrayList<>(l);
        }
        if (v instanceof String s) {
            List<Object> out = new ArrayList<>();
            s.codePoints().forEach(cp -> out.add(new String(Character.toChars(cp))));
            return out;
        }
        if (v instanceof JsMap m) {
            List<Object> out = new ArrayList<>();
            for (Map.Entry<Object, Object> e : m.map.entrySet()) {
                out.add(new ArrayList<>(List.of(unwrapKey(e.getKey()), e.getValue())));
            }
            return out;
        }
        if (v instanceof JsSet s) {
            List<Object> out = new ArrayList<>();
            for (Object x : s.set) {
                out.add(unwrapKey(x));
            }
            return out;
        }
        throw typeError(line, typeOf(v) + " " + abbreviate(str(v)) + " is not iterable"
                + (v instanceof Map<?, ?> ? "; use Object.keys, Object.values or Object.entries" : ""));
    }

    // Map and Set keys: numbers and strings compare by value, objects by identity
    Object wrapKey(Object k) {
        return k instanceof Map<?, ?> || k instanceof List<?> ? new Identity(k) : k;
    }

    static Object unwrapKey(Object k) {
        return k instanceof Identity i ? i.value : k;
    }

    private record Identity(Object value) {
        @Override
        public boolean equals(Object o) {
            return o instanceof Identity i && i.value == value;
        }

        @Override
        public int hashCode() {
            return System.identityHashCode(value);
        }
    }

    // ------------------------------------------------------------------- operators

    private Object binary(String op, Object a, Object b, int line) {
        switch (op) {
            case "+" -> {
                Object pa = primitive(a);
                Object pb = primitive(b);
                if (pa instanceof String || pb instanceof String) {
                    return checkLength(str(pa) + str(pb));
                }
                return num(pa) + num(pb);
            }
            case "-" -> {
                return num(a) - num(b);
            }
            case "*" -> {
                return num(a) * num(b);
            }
            case "/" -> {
                return num(a) / num(b);
            }
            case "%" -> {
                return num(a) % num(b);
            }
            case "**" -> {
                return Math.pow(num(a), num(b));
            }
            case "==" -> {
                return looseEquals(a, b);
            }
            case "!=" -> {
                return !looseEquals(a, b);
            }
            case "===" -> {
                return strictEquals(a, b);
            }
            case "!==" -> {
                return !strictEquals(a, b);
            }
            case "<", ">", "<=", ">=" -> {
                Object pa = primitive(a);
                Object pb = primitive(b);
                int c;
                if (pa instanceof String sa && pb instanceof String sb) {
                    c = sa.compareTo(sb);
                } else {
                    double x = num(pa);
                    double y = num(pb);
                    if (Double.isNaN(x) || Double.isNaN(y)) {
                        return false;
                    }
                    c = Double.compare(x, y);
                    if (x == y) {
                        c = 0;
                    }
                }
                return switch (op) {
                    case "<" -> c < 0;
                    case ">" -> c > 0;
                    case "<=" -> c <= 0;
                    default -> c >= 0;
                };
            }
            case "&" -> {
                return (double) (toInt32(a) & toInt32(b));
            }
            case "|" -> {
                return (double) (toInt32(a) | toInt32(b));
            }
            case "^" -> {
                return (double) (toInt32(a) ^ toInt32(b));
            }
            case "<<" -> {
                return (double) (toInt32(a) << (toInt32(b) & 31));
            }
            case ">>" -> {
                return (double) (toInt32(a) >> (toInt32(b) & 31));
            }
            case ">>>" -> {
                return (double) ((toInt32(a) & 0xFFFFFFFFL) >>> (toInt32(b) & 31));
            }
            case "in" -> {
                if (b instanceof Map<?, ?> m) {
                    return m.containsKey(key(a));
                }
                if (b instanceof List<?> l) {
                    Integer i = index(key(a));
                    return i != null && i < l.size() || key(a).equals("length");
                }
                throw typeError(line, "'in' needs an object on its right");
            }
            case "instanceof" -> {
                return Builtins.instanceOf(this, a, b);
            }
            default -> throw new ScriptError("line " + line + ": unknown operator " + op);
        }
    }

    private Object primitive(Object v) {
        return v instanceof List<?> || v instanceof Map<?, ?> || v instanceof JsDate || v instanceof JsRegex ? str(v) : v;
    }

    private static int toInt32(Object v) {
        double d = num(v);
        if (Double.isNaN(d) || Double.isInfinite(d)) {
            return 0;
        }
        return (int) (long) d;
    }

    static boolean strictEquals(Object a, Object b) {
        if (a instanceof Double x && b instanceof Double y) {
            return x.doubleValue() == y.doubleValue();
        }
        if (a instanceof String x && b instanceof String y) {
            return x.equals(y);
        }
        if (a instanceof Boolean x && b instanceof Boolean y) {
            return x.equals(y);
        }
        return a == b;
    }

    static boolean looseEquals(Object a, Object b) {
        boolean an = a == null || a == UNDEFINED;
        boolean bn = b == null || b == UNDEFINED;
        if (an || bn) {
            return an && bn;
        }
        if (a.getClass() == b.getClass() || a instanceof Map<?, ?> || b instanceof Map<?, ?> || a instanceof List<?>
                || b instanceof List<?>) {
            return strictEquals(a, b);
        }
        return num(a) == num(b);
    }

    // ------------------------------------------------------------------ conversions

    static boolean truthy(Object v) {
        return switch (v) {
            case null -> false;
            case Boolean b -> b;
            case Double d -> d != 0 && !d.isNaN();
            case String s -> !s.isEmpty();
            default -> v != UNDEFINED;
        };
    }

    static double num(Object v) {
        return switch (v) {
            case null -> 0;
            case Double d -> d;
            case Boolean b -> b ? 1 : 0;
            case String s -> {
                String t = s.strip();
                if (t.isEmpty()) {
                    yield 0;
                }
                try {
                    if (t.startsWith("0x") || t.startsWith("0X")) {
                        yield Long.parseLong(t.substring(2), 16);
                    }
                    if (t.equals("Infinity") || t.equals("+Infinity")) {
                        yield Double.POSITIVE_INFINITY;
                    }
                    if (t.equals("-Infinity")) {
                        yield Double.NEGATIVE_INFINITY;
                    }
                    if (!t.matches("[+-]?(\\d+\\.?\\d*|\\.\\d+)([eE][+-]?\\d+)?")) {
                        yield Double.NaN;
                    }
                    yield Double.parseDouble(t);
                } catch (NumberFormatException e) {
                    yield Double.NaN;
                }
            }
            case List<?> l -> l.isEmpty() ? 0 : l.size() == 1 ? num(l.getFirst()) : Double.NaN;
            case JsDate d -> d.time;
            default -> Double.NaN;
        };
    }

    /** A number as JavaScript prints it. */
    static String str(double d) {
        if (Double.isNaN(d)) {
            return "NaN";
        }
        if (Double.isInfinite(d)) {
            return d > 0 ? "Infinity" : "-Infinity";
        }
        if (d == Math.rint(d) && Math.abs(d) < 1e21) {
            return d == 0 ? "0" : new BigDecimal(d).toPlainString();
        }
        double abs = Math.abs(d);
        if (abs >= 1e-6 && abs < 1e21) {
            return new BigDecimal(Double.toString(d)).stripTrailingZeros().toPlainString();
        }
        String s = Double.toString(d).replace(".0E", "E");
        int e = s.indexOf('E');
        return s.substring(0, e) + "e" + (s.charAt(e + 1) == '-' ? "" : "+") + s.substring(e + 1);
    }

    /** Any value as a string, as String(v) does. */
    String str(Object v) {
        return switch (v) {
            case null -> "null";
            case String s -> s;
            case Double d -> str(d.doubleValue());
            case Boolean b -> b.toString();
            case JsError e -> Builtins.describeError(this, e);
            case Map<?, ?> m -> "[object Object]";
            case List<?> l -> {
                StringBuilder b = new StringBuilder();
                for (int i = 0; i < l.size(); i++) {
                    if (i > 0) {
                        b.append(',');
                    }
                    Object x = l.get(i);
                    if (x != null && x != UNDEFINED) {
                        b.append(str(x));
                    }
                }
                yield b.toString();
            }
            case Fn f -> "function " + f.name() + "() { [code] }";
            case JsRegex r -> "/" + r.source + "/" + r.flags;
            case JsDate d -> Builtins.iso(d.time);
            case JsMap m -> "[object Map]";
            case JsSet s -> "[object Set]";
            default -> v == UNDEFINED ? "undefined" : String.valueOf(v);
        };
    }

    static String typeOf(Object v) {
        return switch (v) {
            case null -> "object";
            case String s -> "string";
            case Double d -> "number";
            case Boolean b -> "boolean";
            case Fn f -> "function";
            default -> v == UNDEFINED ? "undefined" : "object";
        };
    }

    /** A script value as plain Java: whole numbers become longs, undefined becomes null. */
    public Object toJava(Object v) {
        return switch (v) {
            case null -> null;
            case Double d -> {
                double x = d;
                if (x == Math.rint(x) && Math.abs(x) < 9e15) {
                    yield (long) x;
                }
                yield Double.isFinite(x) ? x : null;
            }
            case String s -> s;
            case Boolean b -> b;
            case Map<?, ?> m -> {
                Map<String, Object> out = new LinkedHashMap<>();
                for (Map.Entry<?, ?> e : m.entrySet()) {
                    if (e.getValue() != UNDEFINED && !(e.getValue() instanceof Fn)) {
                        out.put(String.valueOf(e.getKey()), toJava(e.getValue()));
                    }
                }
                yield out;
            }
            case List<?> l -> {
                List<Object> out = new ArrayList<>();
                for (Object x : l) {
                    out.add(toJava(x));
                }
                yield out;
            }
            case JsMap m -> {
                Map<String, Object> out = new LinkedHashMap<>();
                for (Map.Entry<Object, Object> e : m.map.entrySet()) {
                    out.put(str(unwrapKey(e.getKey())), toJava(e.getValue()));
                }
                yield out;
            }
            case JsSet s -> {
                List<Object> out = new ArrayList<>();
                for (Object x : s.set) {
                    out.add(toJava(unwrapKey(x)));
                }
                yield out;
            }
            case JsDate d -> Builtins.iso(d.time);
            case JsRegex r -> str(r);
            case Fn f -> null;
            default -> v == UNDEFINED ? null : String.valueOf(v);
        };
    }

    /** Plain Java as a script value: numbers become doubles, maps and lists become ones the script may change. */
    public static Object fromJava(Object v) {
        return switch (v) {
            case null -> null;
            case Number n -> n.doubleValue();
            case String s -> s;
            case Boolean b -> b;
            case Character c -> String.valueOf(c);
            case Map<?, ?> m -> {
                Map<String, Object> out = new LinkedHashMap<>();
                for (Map.Entry<?, ?> e : m.entrySet()) {
                    out.put(String.valueOf(e.getKey()), fromJava(e.getValue()));
                }
                yield out;
            }
            case List<?> l -> {
                List<Object> out = new ArrayList<>();
                for (Object x : l) {
                    out.add(fromJava(x));
                }
                yield out;
            }
            default -> String.valueOf(v);
        };
    }

    Host host() {
        return host;
    }
}
