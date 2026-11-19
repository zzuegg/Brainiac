package io.github.zzuegg.brainiac.studio.agent.js;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

/** The kinds of values scripts hold besides numbers, strings, booleans, objects and arrays. */
final class Values {

    private Values() {
    }

    /** Something a script can call. */
    interface Fn {
        Object call(Interpreter in, Object self, List<Object> args);

        default String name() {
            return "";
        }
    }

    /** A function written in Java, perhaps with properties, such as {@code Number.isInteger}. */
    static final class Native implements Fn {
        final String name;
        final Body body;
        final Map<String, Object> props = new LinkedHashMap<>();
        final Body construct;

        interface Body {
            Object run(Interpreter in, Object self, List<Object> args);
        }

        Native(String name, Body body) {
            this(name, body, null);
        }

        Native(String name, Body body, Body construct) {
            this.name = name;
            this.body = body;
            this.construct = construct;
        }

        @Override
        public Object call(Interpreter in, Object self, List<Object> args) {
            return body.run(in, self, args);
        }

        @Override
        public String name() {
            return name;
        }
    }

    /** A function written in the script. */
    static final class Closure implements Fn {
        final Node.Func func;
        final Scope scope;
        final Object self;   // an arrow's this

        Closure(Node.Func func, Scope scope, Object self) {
            this.func = func;
            this.scope = scope;
            this.self = self;
        }

        @Override
        public Object call(Interpreter in, Object self, List<Object> args) {
            return in.invoke(this, func.arrow() ? this.self : self, args);
        }

        @Override
        public String name() {
            return func.name() == null ? "" : func.name();
        }
    }

    /** A function with its this fixed, from {@code bind}. */
    record Bound(Fn target, Object self, List<Object> head) implements Fn {
        @Override
        public Object call(Interpreter in, Object ignored, List<Object> args) {
            List<Object> all = new ArrayList<>(head);
            all.addAll(args);
            return target.call(in, self, all);
        }
    }

    /** An error object: a plain object that prints as {@code Name: message}. */
    static final class JsError extends LinkedHashMap<String, Object> {
        private static final long serialVersionUID = 1L;
        JsError(String name, String message) {
            put("name", name);
            put("message", message);
        }
    }

    /** A {@code Map}. */
    static final class JsMap {
        final LinkedHashMap<Object, Object> map = new LinkedHashMap<>();
    }

    /** A {@code Set}. */
    static final class JsSet {
        final LinkedHashSet<Object> set = new LinkedHashSet<>();
    }

    /** A regular expression. */
    static final class JsRegex {
        final Pattern pattern;
        final String source;
        final String flags;
        int lastIndex;

        JsRegex(String source, String flags) {
            this.source = source;
            this.flags = flags == null ? "" : flags;
            int f = 0;
            for (char c : this.flags.toCharArray()) {
                switch (c) {
                    case 'i' -> f |= Pattern.CASE_INSENSITIVE | Pattern.UNICODE_CASE;
                    case 'm' -> f |= Pattern.MULTILINE;
                    case 's' -> f |= Pattern.DOTALL;
                    case 'g', 'u', 'y', 'd', 'v' -> {
                        // global and unicode need nothing from the pattern
                    }
                    default -> throw new IllegalArgumentException("unknown regular expression flag " + c);
                }
            }
            this.pattern = Pattern.compile(source, f);
        }

        boolean global() {
            return flags.indexOf('g') >= 0;
        }
    }

    /** A {@code Date}: a moment in milliseconds. */
    static final class JsDate {
        final double time;

        JsDate(double time) {
            this.time = time;
        }
    }

    /** A value a script threw. */
    static final class Thrown extends RuntimeException {
        private static final long serialVersionUID = 1L;
        final transient Object value;

        Thrown(Object value) {
            super(null, null, false, false);
            this.value = value;
        }
    }

    static final class Break extends RuntimeException {
        private static final long serialVersionUID = 1L;
        static final Break INSTANCE = new Break();

        private Break() {
            super(null, null, false, false);
        }
    }

    static final class Continue extends RuntimeException {
        private static final long serialVersionUID = 1L;
        static final Continue INSTANCE = new Continue();

        private Continue() {
            super(null, null, false, false);
        }
    }

    static final class Return extends RuntimeException {
        private static final long serialVersionUID = 1L;
        final transient Object value;

        Return(Object value) {
            super(null, null, false, false);
            this.value = value;
        }
    }

    /** Where names live: a block, a function, or the script. */
    static final class Scope {
        final Scope parent;
        final Map<String, Object> vars = new LinkedHashMap<>();
        java.util.Set<String> constants;

        Scope(Scope parent) {
            this.parent = parent;
        }

        void declare(String name, Object value, boolean constant) {
            vars.put(name, value);
            if (constant) {
                if (constants == null) {
                    constants = new java.util.HashSet<>();
                }
                constants.add(name);
            }
        }

        Scope find(String name) {
            for (Scope s = this; s != null; s = s.parent) {
                if (s.vars.containsKey(name)) {
                    return s;
                }
            }
            return null;
        }
    }
}
