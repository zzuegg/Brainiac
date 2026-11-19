package io.github.zzuegg.brainiac.studio.agent.js;

import static io.github.zzuegg.brainiac.studio.agent.js.Interpreter.UNDEFINED;
import static io.github.zzuegg.brainiac.studio.agent.js.Interpreter.num;
import static io.github.zzuegg.brainiac.studio.agent.js.Interpreter.truthy;

import io.github.zzuegg.brainiac.io.Json;
import io.github.zzuegg.brainiac.studio.agent.js.Values.Bound;
import io.github.zzuegg.brainiac.studio.agent.js.Values.Fn;
import io.github.zzuegg.brainiac.studio.agent.js.Values.JsDate;
import io.github.zzuegg.brainiac.studio.agent.js.Values.JsError;
import io.github.zzuegg.brainiac.studio.agent.js.Values.JsMap;
import io.github.zzuegg.brainiac.studio.agent.js.Values.JsRegex;
import io.github.zzuegg.brainiac.studio.agent.js.Values.JsSet;
import io.github.zzuegg.brainiac.studio.agent.js.Values.Native;
import io.github.zzuegg.brainiac.studio.agent.js.Values.Scope;
import io.github.zzuegg.brainiac.studio.agent.js.Values.Thrown;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.Instant;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;

/** The globals and the methods of strings, arrays, objects, numbers, Map, Set, RegExp and Date. */
final class Builtins {

    private Builtins() {
    }

    private static final Map<String, Native> ARRAY = new HashMap<>();
    private static final Map<String, Native> STRING = new HashMap<>();
    private static final Map<String, Native> OBJECT = new HashMap<>();
    private static final Map<String, Native> NUMBER = new HashMap<>();
    private static final Map<String, Native> FUNCTION = new HashMap<>();
    private static final Map<String, Native> REGEX = new HashMap<>();
    private static final Map<String, Native> DATE = new HashMap<>();
    private static final Map<String, Native> MAP = new HashMap<>();
    private static final Map<String, Native> SET = new HashMap<>();

    private static Object arg(List<Object> args, int i) {
        return i < args.size() ? args.get(i) : UNDEFINED;
    }

    private static boolean given(List<Object> args, int i) {
        return i < args.size() && args.get(i) != UNDEFINED;
    }

    private static int intArg(List<Object> args, int i, int fallback) {
        if (!given(args, i)) {
            return fallback;
        }
        double d = num(args.get(i));
        return Double.isNaN(d) ? 0 : d > Integer.MAX_VALUE ? Integer.MAX_VALUE : d < Integer.MIN_VALUE
                ? Integer.MIN_VALUE : (int) d;
    }

    // a start or end that may count from the end
    private static int relative(List<Object> args, int i, int fallback, int length) {
        int v = intArg(args, i, fallback);
        return v < 0 ? Math.max(0, length + v) : Math.min(v, length);
    }

    private static void def(Map<String, Native> table, String name, Native.Body body) {
        table.put(name, new Native(name, body));
    }

    static Object arrayMethod(Interpreter in, String k) {
        Native n = ARRAY.get(k);
        return n != null ? n : objectMethod(in, k);
    }

    static Object stringMethod(Interpreter in, String k) {
        Native n = STRING.get(k);
        return n != null ? n : objectMethod(in, k);
    }

    static Object objectMethod(Interpreter in, String k) {
        Native n = OBJECT.get(k);
        return n != null ? n : UNDEFINED;
    }

    static Object numberMethod(Interpreter in, String k) {
        Native n = NUMBER.get(k);
        return n != null ? n : UNDEFINED;
    }

    static Object functionMethod(Interpreter in, String k) {
        Native n = FUNCTION.get(k);
        return n != null ? n : UNDEFINED;
    }

    static Object regexMethod(Interpreter in, String k) {
        Native n = REGEX.get(k);
        return n != null ? n : UNDEFINED;
    }

    static Object dateMethod(Interpreter in, String k) {
        Native n = DATE.get(k);
        return n != null ? n : UNDEFINED;
    }

    static Object collectionMember(Interpreter in, Object o, String k) {
        if (k.equals("size")) {
            return (double) (o instanceof JsMap m ? m.map.size() : ((JsSet) o).set.size());
        }
        Native n = (o instanceof JsMap ? MAP : SET).get(k);
        return n != null ? n : UNDEFINED;
    }

    @SuppressWarnings("unchecked")
    private static List<Object> list(Object self) {
        return (List<Object>) self;
    }

    private static String string(Interpreter in, Object self) {
        return self instanceof String s ? s : in.str(self);
    }

    private static Object callback(Interpreter in, List<Object> args) {
        Object f = arg(args, 0);
        if (!(f instanceof Fn)) {
            throw in.error("TypeError", in.str(f) + " is not a function");
        }
        return f;
    }

    // ------------------------------------------------------------------------ arrays

    static {
        def(ARRAY, "push", (in, s, a) -> {
            list(s).addAll(a);
            in.checkSize(list(s));
            return (double) list(s).size();
        });
        def(ARRAY, "pop", (in, s, a) -> list(s).isEmpty() ? UNDEFINED : list(s).removeLast());
        def(ARRAY, "shift", (in, s, a) -> list(s).isEmpty() ? UNDEFINED : list(s).removeFirst());
        def(ARRAY, "unshift", (in, s, a) -> {
            list(s).addAll(0, a);
            in.checkSize(list(s));
            return (double) list(s).size();
        });
        def(ARRAY, "slice", (in, s, a) -> {
            List<Object> l = list(s);
            int from = relative(a, 0, 0, l.size());
            int to = relative(a, 1, l.size(), l.size());
            return new ArrayList<>(from < to ? l.subList(from, to) : List.of());
        });
        def(ARRAY, "splice", (in, s, a) -> {
            List<Object> l = list(s);
            int from = relative(a, 0, 0, l.size());
            int count = given(a, 1) ? Math.max(0, Math.min(intArg(a, 1, 0), l.size() - from)) : l.size() - from;
            List<Object> removed = new ArrayList<>(l.subList(from, from + count));
            l.subList(from, from + count).clear();
            if (a.size() > 2) {
                l.addAll(from, a.subList(2, a.size()));
            }
            return removed;
        });
        def(ARRAY, "concat", (in, s, a) -> {
            List<Object> out = new ArrayList<>(list(s));
            for (Object x : a) {
                if (x instanceof List<?> l) {
                    out.addAll(l);
                } else {
                    out.add(x);
                }
            }
            in.checkSize(out);
            return out;
        });
        def(ARRAY, "join", (in, s, a) -> {
            String sep = given(a, 0) ? in.str(a.getFirst()) : ",";
            StringBuilder b = new StringBuilder();
            List<Object> l = list(s);
            for (int i = 0; i < l.size(); i++) {
                if (i > 0) {
                    b.append(sep);
                }
                Object x = l.get(i);
                if (x != null && x != UNDEFINED) {
                    b.append(in.str(x));
                }
                if (b.length() > 5_000_000) {
                    in.checkLength(b.toString());
                }
            }
            return b.toString();
        });
        def(ARRAY, "reverse", (in, s, a) -> {
            Collections.reverse(list(s));
            return s;
        });
        def(ARRAY, "indexOf", (in, s, a) -> {
            List<Object> l = list(s);
            for (int i = relative(a, 1, 0, l.size()); i < l.size(); i++) {
                if (Interpreter.strictEquals(l.get(i), arg(a, 0))) {
                    return (double) i;
                }
            }
            return -1.0;
        });
        def(ARRAY, "lastIndexOf", (in, s, a) -> {
            List<Object> l = list(s);
            for (int i = l.size() - 1; i >= 0; i--) {
                if (Interpreter.strictEquals(l.get(i), arg(a, 0))) {
                    return (double) i;
                }
            }
            return -1.0;
        });
        def(ARRAY, "includes", (in, s, a) -> {
            Object want = arg(a, 0);
            for (Object x : list(s)) {
                if (Interpreter.strictEquals(x, want) || want instanceof Double d && d.isNaN() && x instanceof Double e
                        && e.isNaN()) {
                    return true;
                }
            }
            return false;
        });
        def(ARRAY, "at", (in, s, a) -> {
            List<Object> l = list(s);
            int i = intArg(a, 0, 0);
            i = i < 0 ? l.size() + i : i;
            return i >= 0 && i < l.size() ? l.get(i) : UNDEFINED;
        });
        def(ARRAY, "fill", (in, s, a) -> {
            List<Object> l = list(s);
            for (int i = relative(a, 1, 0, l.size()); i < relative(a, 2, l.size(), l.size()); i++) {
                l.set(i, arg(a, 0));
            }
            return s;
        });
        def(ARRAY, "flat", (in, s, a) -> flat(list(s), given(a, 0) ? intArg(a, 0, 1) : 1));
        def(ARRAY, "map", (in, s, a) -> {
            Object f = callback(in, a);
            List<Object> l = list(s);
            List<Object> out = new ArrayList<>(l.size());
            for (int i = 0; i < l.size(); i++) {
                out.add(in.callFn(f, arg(a, 1), List.of(nn(l.get(i)), (double) i, l), 0));
            }
            return out;
        });
        def(ARRAY, "flatMap", (in, s, a) -> {
            Object f = callback(in, a);
            List<Object> l = list(s);
            List<Object> out = new ArrayList<>();
            for (int i = 0; i < l.size(); i++) {
                Object v = in.callFn(f, arg(a, 1), List.of(nn(l.get(i)), (double) i, l), 0);
                if (v instanceof List<?> sub) {
                    out.addAll(sub);
                } else {
                    out.add(v);
                }
            }
            in.checkSize(out);
            return out;
        });
        def(ARRAY, "filter", (in, s, a) -> {
            Object f = callback(in, a);
            List<Object> l = list(s);
            List<Object> out = new ArrayList<>();
            for (int i = 0; i < l.size(); i++) {
                if (truthy(in.callFn(f, arg(a, 1), List.of(nn(l.get(i)), (double) i, l), 0))) {
                    out.add(l.get(i));
                }
            }
            return out;
        });
        def(ARRAY, "forEach", (in, s, a) -> {
            Object f = callback(in, a);
            List<Object> l = list(s);
            for (int i = 0; i < l.size(); i++) {
                in.callFn(f, arg(a, 1), List.of(nn(l.get(i)), (double) i, l), 0);
            }
            return UNDEFINED;
        });
        def(ARRAY, "find", (in, s, a) -> {
            Object f = callback(in, a);
            List<Object> l = list(s);
            for (int i = 0; i < l.size(); i++) {
                if (truthy(in.callFn(f, arg(a, 1), List.of(nn(l.get(i)), (double) i, l), 0))) {
                    return l.get(i);
                }
            }
            return UNDEFINED;
        });
        def(ARRAY, "findLast", (in, s, a) -> {
            Object f = callback(in, a);
            List<Object> l = list(s);
            for (int i = l.size() - 1; i >= 0; i--) {
                if (truthy(in.callFn(f, arg(a, 1), List.of(nn(l.get(i)), (double) i, l), 0))) {
                    return l.get(i);
                }
            }
            return UNDEFINED;
        });
        def(ARRAY, "findIndex", (in, s, a) -> {
            Object f = callback(in, a);
            List<Object> l = list(s);
            for (int i = 0; i < l.size(); i++) {
                if (truthy(in.callFn(f, arg(a, 1), List.of(nn(l.get(i)), (double) i, l), 0))) {
                    return (double) i;
                }
            }
            return -1.0;
        });
        def(ARRAY, "some", (in, s, a) -> {
            Object f = callback(in, a);
            List<Object> l = list(s);
            for (int i = 0; i < l.size(); i++) {
                if (truthy(in.callFn(f, arg(a, 1), List.of(nn(l.get(i)), (double) i, l), 0))) {
                    return true;
                }
            }
            return false;
        });
        def(ARRAY, "every", (in, s, a) -> {
            Object f = callback(in, a);
            List<Object> l = list(s);
            for (int i = 0; i < l.size(); i++) {
                if (!truthy(in.callFn(f, arg(a, 1), List.of(nn(l.get(i)), (double) i, l), 0))) {
                    return false;
                }
            }
            return true;
        });
        def(ARRAY, "reduce", (in, s, a) -> reduce(in, list(s), a, false));
        def(ARRAY, "reduceRight", (in, s, a) -> reduce(in, list(s), a, true));
        def(ARRAY, "sort", (in, s, a) -> {
            List<Object> l = list(s);
            sort(in, l, arg(a, 0));
            return s;
        });
        def(ARRAY, "toSorted", (in, s, a) -> {
            List<Object> l = new ArrayList<>(list(s));
            sort(in, l, arg(a, 0));
            return l;
        });
        def(ARRAY, "toReversed", (in, s, a) -> {
            List<Object> l = new ArrayList<>(list(s));
            Collections.reverse(l);
            return l;
        });
        def(ARRAY, "keys", (in, s, a) -> {
            List<Object> out = new ArrayList<>();
            for (int i = 0; i < list(s).size(); i++) {
                out.add((double) i);
            }
            return out;
        });
        def(ARRAY, "values", (in, s, a) -> new ArrayList<>(list(s)));
        def(ARRAY, "entries", (in, s, a) -> {
            List<Object> out = new ArrayList<>();
            for (int i = 0; i < list(s).size(); i++) {
                out.add(new ArrayList<>(List.of((double) i, nn(list(s).get(i)))));
            }
            return out;
        });
    }

    // List.of refuses nulls, and null is a fine script value
    private static Object nn(Object v) {
        return v == null ? NULL : v;
    }

    private static final Object NULL = null;

    private static List<Object> flat(List<?> l, int depth) {
        List<Object> out = new ArrayList<>();
        for (Object x : l) {
            if (x instanceof List<?> sub && depth > 0) {
                out.addAll(flat(sub, depth - 1));
            } else {
                out.add(x);
            }
        }
        return out;
    }

    private static Object reduce(Interpreter in, List<Object> l, List<Object> a, boolean right) {
        Object f = callback(in, a);
        int n = l.size();
        int i = right ? n - 1 : 0;
        int step = right ? -1 : 1;
        Object acc;
        if (given(a, 1) || a.size() > 1) {
            acc = a.get(1);
        } else {
            if (n == 0) {
                throw in.error("TypeError", "reduce of an empty array with no first value");
            }
            acc = l.get(i);
            i += step;
        }
        for (; i >= 0 && i < n; i += step) {
            acc = in.callFn(f, UNDEFINED, java.util.Arrays.asList(acc, l.get(i), (double) i, l), 0);
        }
        return acc;
    }

    private static void sort(Interpreter in, List<Object> l, Object compare) {
        java.util.Comparator<Object> c;
        if (compare instanceof Fn) {
            c = (x, y) -> {
                double d = num(in.callFn(compare, UNDEFINED, java.util.Arrays.asList(x, y), 0));
                return d < 0 ? -1 : d > 0 ? 1 : 0;
            };
        } else {
            c = (x, y) -> {
                if (x == UNDEFINED || y == UNDEFINED) {
                    return x == y ? 0 : x == UNDEFINED ? 1 : -1;
                }
                return in.str(x).compareTo(in.str(y));
            };
        }
        try {
            l.sort(c);
        } catch (IllegalArgumentException e) {   // a compare function that contradicts itself
            List<Object> copy = new ArrayList<>(l);
            for (int i = 1; i < copy.size(); i++) {   // insertion sort never complains
                Object x = copy.get(i);
                int j = i - 1;
                while (j >= 0 && c.compare(copy.get(j), x) > 0) {
                    copy.set(j + 1, copy.get(j));
                    j--;
                }
                copy.set(j + 1, x);
            }
            l.clear();
            l.addAll(copy);
        }
    }

    // ----------------------------------------------------------------------- strings

    static {
        def(STRING, "slice", (in, s, a) -> {
            String t = string(in, s);
            int from = relative(a, 0, 0, t.length());
            int to = relative(a, 1, t.length(), t.length());
            return from < to ? t.substring(from, to) : "";
        });
        def(STRING, "substring", (in, s, a) -> {
            String t = string(in, s);
            int from = Math.max(0, Math.min(intArg(a, 0, 0), t.length()));
            int to = Math.max(0, Math.min(intArg(a, 1, t.length()), t.length()));
            return t.substring(Math.min(from, to), Math.max(from, to));
        });
        def(STRING, "substr", (in, s, a) -> {
            String t = string(in, s);
            int from = relative(a, 0, 0, t.length());
            int len = Math.max(0, Math.min(intArg(a, 1, t.length() - from), t.length() - from));
            return t.substring(from, from + len);
        });
        def(STRING, "indexOf", (in, s, a) -> (double) string(in, s).indexOf(in.str(arg(a, 0)), intArg(a, 1, 0)));
        def(STRING, "lastIndexOf", (in, s, a) -> (double) string(in, s).lastIndexOf(in.str(arg(a, 0))));
        def(STRING, "includes", (in, s, a) -> string(in, s).indexOf(in.str(arg(a, 0)), intArg(a, 1, 0)) >= 0);
        def(STRING, "startsWith", (in, s, a) -> string(in, s).startsWith(in.str(arg(a, 0)), Math.max(0, intArg(a, 1, 0))));
        def(STRING, "endsWith", (in, s, a) -> {
            String t = string(in, s);
            int end = Math.min(intArg(a, 1, t.length()), t.length());
            return t.substring(0, Math.max(0, end)).endsWith(in.str(arg(a, 0)));
        });
        def(STRING, "toUpperCase", (in, s, a) -> string(in, s).toUpperCase(Locale.ROOT));
        def(STRING, "toLowerCase", (in, s, a) -> string(in, s).toLowerCase(Locale.ROOT));
        def(STRING, "toLocaleUpperCase", (in, s, a) -> string(in, s).toUpperCase(Locale.ROOT));
        def(STRING, "toLocaleLowerCase", (in, s, a) -> string(in, s).toLowerCase(Locale.ROOT));
        def(STRING, "trim", (in, s, a) -> string(in, s).strip());
        def(STRING, "trimStart", (in, s, a) -> string(in, s).stripLeading());
        def(STRING, "trimEnd", (in, s, a) -> string(in, s).stripTrailing());
        def(STRING, "charAt", (in, s, a) -> {
            String t = string(in, s);
            int i = intArg(a, 0, 0);
            return i >= 0 && i < t.length() ? String.valueOf(t.charAt(i)) : "";
        });
        def(STRING, "charCodeAt", (in, s, a) -> {
            String t = string(in, s);
            int i = intArg(a, 0, 0);
            return i >= 0 && i < t.length() ? (double) t.charAt(i) : Double.NaN;
        });
        def(STRING, "codePointAt", (in, s, a) -> {
            String t = string(in, s);
            int i = intArg(a, 0, 0);
            return i >= 0 && i < t.length() ? (double) t.codePointAt(i) : UNDEFINED;
        });
        def(STRING, "at", (in, s, a) -> {
            String t = string(in, s);
            int i = intArg(a, 0, 0);
            i = i < 0 ? t.length() + i : i;
            return i >= 0 && i < t.length() ? String.valueOf(t.charAt(i)) : UNDEFINED;
        });
        def(STRING, "repeat", (in, s, a) -> {
            int n = intArg(a, 0, 0);
            String t = string(in, s);
            if (n < 0 || (long) n * t.length() > 5_000_000) {
                throw in.error("RangeError", "repeat count " + n + " is out of range");
            }
            return t.repeat(n);
        });
        def(STRING, "padStart", (in, s, a) -> pad(in, string(in, s), a, true));
        def(STRING, "padEnd", (in, s, a) -> pad(in, string(in, s), a, false));
        def(STRING, "concat", (in, s, a) -> {
            StringBuilder b = new StringBuilder(string(in, s));
            for (Object x : a) {
                b.append(in.str(x));
            }
            return in.checkLength(b.toString());
        });
        def(STRING, "localeCompare", (in, s, a) -> (double) Integer.signum(
                java.text.Collator.getInstance(Locale.ROOT).compare(string(in, s), in.str(arg(a, 0)))));
        def(STRING, "normalize", (in, s, a) -> java.text.Normalizer.normalize(string(in, s),
                java.text.Normalizer.Form.valueOf(given(a, 0) ? in.str(a.getFirst()) : "NFC")));
        def(STRING, "split", (in, s, a) -> {
            String t = string(in, s);
            List<Object> out = new ArrayList<>();
            int limit = given(a, 1) ? intArg(a, 1, Integer.MAX_VALUE) : Integer.MAX_VALUE;
            Object sep = arg(a, 0);
            if (sep == UNDEFINED) {
                out.add(t);
            } else if (sep instanceof JsRegex r) {
                Matcher m = r.pattern.matcher(t);
                int last = 0;
                while (m.find() && out.size() < limit) {
                    if (m.end() == 0 && m.start() == 0 || m.start() == t.length() && m.end() == m.start()) {
                        continue;
                    }
                    out.add(t.substring(last, m.start()));
                    for (int g = 1; g <= m.groupCount(); g++) {
                        out.add(m.group(g) == null ? UNDEFINED : m.group(g));
                    }
                    last = m.end();
                }
                if (out.size() < limit) {
                    out.add(t.substring(last));
                }
            } else {
                String d = in.str(sep);
                if (d.isEmpty()) {
                    for (int i = 0; i < t.length() && out.size() < limit; i++) {
                        out.add(String.valueOf(t.charAt(i)));
                    }
                } else {
                    int last = 0;
                    int i;
                    while ((i = t.indexOf(d, last)) >= 0 && out.size() < limit) {
                        out.add(t.substring(last, i));
                        last = i + d.length();
                    }
                    if (out.size() < limit) {
                        out.add(t.substring(last));
                    }
                }
            }
            return out;
        });
        def(STRING, "replace", (in, s, a) -> replace(in, string(in, s), arg(a, 0), arg(a, 1), false));
        def(STRING, "replaceAll", (in, s, a) -> replace(in, string(in, s), arg(a, 0), arg(a, 1), true));
        def(STRING, "match", (in, s, a) -> {
            String t = string(in, s);
            JsRegex r = arg(a, 0) instanceof JsRegex x ? x : new JsRegex(java.util.regex.Pattern.quote(in.str(arg(a, 0))), "");
            Matcher m = r.pattern.matcher(t);
            if (r.global()) {
                List<Object> all = new ArrayList<>();
                while (m.find()) {
                    all.add(m.group());
                }
                return all.isEmpty() ? null : all;
            }
            return m.find() ? groups(m, t) : null;
        });
        def(STRING, "matchAll", (in, s, a) -> {
            String t = string(in, s);
            JsRegex r = arg(a, 0) instanceof JsRegex x ? x : new JsRegex(java.util.regex.Pattern.quote(in.str(arg(a, 0))), "g");
            Matcher m = r.pattern.matcher(t);
            List<Object> all = new ArrayList<>();
            while (m.find()) {
                all.add(groups(m, t));
            }
            return all;
        });
        def(STRING, "search", (in, s, a) -> {
            String t = string(in, s);
            JsRegex r = arg(a, 0) instanceof JsRegex x ? x : new JsRegex(java.util.regex.Pattern.quote(in.str(arg(a, 0))), "");
            Matcher m = r.pattern.matcher(t);
            return m.find() ? (double) m.start() : -1.0;
        });
    }

    private static String pad(Interpreter in, String t, List<Object> a, boolean start) {
        int len = Math.min(intArg(a, 0, 0), 5_000_000);
        String fill = given(a, 1) ? in.str(a.get(1)) : " ";
        if (t.length() >= len || fill.isEmpty()) {
            return t;
        }
        StringBuilder p = new StringBuilder();
        while (p.length() < len - t.length()) {
            p.append(fill);
        }
        p.setLength(len - t.length());
        return start ? p + t : t + p;
    }

    // an array of the match and its groups, as match() returns; named groups are lost
    private static List<Object> groups(Matcher m, String t) {
        List<Object> out = new ArrayList<>();
        for (int g = 0; g <= m.groupCount(); g++) {
            out.add(m.group(g) == null ? UNDEFINED : m.group(g));
        }
        return out;
    }

    private static String replace(Interpreter in, String t, Object pattern, Object with, boolean all) {
        JsRegex r = pattern instanceof JsRegex x ? x : null;
        java.util.regex.Pattern p = r != null ? r.pattern : java.util.regex.Pattern.compile(java.util.regex.Pattern.quote(in.str(pattern)));
        boolean global = all || r != null && r.global();
        Matcher m = p.matcher(t);
        StringBuilder out = new StringBuilder();
        int last = 0;
        while (m.find()) {
            out.append(t, last, m.start());
            if (with instanceof Fn) {
                List<Object> args = new ArrayList<>();
                for (int g = 0; g <= m.groupCount(); g++) {
                    args.add(m.group(g) == null ? UNDEFINED : m.group(g));
                }
                args.add((double) m.start());
                args.add(t);
                out.append(in.str(in.callFn(with, UNDEFINED, args, 0)));
            } else {
                out.append(expand(in.str(with), m, t));
            }
            last = m.end();
            if (m.end() == m.start()) {
                if (m.end() < t.length()) {
                    out.append(t.charAt(m.end()));
                }
                last = m.end() + 1;
                if (last > t.length()) {
                    break;
                }
                m.region(last, t.length());
            }
            if (!global) {
                break;
            }
            in.checkLength(out.toString().length() > 5_000_000 ? out.toString() : "");
        }
        if (last <= t.length()) {
            out.append(t, Math.min(last, t.length()), t.length());
        }
        return in.checkLength(out.toString());
    }

    // $&, $1, $<name> and $$ in a replacement
    private static String expand(String with, Matcher m, String t) {
        StringBuilder b = new StringBuilder();
        for (int i = 0; i < with.length(); i++) {
            char c = with.charAt(i);
            if (c != '$' || i + 1 >= with.length()) {
                b.append(c);
                continue;
            }
            char d = with.charAt(i + 1);
            if (d == '$') {
                b.append('$');
                i++;
            } else if (d == '&') {
                b.append(m.group());
                i++;
            } else if (d == '`') {
                b.append(t, 0, m.start());
                i++;
            } else if (d == '\'') {
                b.append(t.substring(m.end()));
                i++;
            } else if (Character.isDigit(d)) {
                int j = i + 1;
                int g = d - '0';
                if (j + 1 < with.length() && Character.isDigit(with.charAt(j + 1))
                        && g * 10 + (with.charAt(j + 1) - '0') <= m.groupCount()) {
                    g = g * 10 + (with.charAt(j + 1) - '0');
                    j++;
                }
                if (g >= 1 && g <= m.groupCount()) {
                    b.append(m.group(g) == null ? "" : m.group(g));
                    i = j;
                } else {
                    b.append(c);
                }
            } else if (d == '<') {
                int end = with.indexOf('>', i);
                if (end < 0) {
                    b.append(c);
                    continue;
                }
                try {
                    String g = m.group(with.substring(i + 2, end));
                    b.append(g == null ? "" : g);
                } catch (IllegalArgumentException e) {
                    b.append(with, i, end + 1);
                }
                i = end;
            } else {
                b.append(c);
            }
        }
        return b.toString();
    }

    // ------------------------------------------------------- objects, numbers, functions

    static {
        def(OBJECT, "hasOwnProperty", (in, s, a) -> s instanceof Map<?, ?> m && m.containsKey(in.str(arg(a, 0))));
        def(OBJECT, "toString", (in, s, a) -> in.str(s));
        def(OBJECT, "valueOf", (in, s, a) -> s);
        def(OBJECT, "toLocaleString", (in, s, a) -> in.str(s));
        def(NUMBER, "toFixed", (in, s, a) -> {
            double d = num(s);
            int digits = Math.max(0, Math.min(100, intArg(a, 0, 0)));
            if (!Double.isFinite(d)) {
                return Interpreter.str(d);
            }
            return new BigDecimal(Double.toString(d)).setScale(digits, RoundingMode.HALF_UP).toPlainString();
        });
        def(NUMBER, "toPrecision", (in, s, a) -> {
            double d = num(s);
            if (!given(a, 0) || !Double.isFinite(d)) {
                return Interpreter.str(d);
            }
            return new BigDecimal(Double.toString(d)).round(new java.math.MathContext(Math.max(1, intArg(a, 0, 1))))
                    .toPlainString();
        });
        def(NUMBER, "toString", (in, s, a) -> {
            int radix = intArg(a, 0, 10);
            double d = num(s);
            if (radix == 10 || d != Math.rint(d)) {
                return Interpreter.str(d);
            }
            return Long.toString((long) d, radix);
        });
        def(NUMBER, "toLocaleString", (in, s, a) -> {
            double d = num(s);
            return d == Math.rint(d) ? String.format(Locale.US, "%,d", (long) d)
                    : String.format(Locale.US, "%,.3f", d).replaceAll("0+$", "").replaceAll("\\.$", "");
        });
        def(NUMBER, "valueOf", (in, s, a) -> s);
        def(FUNCTION, "call", (in, s, a) -> in.callFn(s, arg(a, 0), a.size() > 1 ? new ArrayList<>(a.subList(1, a.size()))
                : new ArrayList<>(), 0));
        def(FUNCTION, "apply", (in, s, a) -> in.callFn(s, arg(a, 0), arg(a, 1) instanceof List<?> l
                ? new ArrayList<>(l) : new ArrayList<>(), 0));
        def(FUNCTION, "bind", (in, s, a) -> new Bound((Fn) s, arg(a, 0), a.size() > 1
                ? new ArrayList<>(a.subList(1, a.size())) : List.of()));
        def(FUNCTION, "toString", (in, s, a) -> in.str(s));
    }

    // ------------------------------------------------------- regular expressions, dates

    static {
        def(REGEX, "test", (in, s, a) -> {
            JsRegex r = (JsRegex) s;
            Matcher m = r.pattern.matcher(in.str(arg(a, 0)));
            if (r.global()) {
                if (r.lastIndex > in.str(arg(a, 0)).length() || !m.find(r.lastIndex)) {
                    r.lastIndex = 0;
                    return false;
                }
                r.lastIndex = m.end();
                return true;
            }
            return m.find();
        });
        def(REGEX, "exec", (in, s, a) -> {
            JsRegex r = (JsRegex) s;
            String t = in.str(arg(a, 0));
            Matcher m = r.pattern.matcher(t);
            int from = r.global() ? r.lastIndex : 0;
            if (from > t.length() || !m.find(from)) {
                r.lastIndex = 0;
                return null;
            }
            if (r.global()) {
                r.lastIndex = m.end() == m.start() ? m.end() + 1 : m.end();
            }
            return groups(m, t);
        });
        def(REGEX, "toString", (in, s, a) -> in.str(s));
        def(DATE, "getTime", (in, s, a) -> ((JsDate) s).time);
        def(DATE, "valueOf", (in, s, a) -> ((JsDate) s).time);
        def(DATE, "toISOString", (in, s, a) -> iso(((JsDate) s).time));
        def(DATE, "toJSON", (in, s, a) -> iso(((JsDate) s).time));
        def(DATE, "toString", (in, s, a) -> iso(((JsDate) s).time));
        def(DATE, "toLocaleString", (in, s, a) -> local((JsDate) s).format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")));
        def(DATE, "toLocaleDateString", (in, s, a) -> local((JsDate) s).toLocalDate().toString());
        def(DATE, "toLocaleTimeString", (in, s, a) -> local((JsDate) s).format(DateTimeFormatter.ofPattern("HH:mm:ss")));
        def(DATE, "toDateString", (in, s, a) -> local((JsDate) s).toLocalDate().toString());
        def(DATE, "getFullYear", (in, s, a) -> (double) local((JsDate) s).getYear());
        def(DATE, "getMonth", (in, s, a) -> (double) local((JsDate) s).getMonthValue() - 1);
        def(DATE, "getDate", (in, s, a) -> (double) local((JsDate) s).getDayOfMonth());
        def(DATE, "getDay", (in, s, a) -> (double) (local((JsDate) s).getDayOfWeek().getValue() % 7));
        def(DATE, "getHours", (in, s, a) -> (double) local((JsDate) s).getHour());
        def(DATE, "getMinutes", (in, s, a) -> (double) local((JsDate) s).getMinute());
        def(DATE, "getSeconds", (in, s, a) -> (double) local((JsDate) s).getSecond());
        def(DATE, "getMilliseconds", (in, s, a) -> (double) (local((JsDate) s).getNano() / 1_000_000));
        def(DATE, "getUTCFullYear", (in, s, a) -> (double) utc((JsDate) s).getYear());
        def(DATE, "getUTCMonth", (in, s, a) -> (double) utc((JsDate) s).getMonthValue() - 1);
        def(DATE, "getUTCDate", (in, s, a) -> (double) utc((JsDate) s).getDayOfMonth());
        def(DATE, "getUTCHours", (in, s, a) -> (double) utc((JsDate) s).getHour());
        def(DATE, "getTimezoneOffset", (in, s, a) -> (double) -local((JsDate) s).getOffset().getTotalSeconds() / 60);
    }

    private static java.time.ZonedDateTime local(JsDate d) {
        return Instant.ofEpochMilli((long) d.time).atZone(java.time.ZoneId.systemDefault());
    }

    private static java.time.ZonedDateTime utc(JsDate d) {
        return Instant.ofEpochMilli((long) d.time).atZone(ZoneOffset.UTC);
    }

    static String iso(double time) {
        if (Double.isNaN(time)) {
            return "Invalid Date";
        }
        return DateTimeFormatter.ofPattern("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'").format(Instant.ofEpochMilli((long) time)
                .atZone(ZoneOffset.UTC));
    }

    private static double parseDate(String s) {
        try {
            return Instant.parse(s).toEpochMilli();
        } catch (RuntimeException e) {
            // not an instant
        }
        try {
            return java.time.OffsetDateTime.parse(s).toInstant().toEpochMilli();
        } catch (RuntimeException e) {
            // not with an offset
        }
        try {
            return java.time.LocalDateTime.parse(s).atZone(java.time.ZoneId.systemDefault()).toInstant().toEpochMilli();
        } catch (RuntimeException e) {
            // not a local date and time
        }
        try {
            return java.time.LocalDate.parse(s).atStartOfDay(ZoneOffset.UTC).toInstant().toEpochMilli();
        } catch (RuntimeException e) {
            return Double.NaN;
        }
    }

    // ------------------------------------------------------------------- Map and Set

    static {
        def(MAP, "get", (in, s, a) -> {
            Object v = ((JsMap) s).map.get(in.wrapKey(norm(arg(a, 0))));
            return v == null && !((JsMap) s).map.containsKey(in.wrapKey(norm(arg(a, 0)))) ? UNDEFINED : v;
        });
        def(MAP, "set", (in, s, a) -> {
            ((JsMap) s).map.put(in.wrapKey(norm(arg(a, 0))), arg(a, 1));
            return s;
        });
        def(MAP, "has", (in, s, a) -> ((JsMap) s).map.containsKey(in.wrapKey(norm(arg(a, 0)))));
        def(MAP, "delete", (in, s, a) -> {
            boolean had = ((JsMap) s).map.containsKey(in.wrapKey(norm(arg(a, 0))));
            ((JsMap) s).map.remove(in.wrapKey(norm(arg(a, 0))));
            return had;
        });
        def(MAP, "clear", (in, s, a) -> {
            ((JsMap) s).map.clear();
            return UNDEFINED;
        });
        def(MAP, "keys", (in, s, a) -> {
            List<Object> out = new ArrayList<>();
            for (Object k : ((JsMap) s).map.keySet()) {
                out.add(Interpreter.unwrapKey(k));
            }
            return out;
        });
        def(MAP, "values", (in, s, a) -> new ArrayList<>(((JsMap) s).map.values()));
        def(MAP, "entries", (in, s, a) -> in.iterate(s, 0));
        def(MAP, "forEach", (in, s, a) -> {
            Object f = callback(in, a);
            for (Map.Entry<Object, Object> e : new ArrayList<>(((JsMap) s).map.entrySet())) {
                in.callFn(f, UNDEFINED, java.util.Arrays.asList(e.getValue(), Interpreter.unwrapKey(e.getKey()), s), 0);
            }
            return UNDEFINED;
        });
        def(SET, "add", (in, s, a) -> {
            ((JsSet) s).set.add(in.wrapKey(norm(arg(a, 0))));
            return s;
        });
        def(SET, "has", (in, s, a) -> ((JsSet) s).set.contains(in.wrapKey(norm(arg(a, 0)))));
        def(SET, "delete", (in, s, a) -> ((JsSet) s).set.remove(in.wrapKey(norm(arg(a, 0)))));
        def(SET, "clear", (in, s, a) -> {
            ((JsSet) s).set.clear();
            return UNDEFINED;
        });
        def(SET, "values", (in, s, a) -> in.iterate(s, 0));
        def(SET, "keys", (in, s, a) -> in.iterate(s, 0));
        def(SET, "forEach", (in, s, a) -> {
            Object f = callback(in, a);
            for (Object v : in.iterate(s, 0)) {
                in.callFn(f, UNDEFINED, java.util.Arrays.asList(v, v, s), 0);
            }
            return UNDEFINED;
        });
    }

    // -0 and 0 are one key
    private static Object norm(Object k) {
        return k instanceof Double d && d == 0 ? 0.0 : k;
    }

    // ----------------------------------------------------------------------- globals

    static void install(Interpreter it, Scope g) {
        Map<String, Object> console = new LinkedHashMap<>();
        Native.Body log = (in, s, a) -> {
            StringBuilder b = new StringBuilder();
            for (int i = 0; i < a.size(); i++) {
                if (i > 0) {
                    b.append(' ');
                }
                Object x = a.get(i);
                b.append(x instanceof String str ? str : x instanceof Map<?, ?> || x instanceof List<?>
                        ? stringify(in, x, "", "") : in.str(x));
            }
            in.host().log(b.toString());
            return UNDEFINED;
        };
        for (String n : List.of("log", "info", "warn", "error", "debug")) {
            console.put(n, new Native(n, log));
        }
        g.declare("console", console, true);

        Map<String, Object> json = new LinkedHashMap<>();
        json.put("stringify", new Native("stringify", (in, s, a) -> {
            Object v = arg(a, 0);
            if (v == UNDEFINED || v instanceof Fn) {
                return UNDEFINED;
            }
            Object space = arg(a, 2);
            String indent = space instanceof Double d ? " ".repeat(Math.max(0, Math.min(10, d.intValue())))
                    : space instanceof String str ? str : "";
            return in.checkLength(stringify(in, v, indent, ""));
        }));
        json.put("parse", new Native("parse", (in, s, a) -> {
            try {
                return Interpreter.fromJava(Json.parse(in.str(arg(a, 0))));
            } catch (RuntimeException e) {
                throw in.error("SyntaxError", "JSON.parse: " + e.getMessage());
            }
        }));
        g.declare("JSON", json, true);

        Map<String, Object> math = new LinkedHashMap<>();
        math.put("PI", Math.PI);
        math.put("E", Math.E);
        math.put("LN2", Math.log(2));
        math.put("LN10", Math.log(10));
        math.put("SQRT2", Math.sqrt(2));
        fn(math, "abs", a -> Math.abs(num(arg(a, 0))));
        fn(math, "floor", a -> Math.floor(num(arg(a, 0))));
        fn(math, "ceil", a -> Math.ceil(num(arg(a, 0))));
        fn(math, "round", a -> Math.floor(num(arg(a, 0)) + 0.5));
        fn(math, "trunc", a -> {
            double d = num(arg(a, 0));
            return d < 0 ? Math.ceil(d) : Math.floor(d);
        });
        fn(math, "sign", a -> Math.signum(num(arg(a, 0))));
        fn(math, "sqrt", a -> Math.sqrt(num(arg(a, 0))));
        fn(math, "cbrt", a -> Math.cbrt(num(arg(a, 0))));
        fn(math, "pow", a -> Math.pow(num(arg(a, 0)), num(arg(a, 1))));
        fn(math, "exp", a -> Math.exp(num(arg(a, 0))));
        fn(math, "log", a -> Math.log(num(arg(a, 0))));
        fn(math, "log2", a -> Math.log(num(arg(a, 0))) / Math.log(2));
        fn(math, "log10", a -> Math.log10(num(arg(a, 0))));
        fn(math, "sin", a -> Math.sin(num(arg(a, 0))));
        fn(math, "cos", a -> Math.cos(num(arg(a, 0))));
        fn(math, "tan", a -> Math.tan(num(arg(a, 0))));
        fn(math, "atan", a -> Math.atan(num(arg(a, 0))));
        fn(math, "atan2", a -> Math.atan2(num(arg(a, 0)), num(arg(a, 1))));
        fn(math, "hypot", a -> {
            double s = 0;
            for (Object x : a) {
                s += num(x) * num(x);
            }
            return Math.sqrt(s);
        });
        fn(math, "random", a -> Math.random());
        fn(math, "max", a -> {
            double m = Double.NEGATIVE_INFINITY;
            for (Object x : a) {
                double d = num(x);
                if (Double.isNaN(d)) {
                    return Double.NaN;
                }
                m = Math.max(m, d);
            }
            return m;
        });
        fn(math, "min", a -> {
            double m = Double.POSITIVE_INFINITY;
            for (Object x : a) {
                double d = num(x);
                if (Double.isNaN(d)) {
                    return Double.NaN;
                }
                m = Math.min(m, d);
            }
            return m;
        });
        g.declare("Math", math, true);

        Native object = new Native("Object", (in, s, a) -> a.isEmpty() || arg(a, 0) == null || arg(a, 0) == UNDEFINED
                ? new LinkedHashMap<String, Object>() : arg(a, 0));
        object.props.put("keys", new Native("keys", (in, s, a) -> new ArrayList<Object>(keys(in, arg(a, 0), 0))));
        object.props.put("values", new Native("values", (in, s, a) -> {
            List<Object> out = new ArrayList<>();
            for (String k : keys(in, arg(a, 0), 0)) {
                out.add(in.get(arg(a, 0), k, 0));
            }
            return out;
        }));
        object.props.put("entries", new Native("entries", (in, s, a) -> {
            List<Object> out = new ArrayList<>();
            for (String k : keys(in, arg(a, 0), 0)) {
                out.add(new ArrayList<>(java.util.Arrays.asList(k, in.get(arg(a, 0), k, 0))));
            }
            return out;
        }));
        object.props.put("assign", new Native("assign", (in, s, a) -> {
            Object target = arg(a, 0);
            for (int i = 1; i < a.size(); i++) {
                if (a.get(i) instanceof Map<?, ?> m) {
                    for (Map.Entry<?, ?> e : m.entrySet()) {
                        in.put(target, e.getKey(), e.getValue(), 0);
                    }
                }
            }
            return target;
        }));
        object.props.put("fromEntries", new Native("fromEntries", (in, s, a) -> {
            Map<String, Object> out = new LinkedHashMap<>();
            for (Object e : in.iterate(arg(a, 0), 0)) {
                List<Object> pair = in.iterate(e, 0);
                out.put(in.str(pair.isEmpty() ? UNDEFINED : pair.get(0)), pair.size() > 1 ? pair.get(1) : UNDEFINED);
            }
            return out;
        }));
        object.props.put("freeze", new Native("freeze", (in, s, a) -> arg(a, 0)));
        object.props.put("create", new Native("create", (in, s, a) -> new LinkedHashMap<String, Object>()));
        g.declare("Object", object, true);

        Native array = new Native("Array", (in, s, a) -> newArray(in, a), (in, s, a) -> newArray(in, a));
        array.props.put("isArray", new Native("isArray", (in, s, a) -> arg(a, 0) instanceof List<?>));
        array.props.put("of", new Native("of", (in, s, a) -> new ArrayList<>(a)));
        array.props.put("from", new Native("from", (in, s, a) -> {
            Object src = arg(a, 0);
            List<Object> items;
            if (src instanceof Map<?, ?> m && m.containsKey("length")) {
                items = new ArrayList<>();
                int n = (int) Math.min(num(m.get("length")), 5_000_000);
                for (int i = 0; i < n; i++) {
                    items.add(m.containsKey(String.valueOf(i)) ? m.get(String.valueOf(i)) : UNDEFINED);
                }
            } else {
                items = in.iterate(src, 0);
            }
            if (arg(a, 1) instanceof Fn) {
                for (int i = 0; i < items.size(); i++) {
                    items.set(i, in.callFn(arg(a, 1), UNDEFINED, java.util.Arrays.asList(items.get(i), (double) i), 0));
                }
            }
            return items;
        }));
        g.declare("Array", array, true);

        Native number = new Native("Number", (in, s, a) -> a.isEmpty() ? 0.0 : num(a.getFirst()));
        number.props.put("isInteger", new Native("isInteger", (in, s, a) -> arg(a, 0) instanceof Double d
                && Double.isFinite(d) && d == Math.rint(d)));
        number.props.put("isFinite", new Native("isFinite", (in, s, a) -> arg(a, 0) instanceof Double d && Double.isFinite(d)));
        number.props.put("isNaN", new Native("isNaN", (in, s, a) -> arg(a, 0) instanceof Double d && d.isNaN()));
        number.props.put("parseFloat", new Native("parseFloat", (in, s, a) -> parseFloat(in.str(arg(a, 0)))));
        number.props.put("parseInt", new Native("parseInt", (in, s, a) -> parseInt(in.str(arg(a, 0)), intArg(a, 1, 10))));
        number.props.put("MAX_SAFE_INTEGER", 9007199254740991.0);
        number.props.put("MIN_SAFE_INTEGER", -9007199254740991.0);
        number.props.put("EPSILON", Math.ulp(1.0));
        number.props.put("MAX_VALUE", Double.MAX_VALUE);
        number.props.put("POSITIVE_INFINITY", Double.POSITIVE_INFINITY);
        number.props.put("NEGATIVE_INFINITY", Double.NEGATIVE_INFINITY);
        g.declare("Number", number, true);

        Native string = new Native("String", (in, s, a) -> a.isEmpty() ? "" : in.str(a.getFirst()));
        string.props.put("fromCharCode", new Native("fromCharCode", (in, s, a) -> {
            StringBuilder b = new StringBuilder();
            for (Object x : a) {
                b.append((char) (int) num(x));
            }
            return b.toString();
        }));
        g.declare("String", string, true);
        g.declare("Boolean", new Native("Boolean", (in, s, a) -> truthy(arg(a, 0))), true);
        g.declare("parseInt", new Native("parseInt", (in, s, a) -> parseInt(in.str(arg(a, 0)), intArg(a, 1, 10))), true);
        g.declare("parseFloat", new Native("parseFloat", (in, s, a) -> parseFloat(in.str(arg(a, 0)))), true);
        g.declare("isNaN", new Native("isNaN", (in, s, a) -> Double.isNaN(num(arg(a, 0)))), true);
        g.declare("isFinite", new Native("isFinite", (in, s, a) -> Double.isFinite(num(arg(a, 0)))), true);
        g.declare("NaN", Double.NaN, true);
        g.declare("Infinity", Double.POSITIVE_INFINITY, true);
        g.declare("globalThis", new LinkedHashMap<String, Object>(), true);
        g.declare("encodeURIComponent", new Native("encodeURIComponent", (in, s, a) -> java.net.URLEncoder.encode(
                in.str(arg(a, 0)), java.nio.charset.StandardCharsets.UTF_8).replace("+", "%20")), true);
        g.declare("decodeURIComponent", new Native("decodeURIComponent", (in, s, a) -> java.net.URLDecoder.decode(
                in.str(arg(a, 0)).replace("+", "%2B"), java.nio.charset.StandardCharsets.UTF_8)), true);

        for (String kind : List.of("Error", "TypeError", "RangeError", "SyntaxError", "ReferenceError")) {
            Native.Body make = (in, s, a) -> {
                JsError e = new JsError(kind, given(a, 0) ? in.str(a.getFirst()) : "");
                if (arg(a, 1) instanceof Map<?, ?> opts && opts.containsKey("cause")) {
                    e.put("cause", opts.get("cause"));
                }
                return e;
            };
            g.declare(kind, new Native(kind, make, make), true);
        }
        Native.Body regexp = (in, s, a) -> {
            try {
                return arg(a, 0) instanceof JsRegex r ? new JsRegex(r.source, given(a, 1) ? in.str(a.get(1)) : r.flags)
                        : new JsRegex(in.str(arg(a, 0)), given(a, 1) ? in.str(a.get(1)) : "");
            } catch (RuntimeException e) {
                throw in.error("SyntaxError", "bad regular expression: " + e.getMessage());
            }
        };
        g.declare("RegExp", new Native("RegExp", regexp, regexp), true);
        Native date = new Native("Date", (in, s, a) -> iso(System.currentTimeMillis()), (in, s, a) -> {
            if (a.isEmpty()) {
                return new JsDate(System.currentTimeMillis());
            }
            if (a.size() == 1) {
                Object v = a.getFirst();
                return new JsDate(v instanceof String str ? parseDate(str) : v instanceof JsDate d ? d.time : num(v));
            }
            java.time.LocalDateTime t = java.time.LocalDateTime.of(intArg(a, 0, 1970), 1, 1, 0, 0)
                    .plusMonths(intArg(a, 1, 0)).plusDays(intArg(a, 2, 1) - 1L).plusHours(intArg(a, 3, 0))
                    .plusMinutes(intArg(a, 4, 0)).plusSeconds(intArg(a, 5, 0));
            return new JsDate(t.atZone(java.time.ZoneId.systemDefault()).toInstant().toEpochMilli() + intArg(a, 6, 0));
        });
        date.props.put("now", new Native("now", (in, s, a) -> (double) System.currentTimeMillis()));
        date.props.put("parse", new Native("parse", (in, s, a) -> parseDate(in.str(arg(a, 0)))));
        g.declare("Date", date, true);
        g.declare("Map", new Native("Map", (in, s, a) -> {
            throw in.error("TypeError", "Map needs new");
        }, (in, s, a) -> {
            JsMap m = new JsMap();
            if (given(a, 0)) {
                for (Object e : in.iterate(a.getFirst(), 0)) {
                    List<Object> pair = in.iterate(e, 0);
                    m.map.put(in.wrapKey(norm(pair.isEmpty() ? UNDEFINED : pair.get(0))), pair.size() > 1 ? pair.get(1) : UNDEFINED);
                }
            }
            return m;
        }), true);
        g.declare("Set", new Native("Set", (in, s, a) -> {
            throw in.error("TypeError", "Set needs new");
        }, (in, s, a) -> {
            JsSet set = new JsSet();
            if (given(a, 0)) {
                for (Object x : in.iterate(a.getFirst(), 0)) {
                    set.set.add(in.wrapKey(norm(x)));
                }
            }
            return set;
        }), true);
        // tools answer at once, so a promise is just its value
        Native promise = new Native("Promise", (in, s, a) -> {
            throw in.error("TypeError", "tools answer at once; call them without new Promise");
        });
        promise.props.put("all", new Native("all", (in, s, a) -> in.iterate(arg(a, 0), 0)));
        promise.props.put("resolve", new Native("resolve", (in, s, a) -> arg(a, 0)));
        promise.props.put("allSettled", new Native("allSettled", (in, s, a) -> {
            List<Object> out = new ArrayList<>();
            for (Object v : in.iterate(arg(a, 0), 0)) {
                Map<String, Object> m = new LinkedHashMap<>();
                m.put("status", "fulfilled");
                m.put("value", v);
                out.add(m);
            }
            return out;
        }));
        g.declare("Promise", promise, true);
    }

    private static void fn(Map<String, Object> into, String name, java.util.function.Function<List<Object>, Object> f) {
        into.put(name, new Native(name, (in, s, a) -> f.apply(a)));
    }

    private static Object newArray(Interpreter in, List<Object> a) {
        if (a.size() == 1 && a.getFirst() instanceof Double d) {
            int n = d.intValue();
            if (n != d || n < 0 || n > 5_000_000) {
                throw in.error("RangeError", "invalid array length " + Interpreter.str(d.doubleValue()));
            }
            List<Object> out = new ArrayList<>(n);
            for (int i = 0; i < n; i++) {
                out.add(UNDEFINED);
            }
            return out;
        }
        return new ArrayList<>(a);
    }

    private static double parseFloat(String s) {
        java.util.regex.Matcher m = java.util.regex.Pattern.compile("^\\s*([+-]?(\\d+\\.?\\d*|\\.\\d+)([eE][+-]?\\d+)?|[+-]?Infinity)")
                .matcher(s);
        if (!m.find()) {
            return Double.NaN;
        }
        String t = m.group(1);
        return t.endsWith("Infinity") ? (t.startsWith("-") ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY)
                : Double.parseDouble(t);
    }

    private static double parseInt(String s, int radix) {
        String t = s.strip();
        boolean neg = t.startsWith("-");
        if (t.startsWith("-") || t.startsWith("+")) {
            t = t.substring(1);
        }
        if (radix == 16 && (t.startsWith("0x") || t.startsWith("0X"))) {
            t = t.substring(2);
        } else if (radix == 10 && (t.startsWith("0x") || t.startsWith("0X"))) {
            t = t.substring(2);
            radix = 16;
        }
        if (radix < 2 || radix > 36) {
            return Double.NaN;
        }
        int i = 0;
        while (i < t.length() && Character.digit(t.charAt(i), radix) >= 0) {
            i++;
        }
        if (i == 0) {
            return Double.NaN;
        }
        double v = new java.math.BigInteger(t.substring(0, i), radix).doubleValue();
        return neg ? -v : v;
    }

    static List<String> keys(Interpreter in, Object o, int line) {
        if (o instanceof Map<?, ?> m) {
            List<String> out = new ArrayList<>();
            for (Object k : m.keySet()) {
                out.add(String.valueOf(k));
            }
            return out;
        }
        if (o instanceof List<?> l) {
            List<String> out = new ArrayList<>();
            for (int i = 0; i < l.size(); i++) {
                out.add(String.valueOf(i));
            }
            return out;
        }
        if (o instanceof String s) {
            List<String> out = new ArrayList<>();
            for (int i = 0; i < s.length(); i++) {
                out.add(String.valueOf(i));
            }
            return out;
        }
        if (o == null || o == UNDEFINED) {
            throw in.typeError(line, "cannot list the keys of " + in.str(o));
        }
        return List.of();
    }

    static boolean instanceOf(Interpreter in, Object v, Object type) {
        if (!(type instanceof Native n)) {
            return false;
        }
        return switch (n.name) {
            case "Array" -> v instanceof List<?>;
            case "Object" -> v instanceof Map<?, ?> || v instanceof List<?> || v instanceof Fn || v instanceof JsMap
                    || v instanceof JsSet || v instanceof JsDate || v instanceof JsRegex;
            case "Error" -> v instanceof JsError;
            case "TypeError", "RangeError", "SyntaxError", "ReferenceError" -> v instanceof JsError e
                    && n.name.equals(e.get("name"));
            case "Map" -> v instanceof JsMap;
            case "Set" -> v instanceof JsSet;
            case "Date" -> v instanceof JsDate;
            case "RegExp" -> v instanceof JsRegex;
            default -> false;
        };
    }

    static String describeError(Interpreter in, Object v) {
        if (v instanceof Map<?, ?> m && m.containsKey("message")) {
            Object name = m.get("name");
            String n = name == null || name == UNDEFINED ? "Error" : String.valueOf(name);
            String msg = in.str(m.get("message"));
            return msg.isEmpty() ? n : n + ": " + msg;
        }
        return v instanceof String s ? s : in.str(v);
    }

    // ------------------------------------------------------------------------- JSON

    static String stringify(Interpreter in, Object v, String indent, String at) {
        StringBuilder b = new StringBuilder();
        write(in, b, v, indent, at);
        return b.toString();
    }

    private static void write(Interpreter in, StringBuilder b, Object v, String indent, String at) {
        if (b.length() > 5_000_000) {
            throw new ScriptError("JSON.stringify made more than 5000000 characters");
        }
        switch (v) {
            case null -> b.append("null");
            case String s -> b.append(Json.write(s));
            case Double d -> b.append(Double.isFinite(d) ? Interpreter.str(d.doubleValue()) : "null");
            case Boolean bool -> b.append(bool);
            case JsDate d -> b.append(Json.write(iso(d.time)));
            case List<?> l -> {
                if (l.isEmpty()) {
                    b.append("[]");
                    return;
                }
                String inner = at + indent;
                b.append('[');
                for (int i = 0; i < l.size(); i++) {
                    if (i > 0) {
                        b.append(',');
                    }
                    if (!indent.isEmpty()) {
                        b.append('\n').append(inner);
                    }
                    Object x = l.get(i);
                    write(in, b, x == UNDEFINED || x instanceof Fn ? null : x, indent, inner);
                }
                if (!indent.isEmpty()) {
                    b.append('\n').append(at);
                }
                b.append(']');
            }
            case Map<?, ?> m -> {
                String inner = at + indent;
                b.append('{');
                boolean first = true;
                for (Map.Entry<?, ?> e : m.entrySet()) {
                    Object x = e.getValue();
                    if (x == UNDEFINED || x instanceof Fn) {
                        continue;
                    }
                    if (!first) {
                        b.append(',');
                    }
                    first = false;
                    if (!indent.isEmpty()) {
                        b.append('\n').append(inner);
                    }
                    b.append(Json.write(String.valueOf(e.getKey()))).append(indent.isEmpty() ? ":" : ": ");
                    write(in, b, x, indent, inner);
                }
                if (!first && !indent.isEmpty()) {
                    b.append('\n').append(at);
                }
                b.append('}');
            }
            case JsMap m -> b.append("{}");
            case JsSet s -> b.append("{}");
            default -> b.append(v == UNDEFINED ? "null" : Json.write(in.str(v)));
        }
    }

    static Thrown thrown(String kind, String message) {
        return new Thrown(new JsError(kind, message));
    }
}
