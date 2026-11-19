package io.github.zzuegg.brainiac.studio.agent.js;

import java.util.ArrayList;
import java.util.List;
import java.util.Set;

/** Splits script text into tokens. */
final class Lexer {

    /** What a token is. */
    enum Kind { NAME, KEYWORD, NUMBER, STRING, TEMPLATE, REGEX, PUNCT, EOF }

    /**
     * A token.
     *
     * @param kind what it is
     * @param text its text: the name, the keyword, the punctuator, or a string's value
     * @param number a number's value
     * @param parts a template's pieces: text, then code, then text, and so on
     * @param flags a regex's flags
     * @param line its line, from 1
     * @param newlineBefore whether a line break comes before it
     */
    record Token(Kind kind, String text, double number, List<String> parts, String flags, int line,
            boolean newlineBefore) {

        boolean is(String punct) {
            return (kind == Kind.PUNCT || kind == Kind.KEYWORD) && text.equals(punct);
        }
    }

    static final Set<String> KEYWORDS = Set.of("var", "let", "const", "function", "return", "if", "else", "for",
            "while", "do", "break", "continue", "new", "typeof", "instanceof", "in", "of", "true", "false", "null",
            "undefined", "this", "throw", "try", "catch", "finally", "switch", "case", "default", "delete", "void",
            "async", "await", "class");

    // the soft keywords, which can also name things
    static final Set<String> SOFT = Set.of("of", "async", "undefined");

    private static final String[] PUNCTS = {">>>=", "...", "===", "!==", "**=", "<<=", ">>=", ">>>", "&&=", "||=",
        "??=", "=>", "==", "!=", "<=", ">=", "&&", "||", "??", "?.", "++", "--", "+=", "-=", "*=", "/=", "%=", "**",
        "<<", ">>", "&=", "|=", "^=", "{", "}", "(", ")", "[", "]", ";", ",", "<", ">", "+", "-", "*", "/", "%", "&",
        "|", "^", "!", "~", "?", ":", "=", ".", "#", "@"};

    private final String src;
    private int at;
    private int line = 1;

    private Lexer(String src) {
        this.src = src;
    }

    static List<Token> tokens(String src) {
        return new Lexer(src).all();
    }

    private List<Token> all() {
        List<Token> out = new ArrayList<>();
        while (true) {
            boolean nl = skipSpace();
            if (at >= src.length()) {
                out.add(new Token(Kind.EOF, "", 0, null, null, line, true));
                return out;
            }
            char c = src.charAt(at);
            int startLine = line;
            if (Character.isJavaIdentifierStart(c)) {
                int s = at;
                while (at < src.length() && Character.isJavaIdentifierPart(src.charAt(at))) {
                    at++;
                }
                String w = src.substring(s, at);
                out.add(new Token(KEYWORDS.contains(w) ? Kind.KEYWORD : Kind.NAME, w, 0, null, null, startLine, nl));
            } else if (Character.isDigit(c) || c == '.' && at + 1 < src.length() && Character.isDigit(src.charAt(at + 1))) {
                out.add(new Token(Kind.NUMBER, "", number(), null, null, startLine, nl));
            } else if (c == '"' || c == '\'') {
                out.add(new Token(Kind.STRING, string(c), 0, null, null, startLine, nl));
            } else if (c == '`') {
                out.add(new Token(Kind.TEMPLATE, "", 0, template(), null, startLine, nl));
            } else if (c == '/' && regexAllowed(out)) {
                out.add(regex(startLine, nl));
            } else {
                String p = null;
                for (String q : PUNCTS) {
                    if (src.startsWith(q, at)) {
                        p = q;
                        break;
                    }
                }
                if (p == null) {
                    throw new ScriptError("line " + line + ": unexpected character '" + c + "'");
                }
                // a?.5 is a ? .5, not optional chaining
                if (p.equals("?.") && at + 2 < src.length() && Character.isDigit(src.charAt(at + 2))) {
                    p = "?";
                }
                at += p.length();
                out.add(new Token(Kind.PUNCT, p, 0, null, null, startLine, nl));
            }
        }
    }

    // a slash starts a regex where a value may start
    private static boolean regexAllowed(List<Token> before) {
        if (before.isEmpty()) {
            return true;
        }
        Token t = before.getLast();
        return switch (t.kind()) {
            case NAME, NUMBER, STRING, TEMPLATE, REGEX -> false;
            case KEYWORD -> !Set.of("this", "true", "false", "null", "undefined").contains(t.text());
            case PUNCT -> !Set.of(")", "]", "}").contains(t.text());
            case EOF -> true;
        };
    }

    private boolean skipSpace() {
        boolean nl = false;
        while (at < src.length()) {
            char c = src.charAt(at);
            if (c == '\n') {
                nl = true;
                line++;
                at++;
            } else if (Character.isWhitespace(c) || c == '﻿') {
                at++;
            } else if (src.startsWith("//", at)) {
                while (at < src.length() && src.charAt(at) != '\n') {
                    at++;
                }
            } else if (src.startsWith("/*", at)) {
                int end = src.indexOf("*/", at + 2);
                if (end < 0) {
                    throw new ScriptError("line " + line + ": a comment is not closed");
                }
                for (int i = at; i < end; i++) {
                    if (src.charAt(i) == '\n') {
                        line++;
                        nl = true;
                    }
                }
                at = end + 2;
            } else {
                break;
            }
        }
        return nl;
    }

    private double number() {
        int s = at;
        if (src.startsWith("0x", at) || src.startsWith("0X", at)) {
            at += 2;
            while (at < src.length() && (Character.digit(src.charAt(at), 16) >= 0 || src.charAt(at) == '_')) {
                at++;
            }
            return Long.parseLong(src.substring(s + 2, at).replace("_", ""), 16);
        }
        if (src.startsWith("0b", at) || src.startsWith("0B", at)) {
            at += 2;
            while (at < src.length() && (src.charAt(at) == '0' || src.charAt(at) == '1' || src.charAt(at) == '_')) {
                at++;
            }
            return Long.parseLong(src.substring(s + 2, at).replace("_", ""), 2);
        }
        while (at < src.length() && (Character.isDigit(src.charAt(at)) || src.charAt(at) == '_')) {
            at++;
        }
        if (at < src.length() && src.charAt(at) == '.') {
            at++;
            while (at < src.length() && (Character.isDigit(src.charAt(at)) || src.charAt(at) == '_')) {
                at++;
            }
        }
        if (at < src.length() && (src.charAt(at) == 'e' || src.charAt(at) == 'E')) {
            at++;
            if (at < src.length() && (src.charAt(at) == '+' || src.charAt(at) == '-')) {
                at++;
            }
            while (at < src.length() && Character.isDigit(src.charAt(at))) {
                at++;
            }
        }
        if (at < src.length() && src.charAt(at) == 'n') {
            at++;   // a BigInt is read as a number
            return Double.parseDouble(src.substring(s, at - 1).replace("_", ""));
        }
        return Double.parseDouble(src.substring(s, at).replace("_", ""));
    }

    private String string(char quote) {
        at++;
        StringBuilder b = new StringBuilder();
        while (true) {
            if (at >= src.length() || src.charAt(at) == '\n') {
                throw new ScriptError("line " + line + ": a string is not closed");
            }
            char c = src.charAt(at++);
            if (c == quote) {
                return b.toString();
            }
            if (c == '\\') {
                escape(b);
            } else {
                b.append(c);
            }
        }
    }

    private void escape(StringBuilder b) {
        char e = src.charAt(at++);
        switch (e) {
            case 'n' -> b.append('\n');
            case 't' -> b.append('\t');
            case 'r' -> b.append('\r');
            case 'b' -> b.append('\b');
            case 'f' -> b.append('\f');
            case 'v' -> b.append('\u000B');
            case '0' -> b.append('\0');
            case '\n' -> line++;
            case 'x' -> {
                b.append((char) Integer.parseInt(src.substring(at, at + 2), 16));
                at += 2;
            }
            case 'u' -> {
                if (src.charAt(at) == '{') {
                    int end = src.indexOf('}', at);
                    b.appendCodePoint(Integer.parseInt(src.substring(at + 1, end), 16));
                    at = end + 1;
                } else {
                    b.append((char) Integer.parseInt(src.substring(at, at + 4), 16));
                    at += 4;
                }
            }
            default -> b.append(e);
        }
    }

    // text and code alternate: text, code, text, ..., text
    private List<String> template() {
        at++;
        List<String> parts = new ArrayList<>();
        StringBuilder b = new StringBuilder();
        while (true) {
            if (at >= src.length()) {
                throw new ScriptError("line " + line + ": a template string is not closed");
            }
            char c = src.charAt(at++);
            if (c == '`') {
                parts.add(b.toString());
                return parts;
            }
            if (c == '\\') {
                escape(b);
            } else if (c == '$' && at < src.length() && src.charAt(at) == '{') {
                parts.add(b.toString());
                b.setLength(0);
                at++;
                int s = at;
                int depth = 1;
                while (depth > 0) {
                    if (at >= src.length()) {
                        throw new ScriptError("line " + line + ": a ${ in a template is not closed");
                    }
                    char d = src.charAt(at);
                    if (d == '{') {
                        depth++;
                    } else if (d == '}') {
                        depth--;
                    } else if (d == '\'' || d == '"') {
                        string(d);
                        continue;
                    } else if (d == '`') {
                        template();
                        continue;
                    } else if (d == '\n') {
                        line++;
                    }
                    at++;
                }
                parts.add(src.substring(s, at - 1));
            } else {
                if (c == '\n') {
                    line++;
                }
                b.append(c);
            }
        }
    }

    private Token regex(int startLine, boolean nl) {
        at++;
        StringBuilder b = new StringBuilder();
        boolean inClass = false;
        while (true) {
            if (at >= src.length() || src.charAt(at) == '\n') {
                throw new ScriptError("line " + line + ": a regular expression is not closed");
            }
            char c = src.charAt(at++);
            if (c == '\\') {
                b.append(c).append(src.charAt(at++));
                continue;
            }
            if (c == '[') {
                inClass = true;
            } else if (c == ']') {
                inClass = false;
            } else if (c == '/' && !inClass) {
                break;
            }
            b.append(c);
        }
        int s = at;
        while (at < src.length() && Character.isLetter(src.charAt(at))) {
            at++;
        }
        return new Token(Kind.REGEX, b.toString(), 0, null, src.substring(s, at), startLine, nl);
    }
}
