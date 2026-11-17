package io.github.zzuegg.brainiac.autograd;

import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.TreeSet;

/**
 * Einstein summation, as {@code numpy.einsum} and {@code torch.einsum}: {@code "ij,jk->ik"}
 * multiplies matrices, {@code "bhqd,bhkd->bhqk"} gives attention scores, {@code "ii->i"} is
 * not supported (no letter may repeat within one operand). Without {@code ->} the output has
 * the letters that appear once, in alphabetical order.
 *
 * <p>Each pair of operands becomes one batched matrix product: letters in both operands and
 * the output form the batch, letters in both but not the output are summed by the product,
 * and letters in only one operand and not the output are summed first. Built from
 * {@link Var} operations, so it runs on every device and has gradients.
 *
 * <pre>
 * Var scores = Einsum.of("bqd,bkd->bqk", queries, keys);
 * Tensor outer = Einsum.of("i,j->ij", a, b);
 * </pre>
 */
public final class Einsum {

    private Einsum() {
    }

    /**
     * Evaluates an expression on tensors, without gradients.
     *
     * @param expression such as {@code "ij,jk->ik"}
     * @param operands one tensor per input term
     * @return the result
     */
    public static Tensor of(String expression, Tensor... operands) {
        Var[] vars = new Var[operands.length];
        for (int i = 0; i < vars.length; i++) {
            vars[i] = Var.of(operands[i]);
        }
        return of(expression, vars).value();
    }

    /**
     * Evaluates an expression.
     *
     * @param expression such as {@code "ij,jk->ik"}
     * @param operands one variable per input term
     * @return the result
     */
    public static Var of(String expression, Var... operands) {
        String e = expression.replace(" ", "");
        int arrow = e.indexOf("->");
        String[] terms = (arrow < 0 ? e : e.substring(0, arrow)).split(",", -1);
        if (terms.length != operands.length) {
            throw new IllegalArgumentException(expression + " has " + terms.length + " inputs, but "
                    + operands.length + " operands were given");
        }
        String out;
        if (arrow >= 0) {
            out = e.substring(arrow + 2);
        } else {
            TreeSet<Character> once = new TreeSet<>();
            String all = String.join("", terms);
            for (char c : all.toCharArray()) {
                if (all.indexOf(c) == all.lastIndexOf(c)) {
                    once.add(c);
                }
            }
            StringBuilder b = new StringBuilder();
            once.forEach(b::append);
            out = b.toString();
        }
        int[] sizes = new int[128];
        for (int i = 0; i < terms.length; i++) {
            String t = terms[i];
            if (t.contains(".")) {
                throw new IllegalArgumentException("ellipses are not supported: " + expression);
            }
            if (t.length() != operands[i].shape().rank()) {
                throw new IllegalArgumentException("term " + t + " has " + t.length() + " letters for a "
                        + operands[i].shape() + " operand");
            }
            for (int a = 0; a < t.length(); a++) {
                char c = t.charAt(a);
                if (!Character.isLetter(c) || c >= 128) {
                    throw new IllegalArgumentException("not a letter: " + c + " in " + expression);
                }
                if (t.indexOf(c) != a) {
                    throw new IllegalArgumentException("letter " + c + " repeats in " + t + "; diagonals are not supported");
                }
                int n = operands[i].shape().dim(a);
                if (sizes[c] != 0 && sizes[c] != n) {
                    throw new IllegalArgumentException("letter " + c + " is " + sizes[c] + " and " + n + " long");
                }
                sizes[c] = n;
            }
        }
        for (char c : out.toCharArray()) {
            if (sizes[c] == 0) {
                throw new IllegalArgumentException("output letter " + c + " is in no input");
            }
            if (out.indexOf(c) != out.lastIndexOf(c)) {
                throw new IllegalArgumentException("output letter " + c + " repeats");
            }
        }
        // fold left: each step keeps the letters the output or a later operand needs
        Var acc = operands[0];
        String accTerm = terms[0];
        for (int i = 1; i < terms.length; i++) {
            StringBuilder later = new StringBuilder(out);
            for (int j = i + 1; j < terms.length; j++) {
                later.append(terms[j]);
            }
            String keep = later.toString();
            String[] term = new String[1];
            acc = pair(acc, accTerm, operands[i], terms[i], keep, sizes, term);
            accTerm = term[0];
        }
        acc = sumOut(acc, accTerm, out, null);
        accTerm = only(accTerm, out, null);
        return arrange(acc, accTerm, out);
    }

    /** One product of two operands; {@code term[0]} receives the result's letters. */
    private static Var pair(Var a, String ta, Var b, String tb, String keep, int[] sizes, String[] term) {
        // letters only in one operand, and needed nowhere else, are summed first
        a = sumOut(a, ta, keep, tb);
        ta = only(ta, keep, tb);
        b = sumOut(b, tb, keep, ta);
        tb = only(tb, keep, ta);
        List<Character> batch = new ArrayList<>();
        List<Character> contract = new ArrayList<>();
        List<Character> aOnly = new ArrayList<>();
        List<Character> bOnly = new ArrayList<>();
        for (char c : ta.toCharArray()) {
            if (tb.indexOf(c) >= 0) {
                (keep.indexOf(c) >= 0 ? batch : contract).add(c);
            } else {
                aOnly.add(c);
            }
        }
        for (char c : tb.toCharArray()) {
            if (ta.indexOf(c) < 0) {
                bOnly.add(c);
            }
        }
        Var am = arrange(a, ta, letters(batch, aOnly, contract));
        Var bm = arrange(b, tb, letters(batch, contract, bOnly));
        int nb = product(batch, sizes);
        int m = product(aOnly, sizes);
        int k = product(contract, sizes);
        int n = product(bOnly, sizes);
        Var r = am.reshape(nb, m, k).matmul(bm.reshape(nb, k, n));
        List<Character> dims = new ArrayList<>(batch);
        dims.addAll(aOnly);
        dims.addAll(bOnly);
        int[] shape = new int[dims.size()];
        for (int d = 0; d < shape.length; d++) {
            shape[d] = sizes[dims.get(d)];
        }
        term[0] = letters(dims);
        return shape.length == 0 ? r.reshape() : r.reshape(shape);
    }

    /** Sums the axes of letters needed in neither {@code keep} nor {@code other}. */
    private static Var sumOut(Var v, String t, String keep, String other) {
        for (int a = t.length() - 1; a >= 0; a--) {
            char c = t.charAt(a);
            if (keep.indexOf(c) < 0 && (other == null || other.indexOf(c) < 0)) {
                v = v.sum(a, false);
            }
        }
        return v;
    }

    private static String only(String t, String keep, String other) {
        StringBuilder b = new StringBuilder();
        for (char c : t.toCharArray()) {
            if (keep.indexOf(c) >= 0 || (other != null && other.indexOf(c) >= 0)) {
                b.append(c);
            }
        }
        return b.toString();
    }

    /** Permutes {@code v}, labelled {@code from}, into the order {@code to}. */
    private static Var arrange(Var v, String from, String to) {
        if (from.equals(to)) {
            return v;
        }
        int[] axes = new int[to.length()];
        for (int i = 0; i < axes.length; i++) {
            axes[i] = from.indexOf(to.charAt(i));
        }
        return v.permute(axes);
    }

    @SafeVarargs
    private static String letters(List<Character>... groups) {
        StringBuilder b = new StringBuilder();
        for (List<Character> g : groups) {
            g.forEach(b::append);
        }
        return b.toString();
    }

    private static int product(List<Character> letters, int[] sizes) {
        int p = 1;
        for (char c : letters) {
            p *= sizes[c];
        }
        return p;
    }
}
