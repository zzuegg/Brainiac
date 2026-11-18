package io.github.zzuegg.brainiac.experimental.graph;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.data.RandomSatFormulas;
import io.github.zzuegg.brainiac.experimental.nn.GruCell;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * NeuroSAT (Selsam et al., 2019, <a href="https://arxiv.org/abs/1802.03685">arXiv:1802.03685</a>):
 * a message-passing network on the literal-clause graph of a CNF formula, trained only on
 * the one-bit label "satisfiable". Literal and clause states are updated by recurrent
 * cells (here {@link GruCell}s instead of the paper's LSTMs) for {@code T} rounds:
 * <pre>
 * C' = GRU_C(C, M^T MLP_L(L))                       clauses read their literals
 * L' = GRU_L(L, [M MLP_C(C') || flip(L)])           literals read their clauses and their negation
 * logit = mean_literals MLP_vote(L_T)
 * </pre>
 * where {@code M} is the literal-clause incidence and {@code flip} swaps every literal
 * with its negation. The model is invariant to variable and clause permutations and to
 * negating a variable. More rounds at test time than in training let it solve harder
 * formulas, and on satisfiable formulas the literal embeddings often split into two
 * clusters from which an assignment can be decoded ({@link #decode}).
 *
 * <p>A batch of formulas is one block-diagonal incidence ({@link #batch}).
 */
public final class NeuroSat {

    private final int width;
    private final GruCell literalCell;
    private final GruCell clauseCell;
    private final Var lInit;
    private final Var cInit;
    private final Var[] lMsg;
    private final Var[] cMsg;
    private final Var[] vote;

    /**
     * Formulas as one graph.
     *
     * @param incidence {@code [literals, clauses]}, 1 where a clause contains a literal
     * @param flip for every literal, the index of its negation
     * @param membership {@code [formulas, literals]}, 1 where a literal belongs to a formula
     * @param formulas the formulas
     */
    public record Batch(Tensor incidence, Tensor flip, Tensor membership, List<RandomSatFormulas.Formula> formulas) {
    }

    /**
     * Creates the network.
     *
     * @param width the literal and clause state width
     * @param random the initialization source
     */
    public NeuroSat(int width, SplittableRandom random) {
        this.width = width;
        this.literalCell = new GruCell(2 * width, width, random);
        this.clauseCell = new GruCell(width, width, random);
        this.lInit = Var.param(Tensor.normal(Shape.of(1, width), 0, 1f / (float) Math.sqrt(width), random));
        this.cInit = Var.param(Tensor.normal(Shape.of(1, width), 0, 1f / (float) Math.sqrt(width), random));
        this.lMsg = mlp(random);
        this.cMsg = mlp(random);
        this.vote = new Var[] {GraphMath.weight(width, width, random), GraphMath.bias(width),
            GraphMath.weight(width, 1, random), GraphMath.bias(1)};
    }

    private Var[] mlp(SplittableRandom random) {
        return new Var[] {GraphMath.weight(width, width, random), GraphMath.bias(width),
            GraphMath.weight(width, width, random), GraphMath.bias(width)};
    }

    private static Var apply(Var[] p, Var x) {
        return GraphMath.linear(GraphMath.linear(x, p[0], p[1]).relu(), p[2], p[3]);
    }

    /**
     * Builds the literal-clause graph of several formulas. Literal {@code +v} of a formula
     * with offset {@code o} is node {@code o + v - 1}, literal {@code -v} is
     * {@code o + n + v - 1}.
     *
     * @param formulas the formulas
     * @return the batch
     */
    public static Batch batch(List<RandomSatFormulas.Formula> formulas) {
        int literals = 0;
        int clauses = 0;
        for (RandomSatFormulas.Formula f : formulas) {
            literals += 2 * f.variables();
            clauses += f.clauses().size();
        }
        float[] m = new float[literals * clauses];
        float[] flip = new float[literals];
        float[] member = new float[formulas.size() * literals];
        int lo = 0;
        int co = 0;
        for (int g = 0; g < formulas.size(); g++) {
            RandomSatFormulas.Formula f = formulas.get(g);
            int n = f.variables();
            for (int v = 0; v < n; v++) {
                flip[lo + v] = lo + n + v;
                flip[lo + n + v] = lo + v;
                member[g * literals + lo + v] = 1;
                member[g * literals + lo + n + v] = 1;
            }
            for (int c = 0; c < f.clauses().size(); c++) {
                for (int lit : f.clauses().get(c)) {
                    int node = lo + (lit > 0 ? lit - 1 : n - lit - 1);
                    m[node * clauses + co + c] = 1;
                }
            }
            lo += 2 * n;
            co += f.clauses().size();
        }
        return new Batch(Tensor.of(Shape.of(literals, clauses), m), Tensor.of(Shape.of(literals), flip),
                Tensor.of(Shape.of(formulas.size(), literals), member), List.copyOf(formulas));
    }

    /**
     * Runs the message passing.
     *
     * @param batch the formulas
     * @param rounds the rounds {@code T}
     * @return the final literal states {@code [literals, width]}
     */
    public Var literals(Batch batch, int rounds) {
        int nl = batch.incidence().shape().dim(0);
        int nc = batch.incidence().shape().dim(1);
        Var m = Var.of(batch.incidence());
        Var mt = Var.of(batch.incidence().transpose());
        Var l = lInit.expand(Shape.of(nl, width));
        Var c = cInit.expand(Shape.of(nc, width));
        for (int t = 0; t < rounds; t++) {
            c = clauseCell.step(mt.matmul(apply(lMsg, l)), c);
            l = literalCell.step(Var.concat(1, m.matmul(apply(cMsg, c)), l.gather(0, batch.flip())), l);
        }
        return l;
    }

    /**
     * The satisfiability logits.
     *
     * @param batch the formulas
     * @param rounds the rounds {@code T}
     * @return {@code [formulas]}
     */
    public Var logits(Batch batch, int rounds) {
        Var votes = apply(vote, literals(batch, rounds));
        Tensor mem = batch.membership();
        Tensor count = mem.sum(1, true);
        return Var.of(mem.div(count)).matmul(votes).reshape(mem.shape().dim(0));
    }

    /**
     * Decodes an assignment per formula: the literal embeddings are split into two
     * clusters by 2-means, and each of the two ways of reading the clusters as true and
     * false is tried.
     *
     * @param batch the formulas
     * @param rounds the rounds
     * @return per formula, a satisfying assignment ({@code [v]} for {@code v = 1..n}) or
     *     {@code null} if neither reading satisfies it
     */
    public List<boolean[]> decode(Batch batch, int rounds) {
        float[] l = Var.noGrad(() -> literals(batch, rounds)).value().toArray();
        List<boolean[]> out = new ArrayList<>();
        int lo = 0;
        for (RandomSatFormulas.Formula f : batch.formulas()) {
            int n = f.variables();
            int[] cluster = twoMeans(l, lo, 2 * n);
            boolean[] found = null;
            for (int side = 0; side < 2 && found == null; side++) {
                boolean[] a = new boolean[n + 1];
                for (int v = 0; v < n; v++) {
                    a[v + 1] = cluster[v] == side;
                }
                if (RandomSatFormulas.satisfies(f.clauses(), a)) {
                    found = a;
                }
            }
            out.add(found);
            lo += 2 * n;
        }
        return out;
    }

    private int[] twoMeans(float[] l, int offset, int count) {
        double[][] c = new double[2][width];
        for (int k = 0; k < width; k++) {
            c[0][k] = l[offset * width + k];
            c[1][k] = l[(offset + count / 2) * width + k];
        }
        int[] assign = new int[count];
        for (int it = 0; it < 10; it++) {
            double[][] sum = new double[2][width];
            int[] size = new int[2];
            for (int i = 0; i < count; i++) {
                double d0 = 0;
                double d1 = 0;
                for (int k = 0; k < width; k++) {
                    double x = l[(offset + i) * width + k];
                    d0 += (x - c[0][k]) * (x - c[0][k]);
                    d1 += (x - c[1][k]) * (x - c[1][k]);
                }
                assign[i] = d0 <= d1 ? 0 : 1;
                size[assign[i]]++;
                for (int k = 0; k < width; k++) {
                    sum[assign[i]][k] += l[(offset + i) * width + k];
                }
            }
            for (int s = 0; s < 2; s++) {
                for (int k = 0; k < width && size[s] > 0; k++) {
                    c[s][k] = sum[s][k] / size[s];
                }
            }
        }
        return assign;
    }

    /** @return every parameter */
    public List<Var> parameters() {
        List<Var> p = new ArrayList<>(literalCell.parameters());
        p.addAll(clauseCell.parameters());
        p.add(lInit);
        p.add(cInit);
        p.addAll(List.of(lMsg));
        p.addAll(List.of(cMsg));
        p.addAll(List.of(vote));
        return p;
    }

    @Override
    public String toString() {
        return "NeuroSat(" + width + ")";
    }
}
