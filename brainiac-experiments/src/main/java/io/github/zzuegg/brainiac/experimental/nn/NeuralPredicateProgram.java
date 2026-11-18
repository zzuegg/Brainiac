package io.github.zzuegg.brainiac.experimental.nn;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.tensor.Shape;
import io.github.zzuegg.brainiac.tensor.Tensor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.Predicate;

/**
 * Neural predicates inside a probabilistic logic program, in the style of DeepProbLog and
 * Scallop. A program has {@code slots} objects (such as two digit images), each with a
 * <em>neural predicate</em>: a network gives the probability that the object takes each
 * value of a finite domain ({@code digit(img, d)}). A query, such as
 * {@code sum(img1, img2, 7)}, is proved by the value assignments that satisfy its rule
 * body ({@code digit(img1, a), digit(img2, b), a + b = 7}); each proof's weight is the
 * product of its facts' probabilities and the query's probability is the sum over its
 * proofs (the add-mult semiring; exact here, because two different assignments of the
 * same objects are mutually exclusive). With {@code topK > 0} only the {@code k} most
 * probable proofs of each example are kept (Scallop's top-k proofs provenance), which
 * bounds the cost when proofs are many. The result is a {@link Var}, so the query's
 * log-likelihood trains the neural predicates from labels on the result alone.
 *
 * <p>{@link #ground} enumerates the proofs of a rule body over the domain; for larger
 * programs the caller can supply proofs from any Datalog-style grounding.
 *
 * <p>Papers: Manhaeve et al., "DeepProbLog: Neural Probabilistic Logic Programming"
 * (NeurIPS 2018), <a href="https://arxiv.org/abs/1805.10872">arXiv:1805.10872</a>; Li et al.,
 * "Scallop: A Language for Neurosymbolic Programming" (PLDI 2023),
 * <a href="https://arxiv.org/abs/2304.04812">arXiv:2304.04812</a>.
 */
public final class NeuralPredicateProgram {

    private final int slots;
    private final int domain;

    /**
     * Creates a program over {@code slots} objects with values in {@code 0..domain-1}.
     *
     * @param slots the objects per example
     * @param domain the values of the neural predicate
     */
    public NeuralPredicateProgram(int slots, int domain) {
        this.slots = slots;
        this.domain = domain;
    }

    /**
     * Grounds a rule body: every assignment of values to the slots that satisfies it.
     *
     * @param body the rule body over an assignment {@code values[slot]}
     * @return the proofs
     */
    public List<int[]> ground(Predicate<int[]> body) {
        List<int[]> out = new ArrayList<>();
        int[] a = new int[slots];
        long total = (long) Math.pow(domain, slots);
        for (long code = 0; code < total; code++) {
            long c = code;
            for (int s = 0; s < slots; s++) {
                a[s] = (int) (c % domain);
                c /= domain;
            }
            if (body.test(a)) {
                out.add(a.clone());
            }
        }
        return out;
    }

    /**
     * The probability of each example's query.
     *
     * @param distributions per slot, the neural predicate's probabilities {@code [batch, domain]}
     * @param proofs per example, its proofs (assignments of values to slots)
     * @param topK keep only the {@code k} most probable proofs per example; 0 keeps all
     * @return {@code [batch]} query probabilities
     */
    public Var probability(List<Var> distributions, List<List<int[]>> proofs, int topK) {
        int b = proofs.size();
        List<List<int[]>> kept = new ArrayList<>();
        for (int i = 0; i < b; i++) {
            List<int[]> p = proofs.get(i);
            if (topK > 0 && p.size() > topK) {
                float[][] probs = new float[slots][];
                for (int s = 0; s < slots; s++) {
                    probs[s] = distributions.get(s).value().slice(0, i, i + 1).toArray();
                }
                List<int[]> sorted = new ArrayList<>(p);
                sorted.sort((x, y) -> Double.compare(weight(y, probs), weight(x, probs)));
                p = sorted.subList(0, topK);
            }
            kept.add(p);
        }
        int width = 1;
        for (List<int[]> p : kept) {
            width = Math.max(width, p.size());
        }
        Var product = null;
        for (int s = 0; s < slots; s++) {
            // an extra zero column takes the padding proofs
            Var padded = Var.concat(1, distributions.get(s), Var.of(Tensor.zeros(Shape.of(b, 1))));
            float[] idx = new float[b * width];
            for (int i = 0; i < b; i++) {
                List<int[]> p = kept.get(i);
                for (int j = 0; j < width; j++) {
                    idx[i * width + j] = i * (domain + 1) + (j < p.size() ? p.get(j)[s] : domain);
                }
            }
            Var facts = padded.reshape(b * (domain + 1)).gather(0, Tensor.of(Shape.of(b * width), idx))
                    .reshape(b, width);
            product = product == null ? facts : product.mul(facts);
        }
        return product.sum(1, false);
    }

    private static double weight(int[] proof, float[][] probs) {
        double w = 1;
        for (int s = 0; s < proof.length; s++) {
            w *= probs[s][proof[s]];
        }
        return w;
    }

    /**
     * The negative log-likelihood of the queries, the training loss.
     *
     * @param distributions per slot, {@code [batch, domain]} probabilities
     * @param proofs per example, its proofs
     * @param topK as for {@link #probability}
     * @return the mean of {@code -log P(query)}
     */
    public Var loss(List<Var> distributions, List<List<int[]>> proofs, int topK) {
        return probability(distributions, proofs, topK).add(1e-7f).log().neg().mean();
    }

    @Override
    public String toString() {
        return "NeuralPredicateProgram(" + slots + " slots, domain " + domain + ")";
    }

    /**
     * The proofs of {@code sum(slot values) == target} for each target, cached by target.
     *
     * @param targets per example, the observed sum
     * @return per example, the assignments whose values add up to it
     */
    public List<List<int[]>> sumProofs(int[] targets) {
        List<List<int[]>> byTarget = new ArrayList<>();
        for (int t = 0; t <= slots * (domain - 1); t++) {
            int tt = t;
            byTarget.add(ground(a -> Arrays.stream(a).sum() == tt));
        }
        List<List<int[]>> out = new ArrayList<>();
        for (int t : targets) {
            out.add(byTarget.get(t));
        }
        return out;
    }
}
