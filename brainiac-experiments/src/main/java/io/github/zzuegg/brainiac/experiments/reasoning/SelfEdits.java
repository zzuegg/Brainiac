package io.github.zzuegg.brainiac.experiments.reasoning;

import io.github.zzuegg.brainiac.experimental.llm.SelfEditTrainer;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * SEAL on toy fact injection. A passage states a new fact value-first, {@code v REL e}
 * ("v is e's value"); the question asks for it entity-first, {@code e Q} with answer
 * {@code v}. Facts are injected four at a time (four different entities and values), so a
 * model cannot answer by simply favouring whatever value it saw last. A knowledge model
 * fine-tuned on the raw passages mostly cannot answer (the reversal curse of a left-to-right
 * model); a {@link SelfEditTrainer} editor learns, from recall rewards alone, to rewrite each
 * passage into a training sequence that works, starting from an editor that restates the
 * passage verbatim. Recall after an update on the editor's
 * greedy self-edits is compared with recall after an update on the raw passages.
 */
public final class SelfEdits {

    static final int ENTITIES = 16;
    static final int VALUES = 8;
    static final int REL = ENTITIES + VALUES;
    static final int Q = REL + 1;
    static final int EDIT = REL + 2;
    static final int END = REL + 3;
    static final int VOCAB = REL + 4;
    private static final int UPDATE_STEPS = 10;

    /**
     * Recall of injected facts.
     *
     * @param rawPassage recall after updating on the raw passages
     * @param selfEdit recall after updating on the trained editor's self-edits
     * @param untrainedEdit recall after updating on the untrained editor's self-edits
     * @param oracleEdit recall after updating on the ideal rewrite {@code e Q v}, the ceiling
     */
    public record Result(double rawPassage, double selfEdit, double untrainedEdit, double oracleEdit) {
    }

    private SelfEdits() {
    }

    private static int value(int v) {
        return ENTITIES + v;
    }

    /** A knowledge model that knows both formats: trained on facts about entities 8-15. */
    static WindowLm knowledgeBase(long seed) {
        WindowLm lm = WindowLm.of(VOCAB, 4, 48, seed);
        SplittableRandom r = new SplittableRandom(seed);
        List<int[]> ps = new ArrayList<>();
        List<int[]> cs = new ArrayList<>();
        for (int e = 8; e < ENTITIES; e++) {
            int v = value(r.nextInt(VALUES));
            ps.add(new int[0]);
            cs.add(new int[] {v, REL, e});
            ps.add(new int[] {e, Q});
            cs.add(new int[] {v});
        }
        Optimizer opt = Optimizer.adam(1e-2f);
        for (int s = 0; s < 150; s++) {
            lm.supervisedStep(ps, cs, opt);
        }
        return lm;
    }

    /** Four facts about distinct entities 0-7 with distinct values: {@code {v, REL, e, EDIT}} prompts. */
    static List<int[]> facts(SplittableRandom r) {
        int[] es = shuffled(8, r);
        int[] vs = shuffled(VALUES, r);
        List<int[]> out = new ArrayList<>();
        for (int i = 0; i < 4; i++) {
            out.add(new int[] {value(vs[i]), REL, es[i], EDIT});
        }
        return out;
    }

    private static int[] shuffled(int n, SplittableRandom r) {
        int[] a = new int[n];
        for (int i = 0; i < n; i++) {
            a[i] = i;
        }
        for (int i = n - 1; i > 0; i--) {
            int j = r.nextInt(i + 1);
            int t = a[i];
            a[i] = a[j];
            a[j] = t;
        }
        return a;
    }

    /**
     * The editor's starting point, standing in for a pretrained model asked to rewrite a
     * passage: it restates the passage verbatim, so its first self-edits are the raw passage.
     */
    static WindowLm copyingEditor(long seed) {
        WindowLm editor = WindowLm.of(VOCAB, 4, 64, seed);
        List<int[]> ps = new ArrayList<>();
        List<int[]> cs = new ArrayList<>();
        for (int e = 0; e < ENTITIES; e++) {
            for (int v = 0; v < VALUES; v++) {
                int[] p = {value(v), REL, e, EDIT};
                ps.add(p);
                cs.add(raw(p));
            }
        }
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int s = 0; s < 40; s++) {
            editor.supervisedStep(ps, cs, opt);
        }
        return editor;
    }

    private static int[] raw(int[] prompt) {
        return new int[] {prompt[0], REL, prompt[2]};
    }

    /** How many of the facts the knowledge model answers after an update on {@code docs}. */
    static int recalled(WindowLm base, List<int[]> docs, List<int[]> facts) {
        WindowLm updated = SelfEditTrainer.update(base, docs, UPDATE_STEPS, Optimizer.adam(1e-2f));
        int right = 0;
        for (int[] f : facts) {
            right += updated.greedy(new int[] {f[2], Q}, 1, END)[0] == f[0] ? 1 : 0;
        }
        return right;
    }

    /**
     * @param rounds ReST-EM rounds of the editor
     * @param seed the seed
     * @return recall of new facts about entities 0-7
     */
    public static Result run(int rounds, long seed) {
        WindowLm base = knowledgeBase(seed);
        WindowLm editor = copyingEditor(seed + 1);
        SplittableRandom random = new SplittableRandom(seed);
        // a candidate edit is scored inside an update that also holds three other facts' raw
        // passages; the reward is log p(v | e Q) plus 1 when greedy recall is right
        SelfEditTrainer seal = SelfEditTrainer.of(editor, (prompt, edit) -> {
            List<int[]> docs = new ArrayList<>();
            docs.add(edit);
            SplittableRandom dr = new SplittableRandom(prompt[0] * 31L + prompt[2]);
            for (int[] f : facts(dr)) {
                if (f[2] != prompt[2] && f[0] != prompt[0] && docs.size() < 4) {
                    docs.add(raw(f));
                }
            }
            WindowLm updated = SelfEditTrainer.update(base, docs, UPDATE_STEPS, Optimizer.adam(1e-2f));
            int[] q = {prompt[2], Q};
            return updated.sequenceLogProb(q, new int[] {prompt[0]})
                    + (updated.greedy(q, 1, END)[0] == prompt[0] ? 1 : 0);
        }).edits(3, END, 16, 1.0).finetune(5, () -> Optimizer.adam(3e-3f));
        List<List<int[]>> heldOut = new ArrayList<>();
        SplittableRandom testRandom = new SplittableRandom(seed + 99);
        for (int g = 0; g < 4; g++) {
            heldOut.add(facts(testRandom));
        }
        int untrained = 0;
        for (List<int[]> group : heldOut) {
            List<int[]> docs = new ArrayList<>();
            for (int[] f : group) {
                docs.add(seal.write(f));
            }
            untrained += recalled(base, docs, group);
        }
        for (int r = 0; r < rounds; r++) {
            seal.round(facts(random), random);
        }
        int raw = 0;
        int oracle = 0;
        int edited = 0;
        for (List<int[]> group : heldOut) {
            List<int[]> rawDocs = new ArrayList<>();
            List<int[]> editDocs = new ArrayList<>();
            List<int[]> oracleDocs = new ArrayList<>();
            for (int[] f : group) {
                rawDocs.add(raw(f));
                editDocs.add(seal.write(f));
                oracleDocs.add(new int[] {f[2], Q, f[0]});
            }
            raw += recalled(base, rawDocs, group);
            edited += recalled(base, editDocs, group);
            oracle += recalled(base, oracleDocs, group);
        }
        double n = 4.0 * heldOut.size();
        return new Result(raw / n, edited / n, untrained / n, oracle / n);
    }
}
