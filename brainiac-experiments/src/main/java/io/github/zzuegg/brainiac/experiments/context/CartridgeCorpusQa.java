package io.github.zzuegg.brainiac.experiments.context;

import io.github.zzuegg.brainiac.autograd.Var;
import io.github.zzuegg.brainiac.experimental.llm.CartridgeTrainer;
import io.github.zzuegg.brainiac.experimental.llm.HookedDecoder;
import io.github.zzuegg.brainiac.experimental.llm.SoftPrompts;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.SplittableRandom;

/**
 * Cartridges (<a href="https://arxiv.org/abs/2506.06266">arXiv:2506.06266</a>) on corpus
 * question answering: accuracy against KV-cache bytes. A one-block {@link HookedDecoder} is
 * trained to answer from its context: a corpus is {@value #FACTS} facts, each one token that
 * binds a key to a value, in random order, and the question is a key token whose next token
 * must be the value. Then, for one fixed corpus, the arms are: the full corpus in context;
 * keeping only its last {@code p} tokens (recency eviction, what a fixed KV budget without a
 * selection policy does); a {@code p}-entry {@link CartridgeTrainer} cartridge trained by
 * context distillation on self-study questions; and prefix tuning ({@link SoftPrompts}) of
 * the same size, started from random tokens and trained by next-token prediction on the
 * answers. Bytes are 32-bit keys and values per cached position.
 */
public final class CartridgeCorpusQa {
    /** Facts per corpus. */
    public static final int FACTS = 16;
    private static final int VALUES = 10;
    private static final int QUERY = FACTS * VALUES;
    private static final int ANSWER = QUERY + FACTS;
    private static final int VOCAB = ANSWER + VALUES;

    /**
     * One arm at one budget.
     *
     * @param accuracy share of the corpus's questions answered correctly
     * @param kvBytes cache memory the answer needed
     */
    public record Cell(double accuracy, long kvBytes) {
    }

    private CartridgeCorpusQa() {
    }

    private static int[] corpus(SplittableRandom random) {
        int[] keys = new int[FACTS];
        for (int i = 0; i < FACTS; i++) {
            keys[i] = i;
        }
        for (int i = FACTS - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int t = keys[i];
            keys[i] = keys[j];
            keys[j] = t;
        }
        int[] c = new int[FACTS];
        for (int i = 0; i < FACTS; i++) {
            c[i] = keys[i] * VALUES + random.nextInt(VALUES);
        }
        return c;
    }

    private static int answer(int[] corpus, int key) {
        for (int f : corpus) {
            if (f / VALUES == key) {
                return ANSWER + f % VALUES;
            }
        }
        return -1;
    }

    /** The in-context reader: trained on fresh random corpora. */
    static HookedDecoder reader(int steps, long seed) {
        HookedDecoder m = HookedDecoder.create(HookedDecoder.Settings.tiny(VOCAB).withLayers(1), seed);
        SplittableRandom r = new SplittableRandom(seed + 1);
        Optimizer opt = Optimizer.adam(3e-3f);
        int batch = 32;
        for (int s = 0; s < steps; s++) {
            int[][] seqs = new int[batch][];
            int[][] targets = new int[batch][FACTS + 1];
            for (int b = 0; b < batch; b++) {
                int[] c = corpus(r);
                int key = r.nextInt(FACTS);
                seqs[b] = Arrays.copyOf(c, FACTS + 1);
                seqs[b][FACTS] = QUERY + key;
                Arrays.fill(targets[b], -1);
                targets[b][FACTS] = answer(c, key);
            }
            Var loss = HookedDecoder.loss(m.forward(seqs).logits(), targets);
            loss.backward();
            opt.step(m.parameters());
            m.zeroGrad();
        }
        return m;
    }

    private static double accuracy(HookedDecoder m, int[] context, int[] corpus) {
        int ok = 0;
        for (int key = 0; key < FACTS; key++) {
            int[] seq = Arrays.copyOf(context, context.length + 1);
            seq[context.length] = QUERY + key;
            float[] logits = m.lastLogits(seq);
            int best = ANSWER;
            for (int v = ANSWER; v < VOCAB; v++) {
                if (logits[v] > logits[best]) {
                    best = v;
                }
            }
            ok += best == answer(corpus, key) ? 1 : 0;
        }
        return (double) ok / FACTS;
    }

    /**
     * Runs the comparison.
     *
     * @param budgets cartridge and eviction sizes in cached positions
     * @param readerSteps training steps of the in-context reader
     * @param cartridgeSteps distillation (and prefix-tuning) steps
     * @param seed the seed
     * @return arm name to one cell per budget (the full context repeats its single cell)
     */
    public static Map<String, Cell[]> run(int[] budgets, int readerSteps, int cartridgeSteps, long seed) {
        HookedDecoder model = reader(readerSteps, seed);
        int[] corpus = corpus(new SplittableRandom(seed + 7));
        Map<String, Cell[]> out = new LinkedHashMap<>();
        String[] names = {"full context", "recency eviction", "cartridge (self-study distillation)",
            "prefix tuning (next-token)"};
        for (String n : names) {
            out.put(n, new Cell[budgets.length]);
        }
        Cell full = new Cell(accuracy(model, corpus, corpus), CartridgeTrainer.kvBytes(model, FACTS));
        SplittableRandom r = new SplittableRandom(seed + 9);
        for (int bi = 0; bi < budgets.length; bi++) {
            int p = budgets[bi];
            long bytes = CartridgeTrainer.kvBytes(model, p);
            out.get("full context")[bi] = full;
            out.get("recency eviction")[bi] = new Cell(
                    accuracy(model, Arrays.copyOfRange(corpus, FACTS - p, FACTS), corpus), bytes);

            CartridgeTrainer cartridge = CartridgeTrainer.of(model, corpus, p);
            Optimizer co = Optimizer.adam(1e-2f);
            for (int s = 0; s < cartridgeSteps; s++) {
                cartridge.step(selfStudy(r), co);
            }
            out.get("cartridge (self-study distillation)")[bi] = new Cell(
                    accuracy(cartridge.student(), new int[0], corpus), cartridge.kvBytes());

            HookedDecoder tuned = model.copy();
            List<Var> prefix = SoftPrompts.prefixTuning(tuned, p, seed + bi);
            Optimizer po = Optimizer.adam(1e-2f);
            for (int s = 0; s < cartridgeSteps; s++) {
                List<int[]> qs = selfStudy(r);
                int[][] seqs = qs.toArray(int[][]::new);
                int[][] targets = new int[seqs.length][1];
                for (int i = 0; i < seqs.length; i++) {
                    targets[i][0] = answer(corpus, seqs[i][0] - QUERY);
                }
                Var loss = HookedDecoder.loss(tuned.forward(seqs).logits(), targets);
                loss.backward();
                po.step(prefix);
                tuned.zeroGrad();
            }
            out.get("prefix tuning (next-token)")[bi] = new Cell(accuracy(tuned, new int[0], corpus), bytes);
        }
        return out;
    }

    /** Self-study questions: a batch of the corpus's keys, as the model's own quiz. */
    private static List<int[]> selfStudy(SplittableRandom r) {
        List<int[]> qs = new ArrayList<>();
        for (int i = 0; i < 16; i++) {
            qs.add(new int[] {QUERY + r.nextInt(FACTS)});
        }
        return qs;
    }
}
