package io.github.zzuegg.brainiac.experiments.reasoning;

import io.github.zzuegg.brainiac.data.CharacterTokenizer;
import io.github.zzuegg.brainiac.experimental.llm.ToolRuntime;
import io.github.zzuegg.brainiac.experimental.llm.ToolformerFilter;
import io.github.zzuegg.brainiac.experimental.llm.WindowLm;
import io.github.zzuegg.brainiac.train.Optimizer;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Experiment X24, a tool-using tiny model. A {@link WindowLm} is pretrained on addition and
 * subtraction written with calculator calls ({@code 3+4=[3+4>7]7#}, so it learns to read a
 * tool result), but has never seen a product. The baseline is that model fine-tuned on plain
 * multiplication for a training split of the products. Toolformer instead proposes calls for the multiplication data ({@code [a*b>}, but also wrong ones like
 * {@code [a+b>}), executes them, and keeps only the calls that lower the model's loss on the
 * true answer. Fine-tuned on the kept calls, the model learns to call the calculator for
 * products, and with {@link ToolRuntime} splicing results in it answers held-out products it
 * never saw, which the plain model cannot.
 */
public final class ToolUse {

    /**
     * The outcome.
     *
     * @param candidates candidate calls proposed
     * @param kept calls the filter kept
     * @param keptCorrect kept calls that were the right operation
     * @param rightCallsKept fraction of right-operation candidates that were kept
     * @param plainAccuracy held-out accuracy of the model without tools
     * @param toolAccuracy held-out accuracy of the fine-tuned model with the calculator
     */
    public record Result(int candidates, int kept, int keptCorrect, double rightCallsKept, double plainAccuracy,
            double toolAccuracy) {
    }

    private ToolUse() {
    }

    /**
     * @param pretrainSteps steps of base training
     * @param finetuneSteps steps of tool fine-tuning
     * @param seed the seed
     * @return the outcome
     */
    public static Result run(int pretrainSteps, int finetuneSteps, long seed) {
        CharacterTokenizer tok = new CharacterTokenizer(ArithmeticTask.CHARACTERS);
        int eos = tok.encode("#")[0];
        ToolRuntime runtime = ToolRuntime.of(tok, ToolRuntime.Tool.CALCULATOR, '[', '>', ']');
        SplittableRandom random = new SplittableRandom(seed);

        List<int[]> products = new ArrayList<>();
        for (int a = 0; a <= 9; a++) {
            for (int b = 0; b <= 9; b++) {
                products.add(new int[] {a, b});
            }
        }
        for (int i = products.size() - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int[] t = products.get(i);
            products.set(i, products.get(j));
            products.set(j, t);
        }
        List<int[]> train = products.subList(0, 60);
        List<int[]> test = products.subList(60, products.size());

        // pretraining corpus
        List<int[]> ps = new ArrayList<>();
        List<int[]> cs = new ArrayList<>();
        for (int a = 0; a <= 9; a++) {
            for (int b = 0; b <= 9; b++) {
                for (char op : new char[] {'+', '-'}) {
                    String e = a + String.valueOf(op) + b;
                    String r = ToolRuntime.calculate(e);
                    ps.add(tok.encode(e + "="));
                    cs.add(tok.encode("[" + e + ">" + r + "]" + r + "#"));
                }
            }
        }
        WindowLm base = WindowLm.of(tok.size(), 12, 96, seed);
        Optimizer opt = Optimizer.adam(3e-3f);
        for (int s = 0; s < pretrainSteps; s++) {
            minibatch(base, ps, cs, opt, random);
        }
        // the no-tool baseline: the same model taught the training products directly
        WindowLm plainModel = base.copy();
        List<int[]> pp = new ArrayList<>(ps);
        List<int[]> pc = new ArrayList<>(cs);
        for (int[] ab : train) {
            pp.add(tok.encode(ab[0] + "*" + ab[1] + "="));
            pc.add(tok.encode(ab[0] * ab[1] + "#"));
        }
        Optimizer opt1 = Optimizer.adam(3e-3f);
        for (int s = 0; s < finetuneSteps; s++) {
            minibatch(plainModel, pp, pc, opt1, random);
        }

        // Toolformer: propose, execute, filter
        List<ToolformerFilter.Candidate> candidates = new ArrayList<>();
        for (int[] ab : train) {
            for (char op : new char[] {'*', '+', '-'}) {
                String e = ab[0] + String.valueOf(op) + ab[1];
                candidates.add(new ToolformerFilter.Candidate(tok.encode(ab[0] + "*" + ab[1] + "="),
                        tok.encode("[" + e + ">"), tok.encode(ToolRuntime.calculate(e) + "]"), tok.encode("]"),
                        tok.encode(ab[0] * ab[1] + "#")));
            }
        }
        List<ToolformerFilter.Candidate> kept = ToolformerFilter.of(base, 1.0).filterBestPerPlace(candidates);
        int keptCorrect = 0;
        for (ToolformerFilter.Candidate c : kept) {
            if (tok.decode(c.call()).contains("*")) {
                keptCorrect++;
            }
        }

        // fine-tune a copy on the kept calls (never on tool output), plus the old corpus
        WindowLm tool = base.copy();
        List<int[]> fp = new ArrayList<>();
        List<int[]> fc = new ArrayList<>();
        for (ToolformerFilter.Candidate c : kept) {
            for (int[][] chunk : runtime.trainingChunks(c.prompt(), c.augmented())) {
                fp.add(chunk[0]);
                fc.add(chunk[1]);
            }
        }
        for (int i = 0; i < ps.size(); i++) {
            if (tok.decode(cs.get(i)).startsWith("[")) {
                for (int[][] chunk : runtime.trainingChunks(ps.get(i), cs.get(i))) {
                    fp.add(chunk[0]);
                    fc.add(chunk[1]);
                }
            }
        }
        Optimizer opt2 = Optimizer.adam(3e-3f);
        for (int s = 0; s < finetuneSteps && !fp.isEmpty(); s++) {
            minibatch(tool, fp, fc, opt2, random);
        }

        int plain = 0;
        int withTool = 0;
        for (int[] ab : test) {
            int[] prompt = tok.encode(ab[0] + "*" + ab[1] + "=");
            String want = ab[0] * ab[1] + "#";
            String direct = tok.decode(plainModel.greedy(prompt, 4, eos));
            plain += direct.equals(want) ? 1 : 0;
            String text = runtime.generate(tool, prompt, 16, eos, 0, random).text();
            withTool += text.endsWith("]" + want) ? 1 : 0;
        }
        return new Result(candidates.size(), kept.size(), keptCorrect, (double) keptCorrect / train.size(),
                (double) plain / test.size(), (double) withTool / test.size());
    }

    static void minibatch(WindowLm lm, List<int[]> ps, List<int[]> cs, Optimizer opt, SplittableRandom random) {
        List<int[]> bp = new ArrayList<>();
        List<int[]> bc = new ArrayList<>();
        for (int i = 0; i < 48; i++) {
            int k = random.nextInt(ps.size());
            bp.add(ps.get(k));
            bc.add(cs.get(k));
        }
        lm.supervisedStep(bp, bc, opt);
    }
}
