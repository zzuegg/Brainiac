package io.github.zzuegg.brainiac.llm;

import io.github.zzuegg.brainiac.io.Json;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.function.Function;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Benchmarks for language models, scored as EleutherAI's lm-evaluation-harness scores them:
 * multiple-choice tasks (MMLU, ARC, HellaSwag) by the log-likelihood of each answer after
 * the question, and generative tasks (GSM8K) by an answer extracted from greedy output and
 * compared exactly. Questions take optional few-shot examples in front.
 *
 * <pre>
 * List&lt;Benchmark.Example&gt; arc = Benchmark.load(Path.of("arc_challenge_test.jsonl"), Benchmark::arc);
 * Benchmark.Result r = Benchmark.multipleChoice(generator, arc, arcTrain.subList(0, 5));
 * System.out.println(r);   // accuracy and length-normalized accuracy, with standard errors
 * </pre>
 *
 * <p>Examples load from JSON lines with one of the readers for the common dataset layouts
 * ({@link #mmlu}, {@link #arc}, {@link #hellaswag}, {@link #gsm8k}) or a function of one's own.
 */
public final class Benchmark {

    private Benchmark() {
    }

    /**
     * One question.
     *
     * @param question the text before the answer, with any cue such as {@code "\nAnswer:"}
     * @param choices the answers to pick from, or empty for a generative task
     * @param answer the index of the right choice, or -1 for a generative task
     * @param target the right answer's text: the choice, or what a generative answer must extract to
     * @param solution a worked answer to show in few-shot examples, or the target if there is none
     */
    public record Example(String question, List<String> choices, int answer, String target, String solution) {

        /** A multiple-choice question. */
        public static Example choice(String question, List<String> choices, int answer) {
            return new Example(question, List.copyOf(choices), answer, choices.get(answer), choices.get(answer));
        }

        /** A question answered in free text, with an optional worked solution for few-shot use. */
        public static Example open(String question, String target, String solution) {
            return new Example(question, List.of(), -1, target, solution == null ? target : solution);
        }
    }

    /**
     * Scores.
     *
     * @param correct examples answered right
     * @param normalizedCorrect for multiple choice, examples right when each answer's
     *     log-likelihood is divided by its length in bytes ({@code acc_norm}); else {@code correct}
     * @param total examples
     * @param predictions the chosen index or extracted answer per example
     */
    public record Result(int correct, int normalizedCorrect, int total, List<String> predictions) {

        /** @return the fraction right */
        public double accuracy() {
            return total == 0 ? 0 : (double) correct / total;
        }

        /** @return the fraction right with length-normalized scores */
        public double normalizedAccuracy() {
            return total == 0 ? 0 : (double) normalizedCorrect / total;
        }

        /** @return the standard error of {@link #accuracy}, as lm-eval reports it */
        public double stderr() {
            return stderr(accuracy());
        }

        /** @return the standard error of {@link #normalizedAccuracy} */
        public double normalizedStderr() {
            return stderr(normalizedAccuracy());
        }

        private double stderr(double p) {
            return total < 2 ? 0 : Math.sqrt(p * (1 - p) / (total - 1));
        }

        @Override
        public String toString() {
            return String.format(Locale.ROOT, "acc %.4f ± %.4f, acc_norm %.4f ± %.4f (%d examples)", accuracy(), stderr(),
                    normalizedAccuracy(), normalizedStderr(), total);
        }
    }

    // ---- running

    /**
     * Scores multiple-choice questions by the log-likelihood of {@code " " + choice} after
     * the few-shot examples and the question.
     *
     * @param generator the model with its tokenizer
     * @param examples the questions
     * @param shots solved examples shown before each question, possibly empty
     * @return the scores; predictions are the chosen indices
     */
    public static Result multipleChoice(Generator generator, List<Example> examples, List<Example> shots) {
        String prefix = fewShot(shots);
        int correct = 0;
        int normalized = 0;
        List<String> predictions = new ArrayList<>();
        for (Example e : examples) {
            if (e.choices.isEmpty()) {
                throw new IllegalArgumentException("not a multiple-choice question: " + e.question);
            }
            String context = prefix + prompt(e);
            int best = -1;
            int bestNorm = -1;
            double bestScore = Double.NEGATIVE_INFINITY;
            double bestNormScore = Double.NEGATIVE_INFINITY;
            for (int c = 0; c < e.choices.size(); c++) {
                String cont = " " + e.choices.get(c);
                double ll = logLikelihood(generator, context, cont);
                double norm = ll / Math.max(1, cont.getBytes(StandardCharsets.UTF_8).length);
                if (ll > bestScore) {
                    bestScore = ll;
                    best = c;
                }
                if (norm > bestNormScore) {
                    bestNormScore = norm;
                    bestNorm = c;
                }
            }
            correct += best == e.answer ? 1 : 0;
            normalized += bestNorm == e.answer ? 1 : 0;
            predictions.add(Integer.toString(best));
        }
        return new Result(correct, normalized, examples.size(), List.copyOf(predictions));
    }

    /**
     * Scores free-text answers: greedy generation after the few-shot examples and the question,
     * stopped at a blank line or the next {@code "Question:"}, then {@code extract} applied and
     * compared with the target after {@link #normalize}.
     *
     * @param generator the model with its tokenizer
     * @param examples the questions
     * @param shots solved examples shown before each question, possibly empty
     * @param extract finds the answer in the output, such as {@link #lastNumber}
     * @param maxNewTokens tokens to generate per answer
     * @param batch questions decoded together
     * @return the scores; predictions are the extracted answers
     */
    public static Result generative(Generator generator, List<Example> examples, List<Example> shots,
            Function<String, String> extract, int maxNewTokens, int batch) {
        String prefix = fewShot(shots);
        Generator.Options options = Generator.Options.greedy(maxNewTokens).withStopStrings("\n\n", "Question:");
        int correct = 0;
        List<String> predictions = new ArrayList<>();
        for (int from = 0; from < examples.size(); from += batch) {
            List<Example> part = examples.subList(from, Math.min(examples.size(), from + batch));
            List<String> prompts = part.stream().map(e -> prefix + prompt(e)).toList();
            List<Generator.Output> outs = generator.generate(prompts, options, null);
            for (int i = 0; i < part.size(); i++) {
                String got = extract.apply(outs.get(i).text());
                predictions.add(got);
                if (got != null && normalize(got).equals(normalize(part.get(i).target))) {
                    correct++;
                }
            }
        }
        return new Result(correct, correct, examples.size(), List.copyOf(predictions));
    }

    private static String prompt(Example e) {
        return e.question;
    }

    private static String fewShot(List<Example> shots) {
        StringBuilder b = new StringBuilder();
        for (Example s : shots) {
            b.append(prompt(s)).append(' ').append(s.solution).append("\n\n");
        }
        return b.toString();
    }

    /**
     * {@code log p(continuation | context)}, tokenized as lm-eval does: the whole text at once,
     * the context's tokens being the longest prefix it shares with the context alone, so that a
     * token spanning the boundary counts towards the continuation.
     */
    static double logLikelihood(Generator generator, String context, String continuation) {
        int[] whole = generator.encode(context + continuation);
        int[] ctx = generator.encode(context);
        int n = 0;
        while (n < ctx.length && n < whole.length - 1 && ctx[n] == whole[n]) {
            n++;
        }
        n = Math.max(1, n);
        return Evaluation.logLikelihood(generator.model(), java.util.Arrays.copyOf(whole, n),
                java.util.Arrays.copyOfRange(whole, n, whole.length));
    }

    // ---- answers

    private static final Pattern NUMBER = Pattern.compile("-?[$0-9.,]{2,}|-?[0-9]+");
    private static final Pattern GSM8K = Pattern.compile("#### (-?[0-9.,]+)");

    /**
     * The answer after {@code "#### "}, as GSM8K writes it, else the last number in the text.
     *
     * @param text a generated answer
     * @return the number, or {@code null}
     */
    public static String lastNumber(String text) {
        Matcher m = GSM8K.matcher(text);
        if (m.find()) {
            return m.group(1);
        }
        String last = null;
        Matcher n = NUMBER.matcher(text);
        while (n.find()) {
            last = n.group();
        }
        return last;
    }

    /**
     * Lower case, trimmed, without {@code $}, thousands commas, a trailing period or {@code .0}s,
     * so {@code "$1,234.00."} equals {@code "1234"}.
     *
     * @param answer an answer
     * @return its normal form
     */
    public static String normalize(String answer) {
        String s = answer.strip().toLowerCase(Locale.ROOT).replace("$", "").replace(",", "");
        while (s.endsWith(".")) {
            s = s.substring(0, s.length() - 1);
        }
        if (s.matches("-?[0-9]+\\.0+")) {
            s = s.substring(0, s.indexOf('.'));
        }
        return s.strip();
    }

    // ---- datasets

    /**
     * Reads a dataset file: JSON lines, or Parquet (as Hugging Face publishes datasets) when
     * the name ends in {@code .parquet}.
     *
     * @param file one JSON object per line, or a Parquet file
     * @param reader turns an object into an example, such as {@link #mmlu}
     * @return the examples
     * @throws IOException if the file cannot be read
     */
    public static List<Example> load(Path file, Function<Map<String, Object>, Example> reader) throws IOException {
        List<Example> out = new ArrayList<>();
        if (file.getFileName().toString().endsWith(".parquet")) {
            for (Map<String, Object> row : io.github.zzuegg.brainiac.io.Parquet.read(file)) {
                out.add(reader.apply(row));
            }
            return out;
        }
        for (String line : Files.readAllLines(file, StandardCharsets.UTF_8)) {
            if (!line.isBlank()) {
                out.add(reader.apply(Json.object(Json.parse(line))));
            }
        }
        return out;
    }

    private static final List<String> LETTERS = List.of("A", "B", "C", "D", "E", "F", "G", "H");

    /**
     * MMLU's layout ({@code cais/mmlu}): {@code question}, {@code choices}, {@code answer} as an index.
     * The question lists the choices with letters, and the choices are the letters, as lm-eval's mmlu task.
     *
     * @param row a dataset row
     * @return the example
     */
    public static Example mmlu(Map<String, Object> row) {
        List<String> choices = Json.array(row.get("choices")).stream().map(String::valueOf).toList();
        StringBuilder q = new StringBuilder(Json.string(row, "question", "").strip());
        for (int i = 0; i < choices.size(); i++) {
            q.append('\n').append(LETTERS.get(i)).append(". ").append(choices.get(i));
        }
        return Example.choice(q + "\nAnswer:", LETTERS.subList(0, choices.size()), (int) Json.integer(row, "answer"));
    }

    /**
     * ARC's layout ({@code allenai/ai2_arc}): {@code question}, {@code choices.text},
     * {@code choices.label} and {@code answerKey}.
     *
     * @param row a dataset row
     * @return the example
     */
    public static Example arc(Map<String, Object> row) {
        Map<String, Object> c = Json.object(row.get("choices"));
        List<String> texts = Json.array(c.get("text")).stream().map(String::valueOf).toList();
        List<String> labels = Json.array(c.get("label")).stream().map(String::valueOf).toList();
        int answer = labels.indexOf(Json.string(row, "answerKey", ""));
        if (answer < 0) {
            throw new IllegalArgumentException("answerKey is not among the labels " + labels);
        }
        return Example.choice("Question: " + Json.string(row, "question", "") + "\nAnswer:", texts, answer);
    }

    /**
     * HellaSwag's layout ({@code Rowan/hellaswag}): {@code ctx}, {@code endings} and {@code label}.
     * Scored as lm-eval does, the context being the question and each ending a choice.
     *
     * @param row a dataset row
     * @return the example
     */
    public static Example hellaswag(Map<String, Object> row) {
        List<String> endings = Json.array(row.get("endings")).stream().map(String::valueOf).toList();
        Object label = row.get("label");
        int answer = label instanceof Number n ? n.intValue() : Integer.parseInt(String.valueOf(label));
        String activity = Json.string(row, "activity_label", "");
        String ctx = Json.string(row, "ctx", "");
        return Example.choice(activity.isEmpty() ? ctx : activity + ": " + ctx, endings, answer);
    }

    /**
     * GSM8K's layout ({@code openai/gsm8k}): {@code question}, and {@code answer} as a worked
     * solution ending in {@code "#### number"}.
     *
     * @param row a dataset row
     * @return the example
     */
    public static Example gsm8k(Map<String, Object> row) {
        String solution = Json.string(row, "answer", "");
        return Example.open("Question: " + Json.string(row, "question", "") + "\nAnswer:", lastNumber(solution),
                solution);
    }
}
