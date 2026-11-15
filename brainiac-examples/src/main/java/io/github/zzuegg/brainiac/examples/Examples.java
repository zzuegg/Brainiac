package io.github.zzuegg.brainiac.examples;

/**
 * Runs an example by name: {@code ./gradlew :brainiac-examples:run --args=iris}.
 */
public final class Examples {

    private Examples() {
    }

    /**
     * Runs the named example, or lists them.
     *
     * @param args {@code xor}, {@code spiral} or {@code iris}
     */
    public static void main(String[] args) {
        String name = args.length > 0 ? args[0] : "";
        switch (name) {
            case "xor" -> XorExample.main(new String[0]);
            case "spiral" -> SpiralExample.main(new String[0]);
            case "iris" -> IrisExample.main(new String[0]);
            case "sequence" -> SequenceExample.main(new String[0]);
            case "sentiment" -> SentimentExample.main(new String[0]);
            case "brain" -> BrainExample.main(new String[0]);
            case "rl" -> RlExample.main(java.util.Arrays.copyOfRange(args, 1, args.length));
            case "mnist" -> {
                try {
                    MnistExample.main(java.util.Arrays.copyOfRange(args, 1, args.length));
                } catch (java.io.IOException e) {
                    throw new java.io.UncheckedIOException(e);
                }
            }
            case "gpu" -> GpuExample.main(java.util.Arrays.copyOfRange(args, 1, args.length));
            case "bench" -> BenchmarkExample.main(java.util.Arrays.copyOfRange(args, 1, args.length));
            default -> System.out.println("""
                    Usage: ./gradlew :brainiac-examples:run --args=<example>
                      xor     learn XOR with a tiny network
                      spiral  classify three spiral arms
                      iris    classify iris flowers from CSV
                      sequence  forecast a wave with an LSTM, reverse digits with a transformer
                      sentiment  learn word embeddings to read made-up reviews
                      rl      reinforcement learning: cliff walk, cart-pole, pendulum
                      brain   a growing brain: regions, sparse rewiring, new neurons
                      mnist   recognize handwritten digits with a CNN (downloads MNIST)
                      gpu     list devices, compare CPU and GPU speed
                      bench   run any model on any task: bench <task> [model...] [--epochs N]""");
        }
    }
}
