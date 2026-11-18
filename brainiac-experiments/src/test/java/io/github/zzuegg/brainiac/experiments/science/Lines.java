package io.github.zzuegg.brainiac.experiments.science;

import java.util.Map;

/** Formats result tables on one line for the tests' printouts. */
final class Lines {

    private Lines() {
    }

    static String of(Map<String, double[]> r) {
        StringBuilder b = new StringBuilder();
        r.forEach((k, v) -> {
            b.append(k).append(" {");
            for (int i = 0; i < v.length; i++) {
                b.append(String.format(i == 0 ? "%.4g" : ", %.4g", v[i]));
            }
            b.append("} | ");
        });
        return b.toString();
    }
}
