package io.github.zzuegg.brainiac.experiments.synthesis;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experiments.synthesis.SymbolicRegressionBenchmark.Equation;
import io.github.zzuegg.brainiac.experiments.synthesis.SymbolicRegressionBenchmark.Learner;
import io.github.zzuegg.brainiac.experiments.synthesis.SymbolicRegressionBenchmark.Score;

import org.junit.jupiter.api.Test;

/** Item 7.6.9: GP with lexicase selection against tournament selection, an MLP and NALU. */
class SymbolicRegressionBenchmarkTest {

    @Test
    void lexicaseGpRecoversEquationsThatExtrapolate() {
        StringBuilder line = new StringBuilder("Symbolic regression recovered / extrapolation rel. error:");
        int lexicase = 0;
        int tournament = 0;
        double gpError = 0;
        double mlpError = 0;
        for (Equation eq : Equation.values()) {
            line.append(" | ").append(eq);
            for (Learner l : Learner.values()) {
                if (l == Learner.NALU && eq != Equation.PRODUCT) {
                    continue;
                }
                Score s = SymbolicRegressionBenchmark.run(eq, l, 30, 1);
                line.append(String.format(" %s %s/%.2g", l, s.recovered() ? "yes" : "no", s.extrapolationError()));
                lexicase += l == Learner.GP_LEXICASE && s.recovered() ? 1 : 0;
                tournament += l == Learner.GP_TOURNAMENT && s.recovered() ? 1 : 0;
                gpError += l == Learner.GP_LEXICASE ? Math.log10(s.extrapolationError() + 1e-12) : 0;
                mlpError += l == Learner.MLP ? Math.log10(s.extrapolationError() + 1e-12) : 0;
            }
        }
        System.out.println(line + String.format(" | recovered: lexicase %d, tournament %d of %d", lexicase,
                tournament, Equation.values().length));
        assertTrue(lexicase >= 2, "lexicase GP recovers at least two equations exactly");
        assertTrue(gpError < mlpError, "GP extrapolates better than the MLP (geometric mean error)");
    }
}
