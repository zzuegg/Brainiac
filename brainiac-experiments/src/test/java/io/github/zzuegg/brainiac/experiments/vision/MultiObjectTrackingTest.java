package io.github.zzuegg.brainiac.experiments.vision;

import static org.junit.jupiter.api.Assertions.assertTrue;

import io.github.zzuegg.brainiac.experimental.vision.TrackingByDetection;
import org.junit.jupiter.api.Test;

class MultiObjectTrackingTest {

    @Test
    void kalmanPredictionAndSecondRoundHelp() {
        VideoTracking.TrackScore sort = VideoTracking.track(TrackingByDetection.Mode.SORT, 20, 1);
        VideoTracking.TrackScore bytes = VideoTracking.track(TrackingByDetection.Mode.BYTE_TRACK, 20, 1);
        VideoTracking.TrackScore noKalman = VideoTracking.track(TrackingByDetection.Mode.SORT_NO_KALMAN, 20, 1);
        VideoTracking.TrackScore greedy = VideoTracking.track(TrackingByDetection.Mode.GREEDY_CENTROID, 20, 1);
        System.out.printf("tracking MOTA/IDF1/ID switches: SORT %.3f/%.3f/%d, ByteTrack %.3f/%.3f/%d, SORT without"
                + " Kalman %.3f/%.3f/%d, greedy centroid %.3f/%.3f/%d%n", sort.mota(), sort.idf1(), sort.idSwitches(),
                bytes.mota(), bytes.idf1(), bytes.idSwitches(), noKalman.mota(), noKalman.idf1(),
                noKalman.idSwitches(), greedy.mota(), greedy.idf1(), greedy.idSwitches());
        assertTrue(sort.idSwitches() < noKalman.idSwitches(), "Kalman prediction cuts ID switches");
        assertTrue(bytes.idf1() > sort.idf1(), "ByteTrack's second round lifts IDF1");
        assertTrue(sort.idf1() > greedy.idf1(), "SORT beats greedy centroid association");
    }
}
