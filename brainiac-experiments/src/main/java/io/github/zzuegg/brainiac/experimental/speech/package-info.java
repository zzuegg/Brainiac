/**
 * Speech analysis blocks, written as core code so they can later move to
 * {@code brainiac-audio}: speaker embeddings (x-vector TDNN with statistics pooling, additive
 * angular margin softmax, GE2E, cosine and PLDA scoring), voice activity detection,
 * single-channel speech enhancement and speaker diarisation (embedding clustering and
 * end-to-end neural diarisation).
 */
package io.github.zzuegg.brainiac.experimental.speech;
