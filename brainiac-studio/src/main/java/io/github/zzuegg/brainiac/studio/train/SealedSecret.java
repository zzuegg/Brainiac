package io.github.zzuegg.brainiac.studio.train;

import java.nio.charset.StandardCharsets;
import java.security.GeneralSecurityException;
import java.security.KeyFactory;
import java.security.KeyPair;
import java.security.KeyPairGenerator;
import java.security.MessageDigest;
import java.security.PrivateKey;
import java.security.SecureRandom;
import java.security.spec.X509EncodedKeySpec;
import java.util.Arrays;
import java.util.Base64;
import javax.crypto.Cipher;
import javax.crypto.KeyAgreement;
import javax.crypto.spec.GCMParameterSpec;
import javax.crypto.spec.SecretKeySpec;

/**
 * How the studio hands a secret, such as its Hugging Face token, to a worker so that nobody who
 * only listens to the connection can read it, also when the worker is reached over plain HTTP.
 * Both sides make a one-time X25519 key pair and agree on a key from it and the worker's token,
 * which only they know; the secret travels sealed with AES-256-GCM under that key, bound to its
 * name. A machine in the middle that does not know the worker's token cannot take the place of
 * either side.
 */
public final class SealedSecret {

    private static final byte[] LABEL = "brainiac-worker-secret-v1".getBytes(StandardCharsets.UTF_8);
    private static final int IV = 12;

    private SealedSecret() {
    }

    /**
     * Makes a one-time key pair.
     *
     * @return the pair
     */
    public static KeyPair pair() {
        try {
            return KeyPairGenerator.getInstance("X25519").generateKeyPair();
        } catch (GeneralSecurityException e) {
            throw new IllegalStateException("this Java has no X25519", e);
        }
    }

    /**
     * Writes a pair's public half as text.
     *
     * @param pair the pair
     * @return its public key, Base64
     */
    public static String publicText(KeyPair pair) {
        return Base64.getEncoder().encodeToString(pair.getPublic().getEncoded());
    }

    /**
     * Agrees on the key both sides seal with.
     *
     * @param mine this side's private key
     * @param theirs the other side's public key, as {@link #publicText} wrote it
     * @param workerToken the worker's token, which both sides know
     * @return the key
     * @throws GeneralSecurityException if the other side's key is not a key
     */
    public static byte[] key(PrivateKey mine, String theirs, String workerToken) throws GeneralSecurityException {
        byte[] encoded;
        try {
            encoded = Base64.getDecoder().decode(theirs);
        } catch (IllegalArgumentException e) {
            throw new GeneralSecurityException("not a public key");
        }
        KeyAgreement agreement = KeyAgreement.getInstance("X25519");
        agreement.init(mine);
        agreement.doPhase(KeyFactory.getInstance("X25519").generatePublic(new X509EncodedKeySpec(encoded)), true);
        byte[] shared = agreement.generateSecret();
        MessageDigest md = MessageDigest.getInstance("SHA-256");
        md.update(LABEL);
        md.update(shared);
        md.update((byte) 0);
        md.update(workerToken.getBytes(StandardCharsets.UTF_8));
        Arrays.fill(shared, (byte) 0);
        return md.digest();
    }

    /**
     * Seals a secret.
     *
     * @param key the agreed key
     * @param name what the secret is, which the seal is bound to
     * @param value the secret; empty to take one back
     * @return the sealed secret, Base64
     * @throws GeneralSecurityException if it cannot be sealed
     */
    public static String seal(byte[] key, String name, String value) throws GeneralSecurityException {
        byte[] iv = new byte[IV];
        new SecureRandom().nextBytes(iv);
        Cipher c = Cipher.getInstance("AES/GCM/NoPadding");
        c.init(Cipher.ENCRYPT_MODE, new SecretKeySpec(key, "AES"), new GCMParameterSpec(128, iv));
        c.updateAAD(name.getBytes(StandardCharsets.UTF_8));
        byte[] sealed = c.doFinal(value.getBytes(StandardCharsets.UTF_8));
        byte[] out = Arrays.copyOf(iv, IV + sealed.length);
        System.arraycopy(sealed, 0, out, IV, sealed.length);
        return Base64.getEncoder().encodeToString(out);
    }

    /**
     * Opens a sealed secret.
     *
     * @param key the agreed key
     * @param name what the secret is said to be
     * @param sealed what {@link #seal} gave
     * @return the secret
     * @throws GeneralSecurityException if it was sealed with another key or for another name, or changed on the way
     */
    public static String open(byte[] key, String name, String sealed) throws GeneralSecurityException {
        byte[] in;
        try {
            in = Base64.getDecoder().decode(sealed);
        } catch (IllegalArgumentException e) {
            throw new GeneralSecurityException("not a sealed secret");
        }
        if (in.length < IV + 16) {
            throw new GeneralSecurityException("not a sealed secret");
        }
        Cipher c = Cipher.getInstance("AES/GCM/NoPadding");
        c.init(Cipher.DECRYPT_MODE, new SecretKeySpec(key, "AES"), new GCMParameterSpec(128, in, 0, IV));
        c.updateAAD(name.getBytes(StandardCharsets.UTF_8));
        return new String(c.doFinal(in, IV, in.length - IV), StandardCharsets.UTF_8);
    }
}
