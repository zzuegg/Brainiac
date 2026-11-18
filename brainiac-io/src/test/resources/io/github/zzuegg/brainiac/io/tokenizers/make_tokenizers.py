import json, os, glob, sys, shutil
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, normalizers, processors, trainers, Regex
import sentencepiece as spm
out = sys.argv[1]
corpus = []
for f in sorted(glob.glob('/usr/share/common-licenses/*'))[:8]:
    try: corpus.append(open(f, encoding='utf-8', errors='ignore').read())
    except: pass
extra = ["Grüße aus München! Ça va? Привет мир. 你好，世界。 こんにちは 🙂🚀 naïve café",
         "def f(x):\n    return x**2  # comment\n\n\tTabs and   spaces", "12345 67 8 3.14159 1,000,000",
         "The quick brown fox jumps over the lazy dog. I'm sure you've seen it; they'll say we'd won."]
corpus += extra * 20
open('corpus.txt','w').write("\n".join(corpus))
tests = ["Hello world", " leading space and  double  spaces", "Grüße aus München! 🙂🚀",
         "I'm sure you've seen it; they'll say we'd won.", "def f(x):\n    return x**2\n\n",
         "12345 67 8 3.14159", "你好，世界。 こんにちは", "<|im_start|>user\nHi there<|im_end|>\n",
         "unseen ✓ symbols ⚙ and  nbsp", "", "   ", "tail spaces   "]
LLAMA3 = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
def save(name, tok, add_special=True):
    tok.save(f"{out}/{name}.json")
    cases = []
    for t in tests:
        enc = tok.encode(t, add_special_tokens=add_special)
        cases.append({"text": t, "ids": enc.ids, "decoded": tok.decode(enc.ids, skip_special_tokens=False),
                      "decodedSkip": tok.decode(enc.ids, skip_special_tokens=True)})
    json.dump(cases, open(f"{out}/{name}.expected.json", "w"), ensure_ascii=False, indent=0)

# 1 Llama-3 / Qwen2 style byte-level BPE
t = Tokenizer(models.BPE(ignore_merges=True))
t.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(LLAMA3), behavior="isolated"),
                                           pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
t.decoder = decoders.ByteLevel()
t.normalizer = normalizers.NFC()
t.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=600, special_tokens=specials,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
t.post_processor = processors.Sequence([processors.ByteLevel(trim_offsets=False),
    processors.TemplateProcessing(single="<|endoftext|> $A", pair="<|endoftext|> $A <|endoftext|> $B",
        special_tokens=[("<|endoftext|>", t.token_to_id("<|endoftext|>"))])])
save("bytelevel-bpe", t)
# 2 GPT-2 style: ByteLevel with its own regex and a prefix space
t = Tokenizer(models.BPE())
t.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True, use_regex=True)
t.decoder = decoders.ByteLevel()
t.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=500, special_tokens=["<|endoftext|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
save("gpt2-bpe", t)
# 3 Llama-2 / Mistral style: metaspace BPE with byte fallback
t = Tokenizer(models.BPE(byte_fallback=True, fuse_unk=True, unk_token="<unk>"))
t.normalizer = normalizers.Sequence([normalizers.Prepend("▁"), normalizers.Replace(" ", "▁")])
t.decoder = decoders.Sequence([decoders.Replace("▁", " "), decoders.ByteFallback(), decoders.Fuse(), decoders.Strip(" ", 1, 0)])
t.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=700, special_tokens=["<unk>", "<s>", "</s>"]
        + [f"<0x{i:02X}>" for i in range(256)], show_progress=False))
t.post_processor = processors.TemplateProcessing(single="<s> $A", pair="<s> $A <s> $B",
        special_tokens=[("<s>", t.token_to_id("<s>"))])
# byte tokens belong to the vocabulary, not to the added tokens, as in Llama 2's tokenizer.json
d = json.loads(t.to_str())
d["added_tokens"] = [a for a in d["added_tokens"] if not a["content"].startswith("<0x")]
t = Tokenizer.from_str(json.dumps(d))
save("metaspace-bpe", t)
# 4 T5 style unigram
t = Tokenizer(models.Unigram())
t.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Replace(Regex(" {2,}"), " ")])
t.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")
t.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")
t.train_from_iterator(corpus, trainers.UnigramTrainer(vocab_size=400, special_tokens=["<pad>", "</s>", "<unk>"],
        unk_token="<unk>", show_progress=False))
t.post_processor = processors.TemplateProcessing(single="$A </s>", pair="$A </s> $B </s>",
        special_tokens=[("</s>", t.token_to_id("</s>"))])
save("unigram", t)
# 5 BERT style wordpiece
t = Tokenizer(models.WordPiece(unk_token="[UNK]"))
t.normalizer = normalizers.BertNormalizer(lowercase=True, strip_accents=True, handle_chinese_chars=True)
t.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
t.decoder = decoders.WordPiece()
t.train_from_iterator(corpus, trainers.WordPieceTrainer(vocab_size=500,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"], show_progress=False))
t.post_processor = processors.BertProcessing(("[SEP]", t.token_to_id("[SEP]")), ("[CLS]", t.token_to_id("[CLS]")))
save("wordpiece", t)

# SentencePiece models
for kind, size in (("unigram", 400), ("bpe", 500)):
    spm.SentencePieceTrainer.train(input="corpus.txt", model_prefix=f"{out}/sp-{kind}", vocab_size=size,
        model_type=kind, byte_fallback=True, character_coverage=0.9995, minloglevel=2,
        user_defined_symbols=["<|im_start|>", "<|im_end|>"], normalization_rule_name="identity")
    os.remove(f"{out}/sp-{kind}.vocab")
    sp = spm.SentencePieceProcessor(model_file=f"{out}/sp-{kind}.model")
    cases = [{"text": x, "ids": sp.encode(x), "decoded": sp.decode(sp.encode(x))} for x in tests]
    json.dump(cases, open(f"{out}/sp-{kind}.expected.json", "w"), ensure_ascii=False, indent=0)

# 6b T5 style SentencePiece unigram with nmt_nfkc: full-width commas and letters normalize to
# plain ones before lookup (Wan's Chinese negative prompt is full of "，")
nfkc = ["色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量",
        "a red fox runs through deep snow, quickly.", "Full width text ABC 123, ok", "画面 静止 背景, 低质量."]
open("nfkc.txt", "w").write("\n".join(nfkc * 50))
spm.SentencePieceTrainer.train(input="nfkc.txt", model_prefix=f"{out}/sp-nfkc", vocab_size=120, model_type="unigram",
    normalization_rule_name="nmt_nfkc", character_coverage=1.0, bos_id=-1, eos_id=1, unk_id=2, pad_id=0,
    hard_vocab_limit=False, minloglevel=2)
os.remove(f"{out}/sp-nfkc.vocab")
sp = spm.SentencePieceProcessor(model_file=f"{out}/sp-nfkc.model")
nfkc_tests = [nfkc[0], "画面，静止，背景，低质量", "Ｆｕｌｌ ｗｉｄｔｈ ＡＢＣ １２３，ｏｋ", "a red fox runs through deep snow"]
json.dump([{"text": x, "ids": sp.encode(x)} for x in nfkc_tests], open(f"{out}/sp-nfkc.expected.json", "w"),
          ensure_ascii=False, indent=0)

# GGUF vocabularies, written the way llama.cpp's converter writes them
import gguf
w = gguf.GGUFWriter(f"{out}/vocab.gguf", arch="llama")
d = json.load(open(f"{out}/bytelevel-bpe.json"))
vocab = d["model"]["vocab"]
toks = [None] * (max(vocab.values()) + 1)
for k, v in vocab.items(): toks[v] = k
types = [1] * len(toks)
for a in d["added_tokens"]:
    toks[a["id"]] = a["content"]; types[a["id"]] = 3
w.add_string("tokenizer.ggml.model", "gpt2"); w.add_string("tokenizer.ggml.pre", "llama-bpe")
w.add_array("tokenizer.ggml.tokens", toks); w.add_array("tokenizer.ggml.token_type", types)
w.add_array("tokenizer.ggml.merges", [" ".join(m) for m in d["model"]["merges"]])
w.add_uint32("tokenizer.ggml.bos_token_id", 0); w.add_bool("tokenizer.ggml.add_bos_token", True)
w.add_string("tokenizer.chat_template", "{% for m in messages %}{{ m['content'] }}{% endfor %}")
w.write_header_to_file(); w.write_kv_data_to_file(); w.close()
sp = spm.SentencePieceProcessor(model_file=f"{out}/sp-bpe.model")
w = gguf.GGUFWriter(f"{out}/vocab-llama.gguf", arch="llama")
n = sp.get_piece_size()
w.add_string("tokenizer.ggml.model", "llama")
w.add_array("tokenizer.ggml.tokens", [sp.id_to_piece(i) for i in range(n)])
w.add_array("tokenizer.ggml.scores", [sp.get_score(i) for i in range(n)])
def ty(i):
    if sp.is_unknown(i): return 2
    if sp.is_control(i): return 3
    if sp.is_byte(i): return 6
    if sp.is_unused(i): return 5
    return 4 if sp.id_to_piece(i).startswith("<|") else 1
w.add_array("tokenizer.ggml.token_type", [ty(i) for i in range(n)])
w.add_uint32("tokenizer.ggml.bos_token_id", sp.bos_id()); w.add_uint32("tokenizer.ggml.eos_token_id", sp.eos_id())
w.add_uint32("tokenizer.ggml.unknown_token_id", sp.unk_id()); w.add_bool("tokenizer.ggml.add_bos_token", True)
w.write_header_to_file(); w.write_kv_data_to_file(); w.close()

# 7 a Llama tokenizer as transformers loads it: tokenizer_class LlamaTokenizer rebuilds the
# pipeline (Metaspace pre-tokenizer, "first" without legacy, so no "▁" after special tokens)
from transformers import AutoTokenizer
llama_texts = ["hello world</s>\nthe end", "<s>hi there", "The quick brown fox.</s></s> jumps", " leading space",
               "line\n\nbreaks</s>x", ""]
for legacy in (False, True):
    d = f"{out}/llama-legacy-{str(legacy).lower()}"
    os.makedirs(d, exist_ok=True)
    shutil.copy(f"{out}/metaspace-bpe.json", f"{d}/tokenizer.json")
    json.dump({"tokenizer_class": "LlamaTokenizer", "legacy": legacy, "add_bos_token": True, "add_eos_token": False,
               "bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>"}, open(f"{d}/tokenizer_config.json", "w"))
    tok = AutoTokenizer.from_pretrained(d)
    json.dump([{"text": s, "ids": tok(s)["input_ids"]} for s in llama_texts], open(f"{d}/expected.json", "w"))
