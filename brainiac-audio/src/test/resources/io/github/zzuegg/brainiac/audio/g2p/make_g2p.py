# Reference phonemes from misaki's English G2P (spaCy en_core_web_sm tags, unknown words as ❓).
# Run: python make_g2p.py <this folder>
import json, sys
from misaki import en
sentences = [
    "Hello world! The quick brown fox jumps over the lazy dog.",
    "The weather is nice today. Let us go for a walk in the park.",
    "I can't believe it's already 2024; time flies, doesn't it?",
    "She sells seashells by the seashore.",
    "In 1984, George Orwell's novel was read by 3,500 students.",
    "It costs $5.50, or about £4 and 20 pence.",
    "Dr. Smith arrived at 10:30 on the 21st of March.",
    "An apple a day keeps the doctor away.",
    "The U.S. economy grew 3.5% last quarter.",
    "Please don't touch the well-known artist's paintings!",
    "We're going to the museum, aren't we?",
    "\"Wait,\" he said - \"what's that?\"",
    "NASA launched the rocket at 7 am.",
    "They walked, talked, and laughed for hours.",
    "My phone number is 555-0123.",
    "The children played happily in the gardens.",
    "Kokoro speaks with fifty-four different voices.",
    "Is this the real life? Is this just fantasy?",
    "Brainiac turns text into speech.",
    "The 2nd and 3rd runners finished in 1:02:03.",
]
out = {}
for name, british in (("us", False), ("gb", True)):
    g = en.G2P(trf=False, british=british, fallback=lambda t: ("❓", 1))
    out[name] = [g(s)[0] for s in sentences]
json.dump({"sentences": sentences, **out}, open(f"{sys.argv[1]}/expected.json", "w"), ensure_ascii=False, indent=1)
for s, u in zip(sentences, out["us"]):
    print(u)
