# Code-switching and KG-RAG

Does code-switching break knowledge-graph RAG, and if so, where?

In a normal text RAG system, language effects are spread across the whole
pipeline, so when a code-switched question fails it's hard to say why. A KG-RAG
pipeline has a useful property: once the question's entity is linked to a
Wikidata ID (a QID), everything after that is language-independent. "diabetes",
"मधुमेह" and "madhumeh" all point to the same QID. So if we hold the target QID
fixed and only change how the question is written, we can see exactly which
stage breaks.

## Method

**Data.** Questions come from Mintaka, which has English questions, Hindi
translations, gold Wikidata QIDs and a marked entity mention. I locate the
mention in the Hindi question using the entity's Hindi label and aliases, and
drop QIDs with broken labels (wrong script, vandalism, missing labels).

**Query variants.** Each question is rewritten into eight versions that change
the entity and the surrounding frame separately:

| variant | entity | frame |
|---|---|---|
| `en_latin` | English | English |
| `hi_deva` | Hindi, Devanagari | Hindi, Devanagari |
| `mix_entity_latin` | Hindi, romanised | Hindi, Devanagari |
| `mix_frame_latin` | Hindi, Devanagari | Hindi, romanised |
| `hi_latin` | Hindi, romanised | Hindi, romanised |
| `hi_latin_r` | Hindi, romanised | romanised, English loanwords restored |
| `cs_mixed` | English | Hindi, Devanagari |
| `cs_latin` | English | romanised, loanwords restored |

Mintaka has no romanised Hindi, so I generate it with a rule-based
transliterator (ITRANS plus fixes for nasals, schwa deletion and loanword
vowels), pinned with regression tests. IndicXlit can be swapped in instead.

**Phonetic correspondence.** Entities are grouped by how much the Hindi name
sounds like the English one:

- `CORRESPOND`: transliterated names (Jake Gyllenhaal)
- `HYBRID`: part transliterated, part translated (Mississippi River → misisippi nadi)
- `ALIAS_ONLY`: only a Wikidata alias matches (भारत / इंडिया for India)
- `NO_CORRESPOND`: translated names (दूरभाष for telephone)

This is computed automatically from the label pair, with one threshold
(`CORR_T`) that is swept to check stability.

**Entity linking.** Two matchers rank candidate QIDs:
- lexical: fuzzy string matching with a phonetic fallback
- dense: `multilingual-e5-base`, run on the mention and on the whole question

Each runs against three indexes: English labels only, plus Wikidata's
human-written Hindi aliases, and plus the Hindi label itself (an oracle
ceiling).

**Retrieval and failure localisation.** The linked entity's Wikidata facts are
pulled and written out in English. A question counts as answerable if the gold
answer appears in those facts. To find where things fail, I swap in the gold
QID and rerun retrieval: if performance comes back, linking was the problem.

**Mechanism checks.** Subword fertility (tokens per word) and cross-lingual
alignment (cosine to the English question) test why the dense matcher suffers.
Japanese serves as a control, since its romanisation is standardised.

## Main findings

- Almost all of the loss happens at entity linking. With the gold QID, every
  variant reaches the same retrieval ceiling.
- Romanisation helps lexical matching but hurts dense matching on the same items.
- How much romanisation helps depends on phonetic correspondence. It helps a lot
  for transliterated names and barely at all for translated ones.
- In the full-question setting, romanising the frame hurts more than romanising
  the entity.
- Hindi aliases recover part of the loss, but only about 12% of entities have one.
- The dense penalty tracks weaker cross-lingual alignment, not tokenization.

## Running it

Open `kg_rag_codeswitching_1.2.ipynb` in Colab and run the cells in order.
Wikidata calls are cached, so reruns are fast. A GPU is only needed for the
dense linking part; set `RUN_DENSE=False` to skip it. To use IndicXlit instead
of the rule-based romaniser, set `ROMANIZER_BACKEND = "indicxlit"`.
