# Training data

ARC-1 was trained on public datasets turned into typed decision questions (choice, score, yes/no). Across all
training runs that went into the released checkpoint the model saw 3,810,947 examples from 737 sources.
The complete list, with row counts and licence tags, is in [data_sources.csv](data_sources.csv).

Most sources reached us through the [tasksource](https://github.com/sileod/tasksource) collection and the Hugging
Face Hub. A few kinds of rows were generated: procedurally built decision tasks (`procedural-typed-decisions`,
`synthetic:lookup`), and labels from earlier ARC-1 or open teacher models (`teacher:` prefix).

Some of the public datasets contain text written by language models, for example the responses in preference
datasets like HH-RLHF, PRM800K and HelpSteer. We never collected outputs from closed decision APIs (Jev, OpenAI
Decisions) for training. They were only used for evaluation.

## Licence tags

| tag | rows | share |
|---|---|---|
| commercial | 1,893,286 | 49.7% |
| unspecified | 1,655,175 | 43.4% |
| non-commercial | 262,486 | 6.9% |

`commercial` means the dataset's licence allows commercial use, `unspecified` means we couldn't find a clear
licence, and `non-commercial` means the licence only allows non-commercial use. Because of the last two groups the
model weights are released under CC BY-NC 4.0.

Non-commercial sources: AES2-essay-scoring, BeaverTails, ConTRoL-nli, IntentGrasp/all, PKU-SafeRLHF/helpfulness, PKU-SafeRLHF/safety, add_one_rte, anli/a1, anli/a2, anli/a3, conll2003/ner_tags, contract-nli/contractnli_a/seg, contract-nli/contractnli_b/full, dnc, ekar_english, financial_phrasebank/sentences_allagree, hf:facebook/anli/train_r1, hf:facebook/anli/train_r2, hf:facebook/anli/train_r3, implicit-hate-stg1, language-identification, lexical_relation_classification/BLESS, lexical_relation_classification/CogALexV, lexical_relation_classification/EVALution, lexical_relation_classification/K&H+N, lexical_relation_classification/ROOT09, logiqa-2.0-nli, multilingual/language-identification, multilingual/xlwic/xlwic_de_de, multilingual/xlwic/xlwic_en_ko, multilingual/xlwic/xlwic_fr_fr, multilingual/xlwic/xlwic_it_it, persuasion, phrase_similarity, quail, race-c, riddle_sense, scifact_entailment, sciq, sick/label, sick/relatedness, silicone/dyda_da, silicone/oasis, silicone/sem, summarize_from_feedback/comparisons, toxic-chat/toxicchat0124/jailbreaking, toxic-chat/toxicchat0124/toxicity, webgpt_comparisons.

## Decontamination

Before training, every row that overlapped an evaluation item was removed. A row was dropped when any part of it
(the whole state, a line, a field) equalled an evaluation text of 20 or more characters after normalisation, or when
it contained at least half of the word 8-grams of an evaluation text of 12 or more words. This covered all evaluation
sets, including JevBench and DecideBench, which were used for evaluation only.

## Largest sources

| source | rows | licence tag |
|---|---|---|
| hf:nyu-mll/glue/mnli | 108,857 | unspecified |
| teacher:dair-ai/emotion | 44,420 | unspecified |
| hf:stanfordnlp/snli | 40,242 | commercial |
| hf:nyu-mll/glue/mnli-w3 | 35,208 | unspecified |
| toucan/OSS/tool_choice | 33,951 | commercial |
| toucan/Qwen3/tool_choice | 33,371 | commercial |
| teacher:yelp_review_full | 30,939 | unspecified |
| PKU-SafeRLHF/safety | 28,806 | non-commercial |
| hf:Tobi-Bueck/customer-support-tickets/queue | 26,074 | unspecified |
| teacher:SetFit/sst5 | 24,313 | unspecified |
| teacher:ag_news | 23,671 | unspecified |
| hf:fancyzhx/ag_news | 23,008 | unspecified |
| hf:mteb/banking77 | 22,618 | commercial |
| hf:Tobi-Bueck/customer-support-tickets/priority | 20,090 | unspecified |
| synthetic:lookup | 20,000 | commercial |
| hf:facebook/anli/train_r3 | 19,075 | non-commercial |
| hf:dair-ai/emotion | 18,046 | unspecified |
| ConTRoL-nli | 17,056 | non-commercial |
| hf:SetFit/sst5 | 16,765 | unspecified |
| BeaverTails | 16,108 | non-commercial |
| implicit-hate-stg1 | 15,863 | non-commercial |
| banking77 | 15,844 | commercial |
| FOL-nli | 14,697 | commercial |
| logiqa-2.0-nli | 14,107 | non-commercial |
| logical-entailment | 14,071 | commercial |
| tomi-nli | 13,833 | commercial |
| multilingual/americas_nli/all_languages | 13,546 | commercial |
| oasst2/toxicity | 13,417 | unspecified |
| hf:Tobi-Bueck/customer-support-tickets/type | 13,140 | unspecified |
| LogicNLI | 12,877 | unspecified |
| multilingual/multilingual-NLI-26lang-2mil7 | 12,800 | unspecified |
| acceptability-prediction/rating_votes | 12,521 | commercial |
| go_emotions/simplified | 12,488 | commercial |
| oasst2_pairwise_rlhf_reward | 12,396 | unspecified |
| safe-guard-prompt-injection | 12,260 | unspecified |
| WANLI | 12,003 | commercial |
| multilingual/amazon_reviews_multi/all_languages | 12,000 | unspecified |
| HelpSteer3/preference | 11,887 | commercial |
| hf:bench-llm/or-bench | 11,810 | unspecified |
| multilingual/universal-joy | 11,649 | commercial |
