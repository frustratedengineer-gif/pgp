# Week-6 qualitative examples

## 1. Traced eviction failures: `ours` evicts the evidence and gets it wrong; `ours_utility` keeps it and gets it right

8 example(s) found where the SAME evidence memory was evicted under the original TTL-threshold policy but survived under utility-ranked eviction, AND that specific change flipped the answer from wrong to right (judged, not just EM).

### Example 1 (conv-30)
**Question:** When Jon has lost his job as a banker?
**Reference answer:** 19 January, 2023
**Evicted evidence memory (present for `ours_utility`, gone for `ours`):** "Jon lost his job as a banker the day before the conversation."
**`ours` (TTL threshold) answered:** "I don't have that information yet." -- WRONG (EM=0.0, judge=0.0)
**`ours_utility` (ranked) answered:** "2023-01-19" -- CORRECT (EM=0.0, judge=1.0)

### Example 2 (conv-30)
**Question:** When Gina has lost her job at Door Dash?
**Reference answer:** January, 2023
**Evicted evidence memory (present for `ours_utility`, gone for `ours`):** "Gina lost her job at Door Dash during the month of the conversation."
**`ours` (TTL threshold) answered:** "2023-03-15" -- WRONG (EM=0.0, judge=0.0)
**`ours_utility` (ranked) answered:** "2023-01" -- CORRECT (EM=0.0, judge=1.0)

### Example 3 (conv-30)
**Question:** When was Jon in Paris?
**Reference answer:** 28 January 2023
**Evicted evidence memory (present for `ours_utility`, gone for `ours`):** "Jon visited Paris recently"
**`ours` (TTL threshold) answered:** "I don't have that information yet." -- WRONG (EM=0.0, judge=0.0)
**`ours_utility` (ranked) answered:** "2023-01-28" -- CORRECT (EM=0.0, judge=1.0)

### Example 4 (conv-41)
**Question:** What martial arts has John done?
**Reference answer:** Kickboxing, Taekwondo
**Evicted evidence memory (present for `ours_utility`, gone for `ours`):** "John practices taekwondo."; "John is currently doing kickboxing as a workout."
**`ours` (TTL threshold) answered:** "Kickboxing." -- WRONG (EM=0.0, judge=0.0)
**`ours_utility` (ranked) answered:** "Kickboxing, taekwondo." -- CORRECT (EM=1.0, judge=1.0)

### Example 5 (conv-41)
**Question:** When did John join the online support group?
**Reference answer:** The week before 1 January 2023
**Evicted evidence memory (present for `ours_utility`, gone for `ours`):** "John joined a service-focused online group last week and finds it emotionally rewarding."
**`ours` (TTL threshold) answered:** "I don't have that information yet." -- WRONG (EM=0.0, judge=0.0)
**`ours_utility` (ranked) answered:** "2022-12-25" -- CORRECT (EM=0.0, judge=1.0)

### Example 6 (conv-41)
**Question:** Who gave Maria's family money when she was younger and her family was going through tough times?
**Reference answer:** Her aunt
**Evicted evidence memory (present for `ours_utility`, gone for `ours`):** "When Maria was younger, her family had money problems and had to rely on help from their auntie, teaching her the importance of helping others in need."
**`ours` (TTL threshold) answered:** "I don't have that information yet." -- WRONG (EM=0.0, judge=0.0)
**`ours_utility` (ranked) answered:** "Auntie" -- CORRECT (EM=0.0, judge=1.0)

### Example 7 (conv-42)
**Question:** How long has Nate had his first two turtles?
**Reference answer:** three years
**Evicted evidence memory (present for `ours_utility`, gone for `ours`):** "Nate has had turtles for 3 years that bring him joy and help keep him calm during stressful times."
**`ours` (TTL threshold) answered:** "I don't have that information yet." -- WRONG (EM=0.0, judge=0.0)
**`ours_utility` (ranked) answered:** "2019" -- CORRECT (EM=0.0, judge=1.0)

### Example 8 (conv-42)
**Question:** When did Joanna finish her first screenplay?
**Reference answer:** The Friday before 23January, 2022
**Evicted evidence memory (present for `ours_utility`, gone for `ours`):** "Joanna finished her first full screenplay last Friday, which is a mix of drama and romance."
**`ours` (TTL threshold) answered:** "I don't have that information yet." -- WRONG (EM=0.0, judge=0.0)
**`ours_utility` (ranked) answered:** "2022-01-21" -- CORRECT (EM=0.0, judge=1.0)


## 2. EM penalizes correct-but-differently-worded answers

Shortest, cleanest examples where EM=0 but the LLM judge scored the answer substantively correct (selected for clarity, not cherry-picked for a particular policy).

**Caveat -- not blindly trustworthy:** the judge prompt (`prompts/judge.txt`) does not see memory dates, only question/reference/prediction. A duration-vs-absolute-date pair (e.g. reference "three years" vs. prediction "2019") can only be judged correct if the LLM silently infers the reference date -- plausibly a JUDGE ERROR, not proof EM was too harsh. Inspect entries like this manually before quoting them; most of the table is not this kind, but not all of it is guaranteed clean.

| Question | Reference | Prediction | Why EM=0 despite being correct |
|---|---|---|---|
| How many Prius has Evan owned? | two | 2 | format/wording differs, meaning matches |
| When did Jolene`s mother pass away? | in 2022 | 2022 | format/wording differs, meaning matches |
| Who gave Maria's family money when she was younger and her family was going through tough times? | Her aunt | Auntie | format/wording differs, meaning matches |
| Who gave Maria's family money when she was younger and her family was going through tough times? | Her aunt | Auntie. | format/wording differs, meaning matches |
| How long has Nate had his first two turtles? | three years | 2019 | format/wording differs, meaning matches |
| How many trips have I taken my Canon EOS 80D camera on? | five | Five trips. | format/wording differs, meaning matches |
| How many years passed between Audrey adopting Pixie and her other three dogs? | three years | 3 years | format/wording differs, meaning matches |
| Who gave Maria's family money when she was younger and her family was going through tough times? | Her aunt | Her auntie. | format/wording differs, meaning matches |