| Benchmark | Policy | Mean EM | Mean F1 | Mean Judge (LLM-graded correct) | N |
|---|---|---|---|---|---|
| locomo | fifo | 0.0417 | 0.1294 | 0.1833 | 120 |
| locomo | lru | 0.0833 | 0.1998 | 0.3417 | 120 |
| locomo | no_forget | 0.0917 | 0.1957 | 0.3333 | 120 |
| locomo | ours | 0.0500 | 0.1575 | 0.2000 | 120 |
| locomo | ours_utility | 0.0917 | 0.1949 | 0.3500 | 120 |
| longmemeval | fifo | 0.0800 | 0.1558 | 0.4000 | 25 |
| longmemeval | lru | 0.0800 | 0.1289 | 0.2800 | 25 |
| longmemeval | no_forget | 0.0800 | 0.1525 | 0.3600 | 25 |
| longmemeval | ours | 0.0800 | 0.1563 | 0.3600 | 25 |
| longmemeval | ours_utility | 0.0800 | 0.1563 | 0.3600 | 25 |

EM=0 but judge=CORRECT (EM undercounting genuinely correct answers): 160/725 (22.1%)

| Benchmark | Policy | Question | Reference | Prediction |
|---|---|---|---|---|
| locomo | no_forget | When did Caroline go to the LGBTQ support group? | 7 May 2023 | 2023-05-07 |
| locomo | no_forget | What fields would Caroline be likely to pursue in her educaton? | Psychology, counseling certification | Counseling and mental health. |
| locomo | no_forget | What is Caroline's identity? | Transgender woman | Caroline is part of the transgender community. |
| locomo | no_forget | When did Melanie run a charity race? | The sunday before 25 May 2023 | 2023-05-20 |
| locomo | no_forget | When did Caroline give a speech at a school? | The week before 9 June 2023 | 2023-06-08 |
| locomo | fifo | What fields would Caroline be likely to pursue in her educaton? | Psychology, counseling certification | Counseling and mental health. |
| locomo | fifo | When did Melanie run a charity race? | The sunday before 25 May 2023 | 2023-05-20 |
| locomo | fifo | When did Caroline give a speech at a school? | The week before 9 June 2023 | 2023-06-08 |
| locomo | lru | When did Caroline go to the LGBTQ support group? | 7 May 2023 | 2023-05-07 |
| locomo | lru | What fields would Caroline be likely to pursue in her educaton? | Psychology, counseling certification | Counseling and mental health. |
| locomo | lru | When did Melanie run a charity race? | The sunday before 25 May 2023 | 2023-05-20 |
| locomo | lru | When did Caroline give a speech at a school? | The week before 9 June 2023 | 2023-06-08 |
| locomo | ours | When did Caroline go to the LGBTQ support group? | 7 May 2023 | 2023-05-07 |
| locomo | ours | What fields would Caroline be likely to pursue in her educaton? | Psychology, counseling certification | Counseling and mental health. |
| locomo | ours | What is Caroline's identity? | Transgender woman | Caroline is part of the transgender community. |
| locomo | ours | When did Melanie run a charity race? | The sunday before 25 May 2023 | 2023-05-20 |
| locomo | ours | When did Caroline give a speech at a school? | The week before 9 June 2023 | 2023-06-08 |
| locomo | ours_utility | When did Caroline go to the LGBTQ support group? | 7 May 2023 | 2023-05-07 |
| locomo | ours_utility | What fields would Caroline be likely to pursue in her educaton? | Psychology, counseling certification | Counseling and mental health. |
| locomo | ours_utility | When did Melanie run a charity race? | The sunday before 25 May 2023 | 2023-05-20 |
| ... | ... | (140 more, see raw output) | | |