# E2 Retrieval vs QA Diagnosis (v2)

## Gold doc sources
- HotpotQA: supporting_facts.title (wiki titles).
- MuSiQue: paragraphs where is_supporting=True, using passage ids p{idx:04d}.
- MIRAGE: query_id as gold doc id (mapped_id in doc pool).

## Run root
- /home/wjk/workplace/nq/ano-rag/result_relrag/e1_main_qwen/run_20251225_063718
- Budget: 4096

## HotpotQA

| method | Hit&Correct | Hit&Wrong | Miss&Correct | Miss&Wrong | HC% | HW% | MC% | MW% | total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bm25_rag | 158 | 41 | 1 | 0 | 0.790 | 0.205 | 0.005 | 0.000 | 200 |
| dense_rag | 131 | 29 | 34 | 6 | 0.655 | 0.145 | 0.170 | 0.030 | 200 |
| hybrid_rag | 156 | 41 | 2 | 1 | 0.780 | 0.205 | 0.010 | 0.005 | 200 |
| raptor | 108 | 37 | 39 | 16 | 0.540 | 0.185 | 0.195 | 0.080 | 200 |
| relrag_full | 119 | 35 | 40 | 6 | 0.595 | 0.175 | 0.200 | 0.030 | 200 |

## MuSiQue

| method | Hit&Correct | Hit&Wrong | Miss&Correct | Miss&Wrong | HC% | HW% | MC% | MW% | total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bm25_rag | 19 | 58 | 45 | 77 | 0.095 | 0.291 | 0.226 | 0.387 | 199 |
| dense_rag | 30 | 46 | 41 | 82 | 0.151 | 0.231 | 0.206 | 0.412 | 199 |
| hybrid_rag | 18 | 45 | 48 | 88 | 0.090 | 0.226 | 0.241 | 0.442 | 199 |
| raptor | 28 | 45 | 49 | 77 | 0.141 | 0.226 | 0.246 | 0.387 | 199 |
| relrag_full | 31 | 47 | 47 | 74 | 0.156 | 0.236 | 0.236 | 0.372 | 199 |

## MIRAGE

| method | Hit&Correct | Hit&Wrong | Miss&Correct | Miss&Wrong | HC% | HW% | MC% | MW% | total |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bm25_rag | 186 | 14 | 0 | 0 | 0.930 | 0.070 | 0.000 | 0.000 | 200 |
| dense_rag | 182 | 16 | 0 | 2 | 0.910 | 0.080 | 0.000 | 0.010 | 200 |
| hybrid_rag | 187 | 11 | 1 | 1 | 0.935 | 0.055 | 0.005 | 0.005 | 200 |
| raptor | 177 | 22 | 0 | 1 | 0.885 | 0.110 | 0.000 | 0.005 | 200 |
| relrag_full | 0 | 0 | 93 | 107 | 0.000 | 0.000 | 0.465 | 0.535 | 200 |

## Typical samples

### HotpotQA

#### Hit & Wrong (>=5)
- qid: 5a713ea95542994082a3e6e4
- method: bm25_rag
- gold answer: ['Apalachees']
- retrieved top-5 ids: ['Alvaro Mexia', 'Indigenous peoples of Florida', 'Diplomatic mission', 'United States Mission to the European Union', 'Embassy of the United States, Nairobi']
- model prediction (FINAL): Ais

- qid: 5a7320565542991f9a20c61d
- method: bm25_rag
- gold answer: ['Keith Bostic']
- retrieved top-5 ids: ['Keith Bostic (American football)', 'Keith Bostic', 'Ken Herock', '1987 Pittsburgh Steelers season', '1989 Houston Oilers season']
- model prediction (FINAL): Jerry Glanville

- qid: 5a74c85055429916b0164218
- method: bm25_rag
- gold answer: ['Alistair Grant']
- retrieved top-5 ids: ['British people', 'Australian referendum, 1937', 'Arild Nyquist', 'Valentina Tereshkova', 'William Bonfield']
- model prediction (FINAL): William Bonfield

- qid: 5a75e05c55429976ec32bc5f
- method: bm25_rag
- gold answer: ['9,984']
- retrieved top-5 ids: ['Brown County, Kansas', 'Brown State Fishing Lake', 'Osage State Fishing Lake', 'Commanda Lake', 'Neosho State Fishing Lake']
- model prediction (FINAL): 331 million

- qid: 5a75f0ea5542994ccc91866c
- method: bm25_rag
- gold answer: ['The A41']
- retrieved top-5 ids: ['Backford Cross', 'Strawberry Park, Cheshire', 'Al Haj FAW motors', 'A5117 road', 'U.S. Route 19 Truck (Pittsburgh)']
- model prediction (FINAL): A5117

#### Miss & Correct (>=5)
- qid: 5a722b8655429971e9dc9329
- method: bm25_rag
- gold answer: ['Barton Lee Hazlewood']
- retrieved top-5 ids: ['Jim Shoulders', 'Death of Abdul Wali', 'Boots (album)', 'Boots (EP)', 'The Twenty-fifth Anniversary Anthology']
- model prediction (FINAL): Lee Hazlewood

- qid: 5a7571135542992d0ec05f98
- method: dense_rag
- gold answer: ['New York City']
- retrieved top-5 ids: ['Mathematically Alive', 'Frank L. Schmidt', 'E. Mavis Hetherington', 'David C. Funder', 'Abraham Maslow']
- model prediction (FINAL): New York City

- qid: 5a7625e8554299109176e66a
- method: dense_rag
- gold answer: ['1865']
- retrieved top-5 ids: ['2013 Birthday Honours', 'Birthday Honours', '2014 Birthday Honours', '1951 Birthday Honours', '2015 Birthday Honours']
- model prediction (FINAL): 1865

- qid: 5a77c1505542997042120b1b
- method: dense_rag
- gold answer: ['"We\'ll Burn That Bridge"']
- retrieved top-5 ids: ['Steers &amp; Stripes', 'Alan Jackson albums discography', 'High Lonesome (Randy Travis album)', "A Lot About Livin' (And a Little 'bout Love)", 'Holler Back']
- model prediction (FINAL): We'll Burn That Bridge

- qid: 5a7997a2554299029c4b5f59
- method: dense_rag
- gold answer: ['1943']
- retrieved top-5 ids: ["People's Socialist Republic of Albania", 'Madras Legislative Assembly election, 1967', "Bal'arab bin Himyar", 'Swedish general election, 1982', 'Hungarian Communist Party']
- model prediction (FINAL): 1943


### MuSiQue

#### Hit & Wrong (>=5)
- qid: 2hop__108158_83769
- method: bm25_rag
- gold answer: ["Fred O'Bannion"]
- retrieved top-5 ids: ['p0008', 'p0001', 'p0012', 'p0006', 'p0015']
- model prediction (FINAL): The information provided does not specify who created "Push, Nevada" or if the creator appeared in "Dazed and Confused."

- qid: 2hop__123166_47134
- method: bm25_rag
- gold answer: ['Benny Beaver']
- retrieved top-5 ids: ['p0005', 'p0015', 'p0017', 'p0018', 'p0016']
- model prediction (FINAL): Beavers

- qid: 2hop__14078_49084
- method: bm25_rag
- gold answer: ['Liam Thomas Garrigan', 'Liam Garrigan']
- retrieved top-5 ids: ['p0004', 'p0014', 'p0007', 'p0012', 'p0000']
- model prediction (FINAL): Tony Amendola

- qid: 2hop__14092_8311
- method: bm25_rag
- gold answer: ['1216']
- retrieved top-5 ids: ['p0013', 'p0012', 'p0015', 'p0006', 'p0002']
- model prediction (FINAL): 1284

- qid: 2hop__156658_155922
- method: bm25_rag
- gold answer: ['Sydney Harbour', 'Port Jackson']
- retrieved top-5 ids: ['p0010', 'p0015', 'p0001', 'p0011', 'p0012']
- model prediction (FINAL): Tasman Sea

#### Miss & Correct (>=5)
- qid: 2hop__107548_124896
- method: bm25_rag
- gold answer: ['Nicholas Bacon']
- retrieved top-5 ids: ['p0012', 'p0000', 'p0007', 'p0004', 'p0018']
- model prediction (FINAL): The text does not provide information about the father of Francis Bacon, the artist who painted Head I.

- qid: 2hop__16267_25719
- method: bm25_rag
- gold answer: ['406']
- retrieved top-5 ids: ['p0005', 'p0007', 'p0014', 'p0001', 'p0017']
- model prediction (FINAL): 406

- qid: 2hop__197470_271394
- method: bm25_rag
- gold answer: ['Blair High School']
- retrieved top-5 ids: ['p0010', 'p0018', 'p0001', 'p0019', 'p0004']
- model prediction (FINAL): Blair International Baccalaureate School in Pasadena.

- qid: 2hop__203985_524737
- method: bm25_rag
- gold answer: ['United Soccer League', 'USL', 'USISL', 'Southwest Independent Soccer League', 'SISL']
- retrieved top-5 ids: ['p0004', 'p0014', 'p0001', 'p0016', 'p0009']
- model prediction (FINAL): Southwest Independent Soccer League

- qid: 2hop__207597_63853
- method: bm25_rag
- gold answer: ['Thaba Putsoa']
- retrieved top-5 ids: ['p0004', 'p0018', 'p0013', 'p0009', 'p0012']
- model prediction (FINAL): The source of the river that is the mouth of the Caledon River is Thaba Putsoa in the Maloti Mountains (Drakensberg), Lesotho.


### MIRAGE

#### Hit & Wrong (>=5)
- qid: 24d2e8bb-34e2-4a16-9bf6-3205e37724cc
- method: bm25_rag
- gold answer: ['composer']
- retrieved top-5 ids: ['24d2e8bb-34e2-4a16-9bf6-3205e37724cc', '24d2e8bb-34e2-4a16-9bf6-3205e37724cc', 'b148fb30-aa7c-4a80-8a1b-923ebca0add3', '24d2e8bb-34e2-4a16-9bf6-3205e37724cc', '1760d430-11c2-4556-ab5e-4b0a6438c9fd']
- model prediction (FINAL): baseball player

- qid: 60fd8745-ca21-4159-b5c6-2a732e4658fc
- method: bm25_rag
- gold answer: ['actor', 'actress', 'actors', 'actresses']
- retrieved top-5 ids: ['60fd8745-ca21-4159-b5c6-2a732e4658fc', '60fd8745-ca21-4159-b5c6-2a732e4658fc', '60fd8745-ca21-4159-b5c6-2a732e4658fc', '60fd8745-ca21-4159-b5c6-2a732e4658fc', '60fd8745-ca21-4159-b5c6-2a732e4658fc']
- model prediction (FINAL): South Korean former singer-songwriter and convicted criminal.

- qid: 68d9ab39-5f61-4047-83c6-7e07bba03a83
- method: bm25_rag
- gold answer: ['dentist', 'dentists', 'dental surgeon']
- retrieved top-5 ids: ['68d9ab39-5f61-4047-83c6-7e07bba03a83', '68d9ab39-5f61-4047-83c6-7e07bba03a83', '68d9ab39-5f61-4047-83c6-7e07bba03a83', '68d9ab39-5f61-4047-83c6-7e07bba03a83', '88642e64-82a7-4ca3-b5a2-28d3c2863d4d']
- model prediction (FINAL): Matthew McKay's occupation was a Canadian politician. He served as a Liberal party member of the House of Commons of Canada and held various roles such as Parliamentary Secretary to the Minister of Finance and critic for the Environment. FINAL: Canadian politician.

- qid: 691eaef5-3cd5-4866-824f-47aaf2735de1
- method: bm25_rag
- gold answer: ['actor', 'actress', 'actors', 'actresses']
- retrieved top-5 ids: ['691eaef5-3cd5-4866-824f-47aaf2735de1', '691eaef5-3cd5-4866-824f-47aaf2735de1', '691eaef5-3cd5-4866-824f-47aaf2735de1', '691eaef5-3cd5-4866-824f-47aaf2735de1', '691eaef5-3cd5-4866-824f-47aaf2735de1']
- model prediction (FINAL): Drama teacher

- qid: 69891229-9313-405d-a4b7-2d8d9edbad4d
- method: bm25_rag
- gold answer: ['civil engineer']
- retrieved top-5 ids: ['69891229-9313-405d-a4b7-2d8d9edbad4d', '69891229-9313-405d-a4b7-2d8d9edbad4d', '69891229-9313-405d-a4b7-2d8d9edbad4d', '69891229-9313-405d-a4b7-2d8d9edbad4d', '69891229-9313-405d-a4b7-2d8d9edbad4d']
- model prediction (FINAL): cinematographer

#### Miss & Correct (>=5)
- qid: 70f2d97b-0fee-4340-9bf2-f9d71db39063
- method: hybrid_rag
- gold answer: ['politician', 'political leader', 'political figure', 'polit.', 'pol']
- retrieved top-5 ids: ['19183d8e-60a7-4d68-a838-3360e352a774', '67cf1847-e02d-4b85-b6f5-5f7a3759e8d7', 'ae40a1dd-0bb4-42ab-9432-961e7e3ab6af', 'd595b916-f9b5-441b-92d3-9df795b1ea53', 'd595b916-f9b5-441b-92d3-9df795b1ea53']
- model prediction (FINAL): Canadian politician and wholesaler.

- qid: 0281c798-3494-4ec9-a1ad-1a70e7fcf71c
- method: relrag_full
- gold answer: ['composer']
- retrieved top-5 ids: ['Rinaldo del Mel', 'Atto Melani', 'Kekko Fornarelli', 'Rinaldo del Mel', 'Atto Melani']
- model prediction (FINAL): composer

- qid: 055869b8-82e0-4321-94cb-6dcc0a3933c0
- method: relrag_full
- gold answer: ['actor', 'actress', 'actors', 'actresses']
- retrieved top-5 ids: ['Jonathan Le Billon', 'Under Paris', 'Claire Denis', 'Hate Man', 'Jonathan Le Billon']
- model prediction (FINAL): actor

- qid: 05c2e5a4-28c6-42db-833d-e08b890cf49a
- method: relrag_full
- gold answer: ['actor', 'actress', 'actors', 'actresses']
- retrieved top-5 ids: ['Alleppey Vincent', 'Alleppey Ranganath', 'Alleppey Vincent', 'Alleppey Ranganath', 'Johnson (composer)']
- model prediction (FINAL): Indian film actor in Malayalam cinema.

- qid: 0687ac2a-fed5-48ea-8a00-ae126998736a
- method: relrag_full
- gold answer: ['actor', 'actress', 'actors', 'actresses']
- retrieved top-5 ids: ['Mariví Bilbao', 'Mariví Bilbao', 'Carmen Marai', 'María Elena Marqués', 'Lorena Herrera']
- model prediction (FINAL): actress

## Validation

### Hit@5 vs retrieval_metrics.json
| dataset | method | hit@5_e2 | hit@5_metrics | diff | pass |
| --- | --- | --- | --- | --- | --- |
| HotpotQA | bm25_rag | 0.995 | 0.995 | +0.000 | PASS |
| HotpotQA | dense_rag | 0.800 | 0.800 | +0.000 | PASS |
| HotpotQA | hybrid_rag | 0.985 | 0.983 | +0.002 | PASS |
| HotpotQA | raptor | 0.725 | 0.725 | +0.000 | PASS |
| HotpotQA | relrag_full | 0.770 | 0.770 | +0.000 | PASS |
| MuSiQue | bm25_rag | 0.380 | 0.385 | -0.005 | PASS |
| MuSiQue | dense_rag | 0.385 | 0.385 | +0.000 | PASS |
| MuSiQue | hybrid_rag | 0.310 | 0.315 | -0.005 | PASS |
| MuSiQue | raptor | 0.370 | 0.370 | +0.000 | PASS |
| MuSiQue | relrag_full | 0.395 | 0.395 | +0.000 | PASS |
| MIRAGE | bm25_rag | 1.000 | 1.000 | +0.000 | PASS |
| MIRAGE | dense_rag | 0.990 | 0.990 | +0.000 | PASS |
| MIRAGE | hybrid_rag | 0.990 | 0.990 | +0.000 | PASS |
| MIRAGE | raptor | 0.995 | 0.995 | +0.000 | PASS |
| MIRAGE | relrag_full | 0.000 | 0.000 | +0.000 | PASS |

### Missing qids (if any)

### Random qid samples (id semantics check)
#### HotpotQA (seed=42)
| qid | gold_doc_ids | retrieved_top5_ids | hit |
| --- | --- | --- | --- |
| 5ae0006755429925eb1afbd3 | ["Gum Wall", "Gum Wall", "San Luis Obispo, California"] | ["Gum Wall", "Ouch! (gum)", "Bubblegum Alley", "Philadelphia Baseball Wall of Fame", "Lower Louviers and Chicken Alley"] | hit |
| 5a7a0e1e5542990783324e1a | ["Manchester Terrier", "Scotch Collie", "Scotch Collie"] | ["Scotch Collie", "English Setter", "Florence Nagle", "Viol", "Collie"] | hit |
| 5a738d27554299623ed4abf3 | ["Marco Da Silva (dancer)", "Erika Jayne"] | ["Erika Jayne", "Marco Da Silva (dancer)", "Marco da Silva (French footballer)", "Marquinhos (footballer, born June 1989)", "Café (musician)"] | hit |
| 5ae67dba55429908198fa5f0 | ["Memphis Hustle", "Memphis Hustle", "Southaven, Mississippi", "Southaven, Mississippi"] | ["Memphis Hustle", "Olive Branch, Mississippi", "Ellenbrook, Western Australia", "Lakeland, Tennessee", "Southaven, Mississippi"] | hit |
| 5a88658955429938390d3f47 | ["Rostker v. Goldberg", "Conscription in the United States", "Conscription in the United States"] | ["Rostker v. Goldberg", "Conscription in the United States", "Franklin D. Roosevelt's record on civil rights", "Fernando Tapias Stahelin", "Armed Forces Covenant"] | hit |
| 5a8704f8554299211dda2ba4 | ["Aonghus Mór", "Kingdom of the Isles", "Kingdom of the Isles"] | ["Aonghus Mór", "Lord of Islay", "Kingdom of the Isles", "List of rulers of the Kingdom of the Isles", "List of islands of Tasmania"] | hit |
| 5a85b2d95542997b5ce40028 | ["Eenasul Fateh", "Management consulting"] | ["Management consulting", "Lil Ru", "Eenasul Fateh", "ELDee", "Amaruk Kayshapanta"] | hit |
| 5a7bbb64554299042af8f7cc | ["Annie Morton", "Annie Morton", "Terry Richardson"] | ["Annie Morton", "Gumbo (PJ Morton album)", "Kenton Richardson", "Terry Richardson", "Madonna (book)"] | hit |
| 5ae63dad55429929b0807afe | ["Here We Go Round the Mulberry Bush (film)", "Hunter Davies"] | ["Here We Go Round the Mulberry Bush (film)", "Hunter Davies", "Clive Donner", "Roy Holder", "Here We Go Round the Mulberry Bush (Traffic song)"] | hit |
| 5a79311755429970f5fffe67 | ["Masakazu Katsura", "I&quot;s"] | ["I&quot;s", "The Kindaichi Case Files", "My Bride is a Mermaid", "Clear Skies!", "Silver Spoon (manga)"] | hit |
| 5ae22b8d554299234fd0440f | ["Kasper Schmeichel", "Kasper Schmeichel", "Peter Schmeichel"] | ["Peter Schmeichel", "Sommeren '92", "IFFHS World's Best Club Coach", "IFFHS World's Best Goalkeeper", "Pelé"] | hit |
| 5ac0d83a554299294b219038 | ["Randall Cunningham II", "Bishop Gorman High School"] | ["Randall Cunningham II", "Vashti Cunningham", "Nevada Union High School", "Randall Cunningham", "List of multi-sport athletes"] | hit |
| 5a77c1505542997042120b1b | ["We'll Burn That Bridge", "We'll Burn That Bridge", "Chattahoochee (song)"] | ["We'll Burn That Bridge", "Steers &amp; Stripes", "Chattahoochee (song)", "A Lot About Livin' (And a Little 'bout Love)", "(Who Says) You Can't Have It All"] | hit |
| 5adbf0a255429947ff17385a | ["Laleli Mosque", "Esma Sultan Mansion"] | ["Esma Sultan Mansion", "Laleli Mosque", "Sultan Ahmed Mosque", "Djamaâ el Kebir", "Esma Sultan (daughter of Ahmed III)"] | hit |
| 5ab56e32554299637185c594 | ["Random House Tower", "888 7th Avenue"] | ["Real estate technology", "Random House Tower", "Loan modification in the United States", "Villard (imprint)", "Real estate benchmarking"] | hit |
| 5a74c85055429916b0164218 | ["Alistair Grant", "British people"] | ["British people", "Australian referendum, 1937", "Arild Nyquist", "Valentina Tereshkova", "William Bonfield"] | hit |
| 5a74106b55429979e288289e | ["Sachin Warrier", "Tata Consultancy Services"] | ["William Connolley", "Tata Consultancy Services", "Sachin Warrier", "Alec Muffett", "Muthuchippi Poloru"] | hit |
| 5a77cb335542997042120b3a | ["MEO Rip Curl Pro Portugal", "MEO Rip Curl Pro Portugal", "John John Florence", "John John Florence"] | ["John John Florence", "Coco Ho", "MEO Rip Curl Pro Portugal", "Andy Irons", "Barton Lynch"] | hit |
| 5a84c4135542994c784dda31 | ["Yingkou", "Fuding"] | ["Yingkou", "Yingkou East Railway Station", "Xiapu County", "Bayuquan Railway Station", "Fuding"] | hit |
| 5a85eed75542996432c5713b | ["Mascogos", "Black Seminoles"] | ["Black Seminoles", "Mascogos", "Seminole Nation of Oklahoma", "Dhoolpet", "Seminole"] | hit |
#### MuSiQue (seed=42)
| qid | gold_doc_ids | retrieved_top5_ids | hit |
| --- | --- | --- | --- |
| 3hop2__88342_93066_47738 | ["p0009", "p0017"] | ["p0003", "p0018", "p0002", "p0013", "p0004"] | miss |
| 2hop__2299_38663 | ["p0017"] | ["p0016", "p0010", "p0013", "p0005", "p0004"] | miss |
| 2hop__13106_158105 | ["p0002", "p0003"] | ["p0000", "p0004", "p0011", "p0014", "p0012"] | miss |
| 4hop2__161602_474028_88460_126088 | ["p0000", "p0005", "p0010", "p0012"] | ["p0014", "p0008", "p0019", "p0017", "p0000"] | hit |
| 2hop__554167_451128 | ["p0011", "p0019"] | ["p0003", "p0009", "p0017", "p0012", "p0006"] | miss |
| 2hop__472083_7298 | ["p0010", "p0012"] | ["p0009", "p0016", "p0004", "p0015", "p0003"] | miss |
| 2hop__445963_6098 | ["p0001", "p0016"] | ["p0006", "p0011", "p0012", "p0014", "p0004"] | miss |
| 2hop__279729_20057 | [] | ["p0018", "p0003", "p0011", "p0012", "p0002"] | miss |
| 4hop2__105527_39078_8987_8974 | ["p0004", "p0008", "p0010", "p0019"] | ["p0014", "p0002", "p0011", "p0003", "p0009"] | miss |
| 2hop__215898_67465 | [] | ["p0005", "p0009", "p0006", "p0004", "p0007"] | miss |
| 4hop1__17192_17130_70784_61381 | ["p0004", "p0010", "p0017"] | ["p0017", "p0015", "p0007", "p0010", "p0016"] | hit |
| 3hop1__539312_744503_34754 | ["p0002", "p0008", "p0012"] | ["p0007", "p0005", "p0000", "p0008", "p0009"] | hit |
| 2hop__197470_271394 | ["p0007"] | ["p0010", "p0018", "p0001", "p0019", "p0004"] | miss |
| 3hop1__90327_83076_319330 | ["p0009", "p0010", "p0018"] | ["p0010", "p0009", "p0004", "p0013", "p0005"] | hit |
| 2hop__847760_80026 | ["p0013", "p0014"] | ["p0017", "p0000", "p0014", "p0008", "p0007"] | hit |
| 2hop__14078_49084 | ["p0007", "p0010"] | ["p0004", "p0014", "p0007", "p0012", "p0000"] | hit |
| 2hop__13548_13529 | ["p0010", "p0018"] | ["p0007", "p0004", "p0000", "p0008", "p0017"] | miss |
| 2hop__203985_524737 | ["p0013"] | ["p0004", "p0014", "p0001", "p0016", "p0009"] | miss |
| 2hop__442175_56873 | ["p0017"] | ["p0019", "p0013", "p0003", "p0002", "p0008"] | miss |
| 2hop__458672_20057 | [] | ["p0003", "p0000", "p0012", "p0013", "p0014"] | miss |
#### MIRAGE (seed=42)
| qid | gold_doc_ids | retrieved_top5_ids | hit |
| --- | --- | --- | --- |
| d2e16cd5-565e-443a-b0a1-4a830d981bce | ["d2e16cd5-565e-443a-b0a1-4a830d981bce"] | ["d2e16cd5-565e-443a-b0a1-4a830d981bce", "d2e16cd5-565e-443a-b0a1-4a830d981bce", "d2e16cd5-565e-443a-b0a1-4a830d981bce", "d2e16cd5-565e-443a-b0a1-4a830d981bce", "d2e16cd5-565e-443a-b0a1-4a830d981bce"] | hit |
| 23a40595-4d16-4bae-a06d-90782f3c5d3d | ["23a40595-4d16-4bae-a06d-90782f3c5d3d"] | ["23a40595-4d16-4bae-a06d-90782f3c5d3d", "23a40595-4d16-4bae-a06d-90782f3c5d3d", "23a40595-4d16-4bae-a06d-90782f3c5d3d", "1d640f82-ac32-43a5-8ccc-8a502fc4f342", "23a40595-4d16-4bae-a06d-90782f3c5d3d"] | hit |
| 0735a70f-90c7-40f6-9a3e-79904f20465e | ["0735a70f-90c7-40f6-9a3e-79904f20465e"] | ["0735a70f-90c7-40f6-9a3e-79904f20465e", "0735a70f-90c7-40f6-9a3e-79904f20465e", "0735a70f-90c7-40f6-9a3e-79904f20465e", "0735a70f-90c7-40f6-9a3e-79904f20465e", "0735a70f-90c7-40f6-9a3e-79904f20465e"] | hit |
| f54d10e2-3f25-4f65-91f1-07f412bd4e24 | ["f54d10e2-3f25-4f65-91f1-07f412bd4e24"] | ["f54d10e2-3f25-4f65-91f1-07f412bd4e24", "f54d10e2-3f25-4f65-91f1-07f412bd4e24", "f54d10e2-3f25-4f65-91f1-07f412bd4e24", "60fd8745-ca21-4159-b5c6-2a732e4658fc", "f54d10e2-3f25-4f65-91f1-07f412bd4e24"] | hit |
| 60fd8745-ca21-4159-b5c6-2a732e4658fc | ["60fd8745-ca21-4159-b5c6-2a732e4658fc"] | ["60fd8745-ca21-4159-b5c6-2a732e4658fc", "60fd8745-ca21-4159-b5c6-2a732e4658fc", "60fd8745-ca21-4159-b5c6-2a732e4658fc", "60fd8745-ca21-4159-b5c6-2a732e4658fc", "60fd8745-ca21-4159-b5c6-2a732e4658fc"] | hit |
| 55d7decd-7a10-4a37-9923-5e970c10be77 | ["55d7decd-7a10-4a37-9923-5e970c10be77"] | ["55d7decd-7a10-4a37-9923-5e970c10be77", "55d7decd-7a10-4a37-9923-5e970c10be77", "55d7decd-7a10-4a37-9923-5e970c10be77", "d349e1e9-3296-4cb9-a8e5-2480925cf25a", "7a987b7d-4618-407f-aba4-f400d076c2f6"] | hit |
| 4d5a3c5c-4b7c-4633-a9f9-a39d9e885344 | ["4d5a3c5c-4b7c-4633-a9f9-a39d9e885344"] | ["4d5a3c5c-4b7c-4633-a9f9-a39d9e885344", "4d5a3c5c-4b7c-4633-a9f9-a39d9e885344", "4d5a3c5c-4b7c-4633-a9f9-a39d9e885344", "4d5a3c5c-4b7c-4633-a9f9-a39d9e885344", "4d5a3c5c-4b7c-4633-a9f9-a39d9e885344"] | hit |
| 2d972621-0292-43d0-abb0-7d527d13e97b | ["2d972621-0292-43d0-abb0-7d527d13e97b"] | ["2d972621-0292-43d0-abb0-7d527d13e97b", "2d972621-0292-43d0-abb0-7d527d13e97b", "2d972621-0292-43d0-abb0-7d527d13e97b", "2d972621-0292-43d0-abb0-7d527d13e97b", "2d972621-0292-43d0-abb0-7d527d13e97b"] | hit |
| f4d76401-b38a-41de-ace2-2fb9b1fe448a | ["f4d76401-b38a-41de-ace2-2fb9b1fe448a"] | ["f4d76401-b38a-41de-ace2-2fb9b1fe448a", "f4d76401-b38a-41de-ace2-2fb9b1fe448a", "f4d76401-b38a-41de-ace2-2fb9b1fe448a", "f4d76401-b38a-41de-ace2-2fb9b1fe448a", "f4d76401-b38a-41de-ace2-2fb9b1fe448a"] | hit |
| 20d6d164-7b97-4160-83a3-a333f657364b | ["20d6d164-7b97-4160-83a3-a333f657364b"] | ["20d6d164-7b97-4160-83a3-a333f657364b", "20d6d164-7b97-4160-83a3-a333f657364b", "20d6d164-7b97-4160-83a3-a333f657364b", "20d6d164-7b97-4160-83a3-a333f657364b", "20d6d164-7b97-4160-83a3-a333f657364b"] | hit |
| d86e4b65-10ba-44e1-8b17-e4b952e6f8af | ["d86e4b65-10ba-44e1-8b17-e4b952e6f8af"] | ["d86e4b65-10ba-44e1-8b17-e4b952e6f8af", "d86e4b65-10ba-44e1-8b17-e4b952e6f8af", "d86e4b65-10ba-44e1-8b17-e4b952e6f8af", "d86e4b65-10ba-44e1-8b17-e4b952e6f8af", "d86e4b65-10ba-44e1-8b17-e4b952e6f8af"] | hit |
| b62518aa-0278-4bbb-823b-bd8c0295b153 | ["b62518aa-0278-4bbb-823b-bd8c0295b153"] | ["0b334104-8268-40d9-9ad7-05dbbe9bd30a", "b62518aa-0278-4bbb-823b-bd8c0295b153", "b62518aa-0278-4bbb-823b-bd8c0295b153", "b62518aa-0278-4bbb-823b-bd8c0295b153", "b62518aa-0278-4bbb-823b-bd8c0295b153"] | hit |
| 1a3acc25-e593-4999-8652-612f42aff0f3 | ["1a3acc25-e593-4999-8652-612f42aff0f3"] | ["1a3acc25-e593-4999-8652-612f42aff0f3", "1a3acc25-e593-4999-8652-612f42aff0f3", "1a3acc25-e593-4999-8652-612f42aff0f3", "1a3acc25-e593-4999-8652-612f42aff0f3", "1a3acc25-e593-4999-8652-612f42aff0f3"] | hit |
| c06babd1-3f46-46c9-9300-e542a7abe671 | ["c06babd1-3f46-46c9-9300-e542a7abe671"] | ["c06babd1-3f46-46c9-9300-e542a7abe671", "c06babd1-3f46-46c9-9300-e542a7abe671", "c06babd1-3f46-46c9-9300-e542a7abe671", "c06babd1-3f46-46c9-9300-e542a7abe671", "c06babd1-3f46-46c9-9300-e542a7abe671"] | hit |
| 8f4c2f14-b410-42f9-9855-843cc57c5771 | ["8f4c2f14-b410-42f9-9855-843cc57c5771"] | ["8f4c2f14-b410-42f9-9855-843cc57c5771", "8f4c2f14-b410-42f9-9855-843cc57c5771", "8f4c2f14-b410-42f9-9855-843cc57c5771", "8f4c2f14-b410-42f9-9855-843cc57c5771", "ada03f12-0d44-450b-b662-54ac89504269"] | hit |
| 0846600f-fc75-405c-8acc-53f0e32cfade | ["0846600f-fc75-405c-8acc-53f0e32cfade"] | ["0846600f-fc75-405c-8acc-53f0e32cfade", "0846600f-fc75-405c-8acc-53f0e32cfade", "0846600f-fc75-405c-8acc-53f0e32cfade", "0846600f-fc75-405c-8acc-53f0e32cfade", "d349e1e9-3296-4cb9-a8e5-2480925cf25a"] | hit |
| 075fcad0-3d05-449c-b372-a07ebbe3d334 | ["075fcad0-3d05-449c-b372-a07ebbe3d334"] | ["075fcad0-3d05-449c-b372-a07ebbe3d334", "075fcad0-3d05-449c-b372-a07ebbe3d334", "075fcad0-3d05-449c-b372-a07ebbe3d334", "075fcad0-3d05-449c-b372-a07ebbe3d334", "d349e1e9-3296-4cb9-a8e5-2480925cf25a"] | hit |
| 1d640f82-ac32-43a5-8ccc-8a502fc4f342 | ["1d640f82-ac32-43a5-8ccc-8a502fc4f342"] | ["1d640f82-ac32-43a5-8ccc-8a502fc4f342", "1d640f82-ac32-43a5-8ccc-8a502fc4f342", "1d640f82-ac32-43a5-8ccc-8a502fc4f342", "1d640f82-ac32-43a5-8ccc-8a502fc4f342", "1d640f82-ac32-43a5-8ccc-8a502fc4f342"] | hit |
| 4b0ef0af-25f3-464b-a238-80b0f96395d6 | ["4b0ef0af-25f3-464b-a238-80b0f96395d6"] | ["4b0ef0af-25f3-464b-a238-80b0f96395d6", "4b0ef0af-25f3-464b-a238-80b0f96395d6", "157c3c0a-d614-4437-9b11-66ba60533505", "4b0ef0af-25f3-464b-a238-80b0f96395d6", "4b0ef0af-25f3-464b-a238-80b0f96395d6"] | hit |
| 4f36b3fe-d47b-4d8b-bb44-0773ad84143d | ["4f36b3fe-d47b-4d8b-bb44-0773ad84143d"] | ["4f36b3fe-d47b-4d8b-bb44-0773ad84143d", "4f36b3fe-d47b-4d8b-bb44-0773ad84143d", "4f36b3fe-d47b-4d8b-bb44-0773ad84143d", "4f36b3fe-d47b-4d8b-bb44-0773ad84143d", "d349e1e9-3296-4cb9-a8e5-2480925cf25a"] | hit |

### QA correctness recomputation evidence
Per-query EM/F1 are computed from dataset gold answers with `compute_generation_metrics`, using FINAL answers extracted from `pred_raw.jsonl`.
#### HotpotQA (QA correctness recompute examples)
| qid | gold_answer | pred_final | EM | F1 |
| --- | --- | --- | --- | --- |
| 5ae0006755429925eb1afbd3 | ["San Luis Obispo, California"] | "San Luis Obispo, California." | 1.000 | 1.000 |
| 5a7a0e1e5542990783324e1a | ["Scotch Collie"] | "Scotch Collie" | 1.000 | 1.000 |
| 5a738d27554299623ed4abf3 | ["Erika Jayne"] | "Erika Jayne" | 1.000 | 1.000 |
#### MuSiQue (QA correctness recompute examples)
| qid | gold_answer | pred_final | EM | F1 |
| --- | --- | --- | --- | --- |
| 3hop2__88342_93066_47738 | ["the 2009 season"] | "2012" | 0.000 | 0.000 |
| 2hop__2299_38663 | ["54.7%"] | "50%" | 0.000 | 0.000 |
| 2hop__13106_158105 | ["ease of use and enhanced support for Plug and Play"] | "The provided content does not specify the two features highlighted by a Microsoft executive regarding IPTV in 2007." | 0.000 | 0.000 |
#### MIRAGE (QA correctness recompute examples)
| qid | gold_answer | pred_final | EM | F1 |
| --- | --- | --- | --- | --- |
| d2e16cd5-565e-443a-b0a1-4a830d981bce | ["actor", "actress", "actors", "actresses"] | "actor" | 1.000 | 1.000 |
| 23a40595-4d16-4bae-a06d-90782f3c5d3d | ["composer"] | "Musician and musical composer." | 0.000 | 0.400 |
| 0735a70f-90c7-40f6-9a3e-79904f20465e | ["Klerksdorp"] | "Klerksdorp, North West Province, South Africa." | 0.000 | 0.286 |
