# E2 Retrieval vs QA Diagnosis (v3)

## Gold doc sources
- HotpotQA: supporting_facts.title (wiki titles).
- MuSiQue: prefer artifacts/gold_retrieval_ids.jsonl; fallback to supporting paragraphs p{idx:04d}.
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
| bm25_rag | 18 | 58 | 47 | 77 | 0.090 | 0.290 | 0.235 | 0.385 | 200 |
| dense_rag | 31 | 46 | 41 | 82 | 0.155 | 0.230 | 0.205 | 0.410 | 200 |
| hybrid_rag | 17 | 45 | 50 | 88 | 0.085 | 0.225 | 0.250 | 0.440 | 200 |
| raptor | 29 | 45 | 49 | 77 | 0.145 | 0.225 | 0.245 | 0.385 | 200 |
| relrag_full | 32 | 47 | 47 | 74 | 0.160 | 0.235 | 0.235 | 0.370 | 200 |

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
- qid: 5a8c7595554299585d9e36b6
- method: bm25_rag
- gold answer: ['Chief of Protocol']
- retrieved top-5 ids: ['A Kiss for Corliss', 'Kiss and Tell (1945 film)', 'Secretary of State for Constitutional Affairs', 'Lord High Treasurer', 'Village accountant']
- model prediction (FINAL): Ambassador to Ghana and Czechoslovakia.

- qid: 5a87ab905542996e4f3088c1
- method: bm25_rag
- gold answer: ['3,677 seated']
- retrieved top-5 ids: ['Dwyer Arena', 'Loan modification in the United States', 'Billings Bulls', '2006–07 QMJHL season', 'Lewiston Maineiacs']
- model prediction (FINAL): The capacity of the arena where the Lewiston Maineiacs played their home games is not specified.

- qid: 5ab56e32554299637185c594
- method: bm25_rag
- gold answer: ['no']
- retrieved top-5 ids: ['Real estate technology', 'Random House Tower', 'Loan modification in the United States', 'Villard (imprint)', 'Real estate benchmarking']
- model prediction (FINAL): Yes.

- qid: 5a75e05c55429976ec32bc5f
- method: bm25_rag
- gold answer: ['9,984']
- retrieved top-5 ids: ['Brown County, Kansas', 'Brown State Fishing Lake', 'Osage State Fishing Lake', 'Commanda Lake', 'Neosho State Fishing Lake']
- model prediction (FINAL): 331 million

- qid: 5ae0d4c9554299603e418468
- method: bm25_rag
- gold answer: ['1969 until 1974']
- retrieved top-5 ids: ['Roger O. Egeberg', 'Douglas J. Bennet', 'Steven K. Galson', 'Richard Nixon', 'Robert R. Hood']
- model prediction (FINAL): 1969–1974

#### Miss & Correct (>=5)
- qid: 5a722b8655429971e9dc9329
- method: bm25_rag
- gold answer: ['Barton Lee Hazlewood']
- retrieved top-5 ids: ['Jim Shoulders', 'Death of Abdul Wali', 'Boots (album)', 'Boots (EP)', 'The Twenty-fifth Anniversary Anthology']
- model prediction (FINAL): Lee Hazlewood

- qid: 5a85b2d95542997b5ce40028
- method: dense_rag
- gold answer: ['Eenasul Fateh']
- retrieved top-5 ids: ['James P. Comer', 'Lil Ru', 'Amaruk Kayshapanta', 'Criss Angel', 'ELDee']
- model prediction (FINAL): Eenasul Fateh

- qid: 5a8db19d5542994ba4e3dd00
- method: dense_rag
- gold answer: ['yes']
- retrieved top-5 ids: ['Mendocino County GMO Ban', 'Wilford H. Fawcett House', 'Suffrajett', "Hallelujah! I'm a Bum (album)", 'United States Senate election in Illinois, 1978']
- model prediction (FINAL): Yes, Local H and For Against are both from the United States.

- qid: 5a877e5d5542993e715abf7d
- method: dense_rag
- gold answer: ['David Weissman']
- retrieved top-5 ids: ['Ungod', 'Season of the Witch (2011 film)', 'City of Angels (film)', 'Gone in 60 Seconds (2000 film)', 'Deadfall (1993 film)']
- model prediction (FINAL): David Weissman

- qid: 5ab3b0bf5542992ade7c6e39
- method: dense_rag
- gold answer: ['1999']
- retrieved top-5 ids: ['Steven Adler', "Hammerjack's", "Guns N' Roses discography", 'Last Action Hero', 'Arnold Schwarzenegger filmography']
- model prediction (FINAL): 1999


### MuSiQue

#### Hit & Wrong (>=5)
- qid: 3hop2__655849_223623_162182
- method: bm25_rag
- gold answer: ['South Central Coast']
- retrieved top-5 ids: ['p0012', 'p0006', 'p0009', 'p0016', 'p0013']
- model prediction (FINAL): The provided text does not mention A Lưới or the birthplace of John Phan.

- qid: 4hop1__436202_765799_282674_759393
- method: bm25_rag
- gold answer: ['Green Bay']
- retrieved top-5 ids: ['p0015', 'p0016', 'p0004', 'p0018', 'p0009']
- model prediction (FINAL): Appleton

- qid: 3hop1__539312_744503_34754
- method: bm25_rag
- gold answer: ['blackmail', 'Blackmail']
- retrieved top-5 ids: ['p0007', 'p0005', 'p0000', 'p0008', 'p0009']
- model prediction (FINAL): The provided information does not mention the birthplace of Princess Yaovabha Bongsanid or any country's assistance to Japan.

- qid: 4hop1__342858_131850_33897_81096
- method: bm25_rag
- gold answer: ['Mario Andretti']
- retrieved top-5 ids: ['p0015', 'p0012', 'p0002', 'p0006', 'p0007']
- model prediction (FINAL): The answer cannot be determined from the provided information.

- qid: 2hop__156700_63853
- method: bm25_rag
- gold answer: ['Thaba Putsoa']
- retrieved top-5 ids: ['p0003', 'p0000', 'p0008', 'p0002', 'p0019']
- model prediction (FINAL): The source of the Orange River is not specified in the given information.

#### Miss & Correct (>=5)
- qid: 3hop2__90098_60649_10557
- method: bm25_rag
- gold answer: ['Medieval Latin']
- retrieved top-5 ids: ['p0017', 'p0015', 'p0014', 'p0000', 'p0002']
- model prediction (FINAL): Latin

- qid: 2hop__21104_16335
- method: bm25_rag
- gold answer: ['inter-marriage and conversions in the time of the Roman Empire']
- retrieved top-5 ids: ['p0017', 'p0002', 'p0009', 'p0019', 'p0018']
- model prediction (FINAL): inter-marriage and conversions in the time of the Roman Empire.

- qid: 2hop__499003_853511
- method: bm25_rag
- gold answer: ['Kanine Records']
- retrieved top-5 ids: ['p0017', 'p0019', 'p0008', 'p0014', 'p0000']
- model prediction (FINAL): Sire Records

- qid: 2hop__82910_75184
- method: bm25_rag
- gold answer: ['Claudia Wells']
- retrieved top-5 ids: ['p0006', 'p0016', 'p0005', 'p0014', 'p0002']
- model prediction (FINAL): Claudia Wells

- qid: 2hop__860241_766393
- method: bm25_rag
- gold answer: ['Nobel Prize in Literature', 'Austrian State Prize for European Literature', 'Prix mondial Cino Del Duca', "Grand Prix du roman de l'Académie française", 'Prix Goncourt']
- retrieved top-5 ids: ['p0010', 'p0017', 'p0012', 'p0002', 'p0005']
- model prediction (FINAL): The information is not available in the provided content.


### MIRAGE

#### Hit & Wrong (>=5)
- qid: f80d260e-da2b-4eeb-b123-4646f089d609
- method: bm25_rag
- gold answer: ['songwriter', 'song writer']
- retrieved top-5 ids: ['6e1c9f55-c77a-4527-8170-e98a869f0f30', 'f80d260e-da2b-4eeb-b123-4646f089d609', 'f80d260e-da2b-4eeb-b123-4646f089d609', 'f80d260e-da2b-4eeb-b123-4646f089d609', 'f80d260e-da2b-4eeb-b123-4646f089d609']
- model prediction (FINAL): musician

- qid: 68d9ab39-5f61-4047-83c6-7e07bba03a83
- method: bm25_rag
- gold answer: ['dentist', 'dentists', 'dental surgeon']
- retrieved top-5 ids: ['68d9ab39-5f61-4047-83c6-7e07bba03a83', '68d9ab39-5f61-4047-83c6-7e07bba03a83', '68d9ab39-5f61-4047-83c6-7e07bba03a83', '68d9ab39-5f61-4047-83c6-7e07bba03a83', '88642e64-82a7-4ca3-b5a2-28d3c2863d4d']
- model prediction (FINAL): Matthew McKay's occupation was a Canadian politician. He served as a Liberal party member of the House of Commons of Canada and held various roles such as Parliamentary Secretary to the Minister of Finance and critic for the Environment. FINAL: Canadian politician.

- qid: 8adda8ac-2a34-457c-8f8d-df128285da7b
- method: bm25_rag
- gold answer: ['politician', 'political leader', 'political figure', 'polit.', 'pol']
- retrieved top-5 ids: ['8adda8ac-2a34-457c-8f8d-df128285da7b', '8adda8ac-2a34-457c-8f8d-df128285da7b', '8adda8ac-2a34-457c-8f8d-df128285da7b', '8adda8ac-2a34-457c-8f8d-df128285da7b', '8adda8ac-2a34-457c-8f8d-df128285da7b']
- model prediction (FINAL): Actor and producer.

- qid: 69891229-9313-405d-a4b7-2d8d9edbad4d
- method: bm25_rag
- gold answer: ['civil engineer']
- retrieved top-5 ids: ['69891229-9313-405d-a4b7-2d8d9edbad4d', '69891229-9313-405d-a4b7-2d8d9edbad4d', '69891229-9313-405d-a4b7-2d8d9edbad4d', '69891229-9313-405d-a4b7-2d8d9edbad4d', '69891229-9313-405d-a4b7-2d8d9edbad4d']
- model prediction (FINAL): cinematographer

- qid: 988d764d-5cb4-4063-9841-b2a10498f8d8
- method: bm25_rag
- gold answer: ['film director', 'movie director', 'director', 'motion picture director', 'actor', 'actress', 'actors', 'actresses']
- retrieved top-5 ids: ['988d764d-5cb4-4063-9841-b2a10498f8d8', '988d764d-5cb4-4063-9841-b2a10498f8d8', '988d764d-5cb4-4063-9841-b2a10498f8d8', '988d764d-5cb4-4063-9841-b2a10498f8d8', '988d764d-5cb4-4063-9841-b2a10498f8d8']
- model prediction (FINAL): Burmese politician and former Minister of Home Affairs of Myanmar.

#### Miss & Correct (>=5)
- qid: 70f2d97b-0fee-4340-9bf2-f9d71db39063
- method: hybrid_rag
- gold answer: ['politician', 'political leader', 'political figure', 'polit.', 'pol']
- retrieved top-5 ids: ['19183d8e-60a7-4d68-a838-3360e352a774', '67cf1847-e02d-4b85-b6f5-5f7a3759e8d7', 'ae40a1dd-0bb4-42ab-9432-961e7e3ab6af', 'd595b916-f9b5-441b-92d3-9df795b1ea53', 'd595b916-f9b5-441b-92d3-9df795b1ea53']
- model prediction (FINAL): Canadian politician and wholesaler.

- qid: 647d126c-564b-46c6-a8a7-6eda23449e10
- method: relrag_full
- gold answer: ['politician', 'political leader', 'political figure', 'polit.', 'pol']
- retrieved top-5 ids: ['Kathy Saltzman', 'Melinda and Melinda', 'Honor Harger', 'Jim Boyd (newscaster)', 'Harry Saltzman']
- model prediction (FINAL): Minnesota politician

- qid: e8acd86c-e651-47fa-8440-2eb759a5f5ea
- method: relrag_full
- gold answer: ['cartoonist', 'graphic artist', 'animator', 'illustrator']
- retrieved top-5 ids: ['Eleanor Davis', 'Eleanor Davis', 'Eleanor Layfield Davis', 'Ai (poet)', 'Honor Harger']
- model prediction (FINAL): cartoonist and illustrator.

- qid: 4d5a3c5c-4b7c-4633-a9f9-a39d9e885344
- method: relrag_full
- gold answer: ['politician', 'political leader', 'political figure', 'polit.', 'pol']
- retrieved top-5 ids: ['William Murray, 1st Earl of Mansfield', 'William Murray, 1st Earl of Mansfield', 'William Murray, 1st Earl of Mansfield', 'William Murray, 1st Earl of Mansfield', 'John Dawson Mayne']
- model prediction (FINAL): judge, politician, lawyer, and peer

- qid: 5bede7e0-5a07-480a-ba01-b2e7adffd4e2
- method: relrag_full
- gold answer: ['actor', 'actress', 'actors', 'actresses']
- retrieved top-5 ids: ['Þorsteinn Gylfason', 'Þorsteinn Bachmann', 'Þorsteinn Pálsson', 'Þorsteinn Gylfason', 'Hate Man']
- model prediction (FINAL): film actor

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
