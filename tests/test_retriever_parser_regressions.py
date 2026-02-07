from relrag.retriever.parser import parse_question


def _seed_texts(ir):
    return [seed.text for seed in ir.seeds]


def test_pairwise_members_compare_extracts_two_seeds():
    ir = parse_question("Do the bands 3OH!3 and Moonbabies have the same number of members?")
    assert ir is not None
    assert _seed_texts(ir) == ["3OH!3", "Moonbabies"]
    assert [step.pred for step in ir.pred_chain] == ["has_member_count"]


def test_pairwise_comma_compare_extracts_two_entities():
    ir = parse_question("Which magazine was founded first, Allure or Genre?")
    assert ir is not None
    assert _seed_texts(ir) == ["Allure", "Genre"]
    assert [step.pred for step in ir.pred_chain] == ["founded_on"]


def test_member_of_question_uses_inverse_direction():
    ir = parse_question('Which member of the boy group Shinee released their first studio album "She is"?')
    assert ir is not None
    assert _seed_texts(ir) == ["Shinee"]
    assert [step.pred for step in ir.pred_chain] == ["member_of"]
    assert ir.pred_chain[0].direction == "in"


def test_member_of_clause_prefers_subject_entity_on_left():
    ir = parse_question(
        "In what city did Brian Mannal convene when he was a member of the 188th and 189th General Court?"
    )
    assert ir is not None
    assert _seed_texts(ir) == ["Brian Mannal"]
    assert [step.pred for step in ir.pred_chain] == ["member_of"]
    assert ir.pred_chain[0].direction == "out"


def test_starred_in_question_extracts_work_seed():
    ir = parse_question(
        "Which English actor starred in the 1995 american romantic drama film The Scarlet Letter "
        "alongside Demi Moore and Robert Duvall?"
    )
    assert ir is not None
    assert _seed_texts(ir) == ["The Scarlet Letter"]
    assert [step.pred for step in ir.pred_chain] == ["acted_in"]
    assert ir.pred_chain[0].direction == "in"
