from groundnote.check import check_note, parse_dialogue


def verdicts(note, dialogue):
    return [c.verdict for c in check_note(note, parse_dialogue(dialogue))]


KNEE = """[doctor] so how is the right knee doing ?
[patient] it hurts when i climb stairs .
[doctor] has it ever locked up on you or caught ?
[patient] no , it doesn't .
[doctor] any swelling ?
[patient] yeah , some swelling after soccer ."""


def test_post_bug_negation_flip_is_caught():
    assert verdicts("HPI: The right knee locked.", KNEE) == ["contradicted"]


def test_correct_negation_is_supported():
    assert verdicts("HPI: Denies locking.", KNEE) == ["supported"]


def test_bare_no_answer():
    d = "Doctor: Any fever?\nPatient: No."
    assert verdicts("Reports fever.", d) == ["contradicted"]
    assert verdicts("No fever.", d) == ["supported"]


def test_bare_yes_answer():
    assert verdicts("Reports swelling.", KNEE) == ["supported"]


def test_not_anymore_and_no_longer():
    d = "Doctor: Do you smoke?\nPatient: Not anymore."
    assert verdicts("Patient smokes.", d) == ["contradicted"]
    d = "Doctor: Are you taking ibuprofen?\nPatient: I used to take ibuprofen but no longer."
    assert verdicts("Takes ibuprofen.", d) == ["contradicted"]


def test_double_negation():
    d = "Doctor: Does your knee lock?\nPatient: I wouldn't say it never locks."
    assert verdicts("Knee locks.", d) == ["supported"]


def test_mid_dialogue_correction_latest_wins():
    d = "Doctor: Has the knee locked?\nPatient: No. Actually, it did lock once last week."
    assert verdicts("Knee locked once last week.", d) == ["supported"]
    assert verdicts("Denies knee locking.", d) == ["contradicted"]


def test_list_negation_each_item():
    d = "Doctor: Any fever or chills?\nPatient: No fever and no chills."
    assert verdicts("Denies fever.", d) == ["supported"]
    assert verdicts("Reports chills.", d) == ["contradicted"]


def test_uncertain_answer_is_not_a_denial():
    d = "Doctor: Any history of gout?\nPatient: I don't know, maybe."
    assert verdicts("History of gout.", d) == ["unclear"]


def test_hallucinated_fact_is_unsupported():
    assert verdicts("Prescribed amoxicillin for otitis media.", KNEE) == ["unsupported"]


def test_section_header_and_split_on_but():
    assert verdicts("Plan: Denies locking but reports swelling.", KNEE) == ["supported", "supported"]


def test_mts_style_speakers():
    d = "Doctor: Any allergies to medicines?\nPatient: Yes, penicillin gives me a rash."
    assert verdicts("Allergies: Penicillin.", d) == ["supported"]


def test_question_inside_unpunctuated_speech():
    d = "[doctor] okay and have you ever injured your knees before\n[patient] no never"
    assert verdicts("Denies prior knee injury.", d) == ["supported"]


def test_short_statement_is_not_a_bare_yes():
    d = "Doctor: Does your knee lock or give way?\nPatient: I have a brace."
    assert verdicts("Denies knee locking.", d) == ["unclear"]
