"""Generate synthetic encounters with an open-weight teacher via vLLM: facts (with present/absent status) first,
then a dialogue that expresses every fact, then a sectioned note that uses only those facts.

usage: python -m scripts.gen_synth MODEL N SEED OUT.jsonl
Facts-first gives gold polarity labels, so the student's negation faithfulness can be scored without trusting the checker.
"""
import json
import random
import sys


CHUNK = 200                                            # prompts per vLLM call; results are flushed after each
SECTIONS = ["CHIEF COMPLAINT", "HISTORY OF PRESENT ILLNESS", "PAST MEDICAL HISTORY", "ALLERGIES",
            "PHYSICAL EXAM", "ASSESSMENT", "PLAN"]
COMPLAINTS = """knee pain after a twisting injury|knee pain with clicking|low back pain|neck pain after a car accident|shoulder pain
reaching overhead|wrist injury after a fall|ankle sprain|hip pain when walking|heel pain in the morning|elbow pain from tennis|chest
pain on exertion|palpitations|shortness of breath on stairs|leg swelling|high blood pressure follow-up|diabetes follow-up|high
cholesterol follow-up|thyroid follow-up|fatigue|unintended weight loss|weight gain|headache|migraine|dizziness|fainting episode|numbness
in the hands|tremor|memory concerns|low mood|anxiety|trouble sleeping|cough for three weeks|sore throat|ear pain|sinus congestion|
asthma flare|COPD follow-up|fever and body aches|urinary burning|blood in the urine|kidney stone pain|flank pain|abdominal pain|heartburn|
nausea and vomiting|diarrhea|constipation|rectal bleeding|difficulty swallowing|rash on the arms|itchy scalp|acne|a changing mole|
skin infection on the leg|hair loss|eye redness|blurry vision|hearing loss|ringing in the ears|nosebleeds|dental pain|jaw pain|
pelvic pain|irregular periods|breast lump|erectile dysfunction|testicular pain|prostate symptoms|joint pain in the hands|gout flare|
foot ulcer in a diabetic|wound check after surgery|post-operative knee replacement visit|medication refill visit|annual physical|
pre-operative evaluation|travel vaccination visit|allergic reaction to food|seasonal allergies|bee sting reaction|anemia follow-up|
easy bruising|swollen lymph node|night sweats|cold intolerance|excessive thirst|frequent urination|smoking cessation|alcohol use
concerns|follow-up after an emergency visit|concussion after a sports injury|carpal tunnel symptoms|sciatica|plantar fasciitis|
shin pain in a runner|frozen shoulder|neck lump|hoarseness|snoring and daytime sleepiness|restless legs|leg cramps|varicose veins|
calf pain after a flight|coughing up blood|wheezing in a child|ear infection in a child|fever in a toddler|bedwetting|teen sports
physical|pregnancy nausea|postpartum check|menopause symptoms|osteoporosis screening|fall at home in an older adult""".replace("\n", " ").split("|")
COMPLAINTS = [c.strip() for c in COMPLAINTS]
SETTINGS = ["primary care clinic", "orthopedic clinic", "urgent care", "telehealth video visit", "cardiology clinic",
            "dermatology clinic", "pediatric clinic", "emergency department", "women's health clinic", "sports medicine clinic"]
HARD = ["the patient answers several yes/no questions with a bare 'No.' or 'Nope.'",
        "the patient first denies something, then corrects themselves later ('actually, it did happen once')",
        "the patient says a symptom used to happen but 'not anymore'",
        "the patient volunteers a negative without being asked ('and nothing else is bothering me')",
        "the patient is unsure about one item ('I don't remember', 'maybe')",
        "the patient uses lay words for symptoms ('gives out', 'pins and needles', 'winded')",
        "the patient rambles and includes small talk and filler words ('um', 'like')",
        "a family member joins and adds one detail"]

SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {"type": "array", "minItems": 8, "maxItems": 24, "items": {
            "type": "object",
            "properties": {"section": {"enum": SECTIONS}, "fact": {"type": "string"}, "status": {"enum": ["present", "absent", "uncertain"]}},
            "required": ["section", "fact", "status"]}},
        "dialogue": {"type": "array", "minItems": 16, "maxItems": 70, "items": {
            "type": "object",
            "properties": {"speaker": {"enum": ["doctor", "patient", "family"]}, "text": {"type": "string"}},
            "required": ["speaker", "text"]}},
        "note": {"type": "object", "properties": {s: {"type": "string"} for s in SECTIONS}, "required": SECTIONS},
    },
    "required": ["facts", "dialogue", "note"],
}


# complaint -> (sex or None, min age, max age); everything else is an adult visit (18-85, either sex)
LIMITS = {"erectile dysfunction": ("male", 25, 85), "testicular pain": ("male", 12, 70), "prostate symptoms": ("male", 50, 90),
          "pelvic pain": ("female", 16, 70), "irregular periods": ("female", 13, 50), "breast lump": ("female", 25, 85),
          "pregnancy nausea": ("female", 18, 42), "postpartum check": ("female", 18, 42),
          "menopause symptoms": ("female", 45, 60), "osteoporosis screening": (None, 55, 90),
          "wheezing in a child": (None, 2, 10), "ear infection in a child": (None, 1, 8), "fever in a toddler": (None, 1, 3),
          "bedwetting": (None, 5, 10), "teen sports physical": (None, 13, 18), "acne": (None, 13, 30),
          "fall at home in an older adult": (None, 70, 95), "post-operative knee replacement visit": (None, 55, 85),
          "memory concerns": (None, 60, 90)}


def prompt(rng):
    complaint = rng.choice(COMPLAINTS)
    sex, lo, hi = LIMITS.get(complaint, (None, 18, 85))
    age, sex = rng.randint(lo, hi), sex or rng.choice(["male", "female"])
    hard = rng.sample(HARD, 3)
    return f"""Create one realistic, fully fictional outpatient encounter for training a clinical documentation model.
Patient: {age}-year-old {sex}. Setting: {rng.choice(SETTINGS)}. Reason for visit: {complaint}.

Step 1 "facts": 12-20 atomic clinical facts across the note sections. Present facts must outnumber absent facts at
least 2 to 1: use 3-6 "absent" facts, each a pertinent negative a clinician would actually ask about for THIS
complaint (never a list of unrelated diseases). At most 1 fact is "uncertain". Phrase every fact WITHOUT negation
words; the status alone carries the polarity: a fact "<symptom>" with status "absent", never "no <symptom>".
Step 2 "dialogue": a natural doctor-patient conversation (25-50 turns) in which EVERY fact is stated or clearly
implied, with the polarity given by its status. Exam findings are spoken aloud by the doctor. Make it realistic: {"; ".join(hard)}.
Step 3 "note": a concise clinical note in standard medical style for each section, using ONLY the facts. Write
pertinent negatives explicitly ("Denies <symptom>."). If a section has no facts, write "None reported." or "Not examined."
Every detail in the note (age, numbers, vital signs, doses, durations, follow-up times) must be spoken in the dialogue;
if nobody says it, leave it out of the note. Someone says the patient's age aloud.
Use invented names only. Never include real people, real phone numbers or real identifiers.

Return one JSON object with exactly this shape (no other keys):
{{"facts": [{{"section": "<one of the section names below>", "fact": "...", "status": "present|absent|uncertain"}}],
 "dialogue": [{{"speaker": "doctor|patient|family", "text": "..."}}],
 "note": {{{", ".join(f'"{h}": "..."' for h in SECTIONS)}}}}}"""


def valid(rec):
    """Schema check that also applies when the teacher was not grammar-constrained (gpt-oss)."""
    try:
        return (all(isinstance(rec["note"].get(h), str) and rec["note"][h].strip() for h in SECTIONS)
                and len(rec["dialogue"]) >= 10 and all({"speaker", "text"} <= set(t) for t in rec["dialogue"])
                and len(rec["facts"]) >= 6 and all(f.get("status") in ("present", "absent", "uncertain") and f.get("fact")
                                                   for f in rec["facts"]))
    except (AttributeError, KeyError, TypeError):
        return False


def parse(text):
    """Return the JSON object in the answer; reasoning models may emit analysis text before it."""
    dec, i = json.JSONDecoder(), text.find("{")
    while i >= 0:
        try:
            obj = dec.raw_decode(text[i:])[0]
            if isinstance(obj, dict) and "facts" in obj:
                return obj
        except json.JSONDecodeError:
            pass
        i = text.find("{", i + 1)
    return None


def main(model, n, seed, out):
    rng = random.Random(seed)
    prompts = [[{"role": "user", "content": prompt(rng)}] for _ in range(n)]
    if model.startswith("gemini"):
        return main_gemini(model, prompts, seed, out)
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    reasoning = "gpt-oss" in model                         # grammar would also constrain its analysis channel
    import torch
    llm = LLM(model=model, max_model_len=10240, seed=seed, tensor_parallel_size=torch.cuda.device_count())
    params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=7000 if reasoning else 5000, seed=seed,
                            structured_outputs=None if reasoning else StructuredOutputsParams(json=SCHEMA))
    if reasoning:
        prompts = [[{"role": "system", "content": "Reply with the JSON object only."}] + p for p in prompts]
    kw = {"reasoning_effort": "low"} if reasoning else {"enable_thinking": False}
    ok = toks = 0
    with open(out, "w") as f:
        for start in range(0, n, CHUNK):                   # write per chunk: a timed-out job keeps finished work
            res = llm.chat(prompts[start:start + CHUNK], params, chat_template_kwargs=kw)
            for j, r in enumerate(res):
                i = start + j
                toks += len(r.outputs[0].token_ids)
                rec = parse(r.outputs[0].text)
                if not valid(rec):
                    with open(out + ".rejects", "a") as rj:  # keep raw text so a failing teacher can be diagnosed
                        rj.write(json.dumps({"i": i, "text": r.outputs[0].text, "finish": r.outputs[0].finish_reason}) + "\n")
                    continue                               # truncated or malformed; counted as a failure below
                rec.update(id=f"syn{seed}-{i:06d}", prompt=prompts[i][-1]["content"], teacher=model)
                f.write(json.dumps(rec) + "\n")
                ok += 1
            f.flush()
            print(f"{min(start + CHUNK, n)}/{n} prompts, {ok} valid, {toks} output tokens", flush=True)
    print(f"{ok}/{n} valid records, {toks} output tokens -> {out}")


def main_gemini(model, prompts, seed, out, pause=7.0):
    """Same prompts through the Gemini API (free tier, paced). Same validation as the open models."""
    import time
    from scripts.judge import ask, key
    k, ok, misses = key(), 0, 0
    with open(out, "w") as f:
        for i, p in enumerate(prompts):
            rec = ask(model, k, p[-1]["content"], temperature=0.8)
            misses = 0 if rec else misses + 1
            if misses >= 2:                                # daily free quota (20 requests) used up: stop, keep what we have
                print("quota exhausted, stopping", flush=True)
                break
            if valid(rec):
                rec.update(id=f"syn{seed}-{i:06d}", prompt=p[-1]["content"], teacher=model)
                f.write(json.dumps(rec) + "\n")
                f.flush()                                  # a killed job keeps what it already paid for
                ok += 1
            print(f"{i + 1}/{len(prompts)} done, {ok} valid", flush=True)
            time.sleep(pause)
    print(f"{ok}/{len(prompts)} valid records -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
