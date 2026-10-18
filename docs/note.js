// Browser/Node port of groundnote/scribe.py write_note (greedy only), for transformers.js.
// T is the transformers.js module; the ONNX graph returns last-position logits only.

export const STANDARD = ["CHIEF COMPLAINT", "HISTORY OF PRESENT ILLNESS", "PAST MEDICAL HISTORY", "ALLERGIES",
    "PHYSICAL EXAM", "ASSESSMENT", "PLAN"];

const PENALTY = 1.1;   // repetition_penalty, applied over every id so far (transcript included), as in HF generate

export async function writeNote(T, model, tok, dialogue, { headers = STANDARD, maxLen = 3072, maxNew = 400, onToken } = {}) {
    const enc = (s) => Array.from(tok.encode(s, { add_special_tokens: false }));
    const stop = new Set([enc("<section>")[0], tok.eos_token_id]);
    let ids = enc("<dialogue>\n" + dialogue + "\n");
    const budget = maxLen - maxNew * 3;
    if (ids.length > budget) {                       // same head+tail truncation as training
        const head = Math.floor(budget * 3 / 5);
        ids = ids.slice(0, head).concat(ids.slice(ids.length - (budget - head)));
    }
    let cache = null, seq = [];                      // seq: the token ids whose K/V the cache holds
    const out = [];
    for (const h of headers) {
        ids = ids.concat(enc(`<section> ${h}\n`));
        // keep the cache only up to where it agrees with ids; leave >= 1 token uncached to feed
        let k = 0;
        while (k < Math.min(seq.length, ids.length) && seq[k] === ids[k]) k++;
        k = Math.min(k, ids.length - 1);
        if (cache && k < seq.length) cache = crop(T, cache, k);
        let past = cache ? k : 0, feed = ids.slice(past);
        const gen = [];
        for (let step = 0; step < maxNew; step++) {
            const total = past + feed.length;
            const r = await model({
                input_ids: new T.Tensor("int64", BigInt64Array.from(feed, BigInt), [1, feed.length]),
                attention_mask: T.ones([1, total]),
                past_key_values: cache,
            });
            cache = toCache(T, r);
            past = total;
            const next = argmaxPenalized(r.logits.data, ids, gen);
            if (stop.has(next)) break;
            gen.push(next);
            feed = [next];
            onToken?.(h, gen.length);
        }
        seq = ids.concat(gen).slice(0, past);
        const text = gen.length ? tok.decode(gen, { clean_up_tokenization_spaces: false }).trim() : "";
        out.push([h, text]);
        ids = ids.concat(enc(text + "\n"));
    }
    return out;
}

function argmaxPenalized(logits, ids, gen) {
    const v = Float32Array.from(logits);             // [1, 1, V]
    for (const t of new Set([...ids, ...gen])) v[t] = v[t] < 0 ? v[t] * PENALTY : v[t] / PENALTY;
    let best = 0;
    for (let i = 1; i < v.length; i++) if (v[i] > v[best]) best = i;
    return best;
}

function toCache(T, r) {
    const e = {};
    for (const name in r) if (name.startsWith("present")) e[name.replace("present", "past_key_values")] = r[name];
    return new T.DynamicCache(e);
}

function crop(T, cache, n) {
    const e = {};
    for (const name in cache) e[name] = cache[name].slice(null, null, [0, n], null);   // [batch, heads, seq, dim]
    return new T.DynamicCache(e);
}
