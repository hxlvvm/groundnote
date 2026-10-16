import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.pretrain import build_tok, docs, pack_docs
from scripts.train import SPECIAL, build_tokenizer

DOCS = ["The knee locked twice last week.", "No, it doesn't. Ice and rest.\nFollow up in two weeks."]


def test_tokenizer_survives_init_path_and_pack_round_trips(tmp_path):
    tok = build_tok(DOCS * 20 + ["<dialogue>\nDoctor: hi\n<section> PLAN\nRest.\n<eos>"], str(tmp_path / "tok"))
    assert tok.convert_tokens_to_ids(SPECIAL) == [0, 1, 2, 3]
    ft = build_tokenizer([], str(tmp_path / "ft"), init=str(tmp_path / "tok"))   # train.py --init path
    assert len(ft) == len(tok) and (ft.pad_token_id, ft.eos_token_id) == (0, 1)

    pq.write_table(pa.table({"text": DOCS}), tmp_path / "a.parquet")
    pack_docs(docs([str(tmp_path / "a.parquet")]), str(tmp_path), val_tokens=3)
    ids = np.concatenate([np.memmap(tmp_path / f, np.uint16, mode="r") for f in ("tokens.bin", "val.bin")])
    assert ids[-1] == 1 and (ids == 1).sum() == len(DOCS)
    assert tok.decode(ids.tolist()) == "<eos>".join(DOCS) + "<eos>"
