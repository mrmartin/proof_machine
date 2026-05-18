.PHONY: build test demo clean train train-continuous eval-ood reset-train

build:
	dune build

test: build
	dune runtest
	bash tests/e2e.sh
	python3 tests/test_verify_prefix.py
	python3 tests/synth/test_generator.py

demo: build
	bash tests/e2e.sh

# Resume-aware ExitIt training.  Picks up from hol_expit_ckpt.pt if
# present; otherwise bootstraps from the curated corpus.  Adds
# HOL_EXPIT_SYNTH_PER_ROUND synthetic samples per round (default 2000),
# fsynced into the buffer JSONL so a reboot loses at most the
# in-flight round.  See README's "Resumable continuous training"
# section for the full contract.
train: build
	bash scripts/train_continuous.sh

# Alias for clarity.
train-continuous: train

# Re-evaluate the current checkpoint on the 23-seed test set.
eval-ood: build
	HOL_EXPIT_CKPT_PATH=$(PWD)/hol_expit_ckpt.pt \
	  python3 eval_ood.py --ckpt $(PWD)/hol_expit_ckpt.pt \
	    --methods 1A-T1.0 1B-b \
	    --out runs/eval_ood.csv

# Hard reset: deletes the checkpoint AND the persistent buffer so the
# next `make train` starts from scratch.  Use sparingly.
reset-train:
	rm -f hol_expit_ckpt.pt hol_expit_buffer.jsonl hol_expit_novel.jsonl

clean:
	dune clean
	rm -f examples/*/*.kf examples/*/*.cert examples/*/*.tex
	rm -f examples/*/*.aux examples/*/*.log examples/*/*.pdf
