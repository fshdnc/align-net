# Data

This folder contains the scripts used for processing data. The data preparation can be divided into two parts: candidate generation (generating potential parallel sentences) and gold standard identification.

## Candidates
### `create_faiss_index.py`
Candidate generation with faiss

## Gold standards
### `candidate_labels.py`
Generate a dictionary for the positive sentences for the tatoeba sets
```
$ python3 candidate_labels.py --source-file /home/lhchan/tatoeba/data/eng-fin/dev.src --target-file /home/lhchan/tatoeba/data/eng-fin/dev.trg --positive-dictionary dev-src-positives.json
```

### `merge_labels.py`
The generated positives and negatives are different (src->trg or trg->src). Merge to create the final list of positives.
```
$ python3 merge_labels.py --source-dictionary dev-src-positives.json --target-dictionary dev-trg-positives.json --merged-dictionary dev-positives.json
```

### `dedup.py`: The src and trg sentences contain duplicates, remove the deduplicates in the src and trg files, and remap the indices of the gold standards.
```
# for the dev set
python3 dedup.py --source-file ../tatoeba/data/eng-fin/dev.src --target-file ../tatoeba/data/eng-fin/dev.trg --positive-dictionary dev-positives.json
```