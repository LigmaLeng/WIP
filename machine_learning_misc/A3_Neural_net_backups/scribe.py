import re, gc, copy, fasttext
import numpy as np
import pandas as pd
from .data import *
from . import one_hot
MAX_SEQ_LEN = 256
EMBED_DIM = 300


class Thoth():
    """
    Data Loading function 
    """
    def __init__(self) -> None:
        r_train = pd.read_csv(Raw.train)
        nulls = r_train[r_train.requirements_and_role.isnull() == True].index.to_list()
        r_train = r_train.drop(nulls).reset_index(drop=True)
        mixed_labels = r_train.salary_bin
        labelled_idx = mixed_labels[mixed_labels.isnull() == False].index
        r_train = r_train[labelled_idx]

        self.datasets = {"train" : r_train,
                         "valid" : pd.read_csv(Raw.valid),
                         "test" : pd.read_csv(Raw.test)}
        
        self.pretrained=None
        self.vocab=None
        self._vocab=None
        self.matrix=None


    def pre_process(self):
        # RUN PREPROCESSING STEP
        print("Cleaning Raw data tokens", end="")
        for _data in self.datasets.values():
            _data["requirements_and_role"] = _data["requirements_and_role"].astype(str)
            _data["cleaned"] = _data["requirements_and_role"].map(self._scrub_text)
            _data["cleaned_wordcount"] = _data["cleaned"].str.count(" ") + 1
        print("\rRaw data cleaned")

        print("Pruning lengthy documents", end="")
        for _data in self.datasets.values():
            _data["pre_processed"] = _data["cleaned"].map(self._prune)
            _data["processed_wordcount"] = _data["pre_processed"].str.count(" ") + 1
        print(f"\rDocuments pruned to max length of: {MAX_SEQ_LEN}")

        vocab = pd.concat([_data.cleaned for _data in self.datasets.values()])
        self.inverse_term_freq = copy.deepcopy([term for term in reversed(vocab.str.split().explode().value_counts().index.to_list())])

        self._vocab = pd.concat([_data.pre_processed for _data in self.datasets.values()]).str.split().explode().value_counts().index.to_list()
        
        print("Dataset vocabulary loaded into object")
    
    # DEFINE FUNCTION FOR PREPROCESS MAPPING 
    def _scrub_text(self, dirty):
        # Removing uncleaned apostrophes and trailing chars
        scrub = re.sub(r"('\w*)", "", dirty)
        # Removing backticks
        scrub = re.sub(r"(`)", "", scrub)
        # Removing Numerics and words attached to Numerics
        scrub = re.sub(r"(\b\w*\d+\w*\b)", "", scrub)
        # Removing single chars and a recurring odd char
        scrub = re.sub(r"(\b[a-z]\b)", "",scrub)
        # Trimming white spaces
        scrub = re.sub(r"\s+", " ", scrub)
        scrub = re.sub(r"^\s*", "", scrub)
        clean = re.sub(r"\s*$", "", scrub)
        return clean
    
    def _prune(self, document):
        tokens = document.split()
        for term in self.inverse_term_freq:
            if len(tokens) <= MAX_SEQ_LEN: break
            while term in tokens:
                tokens.remove(term)
        return " ".join(tokens)
    
    def save_pre_processed(self):
        # export preprocessed 
        ...

    def save_labels(self):
        raw_sets = self.datasets[Raw.formatting]
        mixed_unlabeled = raw_sets[Raw.train].salary_bin
        train_labels = np.asarray(mixed_unlabeled[mixed_unlabeled.isnull() == False].to_numpy()).astype(np.uint8)
        np.save(Raw.train_labels, train_labels)
        print("Training labels saved as: {}".format(Raw.train_labels))

        mixed_unlabeled = raw_sets[Raw.valid].salary_bin
        valid_labels = np.asarray(mixed_unlabeled[mixed_unlabeled.isnull() == False].to_numpy()).astype(np.uint8)
        np.save(Raw.valid_labels, valid_labels)
        print("Validation labels saved as: {}".format(Raw.valid_labels))

    def load_pretrained(self):
        self.pretrained = fasttext.load_model(Custom.pretrained_bin)

    def model_guard(fn):
        def wrapper(self):
            if not self.pretrained: raise TypeError(f"{type(self).__name__}.pretrained attribute not defined; avoid error by first calling load_pretrained() on object")
            return fn(self)
        return wrapper

    def vocab_guard(fn):
        def wrapper(self):
            if not self._vocab: raise TypeError(f"required attribute {type(self).__name__}.vocab not defined; call pre-process() to construct processed dataset vocabulary")
            return fn(self)
        return wrapper

    @vocab_guard
    @model_guard
    def get_vocab(self):
        if self.vocab: return self.vocab
        self.vocab = self.pretrained.get_words()
        self.data_vocab_ids = {}
        pretrained_len = len(self.vocab)
        self.relative_ids = []
        oovs=[]
        for i, term in enumerate(self._vocab):
            if term not in self.vocab:
                self.vocab.append(term)
                oovs.append(term)
            else:
                self.data_vocab_ids[term] = i+1
                self.relative_ids.append(self.pretrained.get_word_id(term))
        non_oov_len = len(self.data_vocab_ids.keys())
        for i, term in enumerate(oovs):
            self.data_vocab_ids[term] = non_oov_len + i + 1
        self.relative_ids.extend([pretrained_len + x for x in range(len(oovs))])
        print(f"saving vocab ids relative to pretrained vectors as: {Custom.relative_id}")
        np.save(Custom.relative_id, np.asarray(self.relative_ids, dtype=np.uint32))
        return self.vocab

    @model_guard
    def get_pretrained_matrix(self):
        if self.matrix: return self.matrix
        self.matrix = np.empty(shape=(len(self.get_vocab()), EMBED_DIM), dtype=np.float32)
        for i, term in enumerate(self.vocab):
            self.matrix[i, :] = self.pretrained.get_word_vector(term)
        return self.matrix

    def embed_ids(self, sample):
        corresponding = {Raw.train:Custom.train_vec, Raw.valid:Custom.valid_vec, Raw.test:Custom.test_vec}
        if sample not in[Raw.train, Raw.valid, Raw.test]:
            raise Exception("invalid sample source")
        print(f"\rVectorising sample from {sample}...", end="")
        df = self.datasets[Raw.formatting][sample].pre_processed
        if sample == Raw.train:
            mixed_labels = self.datasets[Raw.formatting][Raw.train].salary_bin
            labelled_idx = mixed_labels[mixed_labels.isnull() == False].index
            df = df[labelled_idx]
        tokenised = df.split().to_list()
        n_instances = len(tokenised)
        matrix = np.zeros((n_instances, 1, MAX_SEQ_LEN, EMBED_DIM), dtype=np.float32)
        n = 0
        for sequence in tokenised:
            i = 0
            for term in sequence:
                matrix[n, 0, i,:] = np.asarray(self.pretrained.get_word_vector(term), dtype=np.float32)
                i+=1
            n += 1
        print(f"\rSample vectorised, proceeding to save...", end="")
        np.save(corresponding[sample], matrix)
        print(f"\rSample saved to {corresponding[sample]}")
        del matrix
        gc.collect()
        return
    
