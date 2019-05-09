import random

import torch
import torchtext
from torchtext import data
from torchtext.vocab import GloVe, Vectors


class load_data():
    """
    Class to handle text data preparation using Torchtext.
    This class allows to preprocess text data from a .csv file and
    to generate batches of train, validation and test data using 
    dynamic padding. It is also possible to load pre-trained
    embedding vectors (300-d GloVe vectors). Note that this class 
    assumes that the .csv file contains a header. Special tokens
    are created to build the vocabulary (i.e. <ukn>, <bos>, <eos>, <pad>)
    and a word has to be present at least 5 times in the corpus to be
    part of the vocabulary.

    Example usage:
    data = load_data(path, 
                     use_embedding, 
                     batch_size, 
                     split_ratio, 
                     seed)
    train_iter, val_iter, test_iter = data.create_datasets()
    """

    def __init__(self, 
                 path, 
                 use_embedding=True, 
                 batch_size=(128, 256, 256), 
                 split_ratio=[0.6, 0.2, 0.2], 
                 seed=True):
        """
        Arguments:
        - path: (string) path to the csv file.
        - use_embedding: (Boolean) if True, load pre-trained 300-d
          GloVe vectors.
        - batch_size: (Tuple[Int]) tuple giving the batch size for train, 
          validation and test datasets.
        - split_ratio: (List[Float]) tuple giving splits ratio for train,
          validation and test datasets. Each float is a float between 0 and 1.
        - seed: (Boolean) if True, use seed for reproductibility.
        """
        self.path = path
        self.use_embedding = use_embedding
        self.train_batch_size = batch_size[0]
        self.val_batch_size = batch_size[1]
        self.test_batch_size = batch_size[2]
        self.split_ratio = split_ratio
        self.seed = seed

        self._select_device()


    def _select_device(self, verbose=True):
        """
        Select device e.g. CPU / GPU
        """
        use_gpu = False
        if use_gpu and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
        if verbose:
            print('Using device:', device)

        self.device = device


    def create_datasets(self):
        """
        Load data, build vocabulary and create Iterator objects
        for train, validation and test data.

        Returns:
        - train_iter : Iterator object for train batches of size self.train_batch_size 
          to iterate over.
        - val_iter : Iterator object for val batches of size self.val_batch_size 
          to iterate over.
        - test_iter : Iterator object for test batches of size self.test_batch_size 
          to iterate over.
        """
        if self.seed:
            random.seed(14)

        # Create fields    
        tokenizer = lambda x: x.split()
        ID = data.Field()
        TEXT = data.Field(tokenize=tokenizer, init_token='<bos>', eos_token='<eos>', lower=True)
        TARGET = data.LabelField(dtype=torch.float)
        train_fields = [('id', None), ('text', TEXT), ('target', TARGET)]

        # Data
        train_data = data.TabularDataset(
            path=self.path,
            format='csv',
            skip_header=True,
            fields=train_fields
        )

        # Split
        train, val, test = train_data.split(split_ratio=[0.6, 0.2, 0.2], random_state=random.getstate())

        # Vocab
        if self.use_embedding:
            TEXT.build_vocab(train_data, vectors=GloVe(name='6B', dim=300), min_freq=5)
            self.embedding_matrix = TEXT.vocab.vectors
        else:
            TEXT.build_vocab(train_data, min_freq=5)
        TARGET.build_vocab(train_data)

        # Iterators
        train_iter = data.BucketIterator(
            train,
            sort_key=lambda x: len(x.text),  # sort sequences by length (dynamic padding)
            batch_size=self.train_batch_size,  # batch size
            device=self.device  # select device (e.g. CPU)
        )

        val_iter = data.BucketIterator(
            val,
            sort_key=lambda x: len(x.text),
            batch_size=self.val_batch_size,
            device=self.device
        )

        test_iter = data.Iterator(
            test,
            batch_size=self.test_batch_size,
            device=self.device,
            train=False,
            sort=False,
            sort_within_batch=False
        )

        return train_iter, val_iter, test_iter
        

    def get_embedding(self):
        """
        Get embedding matrix if used pre-trained GloVe
        vectors
        """
        if self.use_embedding:
            return self.embedding_matrix
        else:
            print("No pre-trained vectors used.")
