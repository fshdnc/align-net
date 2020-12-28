#!/usr/bin/env python3

import numpy
import time

def read_embeddings(embeddings_file):
    with open(embeddings_file, 'rb') as embedding_file:
        embeddings = numpy.load(embedding_file)
    return embeddings

def get_embeddings_batch(embeddings_file, batch_size):
    batch_embeddings = []
    batch_count = 0
    for embedding in read_embeddings(embeddings_file):
        batch_embeddings.append(embedding)
        if len(batch_embeddings) == batch_size:
            batch_embeddings = numpy.stack(batch_embeddings, axis=0)
            start = time.time()
            yield batch_embeddings
            end = time.time()
            batch_count += 1
            print("Batch number %i finished in %f seconds" % (batch_count, end - start)) #, end="\r")
            batch_embeddings = []
    if len(batch_embeddings)!=0:
        batch_embeddings = numpy.stack(batch_embeddings, axis=0)
        start = time.time()
        yield batch_embeddings
        end = time.time()
        batch_count += 1
        print("Batch number %i finished in %f seconds" % (batch_count, end - start)) #, end="\r")
