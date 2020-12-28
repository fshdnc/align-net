#!/usr/bin/env python3

import sys
import os
import numpy
import h5py
sys.path.append("/home/akeele/faiss/python/") # for the faiss module
import faiss
import argparse
from tools import *

# BERT
DIMENSIONS = 768
# LASER: 1024

#def read_embeddings(embeddings_file):
#    with open(embeddings_file, 'rb') as embedding_file:
#        file_size = os.fstat(embedding_file.fileno()).st_size
#        while embedding_file.tell() < file_size:
#            embedding = numpy.fromfile(embedding_file, numpy.float32, DIMENSIONS)
#            yield embedding

#def get_embeddings_batch(embeddings_file, batch_size):
#    batch_embeddings = []
#    batch_count = 0
#    for embedding in read_embeddings(embeddings_file):
#        batch_embeddings.append(embedding)
#        if len(batch_embeddings) == batch_size:
#            batch_embeddings = numpy.stack(batch_embeddings, axis=0)
#            yield batch_embeddings
#            batch_count += 1
#            print("Batch number %i finished" % batch_count, end='\r')
#            batch_embeddings = []


def add_embeddings_to_index(embeddings, gpu_index, batch_size):
    for embeddings_batch in get_embeddings_batch(embeddings, batch_size):
        faiss.normalize_L2(embeddings_batch)
        gpu_index.add(embeddings_batch)

def my_index_cpu_to_gpu_multiple(resources, index, co=None, gpu_now=None):
    '''
    https://gist.github.com/mdouze/bfa06e7dc0869f0c0495928aab25800f
    '''
    vres = faiss.GpuResourcesVector()
    vdev = faiss.IntVector()
    if gpu_nos is None:
        gpu_nos = range(len(resources))
    for i, res in zip(gpu_nos, resources):
        vdev.push_back(i)
        vres.push_back(res)
    index = faiss.index_cpu_to_gpu_multiple(vres, vdev, index, co)
    index.referenced_objects = resources
    return index
        
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--embedding-file", help="File containing sentence embeddings", required=True)
    argparser.add_argument("--index-name", help="Index name to store", required=True)
    argparser.add_argument("--batch-size", help="Batch of embeddings to add to index", required=True)
    argparser.add_argument("--training-size", type=int, help="Number of training embeddings", required=True, default=1500000)
    argparser.add_argument("--gpu", type=int, help="0 CPU only, 1 use GPU", required=True)
    arguments = argparser.parse_args()

    #training_embeddings = numpy.fromfile(arguments.embedding_file, numpy.float32, training_size*DIMENSIONS)
    with open(arguments.embedding_file, "rb") as f:
        training_embeddings = numpy.load(f)
    #training_embeddings.resize(training_embeddings.shape[0] // DIMENSIONS, DIMENSIONS)
    #with h5py.File(arguments.embedding_file, 'r') as f:
    #    training_embeddings = numpy.array(f["data_emb"])

    print("%i training embeddings..." % training_embeddings.shape[0])
    
    faiss.normalize_L2(training_embeddings)
    print("Training embeddings normalized...")
    
    index_cpu = faiss.index_factory(DIMENSIONS, "OPQ32_128,IVF1584,PQ32") #"OPQ32_128,IVF32768,PQ32")
    if arguments.gpu == 1:
        index_gpu = faiss.index_cpu_to_all_gpus(index_cpu)
        index = index_gpu
    else:
        index = index_cpu
    print("Index created...")


    batch_size = int(arguments.batch_size)
    index.train(training_embeddings)
    print("Index trained", index.is_trained)
    add_embeddings_to_index(arguments.embedding_file, index, batch_size)
    print("%i embeddings in index" % index.ntotal)


    # Store index to a file
    if arguments.gpu == 1:
        index = faiss.index_gpu_to_cpu(index)
    print(index.ntotal) # the size of the dataset?
    faiss.write_index(index, arguments.index_name)
