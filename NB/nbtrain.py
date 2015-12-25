#! /usr/bin/python
import argparse
import glob
import os
import math
import string
import heapq

class BagOfWords(object):
    """docstring for BagOfWords"""
    def __init__(self, directory):
        self.directory = directory
        self.bag_of_words = {}
        self.files_cnt = 0
        self.processFile()
        self.weights_p = {}
        self.weights_n = {}


    def word_cond(self,w,f):
        return (w not in string.punctuation and not w.isdigit() and (f >= 5))
    
    def processFile(self):
        files =[file for file in glob.glob(self.directory+"/*")]
        self.files_cnt = len(files)
        for f in files:
            with open(f) as f1:
                for w in f1.read().split():
                    self.bag_of_words.setdefault(w,0)
                    self.bag_of_words[w]+=1
        self.bag_of_words = {w:f for w,f in self.bag_of_words.items() if self.word_cond(w,f)}

    def len(self):
        return len(self.bag_of_words)

    def words(self):
        return (list(self.bag_of_words.keys()))

    def total_files(self):
        return self.files_cnt

    def freq(self,word):
        if word in self.bag_of_words:
            return self.bag_of_words[word]
        else:
            return 0


    def probability(self,vocab):
        prob = {w:((self.freq(w)+1)/(self.len()+len(vocab))) for w in vocab}
        return prob
        


def write_model(filename,p_prob,n_prob,p_prior,n_prior):
    weights_p = {}
    weights_n = {}
    with open(filename,"w") as f:
        f.write(str(math.log(p_prior)) + " " + str(math.log(n_prior)) + "\n")
        for word in list(p_prob.keys()):
            weights_p[word] = math.log(p_prob[word]/n_prob[word])
            weights_n[word] = math.log(n_prob[word]/p_prob[word])
            f.write(word+" "+ str(math.log(p_prob[word])) +" "+ str(math.log(n_prob[word])))
            f.write("\n")
        return weights_p,weights_n

def get_filepaths(directory):
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths

def top_20(weights):
    top = heapq.nlargest(20, weights, key=weights.get)
    for w in top:
        print(w+" ")
    print("\n")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Train the Naive Bayes classifier")
    parser.add_argument('td', metavar="Train Data", help="Train data directory")
    parser.add_argument('model', metavar="model", help="Model filename")
    args = parser.parse_args()
    p_bow = BagOfWords(os.path.join(args.td,"pos"))
    n_bow = BagOfWords(os.path.join(args.td,"neg"))
    vocab = list(set(p_bow.words()) | set(n_bow.words()))
    p_prob = p_bow.probability(vocab)
    n_prob = n_bow.probability(vocab)
    p_files_cnt = p_bow.total_files()
    n_files_cnt = n_bow.total_files()
    total_files = p_files_cnt + n_files_cnt
    p_prior = p_files_cnt/total_files
    n_prior = n_files_cnt/total_files
    weights_p,weights_n = write_model(args.model,p_prob,n_prob,p_prior,n_prior)
    print("Top 20 terms with the highest (log) ratio of positive to negative weight")
    top_20(weights_p)
    print("Top 20 terms with the highest (log) ratio of negative to positive weight")
    top_20(weights_n)






            
