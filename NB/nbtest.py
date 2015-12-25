import glob
import os
import argparse

class Classifier(object):
    """docstring for Classifier"""
    def __init__(self, model):
        super(Classifier, self).__init__()
        self.model = {}
        self.p_prior,self.n_prior = 0,0
        self.p_files = {}
        self.n_files = {}
        self.scores = {}
        self.load_model(model)
        
    def load_model(self,filename):
         with open(filename) as f:
            self.p_prior,self.n_prior = f.readline().split()
            self.p_prior = float(self.p_prior)
            self.n_prior = float(self.n_prior)
            for e in f:
                e = e.split()
                self.model[e[0]] = e[1:]
    
    def classify(self,directory):
        files = get_filepaths(directory)
        self.scores = {file:self.get_scores(file) for file in files}
        return self.scores

    def get_scores(self,file):
        p_score = 0
        n_score = 0
        with open(file) as f:
            for w in f.read().split():
                if w in self.model:
                    p_score+= float(self.model[w][0])
                    n_score+= float(self.model[w][1])
            p_score+=self.p_prior
            n_score+=self.n_prior
            return [p_score,n_score]

    def classified_files(self):
        for f in list(self.scores.keys()):
            if(float(self.scores[f][0]) > float(self.scores[f][1])):
                self.p_files[f] = self.scores[f]
            else:
                self.n_files[f] = self.scores[f]
        return self.p_files,self.n_files

def write_prediction(filename,prediction):
    with open(filename+".csv","w") as f:
        f.write("Filename,Positive_score,Negative_Score"+"\n")
        for w in list(prediction.keys()):
            f.write(w + "," + str(prediction[w][0]) +","+ str(prediction[w][1])+ "\n")

def accuracy(prediction,testSet):
    correct = 0
    for f in prediction:
        if os.path.basename(f) in testSet:
            correct +=1
    return (correct/float(len(testSet))) * 100.0

def get_filepaths(directory):
    file_paths = []  # List which will store all of the full filepaths.

    # Walk the tree.
    for root, directories, files in os.walk(directory):
        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)  # Add it to the list.

    return file_paths

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Test the Naive Bayes classifier")
    parser.add_argument('model', metavar="model", help="Model filename")
    parser.add_argument('td', metavar="td", help="Test data directory")
    parser.add_argument('pfname', metavar="prediction", help="Prediction filename")
    args = parser.parse_args()
    m = Classifier(args.model)
    prediction = m.classify(args.td)
    write_prediction(args.pfname,prediction)
    p_files,n_files = m.classified_files()
    # testSet_p = os.listdir(os.path.join(args.td,"pos"))
    # print(accuracy(list(p_files.keys()),testSet_p))
    # testSet_n = os.listdir(os.path.join(args.td,"neg"))
    # print(accuracy(list(n_files.keys()),testSet_n))
    