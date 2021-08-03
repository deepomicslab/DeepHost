"""DeepHost
Usage:
    DeepHost.py <fasta_file> [--out=<fn>] [--rank=<rn>] [--bacterial=<bn>] [--multiple=<mn>] [--thread=<tn>]
    DeepHost.py (-h | --help)
    DeepHost.py --version

Options:
    -o --out=<fn>   The output file name [default: DeepHost_output.txt].
    -r --rank=<rn>   The predicted taxonomic rank (genus or species) [default: species].
    -b --bacterial=<bn>   The bacterial list from the meta sampling [default: None].
    -m --multiple=<mn>   Return the probabilities of all the hosts for multiple host analysis [default: False].
    -t --thread=<tn>    The number of worker processes to use [default: 1].
    -h --help   Show this screen.
    -v --version    Show version.
"""

from docopt import docopt
import numpy as np
import keras
import warnings
import encode
from multiprocessing import Pool
warnings.filterwarnings("ignore")
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def read_fasta(fasta_file):
    f = open(fasta_file)
    seq = {}
    for line in f:
        if line.startswith(">"):
            acc = line.strip("\n").strip(">")
            seq[acc] = ""
        else:
            seq[acc] += line.strip("\n")
    f.close()

    return seq

def load_label(file_name):
    f = open(file_name)
    label_dic = {}
    for line in f:
        line = line.strip("\n").split("\t")
        label_dic[int(line[0])] = line[1]
    f.close()
    return label_dic

def make_prediction_classes(model, feature_test, seq_collect, label_dic, output_file):
    prediction_classes = model.predict_classes(feature_test)
    f_out = open(output_file, "w")
    count = 0
    for acc in seq_collect:
        f_out.write(acc)
        f_out.write("\t")
        f_out.write(label_dic[prediction_classes[count]])
        f_out.write("\n")
        count += 1
    f_out.close()
    
    return prediction_classes

def make_prediction_prob(model, feature_test, seq_collect, label_dic, output_file, meta_host_list):
    prediction_pro = model.predict(feature_test)
    prediction_classes = []
    for each in prediction_pro:
        pre_dic = dict(zip(list(label_dic.values()), each))
        pre_dic_sort = dict(sorted(pre_dic.items(), key = lambda kv:(kv[1], kv[0]), reverse=True))
        for host in pre_dic_sort:
            if host in meta_host_list:
                prediction_classes.append(host)
                break
    f_out = open(output_file, "w")
    count = 0
    for acc in seq_collect:
        f_out.write(acc)
        f_out.write("\t")
        f_out.write(prediction_classes[count])
        f_out.write("\n")
        count += 1
    f_out.close()
    
    return prediction_pro

def multi_host_analysis(model, feature_test, seq_collect, label_dic, output_file):
    prediction_pro = model.predict(feature_test)
    acc_list = list(seq_collect.keys())
    f_out = open(output_file, "w")
    count = 0
    for each in prediction_pro:
        f_out.write(acc_list[count] + "\t")
        pre_dic = dict(zip(list(label_dic.values()), each))
        pre_dic_sort = dict(sorted(pre_dic.items(), key = lambda kv:(kv[1], kv[0]), reverse=True))
        for host in pre_dic_sort:
            f_out.write(host + ":" + str(pre_dic_sort[host]) + "\t")
        f_out.write("\n")
        count += 1
    f_out.close()

    return prediction_pro

def encode_seq(seq):
    return encode.matrix_encoding(seq, 5)

def main(arguments):
    fasta_file = arguments.get("<fasta_file>")
    output_file = arguments.get("--out")
    tax_rank = arguments.get("--rank")
    meta_host_file = arguments.get("--bacterial")
    return_multi = arguments.get("--multiple")
    thread = int(arguments.get("--thread"))
    seq_collect = read_fasta(fasta_file)
    feature_list = []
    
    print("Encode sequences...")
    seq_list = list(seq_collect.values())
    pool = Pool(thread)
    feature_list = pool.map(encode_seq, seq_list) 
    pool.close()
    pool.join()

    print("make predictions...")
    feature_test = np.array(feature_list).reshape(-1, 6, 32, 32)
    feature_test = np.moveaxis(feature_test, 1, 3)
    if tax_rank == "genus":
        model = keras.models.load_model("CNN_genus_model.h5")
        label_dic = load_label("genus_label.txt")
    elif tax_rank == "species":
        model = keras.models.load_model("CNN_species_model.h5")
        label_dic = load_label("species_label.txt")
    else:
        raise ValueError("Error: The taxonomy rank should be genus or species.")
   
    if return_multi != "False":
        multi_host_analysis(model, feature_test, seq_collect, label_dic, output_file)

    else:
        if meta_host_file == "None":
            make_prediction_classes(model, feature_test, seq_collect, label_dic, output_file)
        else:
            f = open(meta_host_file)
            meta_host_list = []
            for line in f:
                meta_host_list.append(line.strip("\n"))
            f.close()
            make_prediction_prob(model, feature_test, seq_collect, 
                    label_dic, output_file, meta_host_list)
    
    print("Finish. The results can be found on %s. Thanks for using DeepHost."%output_file)

if __name__=="__main__":
    arguments = docopt(__doc__, version="DeepHost 0.1.1")
    main(arguments)
