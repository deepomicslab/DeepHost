"""DeepHost
Usage:
    DeepHost.py <fasta_file> [--out=<fn>] [--rank=<rn>]
    DeepHost.py (-h | --help)
    DeepHost.py --version

Options:
    -o --out=<fn>   The output file name [default: DeepHost_output.txt].
    -r --rank=<rn>   The predicted taxonomic rank (genus or species) [default: species].
    -h --help   Show this screen.
    -v --version    Show version.
"""

from docopt import docopt
import numpy as np
import keras
import warnings
import encode
warnings.filterwarnings("ignore")

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
    label = {}
    for line in f:
        line = line.strip("\n").split("\t")
        label[int(line[0])] = line[1]
    f.close()
    return label

def main(arguments):
    fasta_file = arguments.get("<fasta_file>")
    output_file = arguments.get("--out")
    tax_rank = arguments.get("--rank")
    print(output_file) 
    print(tax_rank) 
    seq_collect = read_fasta(fasta_file)
    feature_list = []
    for acc in seq_collect:
        seq = seq_collect[acc]
        feature_list.append(encode.matrix_encoding(seq, 5))
    
    feature_test = np.array(feature_list).reshape(-1, 6, 32, 32)
    feature_test = np.moveaxis(feature_test, 1, 3)
    if tax_rank == "genus":
        model = keras.models.load_model("CNN_genus_model.h5")
        label = load_label("genus_label.txt")
    elif tax_rank == "species":
        model = keras.models.load_model("CNN_species_model.h5")
        label = load_label("species_label.txt")
    else:
        print("Error: The taxonomy rank should be genus or species.")
    predictions = model.predict_classes(feature_test)
    f_out = open(output_file, "w")
    count = 0
    for acc in seq_collect:
        f_out.write(acc)
        f_out.write("\t")
        f_out.write(label[predictions[count]])
        f_out.write("\n")
        count += 1
    f_out.close()
    

if __name__=="__main__":
    arguments = docopt(__doc__, version="DeepHost 0.1.0")
    main(arguments)
