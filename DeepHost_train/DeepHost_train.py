"""DeepHost
Usage:
    DeepHost_train.py <fasta_file> <host_file> [--rank=<rn>] [--epoch=<en>] [--split=<sn>] [--thread=<tn>]
    DeepHost_train.py (-h | --help)
    DeepHost_train.py --version

Options:
    -r --rank=<rn>   The predicted taxonomic rank (genus or species) [default: species].
    -e --epoch=<en>   The times that CNN will work through the entire training dataset. [default: 20]. 
    -s --split=<sn>   The proportion of the dataset to include in the test split. [default: 0.1].
    -t --thread=<tn>    The number of worker processes to use [default: 1].
    -h --help   Show this screen.
    -v --version    Show version.
"""

from docopt import docopt
import numpy as np
import time
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import  Convolution2D, Flatten, Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import encode
from multiprocessing import Pool


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

def encode_seq(seq):
    
    return encode.matrix_encoding(seq, 5)

def load_data(fasta_file, host_file, rank, test_size, thread):
    # process label
    host_list = []
    f_in = open(host_file)
    f_out = open("%s_label.txt"%rank, "w")
    count = 0
    for line in f_in:
        host = line.strip("\n")
        if host not in host_list:
            host_list.append(host)
            f_out.write(str(count) + "\t" + host + "\n")
            count += 1
    f_in.close()
    f_out.close()

    f_in = open(host_file)
    label_list = []
    for line in f_in:
        host = line.strip("\n")
        label_list.append(host_list.index(host))
    f_in.close()
    y = np.array(label_list)

    # encode
    print("Encode sequences...")
    seq_collect = read_fasta(fasta_file)
    seq_list = list(seq_collect.values())
    pool = Pool(thread)
    x = pool.map(encode_seq, seq_list)
    pool.close()
    pool.join()
    x = np.array(x)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)

    return x_train, y_train, x_test, y_test


def cnn_classifier(n_classes):

    model = Sequential()
        
    model.add(Convolution2D(filters=100, kernel_size=(2,2), strides=(1,1), padding='same', batch_input_shape=(None, 32, 32, 6), activation='relu'))

    model.add(Convolution2D(filters=100, kernel_size=(2,2), strides=(1,1), padding='same', activation='relu'))

    model.add(Flatten())
    model.add(Dense(500,activation='relu'))

    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    adam = Adam(lr=1e-4)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def generate_batch_data(x,y,batch_size):
    count = -1
    while (True):
        count = count + 1
        count_num = count%(x.shape[0])
        yield x[count_num * batch_size:(count_num + 1) * batch_size], y[count_num * batch_size:(count_num + 1) * batch_size]


def training_process(x_train, y_train, x_test, y_test, rank, epoch_n):
    
    n_classes = max(np.max(y_train), np.max(y_test)) + 1
    x_train = x_train.reshape(-1, 6, 32, 32)
    y_train = np_utils.to_categorical(y_train, num_classes=n_classes)
    x_test = x_test.reshape(-1, 6, 32, 32)
    y_test = np_utils.to_categorical(y_test, num_classes=n_classes)
    x_train = np.moveaxis(x_train, 1, 3)
    x_test = np.moveaxis(x_test, 1, 3)
    start_time = time.time()
    model = cnn_classifier(n_classes)
    model.summary()
    model.fit_generator(generate_batch_data(x_train, y_train, 1), epochs=epoch_n, steps_per_epoch=len(y_train)//1)
    if rank == "genus":
        model.save("CNN_genus_model.h5")
    elif rank == "species":
        model.save("CNN_species_model.h5")

    loss,accuracy = model.evaluate(x_test,y_test)
    print('testing accuracy: {}'.format(accuracy))
    print('training took %fs'%(time.time()-start_time))

def main(arguments):
    fasta_file = arguments.get("<fasta_file>")
    host_file = arguments.get("<host_file>")
    rank = arguments.get("--rank")
    epoch_n = int(arguments.get("--epoch"))
    test_size  = float(arguments.get("--split"))
    thread = int(arguments.get("--thread"))

    x_train, y_train, x_test, y_test = load_data(fasta_file, host_file, rank, test_size, thread)
    training_process(x_train, y_train, x_test, y_test, rank, epoch_n)


if __name__ == '__main__':
    arguments = docopt(__doc__, version="DeepHost 0.1.1")
    main(arguments)
