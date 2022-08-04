# DeepHost
DeepHost is a phage host prediction tool.

## Prerequisite
DeepHost is implemented with Python3.8 and keras2.4.3. Following packages should be installed.
+ cython
+ numpy
+ keras


Please install Git LFS and use the following command to clone the repository.
```shell
git lfs clone https://github.com/deepomicslab/DeepHost.git
```
if you can not download DeepHost from github, please use the link: https://drive.google.com/drive/folders/1EXAoorQI-SEwfX-WNGuAFkfNqv_zSXWJ?usp=sharing or contact the maintainer.

Before using DeepHost, please build the Cython file with the command:
```shell
cd DeepHost_scripts
python setup.py build_ext --inplace
cd DeepHost_train
python setup.py build_ext --inplace
```

## Usage
```shell
cd DeepHost_scripts
python DeepHost.py Phage_genomes.fasta --out Output_name.txt --rank species
```
+ Phage\_genomes.fasta contains phage genome sequences in fasta format. DeepHost supports both single genome and multiple genomes in one file. 
+ The input of --out is the filename of the output file, and the default filename is DeepHost\_output.txt. 
+ The input of --rank is the taxonomic rank of predictions, which accepts genus and species (default).
+ The input of --thread is the number of worker processes to use for genome encoding (default:1). 

For example:
```shell
cd DeepHost_scripts
python DeepHost.py ../example/test_data.fasta --thread 10
```

For more information, please use the command:
```shell
python DeepHost.py -h
```
or
```shell
python DeepHost.py --help
```

### Meta data processing
If the phages are digging from the metagenome, we provide a pipeline to use the bacterial genomes to increase the prediction accuracy.

First, use Kraken2 to perform taxonomic classification for all the bacterial sequences with the following command. Kraken\_DB is the name of the Kraken bacterial database. Seq.fasta is the file contains the sequences assembled from meta sequencing data.
```shell
kraken2 --db Kraken_DB  --report report_file.txt seq.fasta
```
From the report files, users can find the bacterial taxonomics. Then taxonomic names should be collected in a file (like host\_species.txt under example/meta/). Note: Please check that all the taxonomic names are scientific names. 

```shell
cd DeepHost_scripts
python DeepHost.py ../example/meta/meta_phage.fasta --bacterial ../example/meta/host_species.txt --thread 10
```

### The probability of all the hosts
In default case, DeepHost will return the most likely host for the input phages. If you want to obtain the probability of all the host taxonomies (72 genus taxonomies or 118 species taxonomies), please use the parameter --multiple True.

For example:
```shell
cd DeepHost_scripts
python DeepHost.py ../example/test_data.fasta --multiple True --thread 10
```

### Train customized models
In case there are some private datasets, DeepHost provides a user-friendly script for users to train their customized models.

```shell
cd DeepHost_train
python DeepHost_train.py phage_genomes.fasta host_information_file.txt --rank species
```
+ phage\_genomes.fasta contains phage genome sequences in fasta format. 
+ host\_information\_file.txt contains the host information (genus or species taxnomomies), which should have the same order with fasta file. 
+ The input of --rank is the taxonomic rank, which accepts genus and species (default). 
+ The input of --epoch is the times that CNN will work through the entire training dataset (default:20). 
+ The input of --split is the proportion of the dataset to include in the test split for validation (default:0.1). 
+ The input of --thread is the number of worker processes to use for genome encoding (default:1). 

For more information, please use the command:
```shell
python DeepHost_train.py -h
```
or
```shell
python DeepHost_train.py --help
```

After training, users can find the trained model CNN\_genus\_model.h5 or CNN\_species\_model.h5 and genus\_label.txt or species\_label.txt under the working fold. Users can replace the model and label information file under the fold DeepHost\_scripts.

Here is a toy example:
```shell
python DeepHost_train.py ../example/test_data.fasta ../example/test_data_species.txt --thread 10
```


## Data
+ data/host\_info.txt: The accesion numbers of phage genomes and their host information obtained from NCBI and EMBL (Jan, 2021); the phage names and their host information obtained form PhageDB (Jul, 2021).
+ data/genus.txt: The 72 genus included in DeepHost.
+ data/species.txt: The 118 species included in DeepHost.

## Maintainer
WANG Ruohan ruohawang2-c@my.cityu.edu.hk

## Reference
@article{ruohan2022deephost,\\
  title={DeepHost: phage host prediction with convolutional neural network},\\
  author={Ruohan, Wang and Xianglilan, Zhang and Jianping, Wang and Shuai Cheng, LI},\\
  journal={Briefings in Bioinformatics},\\
  volume={23},\\
  number={1},\\
  pages={bbab385},\\
  year={2022},\\
  publisher={Oxford University Press}\\
}
