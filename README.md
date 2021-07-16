# DeepHost
DeepHost is a phage host prediction tool.

## Prerequisite
DeepHost is implemented with Python3.6 and keras2.2.4. Following packages should be installed.
+ cython
+ numpy
+ keras


Please install Git LFS and use the following command to clone the repository.
```shell
git lfs clone https://github.com/deepomicslab/DeepHost.git
```


Before using DeepHost, please build the Cython file with the command:
```shell
python setup.py build_ext --inplace
```

## Usage
```shell
cd Scripts
python SeCNV.py Phage_genomes.fasta -o Output_name.txt -r species 
```
Phage\_genomes.fasta contains phage genome sequences in fasta format. DeepHost supports both single genome and multiple genomes in one file. The input of -o is the filename of the output file, and the default filename is DeepHost\_output.txt. The input of -r is the taxonomic rank of predictions, which accepts genus and species (default).

For more information, please use the command:
```shell
python SeCNV.py -h
```
or
```shell
python SeCNV.py --help
```

## Data
+ data/sample\_data.fasta: The sample phage genomes.
+ data/data\_data\_single\_host.txt: The accesion numbers of phage genomes and their host information obtained from NCBI and EMBL (Jan, 2021).

## Maintainer
WANG Ruohan ruohawang2-c@my.cityu.edu.hk
