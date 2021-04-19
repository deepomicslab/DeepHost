# DeepHost
DeepHost, a phage host prediction tool.

## Prerequisite
DeepHost is implemented with Python3.6 and keras2.2.4. Following packages should be installed.
+ cython
+ numpy
+ keras

## Usage
```shell
cd Scripts
python SeCNV.py Phage_genomes.fasta -o Output_name.txt -r species 
```
Phage\_genomes.fasta is phage genome sequences in fasta format. DeepHost supports both single genome and multiple genomes in one file. The input of -o is the filename of the output file, and the default filename is DeepHost\_output.txt. The input of -r is the taxonomic rank of predictions, which accepts genus and species (default).

For more information, please use python SeCNV.py -h or python SeCNV.py --help.

## Sample data
data/sample\_data.fasta: The sample phage genomes.
data/data\_data\_single\_host.txt: The accesion numbers of phage genomes and their host information obtained from NCBI and EMBL (Jan, 2021).

## Maintainer
WANG Ruohan ruohawang2-c@my.cityu.edu.hk
