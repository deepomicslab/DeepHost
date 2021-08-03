f_species = open("/home/ruohawang2/07.DeepHost/revision_1/data/host_information/species_count_sort.txt")
species_choose_list = []
for line in f_species:
    line = line.strip("\n").split("\t")
    species_choose_list.append(line[0])
f_species.close()

acc_host_dic = {}
f_host = open("/home/ruohawang2/07.DeepHost/revision_1/data/host_information/host_info_all.txt")
for line in f_host:
    line = line.strip("\n").split("\t")
    if line[1] in species_choose_list:
        acc_host_dic[line[0].split(".")[0]] = line[1]
f_host.close()

right_count = 0
all_count = 0
f = open("check_result.txt")
f = open("meta_phage_host.txt")
for line in f:
    all_count += 1
    line = line.strip("\n").split("\t")
    if line[1] == acc_host_dic[line[0].split(".")[0]]:
        right_count += 1
f.close()
print(right_count * 1.0 / all_count)
