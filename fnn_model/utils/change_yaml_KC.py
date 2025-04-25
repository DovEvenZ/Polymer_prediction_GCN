import yaml

def change_yaml_KC(path_ym,replace_raw,replace_new):
    data = []
    with open(path_ym,'r') as f:
        for line in f:
            if replace_raw in line:
                check = f'{replace_raw}: {replace_new}\n'
                data.append(check)
            else:
                data.append(line)      

    with open(path_ym,'w') as fw:
        fw.writelines(data)