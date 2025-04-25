import yaml

def change_yaml_KC(path_ym,replace_raw,replace_new):
    data = []
    with open(path_ym,'r') as f:
        for line in f:
            if line.split(':')[0] == replace_raw:
                if '#' in line:
                    last = '#'+'#'.join(line.split('#')[1:])
                else:
                    last = '\n'
                check = f'{replace_raw}: {replace_new} {last}'
                data.append(check)
            else:
                data.append(line)      

    with open(path_ym,'w') as fw:
        fw.writelines(data)