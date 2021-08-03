def parse_atom_line(line):
    data = {}
    data['record'], data['serial_no'], data['name'] = line[:4], int(line[6:11]), line[12:16]
    data['alternate_loc_ind'], data['residue'], data['chain'] = line[16], line[17:20], line[21]
    data['res_seq'], data['code_for_insertions'] = int(line[22:26]), line[26]
    data['x'], data['y'], data['z'] = float(line[30:38]), float(line[38:46]), float(line[46:54])
    data['occ'], data['temp'], data['elem'] = float(line[54:60]), float(line[60:66]), line[76:78]

    for key, value in data.items():
        if isinstance(value, str):
            data[key] = value.strip()
    return data


def split_and_order_chains(datas):
    no_chains = len(list(filter(lambda x: True if x == 'OXT' else False, (x.get('name') for x in datas))))
    no_atoms = len(list(filter(lambda x: True if x == 'ATOM' else False, (x.get('record') for x in datas))))
    i = 0
    chains = [[] for x in range(no_chains)]
    for data in datas:
        if data['record'] == 'TER':
            chains[i - 1].append(data)
            continue
        chains[i].append(data)
        if data['name'] == 'OXT':
            i += 1
    order = []
    for i, chain in enumerate(chains):
        if any([x.get('residue') == 'GLQ' for x in chain]):
            order.insert(0, i)
        else:
            order.append(i)
    chains = [chains[i] for i in order]
    for chain, new_chain in zip(chains, string.ascii_uppercase):
        for value in chain:
            value['chain'] = new_chain
    return chains


def add_ter(chains):
    for i, chain in enumerate(chains):
        for j, element in enumerate(chain):
            if element['record'] == 'ATOM':
                if element['residue'] == 'GLQ' and element['name'] == 'OXT':
                    data = {}
                    data['record'], data['serial_no'] = 'TER', element['serial_no']
                    data['residue'], data['chain'] = element['residue'], element['chain']
                    data['res_seq'], data['code_for_insertions'] = element['res_seq'], element['code_for_insertions']
                    chains[i][j] = data
                if element['residue'] == 'GLY' and j == len(chain) - 1:
                    data = {}
                    data['record'], data['serial_no'] = 'TER', 0
                    data['residue'], data['chain'] = element['residue'], element['chain']
                    data['res_seq'], data['code_for_insertions'] = element['res_seq'], element['code_for_insertions']
                    chains[i].append(data)
    return chains


def build_pdb_line(data, atom_nr, res_number):
    if data['record'] == 'ATOM':
        line = f"{data['record']}  {atom_nr:>5}  {data['name']:<4}{data['alternate_loc_ind']}{data['residue']}"
        line += f" {data['chain']}{res_number:>4}{data['code_for_insertions']}    {data['x']:8.3f}{data['y']:8.3f}{data['z']:8.3f}"
        line += f"{data['occ']:6.2f}{data['temp']:6.2f}          {data['elem']:>2}"
    else:
        line = f"{data['record']}   {atom_nr:>5}      {data['residue']}"
        line += f" {data['chain']}{res_number:>4}{data['code_for_insertions']}"
    return line


def parse_ter_line(line):
    data = {}
    data['record'], data['serial_no'] = line[:4], int(line[6:11])
    data['residue'], data['chain'] = line[17:20], line[21]
    data['res_seq'], data['code_for_insertions'] = int(line[22:26]), line[26]

    for key, value in data.items():
        if isinstance(value, str):
            data[key] = value.strip()

    return data


def create_connect(data):
    line = "CONECT"
    for d in data:
        line += f" {d:>4}"
    return line + '\n'


def parse_pdb_line(line):
    if 'ATOM' in line:
        return parse_atom_line(line)
    elif 'TER' in line:
        return parse_ter_line(line)
    else:
        raise Exception(f"Unknown Record Type in line {line}")


def prepare_pdb_for_gmx(file, verification=None):
    """Function from expansion elephant. Might be deprecated."""
    if verification:
        with open(verification, 'r') as f:
            verification_lines = f.readlines()
    with open(file, 'r') as f:
        lines = f.readlines()

    # define
    leading = []
    datas = []
    old_lines = []
    connects = []

    # parse
    for i, line in enumerate(lines):
        if 'REMARK' in line or 'CRYST' in line or 'MODEL' in line:
            leading.append(line)
        elif 'ATOM' in line or 'TER' in line:
            old_lines.append(line)
            data = parse_pdb_line(line)
            datas.append(data)
        elif 'ENDMDL' in line or 'END' in line:
            pass
        elif 'CONECT' in line:
            connects.append(line)
        else:
            print(i, line)
            raise Exception(f"Unkonwn Record Type in line {line}")

    # rearrange chains
    chains = split_and_order_chains(datas)
    chains = add_ter(chains)
    data = list(itertools.chain(*chains))

    # split some connects
    connects = list(map(lambda x: [int(y) for y in x.split()[1:]], connects))
    replacements = {}

    # build new pdb
    residue = 1
    for i, d in enumerate(data):
        # print(old_lines[i])
        if i > 0:
            if d['residue'] != data[i - 1]['residue']:
                residue += 1
        i += 1
        line = build_pdb_line(d, i, residue) + '\n'
        leading.append(line)

        if any(d['serial_no'] in c for c in connects):
            replacements[d['serial_no']] = i

        if verification:
            if line[:22] != verification_lines[i + 3][:22]:
                print(line, verification_lines[i + 3])
                raise Exception('STOP')

    leading.append('ENDMDL\n')
    # fix connects
    new_connects = []
    for co in connects:
        new = []
        for c in co:
            new.append(replacements[c])
        new_connects.append(new)

    for connect in new_connects:
        leading.append(create_connect(connect))
    leading.append('END')

    with open(file, 'w') as f:
        for line in leading:
            f.write(line)