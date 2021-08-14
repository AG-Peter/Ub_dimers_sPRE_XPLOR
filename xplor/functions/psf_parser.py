import os

class PSFParser:
    def __init__(self, psf):
        if os.path.isfile(psf):
            with open(psf, 'r') as f:
                psf = f.read()
        self.first_line = 'PSF'
        self._psf = psf

        # parse stuff
        self._split_hunks()
        self._parse_remarks()

    def _parse_remarks(self):
        self.remarks = []
        for line in self._psf.splitlines():
            if line.startswith('REMARKS'):
                self.remarks.append(line.lstrip('REMARKS '))

    def _split_hunks(self):
        self.headers = []
        self.hunks  =[]
        for line in self._psf.splitlines():
            if '!N' in line:
                print(line)

    def write(self):
        pass

    def __str__(self):
        return "PSF Parser containing {self.n_atoms}"


class Residue:
    def __init__(self):
        pass


class Bond:
    def __init__(self):
        pass


class Atom:
    def __init__(self):
        pass


class Angle:
    def __init__(self):
        pass


class Dihedral:
    def __init__(self):
        pass


class Impropers:
    def __init__(self):
        pass

class NonbonedInteraction:
    def __init__(self):
        pass


class Donors:
    def __init__(self):
        pass


class Acceptors:
    def __init__(self):
        pass


class Groups:
    def __init__(self):
        pass


