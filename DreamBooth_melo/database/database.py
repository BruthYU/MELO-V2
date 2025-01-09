class TS_item:
    def __init__(self, instance_prompt, index):
        self.instance_prompt = instance_prompt
        self.index = index

    def match(self, head, tail):
        if " ".join([head, tail]) == self.instance_prompt:
            return True
        return False

class TSDB:
    def __init__(self):
        self.table = []

    def TS_insert(self, instance_prompt, index):
        self.table.append(TS_item(instance_prompt, index))

    def TS_search(self, head, tail):
        for idx, row in enumerate(self.table):
            if row.match(head,tail):
                return idx
        return None

    def __getitem__(self, item):
        return self.table[item]




