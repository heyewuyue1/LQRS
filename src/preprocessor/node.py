class Node:
    def __init__(self) -> None:
        self.lc = None
        self.rc = None
        self.operator = None
        self.tables = []
        self.card = None
        self.size_in_bytes = None


    def __init__(self, operator, tables, card, size_in_bytes) -> None:
        self.lc = None
        self.rc = None
        self.tables = tables
        self.operator = operator
        self.card = card
        self.size_in_bytes = size_in_bytes
