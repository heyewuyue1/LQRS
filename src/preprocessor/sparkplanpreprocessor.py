import numpy as np
from utils.logger import setup_custom_logger
from preprocessor.node import Node
from config import EnvironmentConfig

logger = setup_custom_logger("PREPROCESSOR")

class SparkPlanPreprocessor():
    
    def __featurize_not_null_operator(self, node):
        arr = np.zeros(len(self.op_list) + 1)
        arr[self.op_list.index(node.operator)] = 1
        tables = np.zeros(self.table_dim)
        for table in node.tables:
            tables[self.table_list.index(table)] = 1
        return np.concatenate((arr, tables, [node.card, node.size_in_bytes]))

    def __featurize_null_operator(self):
        arr = np.zeros(len(self.op_list) + 1)
        tables = np.zeros(self.table_dim)
        return np.concatenate((arr, tables, [-1, -1]))

    def __featurize(self, tree, i):
        if len(tree) <= 1:
            return self.__featurize_null_operator()
        return self.__featurize_not_null_operator(tree[i]),\
        self.__featurize(tree, tree[i].lc) if tree[i].lc is not None else self.__featurize_null_operator(),\
        self.__featurize(tree, tree[i].rc) if tree[i].rc is not None else self.__featurize_null_operator()
        
    def tranverse_fix_join_tables(self, tree, i):
        if tree[i].lc is not None:
            tree[i].tables = list(set(tree[i].tables + self.tranverse_fix_join_tables(tree, tree[i].lc)))
        if tree[i].rc is not None:
            tree[i].tables = list(set(tree[i].tables + self.tranverse_fix_join_tables(tree, tree[i].rc)))
        return tree[i].tables

    def plan2tree(self, plan, executed_row_counts):
        lines = plan.split('\n')
        tree = []
        colon = 0 
        join_stack = []
        tree.append(Node('Root', [], -1, -1))
        prev_idx = 0

        for i in range(len(lines)):
            if lines[i].strip() == '':
                continue
            tables = []
            if 'Relation' in lines[i]:
                op = 'UnknownQueryStage'
                tables = [lines[i].split('[')[0].split('.')[-1]]
            elif 'Join' in lines[i]:
                op = 'Join'
                join_stack.append(len(tree))
            elif 'BroadcastQueryStage' in lines[i]:
                op = 'BroadcastQueryStage'
            elif 'ShuffleQueryStage' in lines[i]:
                op = 'ShuffleQueryStage'
            elif 'BroadcastQueryStage' and 'ShuffleQueryStage' in lines[i]:
                if lines[i].find('BroadcastQueryStage') > lines[i].find('ShuffleQueryStage'):
                    op = 'BroadcastQueryStage'
                else:
                    op = 'ShuffleQueryStage'
            else:
                continue
            stage = lines[i].split('- ')[-1].strip()
            card = size_in_bytes = -1
            if stage in executed_row_counts:
                if executed_row_counts[stage][0]:
                    card = np.log1p(eval(executed_row_counts[stage][1]))
                    size_in_bytes = np.log1p(eval(executed_row_counts[stage][2]))
                tables = executed_row_counts[stage][3]
            tree.append(Node(op, tables, card, size_in_bytes))
            logger.debug(f"Append Node:\n{op}{tables} {card} {size_in_bytes}")
            if colon <= lines[i].count(':'):
                tree[prev_idx].lc = len(tree) - 1
            else:
                tree[join_stack[-1]].rc = len(tree) - 1
                join_stack.pop()
            colon = lines[i].count(':')
            prev_idx = len(tree) - 1
        self.tranverse_fix_join_tables(tree, 0)
        return tree

    def __init__(self):
        super().__init__()
        self.op_list = ['Join', 'BroadcastQueryStage', 'ShuffleQueryStage', 'UnknownQueryStage']
        self.table_list = []
        with open(EnvironmentConfig.table_file) as f:
            self.table_list = [s.strip() for s in f.readlines()]
        self.table_dim = len(self.table_list)
        logger.info(f"Got {self.table_dim} tables: {self.table_list}")
        self.feature_dim = len(self.op_list) + 1 + self.table_dim + 2
    
    def transform(self, plan, executed_row_counts):
        tree = self.plan2tree(plan, executed_row_counts)
        featurized_tree = self.__featurize(tree, 1)
        return featurized_tree