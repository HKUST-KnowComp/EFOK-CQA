import os.path as osp

from src.structure.knowledge_graph import KnowledgeGraph
from src.structure.knowledge_graph_index import KGIndex


if __name__ == "__main__":
    all_data_folder = ['data/FB15k-237-betae', 'data/FB15k-betae', 'data/NELL-betae']
    for data_folder in all_data_folder:
        kgidx = KGIndex.load(osp.join(data_folder, 'kgindex.json'))
        test_kg = KnowledgeGraph.create(
            triple_files=osp.join(data_folder, 'test_kg.tsv'),
            kgindex=kgidx)
        self_circle_num = 0
        self_circle_rel = set()
        for triple in test_kg.triples:
            h, r, t = triple
            if h == t:
                # print(triple, kgidx.inverse_entity_id_to_name[h], kgidx.inverse_relation_id_to_name[r])
                self_circle_num += 1
                self_circle_rel.add(kgidx.inverse_relation_id_to_name[r])
        print(len(test_kg.triples), self_circle_num, self_circle_rel, len(self_circle_rel))
