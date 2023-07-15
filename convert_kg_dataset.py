import argparse
import os
import os.path as osp

from src.structure.knowledge_graph import KnowledgeGraph
from src.utils.data_util import iter_triple_from_tsv
from src.structure.knowledge_graph_index import KGIndex


def convert_kg_flder(input_folder, output_folder):
    print("processing KG index")
    os.makedirs(output_folder, exist_ok=True)
    kgidx = KGIndex()
    eids = []
    for eid, name in iter_triple_from_tsv(
        osp.join(input_folder, 'map_entity_id_to_text.tsv'),
        to_int=False, check_size=2):
        eid = int(eid)
        kgidx.register_entity(name, eid)
        eids.append(eid)
    assert max(eids) - min(eids) + 1 == len(kgidx.map_entity_name_to_id)

    rids = []
    for rid, name in iter_triple_from_tsv(
        osp.join(input_folder, 'map_relation_id_to_text.tsv'),
        to_int=False,
        check_size=2):
        rid = int(rid)
        kgidx.register_relation(name, rid)
        rids.append(rid)
    assert max(rids) - min(rids) + 1 == len(kgidx.map_relation_name_to_id)

    kgidx.dump(osp.join(output_folder, 'kgindex.json'))
    kgidx = KGIndex.load(osp.join(output_folder, 'kgindex.json'))
    print("done")

    print("process train kg")
    train_kg = KnowledgeGraph.create(
        triple_files=osp.join(input_folder, 'edges_as_id_train.tsv'),
        kgindex=kgidx)
    train_kg.dump(osp.join(output_folder, 'train_kg.tsv'))
    print("done")

    print("process valid kg")
    valid_kg = KnowledgeGraph.create(
        triple_files=osp.join(input_folder, 'edges_as_id_valid.tsv'),
        kgindex=kgidx)
    valid_kg.dump(osp.join(output_folder, 'valid_kg.tsv'))
    print("done")

    print("process test kg")
    test_kg = KnowledgeGraph.create(
        triple_files=osp.join(input_folder, 'edges_as_id_test.tsv'),
        kgindex=kgidx)
    test_kg.dump(osp.join(output_folder, 'test_kg.tsv'))
    print("done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_folder", type=str)
    parser.add_argument("--output_folder", type=str)
    args = parser.parse_args()
    convert_kg_flder(input_folder=args.input_folder,
                     output_folder=args.output_folder)