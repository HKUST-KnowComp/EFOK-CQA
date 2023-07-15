import json


original_filename = 'data/FB15k-237-betae/kgindex_old.json'
new_filename = 'data/FB15k-237-betae/kgindex.json'
entity2wiki_file = 'data/FB15k-237-betae/entity2wikidata.json'
entity2text_file = 'data/FB15k-237-betae/entity2text.txt'

with open(original_filename, 'rt') as f:
    load = json.load(f)
with open(entity2wiki_file, 'rt') as f:
    entity2wiki = json.load(f)
another_entity2text = {}
with open(entity2text_file) as f:
    for line in f:
        entity2text = line.strip()
        entity, text = entity2text.split('\t')
        another_entity2text[entity] = text

new_load = {'r': load['r'], 'e': {}}
for m_str in load['e']:
    if m_str in entity2wiki:
        if entity2wiki[m_str]['description']:
            entity_name = entity2wiki[m_str]['label'] + '(' + entity2wiki[m_str]['description'] + ')'
        else:
            entity_name = entity2wiki[m_str]['label']
    else:
        entity_name = None
    entity_another_name = another_entity2text[m_str] if m_str in another_entity2text else None
    '''
    if entity_name and entity_another_name:
        if entity_name.lower() != entity_another_name.lower():
            print(m_str, entity_name, entity_another_name)
    '''
    if entity_name:
        if entity_name in new_load['e']:
            print(entity_name)
            new_entity_name = entity_name + '2'
            new_load['e'][new_entity_name] = load['e'][m_str]
        else:
            new_load['e'][entity_name] = load['e'][m_str]
    else:
        assert entity_another_name is not None
        new_load['e'][entity_another_name] = load['e'][m_str]

with open(new_filename, 'wt') as f:
    json.dump(new_load, f)
