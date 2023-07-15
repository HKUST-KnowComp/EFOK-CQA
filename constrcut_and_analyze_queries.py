import pandas as pd
import numpy as np

origin_queries = pd.read_csv("data/DNF_EFO2_23_4123166.csv")
variable = [2]
formulas_1 = origin_queries[origin_queries["f_num"]==1]
formulas_2 = origin_queries[origin_queries["f_num"]==2]
if 1 in variable:
    scores = [[] for i in range(3)]
    scores_str = ["" for i in range(3)]
    for c in range(1,4):
            for e  in range(3):
                r_e = np.array(formulas_1["e_num"] == e) & np.array(formulas_1["c_num"] == c)
                if np.any(r_e & ~np.array(formulas_1["multi"]) & ~np.array(formulas_1["cyclic"])):
                    index_red = r_e & ~np.array(formulas_1["multi"]) & ~np.array(formulas_1["cyclic"])
                    avg_r_e_DAG = index_red.sum()
                    scores[c-1].append(avg_r_e_DAG)
                    scores_str[c-1] += "&{}".format(avg_r_e_DAG)
                if np.any(r_e & np.array(formulas_1["multi"])) :
                    index_rem = r_e & np.array(formulas_1["multi"])
                    avg_r_e_M = index_rem.sum()
                    scores[c-1].append(avg_r_e_M)
                    scores_str[c-1] += "&{}".format(avg_r_e_M)
                if np.any(r_e & np.array(formulas_1["cyclic"])):
                    index_rec = r_e & np.array(formulas_1["cyclic"])
                    avg_r_e_C = index_rec.sum()
                    scores[c-1].append(avg_r_e_C)
                    scores_str[c-1] += "&{}".format(avg_r_e_C)
            index_c = np.array(formulas_1["c_num"] == c)
            avg_c = index_c.sum()
            scores_str[c-1] += "&{}".format(avg_c)
            scores[c-1].append(avg_c)

    scores_avge_str = ""
    for e in range(3):
        index_e = np.array(formulas_1["e_num"] == e)
        if np.any(index_e & ~np.array(formulas_1["multi"]) & ~np.array(formulas_1["cyclic"])):
            index_ed = index_e & ~np.array(formulas_1["multi"]) & ~np.array(formulas_1["cyclic"])
            avg_e_DAG = index_ed.sum()
            scores_avge_str += "&{}".format(avg_e_DAG)
        if np.any(index_e & np.array(formulas_1["multi"])):
            index_em = index_e & np.array(formulas_1["multi"])
            avg_e_M = index_em.sum()
            scores_avge_str += "&{}".format(avg_e_M)
        if np.any(index_e & np.array(formulas_1["cyclic"])):
            index_ec = index_e & np.array(formulas_1["cyclic"])
            avg_e_C = index_ec.sum()
            scores_avge_str += "&{}".format(avg_e_C)
    avg = len(formulas_1)

if 2 in variable:
    metric2 = ["marginal_HITS10_0", "HITS10*10_0","HITS10_0"]
    scores = [[] for i in range(3)]
    label = [[] for i in range(3)]
    scores_str = ["" for i in range(len(metric2))]
    id_fc = np.array(formulas_2["c_num"] == 2) & np.array(formulas_2["f_num"] == 2)
    for i in range(len(metric2)):
        metric_index = metric2[i]
        for e  in range(3):
                    id_fce = id_fc & np.array(formulas_2["e_num"] == e)
                    if np.any(id_fce & ~np.array(formulas_2["multi"]) & ~np.array(formulas_2["cyclic"])):
                        id_fced = id_fce & ~np.array(formulas_2["multi"]) & ~np.array(formulas_2["cyclic"])
                        avg_fced = id_fced.sum()
                        label[i].append("DAG")
                        scores[i].append(avg_fced)
                        scores_str[i] += "&{}".format(avg_fced)
#                    else:
#                         scores_str[i] += "&"

                    if np.any(id_fce & np.array(formulas_2["multi"])) :
                        id_fcem = id_fce & np.array(formulas_2["multi"])
                        avg_fcem = id_fcem.sum()
                        label[i].append("Multi")
                        scores[i].append(avg_fcem)
                        scores_str[i] += "&{}".format(avg_fcem)
#                    else:
#                         scores_str[i] += "&"

                    if np.any(id_fce & np.array(formulas_2["cyclic"])):
                        id_fcec = id_fce & np.array(formulas_2["cyclic"])
                        avg_fcec = id_fcec.sum()
                        label[i].append("Cyclic")
                        scores[i].append(avg_fcec)
                        scores_str[i] += "&{}".format(avg_fcec)
#                    else:
#                         scores_str[i] += "&"

        avg_fc_metric =  len(formulas_2[id_fc])
        scores_str[i] += "&{}".format(avg_fc_metric)[:-1]