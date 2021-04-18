from simulate import Population
import time
from pprint import pprint
import os
import sys

POP_COUNT = 1000
GEN_COUNT = 11
MIN_OFFSPRING = 0
MAX_OFFSPRING = 6
MIN_GEN_GROW = 16
MAX_GEN_GROW = 19
DEATH_AGE = 80
MIN_MARRY_AGE = 18
MAX_MARRY_AGE = 60

GENE_HLTHY_WEIGHT = 0.8
GENE_TRAIT_WEIGHT = 0.2
GENE_AFFEC_WEIGHT = 0

ILL_REL_DEG_THRESH = 4  # incl


if __name__ == '__main__':
    header = True
    pop_id = 0
    for run_id in range(int(sys.argv[2])):
        print(f">> Run {run_id}")
        for pop_count in [1000, 4000, 16000]:
            for ill_rel_deg_thr in [0, 2, 3, 4]:
                for cons_marry_chance in [0, 0.2, 0.4, 0.6, 0.8]:
                    print(
                        f"> pop_count {pop_count}, deg {ill_rel_deg_thr}, cons_chance {cons_marry_chance}")
                    pop = Population(
                        uid=str(
                            f"r{str(run_id).zfill(3)}_p{str(pop_id).zfill(3)}"),
                        pop_count=pop_count,
                        gen_count=GEN_COUNT,
                        ill_rel_deg_thr=ill_rel_deg_thr,
                        min_offspring=MIN_OFFSPRING,
                        max_offspring=MAX_OFFSPRING,
                        min_gen_grow=MIN_GEN_GROW,
                        max_gen_grow=MAX_GEN_GROW,
                        death_age=DEATH_AGE,
                        min_marry_age=MIN_MARRY_AGE,
                        max_marry_age=MAX_MARRY_AGE,
                        gene_healthy_weight=GENE_HLTHY_WEIGHT,
                        gene_trait_weight=GENE_TRAIT_WEIGHT,
                        gene_affect_weight=GENE_AFFEC_WEIGHT,
                        cons_marry_chance=cons_marry_chance,
                    )
                    pop.create()
                    pop.simulate()
                    pop.stats_to_csv(sys.argv[1], header)
                    pop_id += 1
                    header = False
