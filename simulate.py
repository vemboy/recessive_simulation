import random
import collections
from typing import List
from enum import Enum
import pprint
import networkx as nx
import pandas as pd


GENE_HLTHY_WEIGHT = 0.5  # 0.966
GENE_TRAIT_WEIGHT = 0.5  # 0.034
GENE_AFFEC_WEIGHT = 0


class GENE(Enum):
    AA = 1  # healthy
    Aa = 2  # trait
    aa = 3  # disorder


class SEX(Enum):
    M = 1
    F = 2


class Person:
    next_id = 0

    def __init__(self, age: int, genes: GENE, sex: SEX):
        self.id = Person.next_id
        self.age = age
        self.genes = genes
        self.sex = sex
        Person.next_id += 1

    def __str__(self):
        return str({
            "id": self.id,
            "age": self.age,
            "genes": self.genes,
            "sex": self.sex,
        })

    def __repr__(self):
        return str(self.id)

    def grow(self, years: int) -> None:
        self.age += years

    def is_dead_of_age(self, death_age) -> bool:
        return self.age > death_age

    def is_dead_of_disorder(self) -> bool:
        return self.genes == GENE.aa


class Population:
    def __init__(
        self,
        uid,
        pop_count,
        gen_count,
        ill_rel_deg_thr,
        min_offspring,
        max_offspring,
        min_gen_grow,
        max_gen_grow,
        death_age,
        min_marry_age,
        max_marry_age,
        gene_healthy_weight,
        gene_trait_weight,
        gene_affect_weight,
        cons_marry_chance,
    ):
        # params
        self.uid = uid
        self.pop_count = pop_count
        self.gen_count = gen_count
        self.ill_rel_deg_thr = ill_rel_deg_thr
        self.min_offspring = min_offspring
        self.max_offspring = max_offspring
        self.min_gen_grow = min_gen_grow
        self.max_gen_grow = max_gen_grow
        self.death_age = death_age
        self.min_marry_age = min_marry_age
        self.max_marry_age = max_marry_age
        self.gene_healthy_weight = gene_healthy_weight
        self.gene_trait_weight = gene_trait_weight
        self.gene_affect_weight = gene_affect_weight
        self.cons_marry_chance = cons_marry_chance

        # setup
        self.relations = nx.Graph()

        # data
        self.groups = {
            "males": [],
            "females": [],
            "kids": [],
            "seniors": [],
            "males_married": [],
            "females_married": []
        }
        self.died_disorder = []
        self.died_age = []
        self.init_size = 0
        self.marry_count = 0
        self.stop_marry_count = 0
        self.offspring_count = 0
        self.cons_marry_count = 0

    def simulate(self):
        for _ in range(self.gen_count):
            self.generation()

    def age(self) -> None:
        for x in self.groups:
            for person in self.groups[x]:
                person.grow(random.randint(
                    self.min_gen_grow, self.max_gen_grow))

    def wants_to_cons_marry(self) -> bool:
        return random.random() < self.cons_marry_chance

    def update_groups(self) -> None:
        self.update_kids()
        self.update_adults()

    def update_kids(self) -> None:
        m_kids_adult_index = []
        f_kids_adult_index = []

        for i in range(len(self.groups["kids"])):
            kid = self.groups["kids"][i]
            if kid.age > self.min_marry_age:
                if kid.sex == SEX.M:
                    m_kids_adult_index.append(i)
                else:
                    f_kids_adult_index.append(i)

        # add adult kids to adults lists
        self.groups["males"] += [self.groups["kids"][i]
                                 for i in m_kids_adult_index]
        self.groups["females"] += [self.groups["kids"][i]
                                   for i in f_kids_adult_index]

        # remove adult kids from kids list
        for i in sorted(m_kids_adult_index + f_kids_adult_index, reverse=True):
            del self.groups["kids"][i]

    def update_adults(self) -> None:
        m_adult_senior_index = []
        f_adult_senior_index = []

        for i in range(len(self.groups["males"])):
            male = self.groups["males"][i]
            if male.age > self.max_marry_age:
                m_adult_senior_index.append(i)

        for i in range(len(self.groups["females"])):
            female = self.groups["females"][i]
            if female.age > self.max_marry_age:
                f_adult_senior_index.append(i)

        # add adult kids to adults lists
        self.groups["seniors"] += [self.groups["males"][i]
                                   for i in m_adult_senior_index]
        self.groups["females"] += [self.groups["females"][i]
                                   for i in f_adult_senior_index]

        # remove senior males from males list
        for i in sorted(m_adult_senior_index, reverse=True):
            del self.groups["males"][i]

        # remove senior females from females list
        for i in sorted(f_adult_senior_index, reverse=True):
            del self.groups["females"][i]

    def kill(self) -> None:
        for g in self.groups:
            dead_people_index = []
            for i, person in enumerate(self.groups[g]):
                if person.is_dead_of_age(self.death_age):
                    dead_people_index.append(i)
                    self.died_age.append(person)
                elif person.is_dead_of_disorder():
                    dead_people_index.append(i)
                    self.died_disorder.append(person)
            for d_i in sorted(dead_people_index, reverse=True):
                del self.groups[g][d_i]

    def get_cousins(self, person):
        lengths = nx.single_source_shortest_path_length(
            self.relations, person, cutoff=3)
        return [p for p, l in lengths.items() if l == 3]

    def rand_marry(self, unmarr_m_to_index, unmarr_f_to_index,
                   marr_m_indexes, marr_f_indexes):
        married_males = []
        for male, m_index in unmarr_m_to_index.items():
            married_female = None
            for female, f_index in unmarr_f_to_index.items():
                # marry
                if (self.person_allowed_to_marry(male, female)):
                    # have children
                    offspring = self.offspring(male, female)
                    for o in offspring:
                        self.add_person(o, "kids", male, female)
                        self.offspring_count += 1

                    # keep track of married
                    marr_m_indexes.append(m_index)
                    marr_f_indexes.append(f_index)

                    married_female = female
                    married_males.append(male)
                    break
                else:
                    # print("#### stopped a mairrage")
                    # degree = len(nx.shortest_path(self.relations, curr_man, curr_wom)) - 1
                    # print(f"{curr_man} and {curr_wom} had a degree of {degree}")
                    self.stop_marry_count += 1

            if married_female is not None:
                del unmarr_f_to_index[married_female]

        for m in married_males:
            del unmarr_m_to_index[m]

    def cons_marry(self, unmarr_m_to_index, unmarr_f_to_index,
                   marr_m_indexes, marr_f_indexes):
        married_males = []
        for male, m_index in unmarr_m_to_index.items():
            married_female = None
            if not self.wants_to_cons_marry():
                continue
            cousins = self.get_cousins(male)
            for female in cousins:
                if female in unmarr_f_to_index:
                    f_index = unmarr_f_to_index[female]
                    if (self.person_allowed_to_marry(male, female)):
                        # have children
                        offspring = self.offspring(male, female)
                        for o in offspring:
                            self.add_person(o, "kids", male, female)
                            self.offspring_count += 1

                        # keep track of married
                        marr_m_indexes.append(m_index)
                        marr_f_indexes.append(f_index)

                        married_female = female
                        married_males.append(male)
                        self.cons_marry_count += 1
                        break
                    else:
                        # print("#### stopped a mairrage")
                        # degree = len(nx.shortest_path(self.relations, curr_man, curr_wom)) - 1
                        # print(f"{curr_man} and {curr_wom} had a degree of {degree}")
                        self.stop_marry_count += 1

            if married_female is not None:
                del unmarr_f_to_index[married_female]

        for m in married_males:
            del unmarr_m_to_index[m]

    def marry(self) -> None:
        random.shuffle(self.groups["males"])
        random.shuffle(self.groups["females"])

        unmarr_m_to_index = {m: m_i for m_i,
                             m in enumerate(self.groups["males"])}
        unmarr_f_to_index = {f: f_i for f_i,
                             f in enumerate(self.groups["females"])}

        marr_m_indexes = []
        marr_f_indexes = []

        self.cons_marry(unmarr_m_to_index, unmarr_f_to_index,
                        marr_m_indexes, marr_f_indexes)
        self.rand_marry(unmarr_m_to_index, unmarr_f_to_index,
                        marr_m_indexes, marr_f_indexes)

        # Add married to married group
        for i in marr_m_indexes:
            self.groups["males_married"].append(self.groups["males"][i])
        for i in marr_f_indexes:
            self.groups["females_married"].append(self.groups["females"][i])

        # Remove married from adult group
        for d_i in sorted(marr_m_indexes, reverse=True):
            del self.groups["males"][d_i]
        for d_i in sorted(marr_f_indexes, reverse=True):
            del self.groups["females"][d_i]

    def size(self) -> int:
        count = 0
        for x in self.groups:
            count += len(self.groups[x])
        return count

    def assign_genes(self, people):
        count = len(people)
        people_genes = random.choices(
            population=list(GENE),
            weights=[GENE_HLTHY_WEIGHT, GENE_TRAIT_WEIGHT, GENE_AFFEC_WEIGHT],
            k=count
        ),
        for i in range(count):
            people[i].genes = people_genes[0][i]

    def create(self):
        count = self.pop_count
        # print(f"### Creating population ({count})")
        self.init_size = count
        people = [person_get_random() for _ in range(count)]
        self.assign_genes(people)
        for p in people:
            if(p.sex == SEX.M and p.age > self.min_marry_age and p.age < self.max_marry_age):
                self.add_person(p, "males", None, None)
            elif(p.sex == SEX.F and p.age > self.min_marry_age and p.age < self.max_marry_age):
                self.add_person(p, "females", None, None)
            elif(p.age < self.min_marry_age):
                self.add_person(p, "kids", None, None)
            else:
                self.add_person(p, "seniors", None, None)

    def generation(self) -> None:
        # print(f"### Generation {self.gen_count} ({self.size()})")
        # print("Aging ", end="", flush=True)
        self.age()
        # print("Updating ", end="", flush=True)
        self.update_groups()
        # print("Killing ", end="", flush=True)
        self.kill()
        # print("Marrying ", end="", flush=True)
        self.marry()
        # print()

    def offspring(self, male: Person, female: Person) -> None:
        self.marry_count += 1
        children = []
        for _ in range(random.randint(self.min_offspring, self.max_offspring)):
            children.append(person_get_offspring(male, female))
        return children

    def add_person(self, person: Person, group: str, father: Person, mother: Person):
        self.groups[group].append(person)

        # Create relations
        self.relations.add_node(person)
        if (father is not None and mother is not None):
            self.relations.add_edge(person, father)
            self.relations.add_edge(person, mother)

    def person_allowed_to_marry(self, p1: Person, p2: Person) -> bool:
        try:
            path = nx.single_source_shortest_path(
                self.relations, p1, cutoff=self.ill_rel_deg_thr)
            if p2 in path.keys():
                return False
            return True
        except nx.NetworkXNoPath:
            return True

    def random_sample_stats(self, group: str):
        p: Person = random.choice(self.groups[group])
        return {
            "id": p.id,
            "age": p.age,
            "genes": p.genes,
            "sex": p.sex,
            "neighbors": [str(n) for n in self.relations.neighbors(p)]
        }

    def stats(self):
        return collections.OrderedDict([
            ("uid", self.uid),
            ("i__start_size", self.init_size),
            ("i__cons_marry_allow_thr", self.ill_rel_deg_thr),
            ("i__cons_marry_chance", self.cons_marry_chance),
            ("d__incident_rate", len(self.died_disorder)/self.offspring_count),
            ("c__gen_count", self.gen_count),
            ("c__min_offspring", self.min_offspring),
            ("c__max_offspring", self.max_offspring),
            ("c__min_gen_grow", self.min_gen_grow),
            ("c__max_gen_grow", self.max_gen_grow),
            ("c__death_age", self.death_age),
            ("c__min_marry_age", self.min_marry_age),
            ("c__max_marry_age", self.max_marry_age),
            ("c__gene_healthy_weight", self.gene_healthy_weight),
            ("c__gene_trait_weight", self.gene_trait_weight),
            ("c__gene_affect_weight", self.gene_affect_weight),
            ("s__stop_marry_prop", self.stop_marry_count/self.marry_count),
            ("s__cons_marry_prop", self.cons_marry_count/self.marry_count),
            ("s__die_dis_count", len(self.died_disorder)),
            ("s__die_age_count", len(self.died_age)),
            ("s__stop_marry_count", self.stop_marry_count),
            ("s__offspring_count", self.offspring_count),
            ("s__cons_marry_count", self.cons_marry_count),
            ("s__marry_count", self.marry_count),
            ("s__end_size", self.size()),
            ("s__end_kid_count", len(self.groups["kids"])),
            ("s__end_senior_count", len(self.groups["seniors"])),
            ("s__end_male_unmarry_count", len(self.groups["males"])),
            ("s__end_female_unmarry_count", len(self.groups["females"])),
            ("s__end_male_marry_count", len(self.groups["males_married"])),
            ("s__end_female_marry_count", len(self.groups["females_married"])),
        ])

    def stats_pd(self):
        data = self.stats()
        return pd.DataFrame(data, columns=data.keys(), index=[data["uid"]])

    def stats_to_csv(self, csvfile, header):
        self.stats_pd().to_csv(
            csvfile, index=False, mode='a', header=header)


def person_get_offspring(p1, p2):
    return Person(
        age=1,
        sex=random.choice(list(SEX)),
        genes=person_inherited_gene(p1, p2)
    )


def person_inherited_gene(p1, p2):
    # AA AA
    if p1.genes == GENE.AA and p2.genes == GENE.AA:
        return GENE.AA

    # aa aa
    if p1.genes == GENE.aa and p2.genes == GENE.aa:
        return GENE.aa

    # AA aa
    if ((p1.genes == GENE.AA and p2.genes == GENE.aa) or
            (p1.genes == GENE.aa and p2.genes == GENE.AA)):
        return GENE.Aa

    # Aa AA
    if ((p1.genes == GENE.Aa and p2.genes == GENE.AA) or
            (p1.genes == GENE.AA and p2.genes == GENE.Aa)):
        return random.choice([GENE.AA, GENE.AA, GENE.Aa, GENE.Aa])

    # Aa aa
    if ((p1.genes == GENE.Aa and p2.genes == GENE.aa) or
            (p1.genes == GENE.aa and p2.genes == GENE.Aa)):
        return random.choice([GENE.Aa, GENE.Aa, GENE.aa, GENE.aa])

    # Aa Aa
    return random.choice([GENE.AA, GENE.Aa, GENE.Aa, GENE.aa])


def person_get_random():
    return Person(
        age=1,
        sex=random.choice(list(SEX)),
        genes=None,
    )
