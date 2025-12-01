"""Credits to this code go to the lecturer of the Historical Linguistics lecture, Harald Hammarström."""


def delta(a, b, insertion_deletion_cost=1, substitution_cost=1):
    if a and b:
        if a == b:
            return 0
        else:
            return substitution_cost
    return insertion_deletion_cost


def D(x, y, insertion_deletion_cost=1, substitution_cost=1):
    lx = len(x)
    ly = len(y)

    d = {}
    d[(0, 0)] = 0
    for i in range(lx):
        d[(i + 1, 0)] = d[(i, 0)] + delta(
            x[i], None, insertion_deletion_cost=1, substitution_cost=substitution_cost
        )
    for j in range(ly):
        d[(0, j + 1)] = d[(0, j)] + delta(
            None, y[j], insertion_deletion_cost=1, substitution_cost=substitution_cost
        )

    for i in range(lx):
        for j in range(ly):
            d[(i + 1, j + 1)] = min(
                d[(i, j)]
                + delta(
                    x[i],
                    y[j],
                    insertion_deletion_cost=1,
                    substitution_cost=substitution_cost,
                ),
                d[(i, j + 1)]
                + delta(
                    x[i],
                    None,
                    insertion_deletion_cost=1,
                    substitution_cost=substitution_cost,
                ),
                d[(i + 1, j)]
                + delta(
                    None,
                    y[j],
                    insertion_deletion_cost=1,
                    substitution_cost=substitution_cost,
                ),
            )

    return d[(lx, ly)]


print()


def tab_to_dict(fn="digphil-hl-lab-mini-forms.tab"):
    with open(fn, "r", encoding="utf-8") as f:
        rows = [row.split("\t") for row in f.read().split("\n") if row.strip()]
    return {row[0]: dict(zip(rows[0][1:], row[1:])) for row in rows[1:]}


# Read the forms and cognacy information
forms = tab_to_dict("3_historical_linguistics/digphil-hl-lab-mini-forms.tab")
cognacy = tab_to_dict("3_historical_linguistics/digphil-hl-lab-mini-cognacy.tab")
print("The forms for BITE are", forms["BITE"])
for lg1, form1 in forms["BITE"].items():
    for lg2, form2 in forms["BITE"].items():
        if lg1 < lg2:
            print(
                lg1,
                form1,
                "and",
                lg2,
                form2,
                "are cognate",
                cognacy["BITE"][lg1] == cognacy["BITE"][lg2],
            )


def cognate_sets(xs=["hello", "yello", "world"], t=0.5):
    dns = {}
    for x1 in xs:
        for x2 in xs:
            if x1 < x2:
                dns[x1, x2] = D(x1, x2) / max(len(x1), len(x2))
    cids = {}
    for pair, dn in dns.items():
        if dn <= t:
            cid = min([cids[x] for x in pair if x in cids] + [len(cids)])
            for x in pair:
                cids[x] = cid
    cognate_sets = {}
    for i, x in enumerate(xs):
        if x in cids:
            cognate_sets[cids[x]] = cognate_sets.get(cids[x], []) + [x]
        else:
            cognate_sets[i + len(xs)] = [x]
    return [set(cset) for cset in cognate_sets.values()]


# print(
#     'Let us divide ["hello", "yello", "world"] into cognate_sets with t = 0.5',
#     cognate_sets(xs=["hello", "yello", "world"], t=0.5),
# )
