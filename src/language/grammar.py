"""
Parse the grammar into the classes
DisjunctiveFormula = List(ConjunctiveFormula)
ConjunctiveFormula = (ConjunctiveFormula)
                   = ConjunctiveFormula|ConjunctiveFormula
                   = ConjunctiveFormula&ConjunctiveFormula
                   = !ConjunctiveFormula
                   = Predicate(Term,Term)

Predicate = r[number]
Term = e[number]
     = u[number]
     = l[number]
     = f[number]
"""
from typing import Union

from .foq import Conjunction, Disjunction, Formula, Lobject, Negation, Atomic, Term, DisjunctiveFormula, \
    ConjunctiveFormula


def remove_outmost_backets(lstr: str):
    if not (lstr[0] == '(' and lstr[-1] == ')'):
        return lstr

    bracket_stack = []
    for i, c in enumerate(lstr):
        if c == '(':
            bracket_stack.append(i)
        elif c == ')':
            left_bracket_index = bracket_stack.pop(-1)

    assert len(bracket_stack) == 0
    if left_bracket_index == 0:
        return lstr[1:-1]
    else:
        return lstr


def remove_brackets(lstr: str):
    new_lstr = remove_outmost_backets(lstr)
    while new_lstr != lstr:
        lstr = new_lstr
        new_lstr = remove_outmost_backets(lstr)
    return lstr


def map_term_name_to_type(name: str):
    c = name[0]
    if c == 'e':
        return Term.EXISTENTIAL, True
    elif c == 'f':
        return Term.FREE, True
    elif c == 'u':
        return Term.UNIVERSAL, True
    elif c == 's':
        return Term.SYMBOL, True
    else:
        assert name.isnumeric()
        term_id = int(name)
        return term_id, False


def parse_term(term_name):
    assert ')' not in term_name
    term_state, is_abstract = map_term_name_to_type(term_name)
    if is_abstract:
        term = Term(state=term_state, name=term_name)
    else:
        term = Term(state=Term.SYMBOL, name="symbol_by_id")
        term.entity_id_list.append(term_state)
    return term


def parse_lstr_to_lformula(lstr: str) -> Formula:
    """
    parse the string a.k.a, lstr to lobject
    This parser has a defect: the outer binary operator always need to be the last one.
    """
    _lstr = remove_brackets(lstr)

    # identify top-level operator
    if lstr[0] == '!':
        sub_lstr = _lstr[1:]
        sub_formula = parse_lstr_to_lformula(sub_lstr)
        if sub_formula.op == 'pred':
            sub_formula.negated = True
        return Negation(formula=sub_formula)

    binary_operator_index = -1
    binary_operator = ""
    for i, c in enumerate(_lstr):
        if c in "&|":
            binary_operator_index = i
            binary_operator = c

    if binary_operator_index >= 0:
        left_lstr = _lstr[:binary_operator_index]
        left_formula = parse_lstr_to_lformula(left_lstr)
        right_lstr = _lstr[binary_operator_index + 1:]
        right_formula = parse_lstr_to_lformula(right_lstr)
        if binary_operator == '&':
            return Conjunction(formulas=[left_formula, right_formula])
        if binary_operator == '|':
            return Disjunction(formulas=[left_formula, right_formula])

    else:  # parse predicate
        assert _lstr[-1] == ')'
        predicate_name, right_lstr = _lstr.split('(')
        right_lstr = right_lstr[:-1]
        term1_name, term2_name = right_lstr.split(',')

        term1 = parse_term(term1_name)
        term2 = parse_term(term2_name)
        if predicate_name.isnumeric():
            predicate_id = int(predicate_name)
            predicate = Atomic(name="predicate_by_id",
                               head=term1,
                               tail=term2)
            predicate.relation_id_list.append(predicate_id)
        else:
            predicate = Atomic(name=predicate_name,
                               head=term1,
                               tail=term2)

        return predicate


def parse_lstr_to_lformula_v2(lstr: str) -> Formula:
    """
    parse the string a.k.a, lstr to Formula
    The improvement is that it considers the bracket first, which is correct.
    """

    _lstr = remove_brackets(lstr)
    if _lstr[0] == '!':
        sub_lstr = _lstr[1:]
        sub_formula = parse_lstr_to_lformula_v2(sub_lstr)
        if sub_formula.op == 'pred':
            sub_formula.negated = True
        return Negation(formula=sub_formula)

    # identify top-level operator
    start_index, finish_index = 0, 0
    find_non_predicate_bracket = False
    finish_searching = False
    while not find_non_predicate_bracket and not finish_searching:
        for i in range(finish_index, len(_lstr)):
            if _lstr[i] == '(':
                finish_index = i
                break
        else:
            finish_searching = True  # has to finish since no bracket found
        if not finish_searching:
            finish_index, find_non_predicate_bracket = find_bracket(_lstr, finish_index)

    if not find_non_predicate_bracket:  # No bracket found
        for i in range(len(_lstr) - 1, -1, -1):   # The last operator is dealt in the outer.
            if _lstr[i] in "&|":
                finish_index = i - 1
                break
        else:  # We deal with the term here
            assert _lstr[-1] == ')'
            predicate_name, right_lstr = _lstr.split('(')
            right_lstr = right_lstr[:-1]
            term1_name, term2_name = right_lstr.split(',')

            term1 = parse_term(term1_name)
            term2 = parse_term(term2_name)
            if predicate_name.isnumeric():
                predicate_id = int(predicate_name)
                predicate = Atomic(name="predicate_by_id",
                                   head=term1,
                                   tail=term2)
                predicate.relation_id_list.append(predicate_id)
            else:
                predicate = Atomic(name=predicate_name,
                                   head=term1,
                                   tail=term2)
            return predicate
    else:  # compute the connective in the outer
        assert _lstr[finish_index + 1] in "&|"
    if finish_index != 0:  # binary operator by bracket or only one
        left_lstr = _lstr[:finish_index + 1]
        left_formula = parse_lstr_to_lformula_v2(left_lstr)
        right_lstr = _lstr[finish_index + 2:]
        right_formula = parse_lstr_to_lformula_v2(right_lstr)
        if _lstr[finish_index + 1] == '&':
            return Conjunction(formulas=[left_formula, right_formula])
        if _lstr[finish_index + 1] == '|':
            return Disjunction(formulas=[left_formula, right_formula])


def parse_lstr_to_disjunctive_formula(lstr: str) -> DisjunctiveFormula:
    lformula = parse_lstr_to_lformula_v2(lstr)
    DNF_formula = DNF_Transformation(lformula)
    if isinstance(DNF_formula, Disjunction):
        formula_list = DNF_formula.formulas
    else:
        formula_list = [DNF_formula]
    conjunctive_formulas_list = [ConjunctiveFormula(formula) for formula in formula_list]
    fof = DisjunctiveFormula(conjunctive_formulas_list)
    return fof


def find_bracket(string, start_index):
    bracket_num = 0
    predicate_in = False
    for i in range(start_index, len(string)):
        if string[i] == '(':
            bracket_num += 1
        elif string[i] == ')':
            bracket_num -= 1
        elif string[i] == 'r':
            predicate_in = True
        if bracket_num == 0:
            return i, predicate_in


def DNF_Transformation(original_formula: Formula) -> Formula:
    def DNF_Step(original_formula: Formula) -> Formula:
        new_formula = union_bubble(negation_sink(original_formula))
        return new_formula
    now_lstr = original_formula.lstr()
    new_formula = DNF_Step(original_formula)
    while now_lstr != new_formula.lstr():
        now_lstr = new_formula.lstr()
        new_formula = DNF_Step(new_formula)
    return new_formula


def negation_sink(now_formula) -> Formula:  # This functions ensures the formula be in negation normal form.
    if now_formula.op == 'neg':
        inner_formula = now_formula.formula
        if inner_formula.op == 'disj':
            new_formula = Conjunction(
                formulas=[negation_sink(Negation(formula=sub_formula)) for sub_formula in inner_formula.formulas])
        elif inner_formula.op == 'conj':
            new_formula = Disjunction(
                formulas=[negation_sink(Negation(formula=sub_formula)) for sub_formula in inner_formula.formulas])
        elif inner_formula.op == 'neg':
            new_formula = inner_formula.formula
        else:  # only possible is predicate here
            new_formula = now_formula
    elif now_formula.op == 'disj':
        new_formula = Disjunction(formulas=[negation_sink(sub_formula) for sub_formula in now_formula.formulas])
    elif now_formula.op == 'conj':
        new_formula = Conjunction(formulas=[negation_sink(sub_formula) for sub_formula in now_formula.formulas])
    else:  # only possible is predicate here
        new_formula = now_formula
    return new_formula


def union_bubble(now_formula) -> Formula:
    if now_formula.op == 'conj':
        new_sub_formula_list = [union_bubble(sub_formula) for sub_formula in now_formula.formulas]
        the_sub_u_formula = None
        other_sub_query_list = []
        for sub_formula in new_sub_formula_list:
            if sub_formula.op == 'disj' and the_sub_u_formula is None:
                the_sub_u_formula = sub_formula
            else:
                other_sub_query_list.append(sub_formula)
        if the_sub_u_formula:
            if len(other_sub_query_list):
                left_formula = other_sub_query_list[0]
            else:
                left_formula = Conjunction(formulas=other_sub_query_list)
            new_formula = Disjunction(
                formulas=[Conjunction(formulas=[sub_sub_formula, copy_formula(left_formula)])
                          for sub_sub_formula in the_sub_u_formula.formulas])
            return union_bubble(new_formula)
        else:
            new_formula = Conjunction(formulas=[union_bubble(sub_formula) for sub_formula in now_formula.formulas])
    elif now_formula.op == 'disj':
        new_formula = Disjunction(formulas=[union_bubble(sub_formula) for sub_formula in now_formula.formulas])
    else:  # The negation and term, since we always do the negation_sink before, we can neglect this situation
        new_formula = now_formula
    return new_formula


def copy_formula(now_formula: Union[Formula, Term]) -> Union[Formula, Term]:
    """
    This function only copy a Formula/Term to an empty term, the grounding is not done.
    """
    op = now_formula.op
    if op == 'term':
        _q = Term(state=now_formula.state, name=now_formula.name)
        return _q
    elif op == 'pred':
        _q = Atomic(name=now_formula.name, head=copy_formula(now_formula.head),
                    tail=copy_formula(now_formula.tail))
        return _q
    elif op == 'neg':
        _q = Negation(copy_formula(now_formula.formula))
        return _q
    elif op == 'disj':
        _q = Disjunction([copy_formula(sq) for sq in now_formula.formulas])
        return _q
    elif op == 'conj':
        _q = Conjunction([copy_formula(sq) for sq in now_formula.formulas])
        return _q
    else:
        raise NotImplementedError


def concate_iu_chains(now_formula: Formula) -> Formula:
    """
    Concatenate i/u
    """
    if now_formula.op in ['disj', 'conj']:
        same_formula_list = []
        other_formula_list = []
        for sub_formula in now_formula.formulas:
            if sub_formula.op == now_formula.op:
                same_formula_list.append(sub_formula)
            else:
                other_formula_list.append(sub_formula)
        if len(same_formula_list) == 0:
            now_formula.formulas = [concate_iu_chains(sub_formula) for sub_formula in now_formula.formulas]
            new_formula = now_formula
        else:
            new_sub_formula_list = other_formula_list
            for sub_same_formula in same_formula_list:
                new_sub_formula_list += sub_same_formula.formulas
            if now_formula.op == 'disj':
                new_formula = Disjunction(formulas=new_sub_formula_list)
            else:
                new_formula = Conjunction(formulas=new_sub_formula_list)
            new_formula = concate_iu_chains(new_formula)
    elif now_formula.op == 'neg':
        new_formula = Negation(formula=concate_iu_chains(now_formula.formula))
    elif now_formula.op in ['pred', 'term']:
        new_formula = now_formula
    else:
        raise NotImplementedError
    return new_formula



