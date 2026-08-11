"""Tier-1 tests for the LIBs-era electrical special-constraint reader (E9-1).

Synthetic-fixture only — a fake ``Restricoes`` returns pandas DataFrames shaped
like idecomp's ``df=True`` accessors, so these run in CI with no real deck.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date

import pandas as pd
import pytest

from cobre_bridge.decomp.libs_electrical import (
    ActivationRule,
    Bucket,
    Comparison,
    DataContext,
    ElectricalAlias,
    ElectricalExpression,
    ElectricalRestriction,
    LibsElectricalModel,
    ParsedTerm,
    ViolationTreatment,
    classify_term,
    classify_terms,
    evaluate_rule,
    is_always_active,
    parse_activation_rule,
    parse_linear_expression,
    read_libs_electrical,
)
from cobre_bridge.decomp.temporal import OperativeStage


class _StubRestricoes:
    """Return a preset DataFrame (or ``None``) for each ``Restricoes`` accessor."""

    def __init__(self, **frames: pd.DataFrame) -> None:
        self._frames = frames

    def __getattr__(self, name: str):  # noqa: ANN204 - test double
        if name.startswith("_"):
            raise AttributeError(name)

        def accessor(df: bool = True) -> pd.DataFrame | None:
            return self._frames.get(name)

        return accessor


def _stage(index: int) -> OperativeStage:
    return OperativeStage(
        index=index,
        start_date=date(2026, 7, 4),
        end_date=date(2026, 7, 11),
        season_id=6,
        block_hours=(168.0,),
    )


def _calendar(n_stages: int) -> list[OperativeStage]:
    return [_stage(i) for i in range(n_stages)]


def _expressao(*rows: tuple) -> pd.DataFrame:
    """EXPRESSAO-ELETRICA rows: (codigo_expressao, identificador_expressao, formula)."""
    return pd.DataFrame(
        rows, columns=["codigo_expressao", "identificador_expressao", "formula"]
    )


def _alias(*rows: tuple) -> pd.DataFrame:
    """ALIAS-ELETRICO rows: (codigo_alias, identificador_alias)."""
    return pd.DataFrame(rows, columns=["codigo_alias", "identificador_alias"])


def _alias_valor(*rows: tuple) -> pd.DataFrame:
    """ALIAS-ELETRICO-VALOR-PERIODO-PATAMAR rows: (codigo_alias, estagio_inicio,
    estagio_fim, patamar, valor)."""
    return pd.DataFrame(
        rows,
        columns=["codigo_alias", "estagio_inicio", "estagio_fim", "patamar", "valor"],
    )


def _re_formula(*rows: tuple) -> pd.DataFrame:
    """RESTRICAO-ELETRICA-FORMULA rows: (codigo_restricao, formula)."""
    return pd.DataFrame(rows, columns=["codigo_restricao", "formula"])


def _re_lim_formula(*rows: tuple) -> pd.DataFrame:
    """RESTRICAO-ELETRICA-LIMITES-FORMULA-PERIODO-PATAMAR rows: (codigo_restricao,
    estagio_inicio, estagio_fim, patamar, limite_inferior, limite_superior)."""
    return pd.DataFrame(
        rows,
        columns=[
            "codigo_restricao",
            "estagio_inicio",
            "estagio_fim",
            "patamar",
            "limite_inferior",
            "limite_superior",
        ],
    )


def _re_inequacao(*rows: tuple) -> pd.DataFrame:
    """RESTRICAO-ELETRICA-INEQUACAO rows: (codigo_restricao, formula, operador,
    formula_limite)."""
    return pd.DataFrame(
        rows, columns=["codigo_restricao", "formula", "operador", "formula_limite"]
    )


def _re_inequacao_per_pat(*rows: tuple) -> pd.DataFrame:
    """RESTRICAO-ELETRICA-INEQUACAO-PERIODO-PATAMAR rows: (codigo_restricao,
    estagio_inicio, estagio_fim, patamar, formula, operador, formula_limite)."""
    return pd.DataFrame(
        rows,
        columns=[
            "codigo_restricao",
            "estagio_inicio",
            "estagio_fim",
            "patamar",
            "formula",
            "operador",
            "formula_limite",
        ],
    )


def _horizonte(*rows: tuple) -> pd.DataFrame:
    """RESTRICAO-ELETRICA-HORIZONTE-PERIODO rows: (codigo_restricao,
    estagio_inicio, estagio_fim)."""
    return pd.DataFrame(
        rows, columns=["codigo_restricao", "estagio_inicio", "estagio_fim"]
    )


def _habilita(*rows: tuple) -> pd.DataFrame:
    """RESTRICAO-ELETRICA-HABILITA rows: (codigo_restricao, codigo_regra_ativacao)."""
    return pd.DataFrame(rows, columns=["codigo_restricao", "codigo_regra_ativacao"])


def _regra(*rows: tuple) -> pd.DataFrame:
    """RESTRICAO-ELETRICA-REGRA-ATIVACAO rows: (codigo_regra_ativacao,
    regra_ativacao)."""
    return pd.DataFrame(rows, columns=["codigo_regra_ativacao", "regra_ativacao"])


def _tratamento_violacao(*rows: tuple) -> pd.DataFrame:
    """RESTRICAO-ELETRICA-TRATAMENTO-VIOLACAO rows: (codigo_restricao,
    custo_violacao)."""
    return pd.DataFrame(rows, columns=["codigo_restricao", "custo_violacao"])


def test_expressions_indexed_by_name() -> None:
    stub = _StubRestricoes(
        expressao_eletrica=_expressao(
            (1, "FNESE", "ener_interc(3,1) - ener_interc(1,3)"),
            (2, "RNE", "ger_usih(10)"),
        ),
    )
    model = read_libs_electrical(stub, _calendar(6))
    assert model is not None
    assert model.expressions["FNESE"].code == 1
    assert model.expressions["FNESE"].formula == "ener_interc(3,1) - ener_interc(1,3)"
    assert model.expressions["RNE"].code == 2
    assert model.expressions["RNE"].formula == "ger_usih(10)"


def test_aliases_with_period_patamar_values_including_na_patamar() -> None:
    stub = _StubRestricoes(
        alias_eletrico=_alias((1, "MMGDSIN")),
        alias_eletrico_valor_periodo_patamar=_alias_valor(
            (1, 1, 2, 1.0, 100.0),
            (1, 3, 3, float("nan"), 250.0),
        ),
    )
    model = read_libs_electrical(stub, _calendar(6))
    assert model is not None
    alias = model.aliases["MMGDSIN"]
    assert alias.code == 1
    # a two-stage range (estagio 1..2) expands to both 0-based stage indices
    assert alias.values[(0, 1)] == 100.0
    assert alias.values[(1, 1)] == 100.0
    # NA patamar -> None ("all blocks")
    assert alias.values[(2, None)] == 250.0


def test_formula_restriction_two_sided_limits_with_sentinel() -> None:
    stub = _StubRestricoes(
        restricao_eletrica_formula=_re_formula(
            (401, "ger_usih(285)+ger_usih(287)"),
        ),
        restricao_eletrica_horizonte_periodo=_horizonte((401, 1, 6)),
        restricao_eletrica_limite_formula_periodo_patamar=_re_lim_formula(
            (401, 1, 6, float("nan"), -1e31, 6300.0),
        ),
    )
    model = read_libs_electrical(stub, _calendar(6))
    assert model is not None
    restriction = model.restrictions[401]
    assert restriction.is_formula is True
    assert restriction.lhs == "ger_usih(285)+ger_usih(287)"
    assert restriction.operator is None
    assert restriction.rhs is None
    assert restriction.stage_start == 0
    assert restriction.stage_end == 5
    # -1E+31 sentinel normalizes to None; the upper side is verbatim
    assert restriction.limits[(0, None)] == (None, 6300.0)
    assert restriction.limits[(5, None)] == (None, 6300.0)


def test_inequacao_restriction_verbatim_lhs_operator_rhs() -> None:
    stub = _StubRestricoes(
        restricao_eletrica_inequacao=_re_inequacao(
            (402, "disp_usih(261) - ger_usih(261)", ">=", "0.05*(val_demanda(1))"),
        ),
    )
    model = read_libs_electrical(stub, _calendar(6))
    assert model is not None
    restriction = model.restrictions[402]
    assert restriction.is_formula is False
    assert restriction.lhs == "disp_usih(261) - ger_usih(261)"
    assert restriction.operator == ">="
    assert restriction.rhs == "0.05*(val_demanda(1))"


def test_inequacao_period_patamar_overrides_read_verbatim() -> None:
    stub = _StubRestricoes(
        restricao_eletrica_inequacao=_re_inequacao(
            (406, "EXPNE", "<=", "14700"),
        ),
        restricao_eletrica_inequacao_periodo_patamar=_re_inequacao_per_pat(
            (406, 1, 6, 1, "EXPNE", "<=", "14700"),
        ),
    )
    model = read_libs_electrical(stub, _calendar(6))
    assert model is not None
    restriction = model.restrictions[406]
    assert len(restriction.overrides) == 1
    override = restriction.overrides[0]
    assert override.stage_start == 0
    assert override.stage_end == 5
    assert override.patamar == 1
    assert override.lhs == "EXPNE"
    assert override.operator == "<="
    assert override.rhs == "14700"


def test_restriction_habilita_rule_and_violation_joined() -> None:
    stub = _StubRestricoes(
        restricao_eletrica_inequacao=_re_inequacao(
            (501, "demanda(1)", ">=", "1000"),
        ),
        restricao_eletrica_habilita=_habilita((501, 10)),
        restricao_eletrica_regra_ativacao=_regra((10, "demanda(1) > 5000")),
        restricao_eletrica_tratamento_violacao=_tratamento_violacao((501, 1234.5)),
    )
    model = read_libs_electrical(stub, _calendar(6))
    assert model is not None
    restriction = model.restrictions[501]
    assert restriction.habilita == 10
    assert model.rules[10] == "demanda(1) > 5000"
    assert restriction.violation == ViolationTreatment(enabled=True, penalty=1234.5)


def test_horizonte_period_out_of_range_raises_value_error() -> None:
    stub = _StubRestricoes(
        restricao_eletrica_inequacao=_re_inequacao(
            (99, "ger_usih(1)", ">=", "0"),
        ),
        restricao_eletrica_horizonte_periodo=_horizonte((99, 1, 7)),
    )
    with pytest.raises(ValueError) as exc_info:
        read_libs_electrical(stub, _calendar(6))
    message = str(exc_info.value)
    assert "7" in message
    assert "6" in message


def test_no_electrical_cards_returns_none() -> None:
    stub = _StubRestricoes()
    assert read_libs_electrical(stub, _calendar(6)) is None


# ---------------------------------------------------------------------------
# parse_linear_expression (TICKET-002)
# ---------------------------------------------------------------------------


def _model(
    expressions: dict[str, str] | None = None,
    aliases: tuple[str, ...] = (),
    restrictions: dict[int, str] | None = None,
) -> LibsElectricalModel:
    """A minimal hand-built model for parser tests: named expressions keyed by
    ``{name: formula}``, bare alias names (their per-(stage,patamar) values are
    irrelevant to the parser, which never resolves them), and restrictions
    keyed by ``{code: lhs}`` (used only to exercise ``re(R)`` expansion)."""
    return LibsElectricalModel(
        expressions={
            name: ElectricalExpression(code=i, name=name, formula=formula)
            for i, (name, formula) in enumerate((expressions or {}).items())
        },
        aliases={
            name: ElectricalAlias(code=i, name=name, values={})
            for i, name in enumerate(aliases)
        },
        restrictions={
            code: ElectricalRestriction(
                code=code,
                lhs=lhs,
                operator=None,
                rhs=None,
                is_formula=True,
                limits={},
                overrides=(),
            )
            for code, lhs in (restrictions or {}).items()
        },
        rules={},
    )


def test_sign_distribution_binary_minus() -> None:
    model = _model()
    terms = parse_linear_expression("disp_usih(261) - ger_usih(261)", model)
    assert terms == [
        ParsedTerm(coefficient=1.0, token="disp_usih", args=(261,)),
        ParsedTerm(coefficient=-1.0, token="ger_usih", args=(261,)),
    ]


def test_sign_distribution_leading_unary_minus() -> None:
    model = _model()
    terms = parse_linear_expression("-ger_usih(5) + ger_usih(6)", model)
    assert terms == [
        ParsedTerm(coefficient=-1.0, token="ger_usih", args=(5,)),
        ParsedTerm(coefficient=1.0, token="ger_usih", args=(6,)),
    ]


def test_scalar_distributed_over_parenthesized_sum() -> None:
    model = _model()
    terms = parse_linear_expression("0.05*(val_demanda(1) + val_demanda(2))", model)
    assert terms == [
        ParsedTerm(coefficient=0.05, token="val_demanda", args=(1,)),
        ParsedTerm(coefficient=0.05, token="val_demanda", args=(2,)),
    ]


def test_standalone_constant_and_explicit_coefficient_terms() -> None:
    model = _model()
    terms = parse_linear_expression(
        "11000 - 0.04*val_demanda(3) + 0.06*ger_pee(11)", model
    )
    assert terms == [
        ParsedTerm(coefficient=11000.0, token="__const__"),
        ParsedTerm(coefficient=-0.04, token="val_demanda", args=(3,)),
        ParsedTerm(coefficient=0.06, token="ger_pee", args=(11,)),
    ]


def test_named_expression_inlined_no_reference_token_remains() -> None:
    model = _model(expressions={"RNE": "ener_interc(3,1) - ener_interc(1,3)"})
    terms = parse_linear_expression("RNE", model)
    assert terms == [
        ParsedTerm(coefficient=1.0, token="ener_interc", args=(3, 1)),
        ParsedTerm(coefficient=-1.0, token="ener_interc", args=(1, 3)),
    ]
    assert all(term.token != "RNE" for term in terms)


def test_named_expression_inlining_multiplies_through_enclosing_coefficient() -> None:
    model = _model(expressions={"RNE": "ener_interc(3,1) - ener_interc(1,3)"})
    terms = parse_linear_expression("0.5*RNE", model)
    assert terms == [
        ParsedTerm(coefficient=0.5, token="ener_interc", args=(3, 1)),
        ParsedTerm(coefficient=-0.5, token="ener_interc", args=(1, 3)),
    ]


def test_nested_named_expression_inlining() -> None:
    model = _model(
        expressions={
            "OUTER": "2*INNER - ger_usih(1)",
            "INNER": "ger_usih(2) + ger_usih(3)",
        }
    )
    terms = parse_linear_expression("OUTER", model)
    assert terms == [
        ParsedTerm(coefficient=2.0, token="ger_usih", args=(2,)),
        ParsedTerm(coefficient=2.0, token="ger_usih", args=(3,)),
        ParsedTerm(coefficient=-1.0, token="ger_usih", args=(1,)),
    ]


def test_re_reference_inlines_restriction_formula() -> None:
    model = _model(restrictions={401: "ger_usih(285) + ger_usih(287)"})
    terms = parse_linear_expression("re(401)", model)
    assert terms == [
        ParsedTerm(coefficient=1.0, token="ger_usih", args=(285,)),
        ParsedTerm(coefficient=1.0, token="ger_usih", args=(287,)),
    ]


def test_alias_reference_kept_as_single_unexpanded_token() -> None:
    model = _model(aliases=("MMGDSECO",))
    terms = parse_linear_expression("MMGDSECO", model)
    assert terms == [
        ParsedTerm(coefficient=1.0, token="alias", alias_name="MMGDSECO"),
    ]


def test_alias_reference_not_a_numeric_value() -> None:
    model = _model(aliases=("MMGDSECO",))
    terms = parse_linear_expression("MMGDSECO", model)
    assert len(terms) == 1
    assert terms[0].token != "__const__"
    assert terms[0].alias_name == "MMGDSECO"


def test_builtin_bare_tokens_kept_unexpanded() -> None:
    model = _model()
    for name in ("demanda_sin", "val_demanda_sin", "carga_ande"):
        terms = parse_linear_expression(name, model)
        assert terms == [ParsedTerm(coefficient=1.0, token=name)]


def test_cyclic_named_expression_reference_raises_value_error() -> None:
    model = _model(expressions={"A": "B", "B": "A"})
    with pytest.raises(ValueError) as exc_info:
        parse_linear_expression("A", model)
    message = str(exc_info.value)
    assert "A" in message
    assert "B" in message


def test_unbalanced_parentheses_raises_value_error() -> None:
    model = _model()
    with pytest.raises(ValueError, match="ger_usih"):
        parse_linear_expression("ger_usih(261", model)


def test_non_numeric_function_argument_raises_value_error() -> None:
    model = _model()
    with pytest.raises(ValueError, match="abc"):
        parse_linear_expression("ger_usih(abc)", model)


def test_unrecognized_bare_identifier_raises_value_error() -> None:
    model = _model()
    with pytest.raises(ValueError, match="ZZTOP"):
        parse_linear_expression("ZZTOP", model)


def test_se_preserved_as_one_structural_term() -> None:
    model = _model()
    terms = parse_linear_expression("se(demanda_sin > 50000, ger_usih(1), 0)", model)
    assert len(terms) == 1
    term = terms[0]
    assert term.token == "se"
    assert term.coefficient == 1.0
    assert term.args == ()
    assert term.condition == "demanda_sin > 50000"
    assert term.branch_true == "ger_usih(1)"
    assert term.branch_false == "0"


def test_se_condition_is_never_evaluated() -> None:
    # A condition referencing an undeclared identifier would raise if the
    # parser tried to evaluate/parse it as its own linear expression; it must
    # not, since se(...)'s cond is opaque, unparsed text (ticket-008's job).
    model = _model()
    terms = parse_linear_expression(
        "se(totally_unparseable !! syntax, ger_usih(1), ger_usih(2))", model
    )
    assert terms[0].condition == "totally_unparseable !! syntax"


def test_se_with_nested_parens_and_commas_in_branches() -> None:
    model = _model()
    terms = parse_linear_expression(
        "se(demanda(1) > 100, ger_usih(1) + ger_usih(2), ger_usih(3))", model
    )
    term = terms[0]
    assert term.condition == "demanda(1) > 100"
    assert term.branch_true == "ger_usih(1) + ger_usih(2)"
    assert term.branch_false == "ger_usih(3)"


# ---------------------------------------------------------------------------
# classify_term / classify_terms (TICKET-003)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "token",
    ["ger_usih", "ger_usit", "ger_pee", "ger_conjh", "ener_interc", "ener_comerc"],
)
def test_classify_term_bucket_a_decision_tokens(token: str) -> None:
    term = ParsedTerm(coefficient=1.0, token=token, args=(1,))
    assert classify_term(term) == Bucket.A


@pytest.mark.parametrize(
    "term",
    [
        ParsedTerm(coefficient=1.0, token="demanda", args=(1,)),
        ParsedTerm(coefficient=1.0, token="val_demanda", args=(1,)),
        ParsedTerm(coefficient=1.0, token="demanda_sin"),
        ParsedTerm(coefficient=1.0, token="val_demanda_sin"),
        ParsedTerm(coefficient=1.0, token="carga_ande"),
        ParsedTerm(coefficient=1.0, token="alias", alias_name="MMGDSECO"),
        ParsedTerm(coefficient=100.0, token="__const__"),
        ParsedTerm(
            coefficient=1.0,
            token="se",
            condition="demanda_sin > 50000",
            branch_true="ger_usih(1)",
            branch_false="0",
        ),
    ],
)
def test_classify_term_bucket_b_input_data_tokens(term: ParsedTerm) -> None:
    assert classify_term(term) == Bucket.B


def test_classify_term_bucket_c_disp_usih() -> None:
    term = ParsedTerm(coefficient=1.0, token="disp_usih", args=(261,))
    assert classify_term(term) == Bucket.C


def test_classify_term_unrecognized_token_raises_value_error() -> None:
    term = ParsedTerm(coefficient=1.0, token="ger_bogus", args=(1,))
    with pytest.raises(ValueError, match="ger_bogus"):
        classify_term(term)


def test_classify_terms_partitions_mixed_expression_into_a_b_c() -> None:
    model = _model()
    terms = parse_linear_expression(
        "disp_usih(261) - ger_usih(261) + 0.05*carga_ande", model
    )
    partition = classify_terms(terms)
    assert len(partition[Bucket.A]) == 1
    assert len(partition[Bucket.B]) == 1
    assert len(partition[Bucket.C]) == 1
    assert partition[Bucket.A][0].token == "ger_usih"
    assert partition[Bucket.B][0].token == "carga_ande"
    assert partition[Bucket.C][0].token == "disp_usih"


def test_classify_terms_present_empty_lists_for_absent_buckets() -> None:
    partition = classify_terms(
        [ParsedTerm(coefficient=1.0, token="ger_usih", args=(1,))]
    )
    assert partition[Bucket.A] == [
        ParsedTerm(coefficient=1.0, token="ger_usih", args=(1,))
    ]
    assert partition[Bucket.B] == []
    assert partition[Bucket.C] == []


def test_classify_terms_order_preserving_within_bucket() -> None:
    terms = [
        ParsedTerm(coefficient=1.0, token="ger_usih", args=(1,)),
        ParsedTerm(coefficient=1.0, token="ger_usit", args=(2,)),
        ParsedTerm(coefficient=1.0, token="ger_pee", args=(3,)),
    ]
    partition = classify_terms(terms)
    assert [term.args for term in partition[Bucket.A]] == [(1,), (2,), (3,)]


# ---------------------------------------------------------------------------
# parse_activation_rule / evaluate_rule / is_always_active (TICKET-004)
# ---------------------------------------------------------------------------


def _dict_context(values: Mapping[str, float]) -> DataContext:
    """A trivial dict-backed :data:`DataContext` test double.

    Keys are a bare token's own name (``"demanda_sin"``), or
    ``"token(args)"`` for a parameterized token (``"demanda(1)"``), or the
    alias's own name for an unresolved ``ALIAS-ELETRICO`` reference. A bare
    numeric constant always resolves to ``1.0`` (its value already lives in
    the term's own ``coefficient`` — see :data:`DataContext`), so it is never
    looked up in *values*.
    """

    def resolve(term: ParsedTerm) -> float:
        if term.token == "__const__":
            return 1.0
        if term.token == "alias":
            assert term.alias_name is not None
            return values[term.alias_name]
        key = (
            f"{term.token}({','.join(str(arg) for arg in term.args)})"
            if term.args
            else term.token
        )
        return values[key]

    return resolve


@pytest.mark.parametrize("op", ["<", "<=", ">", ">=", "==", "!="])
def test_parse_activation_rule_recognizes_each_operator_longest_match(
    op: str,
) -> None:
    model = _model()
    rule = parse_activation_rule(f"demanda_sin {op} 50000", model)
    assert len(rule.comparisons) == 1
    assert rule.comparisons[0].op == op


def test_parse_activation_rule_single_comparison() -> None:
    model = _model()
    rule = parse_activation_rule("demanda_sin >= 50000", model)
    assert rule == ActivationRule(
        comparisons=(
            Comparison(
                left=[ParsedTerm(coefficient=1.0, token="demanda_sin")],
                op=">=",
                right=[ParsedTerm(coefficient=50000.0, token="__const__")],
            ),
        )
    )


def test_parse_activation_rule_alias_and_sum_operands() -> None:
    model = _model(aliases=("MMGDSECO",))
    rule = parse_activation_rule("demanda(3)+demanda(4) < MMGDSECO", model)
    assert len(rule.comparisons) == 1
    comparison = rule.comparisons[0]
    assert comparison.op == "<"
    assert comparison.left == [
        ParsedTerm(coefficient=1.0, token="demanda", args=(3,)),
        ParsedTerm(coefficient=1.0, token="demanda", args=(4,)),
    ]
    assert comparison.right == [
        ParsedTerm(coefficient=1.0, token="alias", alias_name="MMGDSECO"),
    ]


def test_parse_activation_rule_multi_conjunct_splits_on_top_level_ampersand() -> None:
    model = _model()
    rule = parse_activation_rule("demanda_sin >= 50000 & demanda(1) < 1000", model)
    assert len(rule.comparisons) == 2
    first, second = rule.comparisons
    assert first == Comparison(
        left=[ParsedTerm(coefficient=1.0, token="demanda_sin")],
        op=">=",
        right=[ParsedTerm(coefficient=50000.0, token="__const__")],
    )
    assert second == Comparison(
        left=[ParsedTerm(coefficient=1.0, token="demanda", args=(1,))],
        op="<",
        right=[ParsedTerm(coefficient=1000.0, token="__const__")],
    )


def test_parse_activation_rule_bucket_a_operand_raises_value_error() -> None:
    model = _model()
    with pytest.raises(ValueError, match="ger_usih"):
        parse_activation_rule("ger_usih(5) >= 100", model)


def test_parse_activation_rule_conjunct_without_operator_raises_value_error() -> None:
    model = _model()
    with pytest.raises(ValueError, match="no comparison operator"):
        parse_activation_rule("demanda_sin 50000", model)


def test_parse_activation_rule_unrecognized_operator_raises_value_error() -> None:
    model = _model()
    with pytest.raises(ValueError, match="unrecognized comparison operator"):
        parse_activation_rule("demanda_sin <> 50000", model)


def test_parse_activation_rule_unbalanced_parens_raises_value_error() -> None:
    model = _model()
    with pytest.raises(ValueError, match="unbalanced parentheses"):
        parse_activation_rule("demanda(3 >= 5", model)


def test_evaluate_rule_and_of_comparisons_true() -> None:
    model = _model()
    rule = parse_activation_rule("demanda_sin >= 50000 & demanda(1) < 1000", model)
    context = _dict_context({"demanda_sin": 60000.0, "demanda(1)": 900.0})
    assert evaluate_rule(rule, context) is True


def test_evaluate_rule_and_of_comparisons_false_when_one_fails() -> None:
    model = _model()
    rule = parse_activation_rule("demanda_sin >= 50000 & demanda(1) < 1000", model)
    context = _dict_context({"demanda_sin": 60000.0, "demanda(1)": 1500.0})
    assert evaluate_rule(rule, context) is False


def test_evaluate_rule_empty_conjunction_is_always_true() -> None:
    rule = ActivationRule(comparisons=())
    assert evaluate_rule(rule, _dict_context({})) is True


def test_is_always_active_true_for_no_habilita_restriction() -> None:
    assert is_always_active(ActivationRule(comparisons=())) is True


def test_is_always_active_false_for_a_parsed_rule() -> None:
    model = _model()
    rule = parse_activation_rule("demanda_sin >= 50000", model)
    assert is_always_active(rule) is False
