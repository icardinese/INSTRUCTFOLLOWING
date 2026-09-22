"""Locks in the fidelity fixes applied 2026-09-20 after diffing against
Nokia-Bell-Labs/steer-like-the-llm. Each test names the reference file it encodes, so that if a
future change reintroduces one of these gaps the failure says WHY it's a gap rather than just
that a number moved.
"""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from steering.psr.gate import GateState, coefficient, init_gate_state
from steering.psr.reference_config import (
    LOSS_BALANCE_PACKAGES,
    N_EPOCHS_LL,
    N_EPOCHS_MSE,
    USE_COEFF_BIAS,
    WEIGHT_DECAY,
    epochs_for,
    loss_balance,
)


def test_weight_decay_is_the_experiment_value_not_the_dataclass_default():
    """SteeringTrainingArguments' dataclass default is 1e-4, but every config in
    default_model_configs.py overrides it to 1e-6. The experiments are what produced the paper's
    numbers."""
    assert WEIGHT_DECAY == 1e-6


def test_epoch_budget_is_objective_dependent():
    """training_args_focused_psi() -> epochs=15 ; training_args_focused_LL() -> epochs=7.
    Training both endpoints for the same budget confounds 'which objective is better' with
    'which objective converged', which is exactly the confound that made finding F5 unsafe."""
    assert N_EPOCHS_MSE == 15
    assert N_EPOCHS_LL == 7
    assert epochs_for(mse_weight=1.0, nll_weight=0.0) == N_EPOCHS_MSE
    assert epochs_for(mse_weight=0.0, nll_weight=1.0) == N_EPOCHS_LL


def test_blended_loss_falls_back_to_the_longer_budget():
    """A blend is the reference's `combined` objective, which has no published epoch count."""
    assert epochs_for(mse_weight=1.0, nll_weight=0.5) == N_EPOCHS_MSE


def test_coeff_bias_is_frozen_by_default_and_absent_from_the_optimizer():
    """The parameter is faithful (b_{m,l}, paper Section 3.6) but every reported experiment sets
    use_steering_coeff_bias=False -- both base_architecture_focused() and the IFEval override."""
    assert USE_COEFF_BIAS is False
    gate = init_gate_state(hidden_size=8, device="cpu")
    assert gate.coeff_bias.requires_grad is False
    assert gate.coeff_bias.item() == 0.0
    params = gate.parameters()
    assert len(params) == 2, "a frozen coeff_bias must not be handed to the optimizer"
    assert all(p.requires_grad for p in params)


def test_coeff_bias_can_still_be_enabled_for_the_ablation():
    gate = init_gate_state(hidden_size=8, device="cpu", use_coeff_bias=True)
    assert gate.coeff_bias.requires_grad is True
    assert len(gate.parameters()) == 3


def test_coefficient_formula_matches_reference_at_alpha_one():
    """Reference: desired_concept_presence = user_steering_coeffs + steering_coeff_bias, then
    steering_coefficients = desired_concept_presence * location_fit. At alpha=1 (which the paper
    fixes for IFEval, Section 4.3) that is exactly (1 + coeff_bias) * fit."""
    gate = GateState(
        weight=torch.ones(4, 1) * 0.5,
        bias=torch.tensor([0.0]),
        coeff_bias=torch.tensor([0.25]),
    )
    hidden = torch.ones(1, 2, 4)
    coeff, fit = coefficient(gate, hidden, mask=None)
    assert torch.allclose(coeff, (1.0 + 0.25) * fit)


@pytest.mark.parametrize("package", sorted(LOSS_BALANCE_PACKAGES))
def test_loss_balance_packages_are_internally_coherent(package):
    """reg and normalization are coupled: normalization exists only to fix the MSE:reg ratio.
    reg on without normalization is an arbitrary layer-dependent balance; normalization without
    reg is a silent LR rescale. Both packages must therefore agree on the two flags."""
    cfg = loss_balance(package)
    assert (cfg["reg_coeff"] > 0) == cfg["normalize_psi"]


def test_ifeval_package_disables_both():
    """experiments/llm_steer_instruct/eval.py sets regularization_coefficient=None. This is the
    closest analogue to the caveman task and therefore the project default."""
    assert loss_balance("ifeval") == {"reg_coeff": 0.0, "normalize_psi": False}


def test_persona_package_enables_both():
    """base_architecture_focused() uses regularization_coefficient=0.1, and
    training_args_focused_psi() uses normalize_psi_loss=True."""
    assert loss_balance("persona") == {"reg_coeff": 0.1, "normalize_psi": True}


def test_unknown_package_is_rejected_rather_than_silently_defaulted():
    with pytest.raises(ValueError, match="unknown loss-balance package"):
        loss_balance("reg_on_normalize_off")
