"""The hyperparameters Heyman & Vandeputte ACTUALLY RAN, in one place.

Sourced from Nokia-Bell-Labs/steer-like-the-llm:
  - src/ase/experiment_utils/default_model_configs.py  (epochs, lr, weight_decay, objectives)
  - experiments/llm_steer_instruct/eval.py             (the IFEval per-benchmark overrides)
  - src/ase/steering/steering_base.py                  (SteeringTrainingArguments, LossSpecification)

WHY THIS FILE EXISTS: these values were previously duplicated as module-level constants across
six trainers (proper, all_layer, single_gate, clamp_gate, conceptor, conceptor/matrix,
conceptor/selfproj), several of which had silently drifted from the reference. A shared module
makes "which hyperparameters is this project claiming to replicate" answerable by reading one
file, and makes drift impossible rather than merely discouraged.

IMPORTANT DISTINCTION, because it bit this project once already: the reference's
`SteeringTrainingArguments` DATACLASS DEFAULT for weight_decay is 1e-4, but every actual
experiment config overrides it to 1e-6. Reading the dataclass and reading the experiments gives
two different answers; the experiments are what produced the paper's numbers. Same trap applies
to FocusedSteeredModelConfig.use_steering_coeff_bias (dataclass default True, every experiment
False).
"""

# --- Optimizer / schedule (default_model_configs.py) ---------------------------------------
# Identical across every objective and benchmark in the reference.
LR = 1e-3
WEIGHT_DECAY = 1e-6          # NOT the dataclass default of 1e-4 -- see module docstring
OPTIMIZER = "adamw"          # optim.AdamW, not Adam: decoupled decay
LR_SCHEDULE = "linear"       # get_scheduler("linear", num_warmup_steps=0, num_training_steps=N)
DATA_SHUFFLE_SEED = 123      # SteeringTrainingArguments.data_shuffle_seed

# --- Epochs are OBJECTIVE-DEPENDENT ---------------------------------------------------------
# training_args_focused_psi() -> epochs=15 ; training_args_focused_LL() -> epochs=7.
# This asymmetry is the authors' own evidence that the MSE objective converges more slowly than
# loglikelihood. Training both for the same budget (this project previously used 3 for both)
# confounds "which objective is better" with "which objective converged".
N_EPOCHS_MSE = 15
N_EPOCHS_LL = 7


def epochs_for(mse_weight: float, nll_weight: float) -> int:
    """Reference epoch budget for a (mse_weight, nll_weight) endpoint. Only the two
    mutually-exclusive endpoints are reference configurations; a blend is this project's own
    extension (the reference's `combined` objective) and has no published epoch count, so it
    falls back to the longer MSE budget."""
    if nll_weight > 0.0 and mse_weight == 0.0:
        return N_EPOCHS_LL
    return N_EPOCHS_MSE


# --- The reg / normalization PACKAGE ---------------------------------------------------------
# regularization_coefficient and normalize_psi_loss are NOT independent knobs. Normalization
# exists solely to fix the MSE:regularization ratio (raw MSE magnitude varies substantially by
# layer and model, so without it that ratio is arbitrary and layer-dependent). Consequences:
#
#   reg ON  + normalize ON   -> coherent. The persona-vectors / AxBench configuration.
#   reg OFF + normalize OFF  -> coherent. The IFEval configuration (regularization_coefficient
#                               =None in experiments/llm_steer_instruct/eval.py). With a
#                               single-term loss, normalization is a pure LR rescale, so leaving
#                               it off keeps the reference LR of 1e-3 meaning what it meant when
#                               they tuned it.
#   reg ON  + normalize OFF  -> INCOHERENT. Arbitrary, layer-dependent loss balance. (This was
#                               this project's state before 2026-09-20.)
#   reg OFF + normalize ON   -> INCOHERENT. Silently trains at 1e-3/avg_psi instead of 1e-3.
#
# Exposed as named packages rather than two booleans so the incoherent cells aren't reachable
# by accident.
LOSS_BALANCE_PACKAGES = {
    # Instruction-following. The closest analogue to caveman, and the default here.
    "ifeval": {"reg_coeff": 0.0, "normalize_psi": False},
    # Persona vectors / AxBench.
    "persona": {"reg_coeff": 0.1, "normalize_psi": True},
}
DEFAULT_LOSS_BALANCE = "ifeval"


def loss_balance(package: str = DEFAULT_LOSS_BALANCE) -> dict:
    if package not in LOSS_BALANCE_PACKAGES:
        raise ValueError(
            f"unknown loss-balance package {package!r}; expected one of "
            f"{sorted(LOSS_BALANCE_PACKAGES)}. These are deliberately packages rather than "
            f"independent flags -- see this module's docstring for why the mixed settings are "
            f"incoherent."
        )
    return dict(LOSS_BALANCE_PACKAGES[package])


# --- Architecture flags that every reported experiment sets -----------------------------------
# FocusedSteeredModelConfig's dataclass default is use_steering_coeff_bias=True, but
# base_architecture_focused() sets it False, and the IFEval config re-asserts False. The
# parameter itself IS faithful (it is b_{m,l} in paper Section 3.6, and the reference computes
# `user_steering_coeffs + steering_coeff_bias`, which equals this project's `1.0 + coeff_bias`
# at alpha=1) -- it is simply disabled in every result the paper reports.
USE_COEFF_BIAS = False

# steering_location: "answer_only" in the dataclass default, in IFEval, and in AxBench;
# "question_and_answer" only in the persona-vectors eval. Kept here for documentation; the
# actual masking lives in steering/psr/gate.py::answer_only_mask.
STEERING_LOCATION = "answer_only"
