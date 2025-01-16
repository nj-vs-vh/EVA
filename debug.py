from cr_knee_fit.inference import make_logposterior, logprior

from cr_knee_fit.inference import initial_guess_model
from cr_knee_fit.model import ModelConfig
from cr_knee_fit.types_ import Experiment, FitData, Primary


PRIMARIES = [Primary.H, Primary.He]
EXPERIMENTS = [Experiment.AMS02, Experiment.DAMPE]
R_BOUNDS = (7e2, 1e5)  # GV

fit_data = FitData.load(EXPERIMENTS, PRIMARIES, R_BOUNDS)
config = ModelConfig(
    primaries=PRIMARIES,
    shifted_experiments=[e for e in EXPERIMENTS if e is not Experiment.AMS02],
)

init_model = initial_guess_model(config)
lp = make_logposterior(fit_data, config)
print(lp(init_model.pack()))
