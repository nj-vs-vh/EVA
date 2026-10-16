import numpy as np
from twopop_kneebump import run_2_or_3_pop

from cr_knee_fit.local import LocalRunOptions

if __name__ == "__main__":
    outdir = "3pop_significance"

    opts = LocalRunOptions(
        mcmc=False,
        export=False,
        overwrite=True,
        plots_only=True,
        no_stdout=False,
        custom_run_name=None,
        args_raw=None,
    )

    for is_3pop in (False, True):
        kw = dict(  # noqa: C408
            opts=opts,
            omit_detailed_plots=True,
            add_population_3=is_3pop,
        )
        for lhaaso_him in ("sibyll", "epos", "qgsjet"):
            dirname = f"{outdir}/{'3pop' if is_3pop else '2pop'}/lhaaso_{lhaaso_him}"
            run_2_or_3_pop(
                analysis_name=f"{dirname}/uncorrelated",
                lhaaso_him=lhaaso_him,
                chi2_method="dimidated",
                **kw,  # type: ignore
            )
            for lambda_syst in np.geomspace(0.01, 10, 15):
                run_2_or_3_pop(
                    analysis_name=f"{dirname}/lambda={lambda_syst}",
                    lhaaso_him=lhaaso_him,
                    chi2_method="correlated",
                    lambda_syst=lambda_syst,
                    **kw,  # type: ignore
                )
