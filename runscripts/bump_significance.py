from cr_knee_fit.local import LocalRunOptions
from runscripts.local_source import generate_populations_with_bump
from runscripts.minimal import generate_minimal_populations, run

if __name__ == "__main__":
    # FIXME: seed the minimizer with the optimal point
    for lambda_syst in (0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
        for bump in (False, True):
            for lhaaso_him in ("sibyll", "qgsjet", "epos"):
                analysis_name = (
                    f"bump_significance/{lhaaso_him}/{lambda_syst=}/{'bump' if bump else 'minimal'}"
                )
                print()
                run(
                    analysis_name=analysis_name,
                    lhaaso_him=lhaaso_him,
                    lambda_syst=lambda_syst,
                    opts=LocalRunOptions(
                        mcmc=False,
                        export=False,
                        overwrite=True,
                        plots_only=True,
                        no_stdout=False,
                        custom_run_name=analysis_name,
                        args_raw=None,
                    ),
                    omit_detailed_plots=True,
                    generate_populations=(
                        generate_populations_with_bump if bump else generate_minimal_populations
                    ),
                )
