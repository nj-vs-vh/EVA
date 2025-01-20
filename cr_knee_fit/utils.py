def add_log_margin(min: float, max: float, log_margin: float = 0.1) -> tuple[float, float]:
    frac = max / min
    margin = frac**log_margin
    return min / margin, max * margin
