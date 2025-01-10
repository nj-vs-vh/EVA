# EVA

A novel interpretation of the cosmic ray knee.

## Development

### Virtual environment

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Data preparation

```bash
cd data
python extract.py
python transform.py
```

Now `data/output` should contain text files with data prepared for the fit.

### Running fit

```bash
mkdir output
python model/fit_composition.py
python model/fit_knee.py
# TBD
```
