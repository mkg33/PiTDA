# PiTDA
Projects in TDA 2024

Usage instructions for the `tda_final.py` file.

USAGE: `python3.12 tda_final.py`
Note: tested with 3.12 because there were setup problems with `Gudhi` when using 3.13.

OPTIONAL: Specify test cases to run by their numbers or ranges.
Examples: '1', '1-5', '10 15 20-25'.
USAGE: `python3.12 tda_final.py -t 40-100`.
If you type a test case greater than 60, the code will automatically generate
further examples with random regular graphs (see function below).
You can set the parameters yourself (number of notes and graph degree)
by modifying the code (where the add_random_regular_test_cases function is called).

OPTIONAL: pass the parameter `-nv` to disable visualization.
Note that visualization is available exclusively for planar graphs.
USAGE: `python3.12 tda_final.py -t 40-100 -nv`.

OPTIONAL: pass the parameter -a to abort the computation once the timeout has been reached.
Timeout: 600 seconds (10 minutes) by default. You can modify it yourself and set it to a custom value, if you wish.
USAGE: `python3.12 tda_final.py -t 40-100 -nv -a`.

See the sample outputs and performance plots in the `output` directory.