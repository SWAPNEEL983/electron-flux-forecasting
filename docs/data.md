# Data

**Source:** ISRO GRASP electron and proton flux, publicly available from [add portal link here].
The data is not redistributed in this repository.

**Period used:** 2017-07-01 to 2018-09-01, 5-minute resolution (122,976 grid slots).

## Expected input file
`grasp_combined.csv` with these columns (this is what `load_raw` in `src/core.py` reads):

| Column | Meaning |
|---|---|
| `timestamp` | sample time, parsed as a date-time |
| `electron_flux` | electron flux |
| `proton_flux` | proton flux |
| `source_file` | name of the downloaded file the row came from |

**How it was built:** [add one line, for example "downloaded files concatenated, one row per sample, source file recorded"].

## What the notebook does with it
It snaps the samples to a regular 5-minute grid without dropping rows, keeps the raw values, adds quality flags,
and writes the grid and a manifest. Zero proton values cannot be logged and are treated as missing.
Compare the grid digest in `docs/protocol.md` to confirm you have the same data.
