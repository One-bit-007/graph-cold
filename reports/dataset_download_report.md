# Dataset Download Report

- Dataset: cicids2017
- Mode: local-zip
- Download attempted: False
- Download success: True
- Official source used: True
- Third-party mirror used: False
- Manual action required: False
- Files present: 8/8
- Dataset hash: `2585508ac445a94a3eb2244aa64778678928d201555396b4b9afc1ed6a2f1ab4`
- Audit passed: True

## Manual Instructions
- Open the official CICIDS-2017 page: https://www.unb.ca/cic/datasets/ids-2017.html
- Use the official Download this dataset link and complete the CIC form if required.
- Download MachineLearningCSV.zip, not the PCAP bundle.
- Run: python scripts/download_cicids2017.py --mode local-zip --zip path/to/MachineLearningCSV.zip --out data/cicids2017
