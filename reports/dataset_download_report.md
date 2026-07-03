# Dataset Download Report

- Dataset: cicids2017
- Mode: instructions
- Download attempted: False
- Download success: False
- Official source used: True
- Third-party mirror used: False
- Manual action required: True
- Files present: 0/8
- Dataset hash: `None`
- Audit passed: False

## Manual Instructions
- Open the official CICIDS-2017 page: https://www.unb.ca/cic/datasets/ids-2017.html
- Use the official Download this dataset link and complete the CIC form if required.
- Download MachineLearningCSV.zip, not the PCAP bundle.
- Run: python scripts/download_cicids2017.py --mode local-zip --zip path/to/MachineLearningCSV.zip --out data/cicids2017
