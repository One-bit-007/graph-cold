# D9.5 Baseline Smoke Gate

- Seed: 42
- Passed methods: Decoupling

## Rows
- cicids2017 clean Decoupling: Macro-F1=0.6184000947901945, ERR_final=1.0, retained=0.05505243504510635, passed=True, reason=none
- cicids2017 clean FINE-style: Macro-F1=0.9902675229672636, ERR_final=1.0, retained=1.0, passed=True, reason=none
- cicids2017 symmetric Decoupling: Macro-F1=0.6026826729017022, ERR_final=0.10372124011169304, retained=0.04099625792553605, passed=True, reason=none
- cicids2017 symmetric FINE-style: Macro-F1=0.44757567024350975, ERR_final=0.6759151836237092, retained=0.8000092005222089, passed=False, reason=macro_f1_below_0_50
- cesnet_tls_year22 clean Decoupling: Macro-F1=0.803891474673512, ERR_final=1.0, retained=0.037342276938178695, passed=True, reason=none
- cesnet_tls_year22 clean FINE-style: Macro-F1=0.9948935194041222, ERR_final=1.0, retained=1.0, passed=True, reason=none
- cesnet_tls_year22 symmetric Decoupling: Macro-F1=0.7748837545982462, ERR_final=0.2593238686694197, retained=0.1370580175463711, passed=True, reason=none
- cesnet_tls_year22 symmetric FINE-style: Macro-F1=0.8001510502208922, ERR_final=0.7731361250066917, retained=0.8002004130963823, passed=True, reason=none

## Failed
- FINE-style: cicids2017/symmetric: macro_f1_below_0_50
