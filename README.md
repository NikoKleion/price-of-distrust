# price-of-distrust

Key rate cost of trusted relay vs carry only on a quantum repeater chain, plus routing algebra isotonicity checks.

## Usage

Python 3.10+, numpy.

```
cd routing
python isotonicity_check.py
python t9_label_check.py

cd ../chain
python price_of_distrust.py
```

Outputs are in `results/`.

## References

1. Tang et al. Routing in Non-Isotonic Quantum Networks. arXiv:2511.20628
2. Ercetin, Gedik. Fidelity-Age-Aware Scheduling in Quantum Repeater Networks. arXiv:2602.09562
3. Bacciottini et al. Leveraging Internet Principles to Build a Quantum Network. arXiv:2410.08980
4. Pirandola et al. Fundamental limits of repeaterless quantum communications. arXiv:1510.08863
5. Sobrinho. An Algebraic Theory of Dynamic Network Routing. doi:10.1109/TNET.2005.857111
6. Caleffi. Optimal Routing for Quantum Networks. doi:10.1109/ACCESS.2017.2763325
