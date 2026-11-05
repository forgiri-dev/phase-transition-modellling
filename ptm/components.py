"""
Pure-component critical constants and acentric factors.

Critical temperature/pressure are taken from this project's own NIST
Chemistry WebBook extraction (see data/raw/extract.py and data/vle/*.csv),
so the Peng-Robinson critical point matches the reference curve used for
validation exactly, by construction (that is *not* true of a mixture's
critical point, which the EOS has to predict).

Acentric factors omega are the standard literature values tabulated in:
  Poling, B.E.; Prausnitz, J.M.; O'Connell, J.P. "The Properties of Gases
  and Liquids", 5th ed., McGraw-Hill, 2001, Appendix A.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Component:
    name: str
    cas: str
    Tc: float       # K
    Pc: float       # bar
    omega: float    # acentric factor, dimensionless
    Mw: float       # g/mol

    @property
    def Pc_Pa(self) -> float:
        return self.Pc * 1e5


METHANE = Component("methane", "74-82-8", Tc=190.56, Pc=45.99, omega=0.0115, Mw=16.043)
ETHANE = Component("ethane", "74-84-0", Tc=305.32, Pc=48.72, omega=0.0995, Mw=30.070)
PROPANE = Component("propane", "74-98-6", Tc=369.89, Pc=42.51, omega=0.1523, Mw=44.097)
NBUTANE = Component("n-butane", "106-97-8", Tc=425.12, Pc=37.96, omega=0.2002, Mw=58.123)
CO2 = Component("carbon dioxide", "124-38-9", Tc=304.13, Pc=73.77, omega=0.2236, Mw=44.010)
# n-pentane: used only to anchor the log-log Tc/Pc/omega-vs-M regression in
# continuous.py (extrapolated to C7+ pseudo-components); not part of the
# validated binary/ternary systems. Values from Poling et al., Appendix A.
NPENTANE = Component("n-pentane", "109-66-0", Tc=469.70, Pc=33.70, omega=0.2510, Mw=72.150)

REGISTRY = {c.name: c for c in (METHANE, ETHANE, PROPANE, NBUTANE, CO2, NPENTANE)}


def get(name: str) -> Component:
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"Unknown component {name!r}; available: {sorted(REGISTRY)}")
