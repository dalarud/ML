"""Kommandorad: python -m bolan {hamta,analysera,backtest,validera}."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .data import ROOT


def main(argv=None):
    ap = argparse.ArgumentParser(prog="bolan", description="ML-verktyg för planering av bolåneräntor")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("hamta", help="Hämta senaste data från SCB och Riksbanken")
    a = sub.add_parser("analysera", help="Full analys + rekommendation + HTML-rapport")
    standard = ROOT / "config" / "min_profil.toml"
    if not standard.exists():
        standard = ROOT / "config" / "profil.toml"
    a.add_argument("--profil", default=str(standard),
                   help="Profil (standard: config/min_profil.toml om den finns, annars exempelprofilen)")
    a.add_argument("--rapport", default=str(ROOT / "rapport" / "bolaneplan.html"))
    a.add_argument("--riskaversion", type=float, help="Överstyr λ i profilen")
    a.add_argument("--omtrana", action="store_true", help="Träna om modellerna även om cache finns")
    sub.add_parser("backtest", help="Historisk backtest av strategier")
    sub.add_parser("validera", help="Visa korsvalideringsresultat för modellerna")
    args = ap.parse_args(argv)
    pd.set_option("display.width", 200)

    if args.cmd == "hamta":
        from .data import hamta_allt
        hamta_allt()
    elif args.cmd == "analysera":
        from .recommend import analysera, las_profil, sammanfatta
        from .report import bygg_rapport
        prof = las_profil(args.profil)
        if args.riskaversion is not None:
            prof.riskaversion = args.riskaversion
        res = analysera(prof, tvinga_traning=args.omtrana)
        print()
        print("\n".join(sammanfatta(res)))
        kol = ["bank", "strategi", "snittranta", "snittranta_p95", "forvantad_kostnad", "cvar", "poang"]
        print("\nTopp 10:")
        print(res["resultat"][kol].head(10).round(2).to_string(index=False))
        ut = bygg_rapport(res, Path(args.rapport))
        print(f"\nRapport: {ut}")
    elif args.cmd == "backtest":
        from .backtest import kor_ml_timing, rullande_fonster, sammanfattning
        from .data import bygg_panel
        p = bygg_panel()
        rf = rullande_fonster(p)
        print("Andel 5-årsfönster där bindningstiden var billigast:")
        print(rf.idxmin(axis=1).value_counts(normalize=True).round(2).to_string())
        s, _ = kor_ml_timing(p)
        print("\nML-timing (walk-forward) vs statiska strategier:")
        print(sammanfattning(s).round(2).to_string())
    elif args.cmd == "validera":
        from .data import bygg_panel
        from .recommend import tranade_modeller
        c = tranade_modeller(bygg_panel())
        for k, v in c["resultat"].items():
            print(f"\n== {k}  (ensemble {v.ensemble:+.2f}, vikter {list(v.vikter)})")
            print(v.cv.round(3).to_string())


if __name__ == "__main__":
    main()
