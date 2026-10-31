from pathlib import Path


def guvenli_cikti_yolu(path: Path) -> Path:
    """
    Var olan dosyanin uzerine yazmaz.

    x.xlsx yoksa    -> x.xlsx
    x.xlsx varsa    -> x_V1.xlsx
    x_V1 varsa      -> x_V2.xlsx
    """
    path = Path(path)

    if not path.exists():
        return path

    version = 1

    while True:
        aday = path.with_name(
            f"{path.stem}_V{version}{path.suffix}"
        )

        if not aday.exists():
            return aday

        version += 1


def guvenli_cikti_grubu(*paths):
    """
    Birbirine ait Excel/CSV/rapor dosyalarina
    ayni surum numarasini verir.

    Ornek:
        x.xlsx
        x.csv
        rapor.txt

    bunlardan herhangi biri varsa tum yeni ciktilar
    _V1, sonra _V2 ... olarak birlikte uretilir.
    """
    paths = [Path(p) for p in paths]

    if not any(p.exists() for p in paths):
        return tuple(paths)

    version = 1

    while True:
        adaylar = [
            p.with_name(
                f"{p.stem}_V{version}{p.suffix}"
            )
            for p in paths
        ]

        if not any(p.exists() for p in adaylar):
            return tuple(adaylar)

        version += 1
