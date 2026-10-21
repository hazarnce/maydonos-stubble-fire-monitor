"""Create a reviewable text bundle without local secrets, archives or caches."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


SKIP_DIRS = {".venv", "__pycache__", ".ruff_cache", ".pip-tmp", "cache"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".toml", ".bat", ".js", ".css", ".html", ".yml", ".yaml", ".svg", ".geojson"}
SKIP_ROOTS = {"data", "ml/models", "logs"}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    files: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or path == root / "MaydonOs_tum_dosyalar.txt":
            continue
        relative = path.relative_to(root)
        if any(part in SKIP_DIRS for part in relative.parts) or relative.parts[0] in {"data", "models", "logs"} or relative.parts[:2] == ("ml", "models"):
            continue
        if relative.as_posix() == ".streamlit/secrets.toml" or path.suffix.lower() == ".zip":
            continue
        if path.name == ".gitignore" or path.suffix.lower() in TEXT_SUFFIXES:
            files.append(relative.as_posix())
    files.sort()
    sections = [
        "MAYDONOS PROJE DOSYALARI — BİRLEŞİK METİN PAKETİ",
        "=" * 72,
        f"Proje klasörü: {root}",
        "Oluşturulma: " + datetime.now(ZoneInfo("Europe/Istanbul")).isoformat(timespec="seconds"),
        f"Dosya sayısı: {len(files)}",
        "Gizli anahtarlar (.streamlit/secrets.toml/.env), sanal ortamlar, önbellekler, veri setleri, model ağırlıkları ve ikili arşivler eklenmedi.",
        "Python modülleri, web arayüzü (HTML/CSS/JS), başlatıcılar, bağımlılık listeleri ve dokümantasyon aşağıdadır.",
        "",
        "DOSYA İÇİNDEKİLERİ",
        "-" * 72,
    ]
    for name in files:
        path = root / name
        sections.append(f"- {name} ({path.stat().st_size:,} bayt)")
    for name in files:
        path = root / name
        raw = path.read_bytes()
        try:
            content = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            content = raw.decode("cp1254")
        sections.extend(["", "=" * 96, f"DOSYA: {name}", "=" * 96, content.rstrip(), ""])
    output = root / "MaydonOs_tum_dosyalar.txt"
    output.write_text("\n".join(sections), encoding="utf-8")
    print(f"Paket yazıldı: {output} · {len(files)} dosya · {output.stat().st_size:,} bayt")


if __name__ == "__main__":
    main()
