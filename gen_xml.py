#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Génère un fichier XML correctement formaté et indenté à partir :
- d'un template Jinja2,
- d'un YAML de variables par défaut (optionnel),
- d'un YAML de variables spécifiques (obligatoire, passé en argument).

Comportement :
  - Tente de valider et d'indenter le rendu XML.
  - Si le XML est mal formé, affiche un WARNING et écrit tout de même le rendu brut.
  - Supprime l'en-tête XML dans tous les cas.
  - Active l'extension Jinja2 'do' pour permettre des opérations sans sortie.
  - Tolère les variables manquantes (rendu = chaîne vide au lieu d'erreur).

Chemins par défaut préconfigurés :
  - Template : ./templates/template.xml.j2
  - Defaults : ./config/defaults.yaml (ignoré s'il n'existe pas)

Résolution intelligente des chemins :
  - On tente d'abord le chemin fourni en CLI (s'il y en a un).
  - Sinon, on essaie le chemin par défaut relatif au CWD.
  - Puis relatif au dossier du script.
  - Puis relatif au dossier du YAML spécifique.

Dépendances :
  pip install Jinja2 PyYAML

Exemples :
  python gen_xml.py envs/clientA.yaml
  python gen_xml.py envs/clientA.yaml --template templates/config.xml.j2
  python gen_xml.py envs/clientA.yaml --defaults config/defaults.yaml
  python gen_xml.py envs/clientA.yaml -t templates/config.xml.j2 -d config/defaults.yaml
  python gen_xml.py envs/clientA.yaml -o out/clientA.xml
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import io
from typing import Any, Mapping, MutableMapping, Dict, Iterable

# Dépendances externes
try:
    import yaml  # PyYAML
except ImportError:
    print("Erreur: PyYAML n'est pas installé. Installez-le avec : pip install PyYAML", file=sys.stderr)
    sys.exit(1)

try:
    from jinja2 import Environment, FileSystemLoader, TemplateNotFound, Undefined
except ImportError:
    print("Erreur: Jinja2 n'est pas installé. Installez-le avec : pip install Jinja2", file=sys.stderr)
    sys.exit(1)

import xml.etree.ElementTree as ET
from xml.dom import minidom


# Chemins par défaut préconfigurés
DEFAULT_TEMPLATE_REL = Path("templates/template.xml.j2")
DEFAULT_DEFAULTS_REL = Path("config/defaults.yaml")


def read_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Fichier YAML introuvable: {path}")
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                raise ValueError(f"Le contenu YAML doit être un mapping (dict) à la racine: {path}")
            return data
    except yaml.YAMLError as e:
        raise ValueError(f"YAML invalide dans {path} : {e}") from e


def deep_merge(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Fusion récursive : les valeurs de 'override' prennent le pas sur 'base'."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_merge(base[k], v)  # type: ignore[index]
        else:
            base[k] = v
    return base


def resolve_with_fallbacks(
    preferred: Path | None,
    defaults_rel: Path,
    anchors: Iterable[Path],
    must_exist: bool = True
) -> Path | None:
    """
    Tente, dans l'ordre :
      1) `preferred` (si fourni)
      2) `defaults_rel` relatif à chaque ancre de `anchors`
    Retourne le premier chemin existant (si must_exist=True), sinon l'ultime candidat.
    """
    candidates: list[Path] = []
    if preferred is not None:
        candidates.append(preferred)
    for anchor in anchors:
        candidates.append((anchor / defaults_rel).resolve())

    if must_exist:
        for c in candidates:
            if c.exists():
                return c
        return candidates[0] if candidates else None
    else:
        for c in candidates:
            if c.exists():
                return c
        return candidates[-1] if candidates else None


def load_context(defaults_path: Path | None, specific_yaml: Path) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {}
    if defaults_path and defaults_path.exists():
        defaults = read_yaml(defaults_path)
    specific = read_yaml(specific_yaml)
    return deep_merge(defaults.copy(), specific)


def render_jinja_xml(template_path: Path, context: Dict[str, Any]) -> str:
    if not template_path.exists():
        raise FileNotFoundError(f"Template introuvable: {template_path}")
    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        undefined=Undefined,               # ✅ tolérant : variables manquantes => chaîne vide
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,
        extensions=["jinja2.ext.do"],      # ✅ Extension 'do' activée
    )
    try:
        template = env.get_template(template_path.name)
    except TemplateNotFound as e:
        raise FileNotFoundError(f"Template introuvable: {template_path}") from e

    try:
        rendered = template.render(**context)
    except Exception as e:
        raise RuntimeError(f"Erreur lors du rendu Jinja2 : {e}") from e

    if not rendered.strip():
        # Avec Undefined tolérant, un rendu entièrement vide est suspect.
        raise ValueError("Rendu Jinja2 vide. Vérifiez le template et les variables.")
    return rendered


def strip_xml_declaration(text: str) -> str:
    """
    Supprime la ligne d'en-tête XML si présente.
    Gère les éventuels BOM/espaces en début de fichier.
    """
    lines = text.splitlines()
    out_lines = []
    skipped = False
    for i, ln in enumerate(lines):
        s = ln.lstrip("\ufeff \t")
        if i == 0 and s.startswith("<?xml") and s.rstrip().endswith("?>"):
            skipped = True
            continue
        out_lines.append(ln)
    return "\n".join(out_lines) if skipped else text


def validate_and_pretty_xml(xml_text: str) -> str:
    """
    Valide le XML puis retourne une version indentée **sans en-tête XML**.
    - Utilise ElementTree.indent quand disponible (Python 3.9+).
    - Repli sur minidom sinon, avec suppression de l'en-tête.
    Lève ValueError si le XML est mal formé.
    """
    try:
        root = ET.fromstring(xml_text.encode("utf-8"))  # validation bien-formed
        tree = ET.ElementTree(root)
        try:
            ET.indent(tree, space="  ", level=0)  # Python 3.9+
            buf = io.BytesIO()
            # ⚠️ Pas d'en-tête XML
            tree.write(buf, encoding="utf-8", xml_declaration=False)
            return buf.getvalue().decode("utf-8")
        except AttributeError:
            dom = minidom.parseString(xml_text.encode("utf-8"))
            pretty_bytes = dom.toprettyxml(indent="  ", encoding="utf-8")
            pretty = pretty_bytes.decode("utf-8")
            # Supprimer la ligne d'en-tête et les lignes vides
            lines = [ln for ln in pretty.splitlines() if ln.strip() and not ln.startswith("<?xml")]
            return "\n".join(lines)
    except ET.ParseError as e:
        raise ValueError(f"Le rendu n'est pas un XML bien formé : {e}") from e


def compute_output_path(specific_yaml: Path, override: Path | None) -> Path:
    return override if override else specific_yaml.with_suffix(".xml")


def main():
    parser = argparse.ArgumentParser(
        prog="gen_xml.py",
        description="Génère un XML formaté depuis un template Jinja2 et deux fichiers YAML (défauts + spécifiques)."
    )
    parser.add_argument(
        "specific",
        type=Path,
        help="Chemin du fichier YAML de variables spécifiques (obligatoire)."
    )
    parser.add_argument(
        "-t", "--template",
        type=Path,
        default=None,  # résolu via DEFAULT_TEMPLATE_REL
        help=f"Chemin du template Jinja2 (défaut: {DEFAULT_TEMPLATE_REL})."
    )
    parser.add_argument(
        "-d", "--defaults",
        type=Path,
        default=None,  # résolu via DEFAULT_DEFAULTS_REL (optionnel)
        help=f"Chemin du YAML de variables par défaut (défaut: {DEFAULT_DEFAULTS_REL}, ignoré s'il n'existe pas)."
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        help="Chemin de sortie XML (optionnel). Par défaut, même dossier/nom que le YAML spécifique avec extension .xml."
    )
    args = parser.parse_args()

    # Points d'ancrage pour la résolution des chemins
    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent
    specific_dir = args.specific.resolve().parent

    # Résolution du template (doit exister)
    template_path = resolve_with_fallbacks(
        preferred=args.template,
        defaults_rel=DEFAULT_TEMPLATE_REL,
        anchors=[cwd, script_dir, specific_dir],
        must_exist=True
    )
    if not template_path or not template_path.exists():
        print(f"❌ Template introuvable. Essayé: {args.template or DEFAULT_TEMPLATE_REL}", file=sys.stderr)
        sys.exit(1)

    # Résolution du defaults.yaml (optionnel)
    defaults_path = resolve_with_fallbacks(
        preferred=args.defaults,
        defaults_rel=DEFAULT_DEFAULTS_REL,
        anchors=[cwd, script_dir, specific_dir],
        must_exist=False
    )

    try:
        context = load_context(defaults_path if defaults_path and defaults_path.exists() else None, args.specific)
        rendered = render_jinja_xml(Path(template_path), context)

        # Suppression d'en-tête XML éventuelle dès maintenant (couvre aussi le fallback)
        rendered_no_decl = strip_xml_declaration(rendered)

        # Essai de validation + pretty
        try:
            pretty_xml = validate_and_pretty_xml(rendered_no_decl)
            output_text = pretty_xml
        except ValueError as ve:
            # ⚠️ Warning (pas d'échec), on garde le rendu brut (sans en-tête)
            print(f"⚠️  WARNING: {ve}", file=sys.stderr)
            output_text = rendered_no_decl

        out_path = compute_output_path(args.specific, args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(output_text, encoding="utf-8")

        print(f"✅ XML généré : {out_path}")
    except Exception as e:
        print("❌ Erreur :", str(e), file=sys.stderr)
        # Pour debug détaillé, décommentez :
        # import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
