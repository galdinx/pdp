#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Génère un fichier XML correctement formaté et indenté à partir :
- d'un template Jinja2 (tolérant aux variables manquantes),
- d'un YAML de variables par défaut (optionnel),
- d'un YAML de variables spécifiques (obligatoire, passé en argument).

Caractéristiques :
  - Pretty-print forcé avec minidom (pas d'ElementTree pour la sortie).
  - Si le XML est mal formé, affiche un WARNING et écrit le rendu brut tel quel.
  - Supprime la déclaration XML (<?xml ...?>) dans tous les cas.
  - Active l'extension Jinja2 'do'.

Chemins par défaut :
  - Template : ./templates/template.xml.j2
  - Defaults : ./config/defaults.yaml (ignoré s'il n'existe pas)

Dépendances :
  pip install Jinja2 PyYAML
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
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

from xml.dom import minidom
from xml.parsers.expat import ExpatError


# Chemins par défaut
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
    """Fusion récursive : les valeurs de 'override' remplacent celles de 'base'."""
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
        undefined=Undefined,               # tolérant : variables manquantes => chaîne vide
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,
        extensions=["jinja2.ext.do"],      # extension 'do'
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
        raise ValueError("Rendu Jinja2 vide. Vérifiez le template et les variables.")
    return rendered


def strip_xml_declaration(text: str) -> str:
    """
    Supprime la déclaration XML si présente (gère BOM/espaces initiaux).
    """
    lines = text.splitlines()
    if not lines:
        return text
    first = lines[0].lstrip("\ufeff \t")
    if first.startswith("<?xml") and first.rstrip().endswith("?>"):
        return "\n".join(lines[1:])
    return text


def pretty_with_minidom(xml_text: str) -> str:
    """
    Pretty-print forcé avec minidom.
    - Valide la bonne formation (lève ValueError si invalide).
    - Supprime la déclaration XML et les lignes vides superflues.
    - Conserve les préfixes de namespaces du texte source.
    """
    try:
        dom = minidom.parseString(xml_text.encode("utf-8"))
    except ExpatError as e:
        raise ValueError(f"Le rendu n'est pas un XML bien formé : {e}") from e

    pretty_bytes = dom.toprettyxml(indent="  ", encoding="utf-8")
    pretty = pretty_bytes.decode("utf-8")

    # Retirer déclaration XML et lignes vides
    lines = []
    for i, ln in enumerate(pretty.splitlines()):
        if i == 0 and ln.startswith("<?xml"):
            continue
        if ln.strip():
            lines.append(ln)
    return "\n".join(lines)


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

        # Supprimer toute déclaration XML éventuelle dès le rendu (cohérent avec la politique de sortie)
        rendered_no_decl = strip_xml_declaration(rendered)

        # Pretty-print forcé via minidom (avec warning non bloquant si mal formé)
        try:
            output_text = pretty_with_minidom(rendered_no_decl)
        except ValueError as ve:
            print(f"⚠️  WARNING: {ve}", file=sys.stderr)
            output_text = rendered_no_decl  # on écrit le brut (sans en-tête)

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