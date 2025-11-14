#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Génère des fichiers XML correctement formatés et indentés à partir :
- d'un template Jinja2 (tolérant aux variables manquantes),
- d'un YAML de variables par défaut (optionnel),
- d'un ou plusieurs YAML spécifiques (fichier ou répertoire).

Fonctionnement :
  - Pour chaque YAML, génère 2 sorties :
      1) payloadOnly = False  → fichier normal      (ex: foo.xml)
      2) payloadOnly = True   → fichier Webchecker  (ex: foo_webchecker.xml)
  - Entrée : fichier YAML OU répertoire (avec --recursive pour descendre).
  - Pretty-print forcé via minidom.
  - XML mal formé : WARNING et écriture du rendu brut tel quel.
  - Par défaut, **la déclaration XML est conservée**.
    → Option -x / --no-xml-declaration pour **la supprimer**.
  - Jinja2 tolérant aux variables manquantes + extension 'do'.
  - Filtre Jinja 'required' (simple) pour rendre certaines variables obligatoires.
  - Chemins par défaut :
      - Template : ./templates/template.xml.j2
      - Defaults : ./config/defaults.yaml (ignoré s'il n'existe pas).

Dépendances :
  pip install Jinja2 PyYAML
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Dict, Iterable, List, Tuple
import copy

# Dépendances externes
try:
    import yaml  # PyYAML
except ImportError:
    print("Erreur: PyYAML n'est pas installé. Installez-le avec : pip install PyYAML", file=sys.stderr)
    sys.exit(1)

try:
    from jinja2 import Environment, FileSystemLoader, TemplateNotFound, Undefined
    from jinja2.runtime import Undefined as RTUndefined  # pour détecter les valeurs indéfinies dans le filtre
except ImportError:
    print("Erreur: Jinja2 n'est pas installé. Installez-le avec : pip install Jinja2", file=sys.stderr)
    sys.exit(1)

from xml.dom import minidom
from xml.parsers.expat import ExpatError


# Chemins par défaut
DEFAULT_TEMPLATE_REL = Path("templates/template.xml.j2")
DEFAULT_DEFAULTS_REL = Path("config/defaults.yaml")


# -------------------- Utilitaires YAML & fusion --------------------

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
      1) chemin préféré (si fourni)
      2) chemin par défaut relatif à chaque ancre de `anchors`
    Retourne le premier existant (si must_exist=True), sinon l'ultime candidat.
    """
    candidates: List[Path] = []
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


# -------------------- Filtre Jinja : required (simple) --------------------

def _is_effectively_empty(value: Any) -> bool:
    """Vrai si value est vide : '', whitespace, [], (), set(), {}."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    try:
        if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
            return len(value) == 0  # type: ignore[arg-type]
    except Exception:
        pass
    return False


def required(value: Any, name: str | None = None, allow_empty: bool = False) -> Any:
    """
    Filtre Jinja (simple) : rend la valeur obligatoire.
      - Lève ValueError si la valeur est Undefined/None.
      - Si allow_empty=False (défaut), lève aussi si '' / whitespace / [] / () / {} / set().
      - Retourne la valeur inchangée sinon (pour chainage).

    Usage :
      {{ env.endpoint | required('env.endpoint') }}
      {{ comment | required('comment', allow_empty=True) }}
    """
    if isinstance(value, RTUndefined):
        label = f"'{name}'" if name else "variable requise"
        raise ValueError(f"Variable requise manquante : {label}")
    if value is None:
        label = f"'{name}'" if name else "variable requise"
        raise ValueError(f"Variable requise manquante (None) : {label}")
    if not allow_empty and _is_effectively_empty(value):
        label = f"'{name}'" if name else "variable requise"
        raise ValueError(f"Variable requise vide : {label}")
    return value


# -------------------- Rendu / Sortie --------------------

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
    # Filtres et globals personnalisés
    env.filters["required"] = required

    try:
        template = env.get_template(template_path.name)
    except TemplateNotFound as e:
        raise FileNotFoundError(f"Template introuvable: {template_path}") from e

    try:
        rendered = template.render(**context)
    except Exception as e:
        # Erreur générique de rendu (dont ValueError venant de |required)
        raise RuntimeError(f"Erreur lors du rendu Jinja2 : {e}") from e

    if not rendered.strip():
        raise ValueError("Rendu Jinja2 vide. Vérifiez le template et les variables.")
    return rendered


def strip_xml_declaration(text: str) -> str:
    """Supprime la déclaration XML si présente (gère BOM/espaces initiaux)."""
    lines = text.splitlines()
    if not lines:
        return text
    first = lines[0].lstrip("\ufeff \t")
    if first.startswith("<?xml") and first.rstrip().endswith("?>"):
        return "\n".join(lines[1:])
    return text


def pretty_with_minidom(xml_text: str, keep_decl: bool) -> str:
    """
    Pretty-print forcé avec minidom.
    - Valide la bonne formation (lève ValueError si invalide).
    - Conserve les préfixes de namespaces.
    - Conserve ou retire la déclaration XML selon keep_decl.
    """
    try:
        dom = minidom.parseString(xml_text.encode("utf-8"))
    except ExpatError as e:
        raise ValueError(f"Le rendu n'est pas un XML bien formé : {e}") from e

    # toprettyxml avec encoding génère une déclaration XML
    pretty_bytes = dom.toprettyxml(indent="  ", encoding="utf-8")
    pretty = pretty_bytes.decode("utf-8")

    lines: List[str] = []
    for i, ln in enumerate(pretty.splitlines()):
        if i == 0 and ln.startswith("<?xml") and not keep_decl:
            # on supprime la déclaration si demandé
            continue
        if ln.strip():
            lines.append(ln)
        else:
            # retire les lignes totalement vides
            pass
    return "\n".join(lines)


def compute_output_path_for_base(specific_yaml: Path, override: Path | None) -> Path:
    """Chemin du fichier 'base' (payloadOnly=False)."""
    return override if (override and specific_yaml.is_file()) else specific_yaml.with_suffix(".xml")


def webchecker_path_from(base_xml: Path) -> Path:
    """Insère le suffixe '_webchecker' avant l'extension. Ex: foo.xml -> foo_webchecker.xml"""
    suffix = base_xml.suffix  # '.xml' attendu
    stem = base_xml.stem
    return base_xml.with_name(f"{stem}_webchecker{suffix or '.xml'}")


def list_yaml_files(root: Path, recursive: bool) -> List[Path]:
    """Liste les fichiers .yaml et .yml dans 'root' (évent. récursif)."""
    patterns = ["*.yaml", "*.yml"]
    files: List[Path] = []
    if recursive:
        for pat in patterns:
            files.extend(root.rglob(pat))
    else:
        for pat in patterns:
            files.extend(root.glob(pat))
    return sorted([p for p in files if p.is_file()])


def render_and_write_variant(
    template_path: Path,
    context: Dict[str, Any],
    payload_only_bool: bool,
    output_path: Path,
    keep_decl: bool
) -> Tuple[bool, bool]:
    """
    Rend une variante donnée (payloadOnly booléen) et écrit le fichier.
    Retourne (ok, warned).
    """
    ctx = copy.deepcopy(context)

    # Booléen pour éviter les pièges des chaînes "truthy"
    ctx["payloadOnly"] = payload_only_bool
    # Compat : version chaîne si le template compare à 'yes'/'no'
    ctx["payloadOnly_str"] = "yes" if payload_only_bool else "no"
    # Alias snake_case
    ctx["payload_only"] = payload_only_bool
    # Regroupement optionnel
    flags = ctx.get("flags") or {}
    if not isinstance(flags, dict):
        flags = {}
    flags["payloadOnly"] = payload_only_bool
    ctx["flags"] = flags

    try:
        rendered = render_jinja_xml(template_path, ctx)

        # Si keep_decl=False, on supprime une éventuelle déclaration avant pretty
        rendered_text = rendered if keep_decl else strip_xml_declaration(rendered)

        warned = False
        try:
            output_text = pretty_with_minidom(rendered_text, keep_decl=keep_decl)
        except ValueError as ve:
            print(f"⚠️  WARNING ({output_path.name}) : {ve}", file=sys.stderr)
            # En fallback brut, respecter aussi le choix keep_decl (on ne rajoute pas / n'enlève pas)
            output_text = rendered_text
            warned = True

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text, encoding="utf-8")
        print(f"✅ XML généré : {output_path}")
        return (True, warned)
    except Exception as e:
        print(f"❌ Erreur lors du rendu/écriture de {output_path.name} : {e}", file=sys.stderr)
        return (False, False)


def process_one_yaml(
    specific_yaml: Path,
    template_cli: Path | None,
    defaults_cli: Path | None,
    output_cli: Path | None,
    cwd: Path,
    script_dir: Path,
    keep_decl: bool
) -> Tuple[int, int, int]:
    """
    Traite un fichier YAML spécifique et génère 2 fichiers (payloadOnly=False & True).
    Retourne (ok_count, warn_count, err_count).
    """
    specific_dir = specific_yaml.resolve().parent

    template_path = resolve_with_fallbacks(
        preferred=template_cli,
        defaults_rel=DEFAULT_TEMPLATE_REL,
        anchors=[cwd, script_dir, specific_dir],
        must_exist=True
    )
    if not template_path or not template_path.exists():
        print(f"❌ Template introuvable pour {specific_yaml}. Essayé: {template_cli or DEFAULT_TEMPLATE_REL}", file=sys.stderr)
        return (0, 0, 1)

    defaults_path = resolve_with_fallbacks(
        preferred=defaults_cli,
        defaults_rel=DEFAULT_DEFAULTS_REL,
        anchors=[cwd, script_dir, specific_dir],
        must_exist=False
    )

    try:
        context = load_context(defaults_path if defaults_path and defaults_path.exists() else None, specific_yaml)

        base_out = compute_output_path_for_base(specific_yaml, output_cli)
        webchecker_out = webchecker_path_from(base_out)

        # 1) payloadOnly = False  -> base_out
        ok1, warn1 = render_and_write_variant(template_path, context, False, base_out, keep_decl=keep_decl)
        # 2) payloadOnly = True   -> webchecker_out
        ok2, warn2 = render_and_write_variant(template_path, context, True, webchecker_out, keep_decl=keep_decl)

        ok_count = (1 if ok1 else 0) + (1 if ok2 else 0)
        warn_count = (1 if warn1 else 0) + (1 if warn2 else 0)
        err_count = (0 if ok1 else 1) + (0 if ok2 else 1)
        return (ok_count, warn_count, err_count)

    except Exception as e:
        print(f"❌ Erreur sur {specific_yaml} : {e}", file=sys.stderr)
        return (0, 0, 1)


def main():
    parser = argparse.ArgumentParser(
        prog="gen_xml.py",
        description="Génère 2 XML (payloadOnly=False et payloadOnly=True) depuis un template Jinja2 et des YAML (fichier ou répertoire)."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Chemin d'un fichier YAML spécifique OU d'un répertoire contenant des fichiers YAML."
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
        help="Chemin de sortie pour la variante payloadOnly=False (mode fichier uniquement). "
             "La variante payloadOnly=True sera écrite au même emplacement avec suffixe '_webchecker'. "
             "Ignorée en mode répertoire."
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="En mode répertoire, traite aussi les sous-répertoires."
    )
    parser.add_argument(
        "-x", "--no-xml-declaration",
        action="store_true",
        help="Supprimer la déclaration XML en tête de fichier (par défaut elle est CONSERVÉE)."
    )
    args = parser.parse_args()

    cwd = Path.cwd()
    script_dir = Path(__file__).resolve().parent

    if not args.input.exists():
        print(f"❌ Chemin introuvable : {args.input}", file=sys.stderr)
        sys.exit(1)

    # Par défaut, garder la déclaration XML ; -x la désactive
    keep_decl = not args.no_xml_declaration

    total_yaml = 0
    total_outputs = 0  # 2 par YAML
    ok_count = 0
    warn_count = 0
    err_count = 0

    if args.input.is_file():
        total_yaml = 1
        ok, warn, err = process_one_yaml(
            specific_yaml=args.input,
            template_cli=args.template,
            defaults_cli=args.defaults,
            output_cli=args.output,   # autorisé en mode fichier
            cwd=cwd,
            script_dir=script_dir,
            keep_decl=keep_decl
        )
        ok_count += ok
        warn_count += warn
        err_count += err
        total_outputs += 2

    else:
        if args.output:
            print("ℹ️  Info: option --output ignorée en mode répertoire ; chaque YAML génère ses sorties à côté du fichier.", file=sys.stderr)

        yaml_files = list_yaml_files(args.input, recursive=args.recursive)
        if not yaml_files:
            print("⚠️  Aucun fichier YAML (*.yaml|*.yml) trouvé dans le répertoire fourni.", file=sys.stderr)
            sys.exit(0)

        total_yaml = len(yaml_files)
        total_outputs = total_yaml * 2
        print(f"🔎 {total_yaml} fichier(s) YAML détecté(s) → {total_outputs} sortie(s) attendue(s).")

        for yml in yaml_files:
            ok, warn, err = process_one_yaml(
                specific_yaml=yml,
                template_cli=args.template,
                defaults_cli=args.defaults,
                output_cli=None,      # ignoré en mode répertoire
                cwd=cwd,
                script_dir=script_dir,
                keep_decl=keep_decl
            )
            ok_count += ok
            warn_count += warn
            err_count += err

    print(f"\nRésumé : {ok_count}/{total_outputs} OK — {warn_count} warning(s) — {err_count} erreur(s) — "
          f"{total_yaml} YAML traité(s).")
    sys.exit(0 if err_count == 0 else 1)


if __name__ == "__main__":
    main()