from __future__ import annotations

import copy
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parent.parent

REFERENCE_ATTRS = {
    "material",
    "mesh",
    "texture",
    "hfield",
    "joint",
    "body",
    "site",
    "geom",
    "tendon",
    "flex",
}

NAMED_TAGS = {
    "body",
    "geom",
    "joint",
    "site",
    "camera",
    "light",
    "mesh",
    "texture",
    "material",
    "hfield",
}


def _xml_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def _prefix_value(value: str, mapping: Dict[str, str]) -> str:
    parts = value.split()
    if len(parts) > 1:
        return " ".join(mapping.get(part, part) for part in parts)
    return mapping.get(value, value)


def _namespace_tree(root: ET.Element, prefix: str, source_dir: Path) -> None:
    name_map: Dict[str, str] = {}

    for elem in root.iter():
        name = elem.attrib.get("name")
        if name and elem.tag in NAMED_TAGS:
            new_name = f"{prefix}_{name}"
            name_map[name] = new_name
            elem.set("name", new_name)

    for elem in root.iter():
        for attr in REFERENCE_ATTRS:
            value = elem.attrib.get(attr)
            if value:
                elem.set(attr, _prefix_value(value, name_map))
        file_attr = elem.attrib.get("file")
        if file_attr and not Path(file_attr).is_absolute():
            elem.set("file", _xml_path(source_dir / file_attr))


def import_mjcf_object(
    source_xml: str | Path,
    prefix: str,
    body_name: str,
    pos: Iterable[float] = (0.0, 0.0, 0.0),
    quat: Iterable[float] = (1.0, 0.0, 0.0, 0.0),
    freejoint: bool = False,
    geom_group: str | None = None,
) -> Tuple[List[ET.Element], ET.Element]:
    """Return namespaced asset elements and a wrapper body for one MJCF object."""

    path = (PROJECT_ROOT / source_xml).resolve() if not Path(source_xml).is_absolute() else Path(source_xml)
    root = ET.parse(path).getroot()
    _namespace_tree(root, prefix, path.parent)

    assets = [copy.deepcopy(child) for child in root.findall("./asset/*")]
    wrapper = ET.Element(
        "body",
        {
            "name": body_name,
            "pos": " ".join(f"{value:.5f}" for value in pos),
            "quat": " ".join(f"{value:.8g}" for value in quat),
        },
    )
    if freejoint:
        ET.SubElement(wrapper, "freejoint", {"name": f"{body_name}_free"})

    for child in root.findall("./worldbody/*"):
        copied = copy.deepcopy(child)
        if geom_group is not None:
            for geom in copied.iter("geom"):
                geom.set("group", geom_group)
        wrapper.append(copied)

    return assets, wrapper


def elements_to_xml(elements: Iterable[ET.Element]) -> str:
    return "\n".join(ET.tostring(elem, encoding="unicode") for elem in elements)

