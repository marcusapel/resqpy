#!/usr/bin/env python3
"""
Eclipse GRDECL -> RESQML 2.0.1 (in-memory via resqpy) -> ETP 1.2 publisher
==========================================================================

Highlights
----------
- Parses a single Eclipse GRDECL grid (Cartesian with DX/DY/DZ/TOPS or Corner-Point with COORD/ZCORN)
  and common colocated properties (PORO, PERMX, PRESSURE, SWAT, SOIL, ACTNUM). Falls back to a tiny
  2x2x3 demo grid if no file is supplied.
- Builds EnergyML XML objects in memory using **resqpy** only (no EPC/HDF written):
  * eml20.LocalDepth3dCrs
  * resqml20.IjkGridRepresentation
  * resqml20.ContinuousProperty (for the supported GRDECL properties)
- Publishes the XML objects to an ETP 1.2 RDDMS Store using Protocol 3 PutDataObjects.
- Streams large arrays (geometry & properties) via **Protocol 9 DataArray** using 'uuid-path' style
  array identifiers; gzip is used for large (> 4 MiB) payloads.
- Opens & reuses an ETP **Protocol 18 Transaction** for the target dataspace and commits on success.

Notes
-----
* The script relies on *resqpy* for XML authoring. For corner-point grids, geometry is explicit and
  will require streaming arrays. For Cartesian regular grids, geometry uses lattice representation in
  XML and typically has no large arrays to stream.
* If the `resdata` GRDECL parser is available it will be used; otherwise a small built-in parser is
  used for the subset of keywords needed here.
* Object & array UUIDs are deterministic using UUIDv5 to ensure stable 'uuid-path' addressing.

Usage
-----
  python ecl2etp.py --eclgrd PATH --log
  python ecl2etp.py --dry-run --log

Environment / Config
--------------------
Reads ETP connection and dataspace settings from `config_etp.py` (placed next to this script).

"""
from __future__ import annotations
import argparse
import asyncio
import gzip
import logging
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# --- Local configuration (token is never logged) -----------------------------------------------
try:
    from config_etp import Config  # type: ignore
except Exception:
    print("FATAL: Missing config_etp.py next to this script.", file=sys.stderr)
    raise

# --- RESQML authoring via resqpy ----------------------------------------------------------------
from resqpy.model import Model
from resqpy.crs import Crs
from resqpy.grid import RegularGrid, Grid

# Optional: import helper for creating CP grids directly from COORD/ZCORN
try:  # resqpy >= 5 provides this helper
    from resqpy.rq_import import grid_from_cp  # type: ignore
except Exception:
    grid_from_cp = None

# Property helpers
try:
    from resqpy.property import PropertyCollection  # type: ignore
except Exception:
    PropertyCollection = None  # noqa: N816


# --- ETP / etptypes ----------------------------------------------------------------------------
from etptypes.energistics.etp.v12.datatypes.object.data_object import DataObject  # type: ignore
from etptypes.energistics.etp.v12.datatypes.object.resource import Resource  # type: ignore
from etptypes.energistics.etp.v12.datatypes.object.active_status_kind import ActiveStatusKind  # type: ignore

# DataArray (Protocol 9)
try:
    from etptypes.energistics.etp.v12.datatypes.data_array_types import (  # type: ignore
        DataArray, DataArrayIdentifier,
    )
    from etptypes.energistics.etp.v12.datatypes.array_types import (  # type: ignore
        AnyArray, ArrayOfFloat, ArrayOfDouble, ArrayOfInt,
    )
    from etptypes.energistics.etp.v12.protocol.data_array.put_data_arrays import PutDataArrays  # type: ignore
except Exception:
    DataArray = None  # type: ignore
    DataArrayIdentifier = None  # type: ignore
    AnyArray = ArrayOfFloat = ArrayOfDouble = ArrayOfInt = None  # type: ignore
    PutDataArrays = None  # type: ignore

# Store (Protocol 3)
from etptypes.energistics.etp.v12.protocol.store.put_data_objects import PutDataObjects  # type: ignore

# Transaction (Protocol 18)
from etptypes.energistics.etp.v12.protocol.transaction.start_transaction import StartTransaction  # type: ignore
from etptypes.energistics.etp.v12.protocol.transaction.commit_transaction import CommitTransaction  # type: ignore


# ============================================================
#                         GRDECL parsing
# ============================================================

@dataclass
class GrdeclModel:
    kind: str  # 'Cartesian' or 'CornerPoint'
    nx: int
    ny: int
    nz: int
    DX: Optional[np.ndarray] = None
    DY: Optional[np.ndarray] = None
    DZ: Optional[np.ndarray] = None
    TOPS: Optional[np.ndarray] = None
    COORD: Optional[np.ndarray] = None
    ZCORN: Optional[np.ndarray] = None
    ACTNUM: Optional[np.ndarray] = None
    properties: Dict[str, np.ndarray] = None


_GRDECL_PROPS = {"PORO", "PERMX", "PERMY", "PERMZ", "PRESSURE", "SWAT", "SOIL"}


# Try to use Equinor resdata if available
def _read_with_resdata(path: str) -> Optional[GrdeclModel]:
    try:
        # resdata uses this entry point
        from resdata.grid import rd_grid  # type: ignore
    except Exception:
        return None

    try:
        g = rd_grid.RdGrid(path)
        nx, ny, nz = int(g.nx), int(g.ny), int(g.nz)
        if hasattr(g, 'coord') and hasattr(g, 'zcorn') and g.coord is not None and g.zcorn is not None:
            kind = 'CornerPoint'
            COORD = np.asarray(g.coord, dtype=float).reshape(-1)
            ZCORN = np.asarray(g.zcorn, dtype=float).reshape(-1)
            ACTNUM = np.asarray(getattr(g, 'actnum', None), dtype=int).reshape(-1) if getattr(g, 'actnum', None) is not None else None
            props: Dict[str, np.ndarray] = {}
            for p in _GRDECL_PROPS:
                if hasattr(g, p.lower()):
                    arr = getattr(g, p.lower())
                    if arr is not None:
                        props[p] = np.asarray(arr, dtype=float).reshape(nx * ny * nz)
            return GrdeclModel(kind=kind, nx=nx, ny=ny, nz=nz, COORD=COORD, ZCORN=ZCORN, ACTNUM=ACTNUM, properties=props)
        # Cartesian fallback if DX/DY/DZ present
        if hasattr(g, 'dx') and hasattr(g, 'dy') and hasattr(g, 'dz'):
            kind = 'Cartesian'
            DX = np.asarray(g.dx, dtype=float).reshape(nx * ny * nz)
            DY = np.asarray(g.dy, dtype=float).reshape(nx * ny * nz)
            DZ = np.asarray(g.dz, dtype=float).reshape(nx * ny * nz)
            TOPS = np.asarray(getattr(g, 'tops', None), dtype=float).reshape(nx * ny * nz) if getattr(g, 'tops', None) is not None else None
            ACTNUM = np.asarray(getattr(g, 'actnum', None), dtype=int).reshape(-1) if getattr(g, 'actnum', None) is not None else None
            props: Dict[str, np.ndarray] = {}
            for p in _GRDECL_PROPS:
                val = getattr(g, p.lower(), None)
                if val is not None:
                    props[p] = np.asarray(val, dtype=float).reshape(nx * ny * nz)
            return GrdeclModel(kind=kind, nx=nx, ny=ny, nz=nz, DX=DX, DY=DY, DZ=DZ, TOPS=TOPS, ACTNUM=ACTNUM, properties=props)
    except Exception as e:
        logging.warning("resdata parsing failed (%s); will try built-in parser", e)
        return None

    return None


# Lightweight built-in parser for DIMENS/SPECGRID, DX/DY/DZ/TOPS, COORD/ZCORN and properties
_re_rep = re.compile(r"^(\d+)\*(.+)$")


def _strip_comments(text: str) -> str:
    return re.sub(r"\s*\-\-.*$", "", text, flags=re.MULTILINE)


def _expand_rep_token(tok: str) -> List[str]:
    m = _re_rep.match(tok)
    if m:
        count, value = int(m.group(1)), m.group(2)
        return [value] * count
    return [tok]


def _parse_values(tokens: Iterable[str]) -> List[str]:
    out: List[str] = []
    for t in tokens:
        out.extend(_expand_rep_token(t))
    return out


def _to_float(vals: List[str]) -> np.ndarray:
    return np.array([float(v) for v in vals], dtype=float)


def _to_int(vals: List[str]) -> np.ndarray:
    return np.array([int(float(v)) for v in vals], dtype=int)


def _read_with_builtin(path: str) -> GrdeclModel:
    text = open(path, 'r', encoding='utf-8', errors='ignore').read()
    text = _strip_comments(text)
    blocks = [b.strip() for b in text.split('/') if b.strip()]

    nx = ny = nz = None  # type: ignore
    kind: Optional[str] = None
    accum: Dict[str, List[str]] = {}

    for b in blocks:
        toks = b.split()
        if not toks:
            continue
        kw = toks[0].upper()
        vals = _parse_values(toks[1:])
        if kw == 'DIMENS':
            nx, ny, nz = map(int, vals[:3])
            kind = 'Cartesian'
        elif kw == 'SPECGRID':
            nx, ny, nz = map(int, vals[:3])
            kind = 'CornerPoint'
        else:
            accum[kw] = vals

    if nx is None or ny is None or nz is None:
        raise ValueError('Missing DIMENS or SPECGRID in GRDECL')

    n = nx * ny * nz

    def _ensure_len(name: str, expected: int, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return None
        if len(v) != expected:
            raise ValueError(f"{name} length {len(v)} != expected {expected}")
        return v

    DX = DY = DZ = TOPS = COORD = ZCORN = ACTNUM = None
    props: Dict[str, np.ndarray] = {}

    if kind == 'Cartesian':
        DX = _to_float(_ensure_len('DX', n, accum.get('DX')) or ['100'] * n)
        DY = _to_float(_ensure_len('DY', n, accum.get('DY')) or ['100'] * n)
        DZ = _to_float(_ensure_len('DZ', n, accum.get('DZ')) or ['10'] * n)
        if accum.get('TOPS'):
            TOPS = _to_float(_ensure_len('TOPS', n, accum.get('TOPS')))
        if accum.get('ACTNUM'):
            ACTNUM = _to_int(_ensure_len('ACTNUM', n, accum.get('ACTNUM')))
    else:
        if accum.get('COORD'):
            COORD = _to_float(_ensure_len('COORD', 6 * (nx + 1) * (ny + 1), accum.get('COORD')))
        if accum.get('ZCORN'):
            ZCORN = _to_float(_ensure_len('ZCORN', 8 * n, accum.get('ZCORN')))
        if accum.get('ACTNUM'):
            ACTNUM = _to_int(_ensure_len('ACTNUM', n, accum.get('ACTNUM')))

    for p in _GRDECL_PROPS:
        if accum.get(p):
            props[p] = _to_float(_ensure_len(p, n, accum.get(p)))

    return GrdeclModel(kind=kind or 'Cartesian', nx=nx, ny=ny, nz=nz,
                       DX=DX, DY=DY, DZ=DZ, TOPS=TOPS, COORD=COORD, ZCORN=ZCORN,
                       ACTNUM=ACTNUM, properties=props)


def read_grdecl(path: str) -> GrdeclModel:
    m = _read_with_resdata(path)
    if m is not None:
        return m
    return _read_with_builtin(path)


# --- Demo bundle -------------------------------------------------------------------------------
@dataclass
class GridBundle:
    ni: int
    nj: int
    nk: int
    dx: float = 100.0
    dy: float = 100.0
    dz: float = 10.0
    origin: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    properties: Optional[Dict[str, np.ndarray]] = None


def default_bundle() -> GridBundle:
    ni, nj, nk = 2, 2, 3
    nc = ni * nj * nk
    props = {
        'PERMX': np.full(nc, 100.0),
        'PORO': np.full(nc, 0.20),
        'PRESSURE': np.full(nc, 2.5e7),
        'SWAT': np.full(nc, 0.30),
        'SOIL': np.full(nc, 0.70),
    }
    return GridBundle(ni, nj, nk, properties=props)


def _mean_unique(name: str, arr: np.ndarray, axis: str) -> float:
    v = float(np.mean(arr))
    if not np.allclose(arr, v, rtol=0, atol=0):
        raise NotImplementedError(
            f"{name} not uniform ({axis}); lattice exporter requires uniform spacing per axis"
        )
    return v


def grdecl_to_bundle_cartesian(g: GrdeclModel) -> GridBundle:
    if g.DX is None or g.DY is None or g.DZ is None:
        raise ValueError('Cartesian GRDECL requires DX/DY/DZ arrays')
    DX3 = g.DX.reshape(g.nz, g.ny, g.nx)
    DY3 = g.DY.reshape(g.nz, g.ny, g.nx)
    DZ3 = g.DZ.reshape(g.nz, g.ny, g.nx)
    dx = _mean_unique('DX', DX3.mean(axis=(0, 1)), 'I')
    dy = _mean_unique('DY', DY3.mean(axis=(0, 2)), 'J')
    dz = _mean_unique('DZ', DZ3.mean(axis=(1, 2)), 'K')
    return GridBundle(ni=g.nx, nj=g.ny, nk=g.nz, dx=float(dx), dy=float(dy), dz=float(dz),
                      origin=(0.0, 0.0, 0.0), properties=g.properties or {})


# ============================================================
#                      RESQML object build
# ============================================================

OSDU_PROP_MAP: Dict[str, Dict[str, str]] = {
    'PORO':  {"osdu_ref": "osdu:reference-data--PropertyNameType:Porosity:1.0.0",   "osdu_name": "Porosity",  "display": "Porosity",  "uom": "frac"},
    'PERMX': {"osdu_ref": "osdu:reference-data--PropertyNameType:Permeability:1.0.0", "osdu_name": "Permeability", "display": "Permeability X", "uom": "mD"},
    'PERMY': {"osdu_ref": "osdu:reference-data--PropertyNameType:Permeability:1.0.0", "osdu_name": "Permeability", "display": "Permeability Y", "uom": "mD"},
    'PERMZ': {"osdu_ref": "osdu:reference-data--PropertyNameType:Permeability:1.0.0", "osdu_name": "Permeability", "display": "Permeability Z", "uom": "mD"},
    'PRESSURE': {"osdu_ref": "osdu:reference-data--PropertyNameType:Pressure:1.0.0", "osdu_name": "Pressure", "display": "Pressure", "uom": "Pa"},
    'SWAT': {"osdu_ref": "osdu:reference-data--PropertyNameType:WaterSaturation:1.0.0", "osdu_name": "WaterSaturation", "display": "Water Saturation", "uom": "frac"},
    'SOIL': {"osdu_ref": "osdu:reference-data--PropertyNameType:OilSaturation:1.0.0", "osdu_name": "OilSaturation", "display": "Oil Saturation", "uom": "frac"},
}


@dataclass
class Built:
    model: Model
    crs: Crs
    grid: Grid
    prop_arrays: Dict[str, np.ndarray]


def _uuid5(ns: uuid.UUID, name: str) -> str:
    return str(uuid.uuid5(ns, name))


def _safe_crs(model: Model, *, title: str, ns: uuid.UUID) -> Crs:
    # resqpy changed kw name across versions: z_inc_down vs z_increase_down
    kwargs = dict(model=model, xy_units='m', z_units='m', title=title)
    try:
        crs = Crs(**kwargs, z_inc_down=True)  # type: ignore
    except TypeError:
        crs = Crs(**kwargs, z_increase_down=True)  # type: ignore
    # set deterministic uuid
    try:
        crs.uuid = _uuid5(ns, 'eml:LocalDepth3dCrs:m+down')
    except Exception:
        pass
    crs.create_xml(add_as_part=True)
    return crs


def build_resqpy_cartesian(bundle: GridBundle, *, ns: uuid.UUID) -> Built:
    model = Model(new_epc=True)
    crs = _safe_crs(model, title='LocalDepth3dCrs (XY m, Z m +down)', ns=ns)
    # Regular lattice grid (explicit geometry not required)
    grid = RegularGrid(model,
                       extent_kji=(bundle.nk, bundle.nj, bundle.ni),
                       dxyz=(bundle.dx, bundle.dy, bundle.dz),
                       origin=(bundle.origin[0], bundle.origin[1], bundle.origin[2]),
                       title=f'Cartesian IJK Grid ({bundle.ni}x{bundle.nj}x{bundle.nk})')
    grid.set_crs(crs)
    try:
        grid.uuid = _uuid5(ns, f'resqml:IjkGrid:{bundle.ni}x{bundle.nj}x{bundle.nk}:lattice')
    except Exception:
        pass
    grid.create_xml(add_as_part=True)
    return Built(model=model, crs=crs, grid=grid, prop_arrays=bundle.properties or {})


def build_resqpy_cornerpoint(g: GrdeclModel, *, ns: uuid.UUID) -> Built:
    assert g.COORD is not None and g.ZCORN is not None
    model = Model(new_epc=True)
    crs = _safe_crs(model, title='LocalDepth3dCrs (XY m, Z m +down)', ns=ns)

    if grid_from_cp is not None:
        grid = grid_from_cp(parent_model=model,
                            extent_kji=(g.nz, g.ny, g.nx),
                            zcorn=g.ZCORN.reshape(-1),
                            coord=g.COORD.reshape(-1),
                            crs_uuid=crs.uuid,
                            title=f'CornerPoint IJK Grid ({g.nx}x{g.ny}x{g.nz})')
    else:
        # Fallback: create an empty Grid and let resqpy infer from points after write_hdf5; here we just
        # create the shell XML – array streaming will supply geometry to the store.
        grid = Grid(model, title=f'CornerPoint IJK Grid ({g.nx}x{g.ny}x{g.nz})')
        grid.extent_kji = (g.nz, g.ny, g.nx)
        grid.set_crs(crs)

    try:
        grid.uuid = _uuid5(ns, f'resqml:IjkGrid:{g.nx}x{g.ny}x{g.nz}:cornerpoint')
    except Exception:
        pass

    grid.create_xml(add_as_part=True)
    return Built(model=model, crs=crs, grid=grid, prop_arrays=g.properties or {})


# --- Property creation --------------------------------------------------------------------------

def _grid_kji_shape(grid: Grid) -> Tuple[int, int, int]:
    nk, nj, ni = grid.extent_kji
    return nk, nj, ni


def _add_property_arrays(built: Built, *, ns: uuid.UUID) -> List[uuid.UUID]:
    """Create ContinuousProperty XML parts for each array; return list of property UUIDs."""
    if PropertyCollection is None:
        logging.warning('resqpy.property not available; skipping properties')
        return []

    nk, nj, ni = _grid_kji_shape(built.grid)
    pc = PropertyCollection(support=built.grid)
    uuids: List[uuid.UUID] = []

    for kw, arr in (built.prop_arrays or {}).items():
        try:
            md = OSDU_PROP_MAP.get(kw, None)
            uom = (md or {}).get('uom', 'Euc')
            title = (md or {}).get('display', kw)

            # Ensure shape (nk,nj,ni)
            a = np.asarray(arr)
            if a.size != nk * nj * ni:
                logging.warning('Property %s has %d values; expected %d – skipping', kw, a.size, nk*nj*ni)
                continue
            a3 = a.reshape(nk, nj, ni)

            # resqpy PropertyCollection has add_cached_array() in newer versions
            if hasattr(pc, 'add_cached_array'):
                prop_uuid = uuid.UUID(_uuid5(ns, f'prop:{kw}'))
                pc.add_cached_array(array=a3,
                                    source_info='ecl2etp', 
                                    keyword=kw,
                                    support_uuid=built.grid.uuid,
                                    property_kind='continuous',
                                    uom=uom,
                                    title=title,
                                    dtype=a3.dtype)
                # Gather UUID via collection APIs
                try:
                    part = pc.singleton(citation_title=title)
                    if part is not None:
                        uuids.append(pc.uuid_for_part(part))
                except Exception:
                    pass
            else:
                # Fallback: add using grid property interface if available
                try:
                    if hasattr(built.grid, 'add_property'):
                        built.grid.add_property(a3, property_kind='continuous', uom=uom, title=title)
                except Exception as e:
                    logging.warning('Could not add property %s: %s', kw, e)
        except Exception as e:
            logging.warning('Property %s skipped: %s', kw, e)

    # Ensure XML is generated for properties present in collection
    try:
        if hasattr(pc, 'create_xml_for_parts'):  # new API
            pc.create_xml_for_parts()
        elif hasattr(pc, 'write_hdf5_for_imported_list'):  # don't write; but generating XML is fine
            pass
    except Exception:
        pass

    return uuids


# ============================================================
#                  ETP helpers (connect + publish)
# ============================================================

DATAOBJECT_FORMAT_FIELD = 'format_'
DATAOBJECT_FORMAT_XML = 'xml'


def _uuid_str(u) -> str:
    s = str(u)
    return s.strip('{}').lower()


def _qualified_name(package: str, type_name: str) -> str:
    return ('resqml20' if package == 'resqml' else 'eml20') + '.' + type_name


def _eml_uri(dataspace_uri: str, qualified: str, uuid_str: str) -> str:
    return f"{dataspace_uri}/{qualified}({uuid_str})"


def _as_resource(uri: str, name: Optional[str] = None) -> Resource:
    now = int(time.time() * 1000)
    return Resource(uri=uri, name=name or uri.split('/')[-1], lastChanged=now,
                    storeLastWrite=now, storeCreated=now, activeStatus=ActiveStatusKind.ACTIVE)


def _to_data_object(xml_bytes: bytes, *, dataspace_uri: str, type_name: str, package: str,
                    uuid_str: str, title: str) -> DataObject:
    qualified = _qualified_name(package, type_name)
    uri = _eml_uri(dataspace_uri, qualified, uuid_str)
    dobj = DataObject(resource=_as_resource(uri, title), data=xml_bytes)
    setattr(dobj, DATAOBJECT_FORMAT_FIELD, DATAOBJECT_FORMAT_XML)
    return dobj


def xml_bytes_from_resqpy(obj) -> bytes:
    from xml.etree import ElementTree as ET
    return ET.tostring(obj.root, encoding='utf-8', xml_declaration=True)


# Connection

def _prepare_env(config: Config) -> None:
    os.environ['ETP_URL'] = config.rddms_host
    os.environ['DATA_PARTITION_ID'] = config.data_partition_id
    os.environ['DATA_PARTITION'] = config.data_partition_id
    os.environ['DATASPACE_URI'] = config.dataspace_uri
    # also set short form if some libs expect it
    m = re.match(r"^eml:///dataspace\('([^']+)'\)$", config.dataspace_uri.strip())
    if m:
        os.environ['DATASPACE'] = m.group(1)
    if config.token:
        os.environ['AUTHORIZATION'] = config.token if config.token.lower().startswith('bearer ') else f"Bearer {config.token}"


def _norm_auth(token: Optional[str]) -> str:
    if not token:
        return ''
    t = token.strip()
    return t if t.lower().startswith('bearer ') else f'Bearer {t}'


def _etp_headers(config: Config) -> Dict[str, str]:
    return {'data-partition-id': config.data_partition_id}


def _redact_token(v: str) -> str:
    if not v:
        return ''
    t = v.strip()
    if t.lower().startswith('bearer '):
        core = t[7:]
        return 'Bearer ' + (f"{core[:8]}…{core[-4:]}" if len(core) > 14 else '***')
    return '***'


async def _connect(config: Config, *, log: bool = False):
    _prepare_env(config)
    from pyetp.client import connect  # type: ignore
    import inspect
    auth = _norm_auth(config.token)
    headers = _etp_headers(config)
    sig = inspect.signature(connect)
    kwargs = {}
    if 'authorization' in sig.parameters:
        kwargs['authorization'] = auth
    if 'additional_headers' in sig.parameters:
        kwargs['additional_headers'] = headers
    elif 'headers' in sig.parameters:
        kwargs['headers'] = headers
    elif 'extra_headers' in sig.parameters:
        kwargs['extra_headers'] = headers
    if log:
        logging.info('ETP connect: url=%s partition=%s headers=%s auth=%s',
                     os.environ.get('ETP_URL'), config.data_partition_id,
                     {k: v for k, v in headers.items()}, _redact_token(auth))
        logging.info('ETP dataspace: %s', config.dataspace_uri)
    return await connect(**kwargs)


# Transaction handling
class TxManager:
    def __init__(self) -> None:
        self._by_ds: Dict[str, str] = {}

    def get(self, ds_key: str) -> Optional[str]:
        return self._by_ds.get(ds_key)

    def set(self, ds_key: str, tx_id: str) -> None:
        self._by_ds[ds_key] = tx_id

    def clear(self, ds_key: str) -> None:
        self._by_ds.pop(ds_key, None)


def _dataspace_name_from_uri(uri: str) -> str:
    m = re.match(r"^eml:///dataspace\('([^']+)'\)$", uri.strip())
    return m.group(1) if m else uri.strip()


def _get_tx_manager(client) -> TxManager:
    mgr = getattr(client, '_tx_manager', None)
    if mgr is None:
        mgr = TxManager()
        setattr(client, '_tx_manager', mgr)
    return mgr


def _tx_already_active(exc: Exception) -> bool:
    s = str(exc)
    return ("code=15" in s) or ("ETP-15" in s) or ("already active" in s.lower())


async def _open_or_reuse_transaction(client, dataspace_uri: str, *, log: bool = False) -> str:
    txm = _get_tx_manager(client)
    ds_key = _dataspace_name_from_uri(dataspace_uri)
    cached = txm.get(ds_key)
    if cached:
        if log:
            logging.info('Reusing transaction for %s: %s', ds_key, cached)
        return cached

    candidates = [[dataspace_uri], [f"eml:///dataspace('{ds_key}')"], [ds_key]]
    last_exc: Optional[Exception] = None
    for uris in candidates:
        try:
            if log:
                logging.info('P18 StartTransaction URIs=%s', uris)
            st = StartTransaction(dataspaceUris=uris, readOnly=False)
            resp = await client.send(st)
            tx_id = getattr(resp, 'transactionUuid', None) or getattr(resp, 'transactionId', None)
            if not tx_id:
                raise RuntimeError('Server did not return a transaction id')
            txm.set(ds_key, tx_id)
            return tx_id
        except Exception as e:
            last_exc = e
            if _tx_already_active(e):
                raise RuntimeError(
                    'RDDMS reports a write transaction is already active for this dataspace. ' \
                    'Commit/rollback the other session or retry later. Last error: ' + str(e)
                ) from e
    raise RuntimeError(f'Could not open transaction for {dataspace_uri}. Last error: {last_exc}')


async def _commit_if_open(client, dataspace_uri: str, *, log: bool = False) -> None:
    txm = _get_tx_manager(client)
    ds_key = _dataspace_name_from_uri(dataspace_uri)
    tx_id = txm.get(ds_key)
    if not tx_id:
        return
    try:
        try:
            await client.send(CommitTransaction(transactionUuid=tx_id))
        except TypeError:
            await client.send(CommitTransaction(transactionId=tx_id))
        if log:
            logging.info('Committed transaction: %s', tx_id)
    finally:
        txm.clear(ds_key)


# DataArray sender

def _as_any_array(arr: np.ndarray):
    """Try to wrap a numpy array into etptypes AnyArray. Fallback to bytes if not available."""
    if AnyArray is None:
        return None
    arr_c = np.ascontiguousarray(arr)
    if arr_c.dtype in (np.float64,):
        return AnyArray(item=ArrayOfDouble(values=arr_c.ravel(order='C').tolist()))
    if arr_c.dtype in (np.float32,):
        return AnyArray(item=ArrayOfFloat(values=arr_c.astype(np.float32).ravel(order='C').tolist()))
    if arr_c.dtype in (np.int32, np.int64, np.uint32, np.uint64, np.int16, np.uint16, np.int8, np.uint8):
        return AnyArray(item=ArrayOfInt(values=arr_c.astype(np.int64).ravel(order='C').tolist()))
    # default: promote to double
    return AnyArray(item=ArrayOfDouble(values=arr_c.astype(np.float64).ravel(order='C').tolist()))


def _gzip_if_needed(b: bytes, compress_over: int) -> Tuple[bytes, Optional[str]]:
    if len(b) > compress_over:
        return gzip.compress(b), 'gzip'
    return b, None


async def put_arrays(client, arrays: List[Tuple[str, str, np.ndarray]], *, inline_threshold: int,
                     compress_over: int, log: bool = False) -> None:
    """Send numpy arrays via Protocol 9. Each tuple is (uri, pathInResource, array)."""
    if PutDataArrays is None or DataArray is None or DataArrayIdentifier is None:
        logging.warning('Protocol 9 types not available in etptypes; skipping DataArray send')
        return

    items = {}
    for uri, path, arr in arrays:
        if arr.nbytes <= inline_threshold:
            # Try to use AnyArray typed transfer for small data
            any_arr = _as_any_array(arr)
            if any_arr is not None:
                da = DataArray(array=any_arr, dimensions=list(arr.shape))
            else:
                raw = np.ascontiguousarray(arr).tobytes(order='C')
                raw, comp = _gzip_if_needed(raw, compress_over)
                da = DataArray(data=raw, dimensions=list(arr.shape))
                if comp:
                    try:
                        setattr(da, 'compression', comp)
                    except Exception:
                        pass
        else:
            # Large: send as bytes (optionally gzipped)
            raw = np.ascontiguousarray(arr).tobytes(order='C')
            raw, comp = _gzip_if_needed(raw, compress_over)
            da = DataArray(data=raw, dimensions=list(arr.shape))
            if comp:
                try:
                    setattr(da, 'compression', comp)
                except Exception:
                    pass
        dai = DataArrayIdentifier(uri=uri, pathInResource=path)
        items[dai] = da
        if log:
            logging.info('P09 planned: %s [%s], %d bytes%s', uri, path, arr.nbytes,
                         ' (gz)' if arr.nbytes > compress_over else '')

    if not items:
        return

    try:
        if hasattr(client, 'data_array') and hasattr(client.data_array, 'put_data_arrays'):
            await client.data_array.put_data_arrays(items)
        else:
            await client.send(PutDataArrays(dataArrays=items))
        if log:
            logging.info('P09 PutDataArrays OK: %d arrays', len(items))
    except Exception as e:
        logging.error('P09 PutDataArrays failed: %s', e)
        raise


# ============================================================
#                           Publisher
# ============================================================

async def publish(built: Built, *, dataspace_uri: str, inline_threshold: int = 1024,
                  compress_over: int = 4 * 1024 * 1024, dry_run: bool = False, log: bool = False,
                  xml_preview: int = 0) -> bool:
    # Prepare XML bytes
    crs_xml = xml_bytes_from_resqpy(built.crs)
    grid_xml = xml_bytes_from_resqpy(built.grid)

    crs_uuid = _uuid_str(built.crs.uuid)
    grid_uuid = _uuid_str(built.grid.uuid)

    crs_do = _to_data_object(crs_xml, dataspace_uri=dataspace_uri, type_name='LocalDepth3dCrs',
                             package='eml', uuid_str=crs_uuid, title='LocalDepth3dCrs (m, +down)')
    grid_do = _to_data_object(grid_xml, dataspace_uri=dataspace_uri, type_name='IjkGridRepresentation',
                              package='resqml', uuid_str=grid_uuid, title=built.grid.title)

    # Create property XMLs if any
    prop_uuids = _add_property_arrays(built, ns=uuid.uuid5(uuid.NAMESPACE_URL, dataspace_uri))
    prop_dobjs: List[DataObject] = []
    for pu in prop_uuids:
        # locate the part in the model and serialize
        try:
            part = built.model.part_for_uuid(pu)
            if part is None:
                continue
            root = built.model.root_for_part(part)
            if root is None:
                continue
            from xml.etree import ElementTree as ET
            pxml = ET.tostring(root, encoding='utf-8', xml_declaration=True)
            prop_do = _to_data_object(pxml, dataspace_uri=dataspace_uri, type_name='ContinuousProperty',
                                      package='resqml', uuid_str=_uuid_str(pu), title=str(root.get('title') or 'Property'))
            prop_dobjs.append(prop_do)
        except Exception as e:
            logging.warning('Skip property uuid %s: %s', pu, e)

    all_dobjs = [crs_do, grid_do] + prop_dobjs

    if log:
        logging.info('Will publish %d objects:', len(all_dobjs))
        for d in all_dobjs:
            logging.info(' URI: %s', d.resource.uri)
    if xml_preview > 0:
        logging.info('Preview CRS xml: %s', crs_xml[:xml_preview])
        logging.info('Preview GRID xml: %s', grid_xml[:xml_preview])

    if dry_run:
        logging.info('DRY-RUN: built XML objects (no ETP send)')
        return True

    # Connect
    config = Config.from_env()
    client = await _connect(config, log=log)

    try:
        # Ensure write transaction
        _ = await _open_or_reuse_transaction(client, dataspace_uri, log=log)

        # Put objects (CRS -> Grid -> Properties)
        async def _put_one(do: DataObject) -> None:
            try:
                if hasattr(client, 'store') and hasattr(client.store, 'put_data_objects'):
                    await client.store.put_data_objects({do.resource.uri: do})
                else:
                    await client.send(PutDataObjects(dataObjects={do.resource.uri: do}))
                if log:
                    logging.info('Put OK: %s (%d bytes)', do.resource.uri, len(do.data or b''))
            except Exception as eg:
                logging.error('PutDataObjects error: %s', eg)
                raise

        await _put_one(crs_do)
        await _put_one(grid_do)
        for p in prop_dobjs:
            await _put_one(p)

        # --- Protocol 9 arrays (if any) --------------------------------------------------------
        arrays: List[Tuple[str, str, np.ndarray]] = []

        # For corner-point grids, stream points if available in cache
        try:
            if hasattr(built.grid, 'points_cached') and built.grid.points_cached is not None:
                pts = np.asarray(built.grid.points_cached, dtype=np.float64)
                arrays.append((grid_do.resource.uri, 'uuid-path:points', pts))
        except Exception:
            pass

        # Stream properties (found in prop_uuids) as flat arrays (nk,nj,ni)
        for pu in prop_uuids:
            try:
                part = built.model.part_for_uuid(pu)
                if part is None:
                    continue
                # Get array values via property collection
                if PropertyCollection is not None:
                    pc = PropertyCollection(support=built.grid)
                    title = built.model.title_for_part(part)
                    a = pc.single_array_ref(citation_title=title) if hasattr(pc, 'single_array_ref') else None
                else:
                    a = None
                if a is None:
                    # Try to fetch via model helper
                    a = built.model.h5_array_ref(part)
                if a is not None:
                    arrays.append((_eml_uri(dataspace_uri, 'resqml20.ContinuousProperty', _uuid_str(pu)),
                                   'uuid-path:values', np.asarray(a)))
            except Exception as e:
                logging.warning('Skip array for property %s: %s', pu, e)

        if arrays:
            await put_arrays(client, arrays, inline_threshold=inline_threshold,
                             compress_over=compress_over, log=log)

        # Commit
        await _commit_if_open(client, dataspace_uri, log=log)

        logging.info('OK: Published %d object(s)%s', len(all_dobjs), (f" + {len(arrays)} arrays" if arrays else ''))
        return True

    finally:
        try:
            await client.close()
        except Exception:
            pass


# ============================================================
#                                 CLI
# ============================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Eclipse GRDECL -> RESQML 2.0.1 (resqpy) -> ETP 1.2 publisher')
    p.add_argument('--eclgrd', dest='grdecl_path', default=None, help='Path to Eclipse .GRDECL file')
    p.add_argument('--dry-run', action='store_true', help='Build XML payloads only (no ETP send)')
    p.add_argument('--log', action='store_true', help='Enable logging (connection headers & URIs; token redacted)')
    p.add_argument('--inline-threshold', type=int, default=1024, help='Inline arrays <= this many bytes (else P9)')
    p.add_argument('--compress-over', type=int, default=4*1024*1024, help='Gzip arrays strictly larger than this size in bytes')
    p.add_argument('--max-xml-preview', type=int, default=0, help='Log first N bytes of XML payloads (0 disables)')
    return p


def main(argv: Optional[List[str]] = None) -> int:
    ns = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if ns.log else logging.WARNING, format='%(levelname)s %(message)s')

    nsid = uuid.uuid5(uuid.NAMESPACE_URL, f"dataspace:{Config.from_env().dataspace_uri}")

    # Build source
    if ns.grdecl_path:
        g = read_grdecl(ns.grdecl_path)
        if g.kind == 'CornerPoint':
            built = build_resqpy_cornerpoint(g, ns=nsid)
        else:
            built = build_resqpy_cartesian(grdecl_to_bundle_cartesian(g), ns=nsid)
    else:
        built = build_resqpy_cartesian(default_bundle(), ns=nsid)

    ok = asyncio.run(publish(built, dataspace_uri=Config.from_env().dataspace_uri, inline_threshold=int(ns.inline_threshold),
                             compress_over=int(ns.compress_over), dry_run=bool(ns.dry_run), log=bool(ns.log),
                             xml_preview=int(ns.max_xml_preview)))
    return 0 if ok else 2


if __name__ == '__main__':
    raise SystemExit(main())
