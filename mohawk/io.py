"""Data import and montage handling (port of ``dataimport.m``).

Reads EEG in the formats the original pipeline supported (EGI RAW/MFF, EDF,
BrainVision VHDR, EEGLAB SET, FIF), assigns a channel montage, and removes the
peripheral electrodes that MOHAWK excludes for 128/256-channel EGI nets.
"""

from __future__ import annotations

import os

import mne

# Peripheral channels excluded for high-density EGI nets (dataimport.m).
CHANEXCL_128 = [
    "E1", "E8", "E14", "E17", "E21", "E25", "E32", "E38", "E43", "E44", "E48",
    "E49", "E56", "E57", "E63", "E64", "E68", "E69", "E73", "E74", "E81", "E82",
    "E88", "E89", "E94", "E95", "E99", "E100", "E107", "E113", "E114", "E119",
    "E120", "E121", "E125", "E126", "E127", "E128",
]
CHANEXCL_256 = [
    "E31", "E67", "E73", "E82", "E91", "E92", "E93", "E94", "E102", "E103",
    "E104", "E105", "E111", "E112", "E113", "E114", "E120", "E121", "E122",
    "E123", "E133", "E134", "E135", "E136", "E145", "E146", "E147", "E148",
    "E156", "E157", "E158", "E165", "E166", "E167", "E168", "E174", "E175",
    "E176", "E177", "E187", "E188", "E189", "E190", "E199", "E200", "E201",
    "E208", "E209", "E216", "E217", "E218", "E219", "E225", "E226", "E227",
    "E228", "E229", "E230", "E231", "E232", "E233", "E234", "E235", "E236",
    "E237", "E238", "E239", "E240", "E241", "E242", "E243", "E244", "E245",
    "E246", "E247", "E248", "E249", "E250", "E251", "E252", "E253", "E254",
    "E255", "E256",
]

# Map EGI channel counts to the MNE built-in GSN-HydroCel montage names.
_GSN_MONTAGE = {
    32: "GSN-HydroCel-32",
    64: "GSN-HydroCel-64_1.0",
    128: "GSN-HydroCel-128",
    129: "GSN-HydroCel-129",
    256: "GSN-HydroCel-256",
    257: "GSN-HydroCel-257",
}

_READERS = {
    ".edf": lambda f: mne.io.read_raw_edf(f, preload=True, verbose="ERROR"),
    ".bdf": lambda f: mne.io.read_raw_bdf(f, preload=True, verbose="ERROR"),
    ".vhdr": lambda f: mne.io.read_raw_brainvision(f, preload=True, verbose="ERROR"),
    ".set": lambda f: mne.io.read_raw_eeglab(f, preload=True, verbose="ERROR"),
    ".fif": lambda f: mne.io.read_raw_fif(f, preload=True, verbose="ERROR"),
    ".mff": lambda f: mne.io.read_raw_egi(f, preload=True, verbose="ERROR"),
    ".raw": lambda f: mne.io.read_raw_egi(f, preload=True, verbose="ERROR"),
}


def read_raw(path: str) -> mne.io.BaseRaw:
    """Read an EEG file, dispatching on extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext not in _READERS:
        raise ValueError(f"Unsupported file type: {ext!r} ({path})")
    raw = _READERS[ext](path)
    raw.pick("eeg") if "eeg" in raw else raw.pick_types(eeg=True, exclude=[])
    return raw


def strip_eeg_prefix(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Strip a leading ``"EEG "`` from channel labels (EDF case, dataimport.m)."""
    rename = {
        ch: ch[4:]
        for ch in raw.ch_names
        if ch.startswith("EEG ") and len(ch) > 4
    }
    if rename:
        raw.rename_channels(rename)
    return raw


def set_montage(raw: mne.io.BaseRaw, montage: str | None = None) -> mne.io.BaseRaw:
    """Assign a montage.

    If ``montage`` is given it is used directly.  Otherwise a GSN-HydroCel
    montage is chosen for EGI nets by channel count, falling back to the
    ``standard_1005`` 10-5 montage (matching ``standard-10-5-cap385.elp``).
    """
    if montage is None:
        nch = len(raw.ch_names)
        montage = _GSN_MONTAGE.get(nch, "standard_1005")
    dig = mne.channels.make_standard_montage(montage)
    raw.set_montage(dig, on_missing="ignore", match_case=False)
    return raw


def remove_peripheral_channels(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    """Drop peripheral EGI electrodes for 128/256-channel nets (dataimport.m)."""
    nch = len(raw.ch_names)
    exclude = []
    if nch in (128, 130):
        exclude = CHANEXCL_128
    elif nch == 256:
        exclude = CHANEXCL_256
    present = [ch for ch in exclude if ch in raw.ch_names]
    if present:
        raw.drop_channels(present)
    return raw
