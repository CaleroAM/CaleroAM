#!/usr/bin/env python3
"""
Generate the high-fidelity animated SVG terminal banner for CaleroAM.
Uses optimal transport and Floyd-Steinberg dither / particle stippling inspired by macu-dev,
cycling seamlessly between Tux (Linux), NixOS snowflake, and OpenClaw lobster.
"""

from pathlib import Path
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parent.parent
SOURCE = Path(__file__).resolve().parent / "source"
ASSETS = ROOT / "assets"

CX = 800.0
CY = 176.0
COLOR_DOT = "#ffffff"
COLOR_LABEL = "#ffffff"
TRAVELLER_COUNT = 650
HOLD_COUNT = 1850

def load_tux():
    im = Image.open(SOURCE / "tux.png").convert("RGBA")
    im.thumbnail((140, 140), Image.Resampling.LANCZOS)
    alpha = np.array(im.getchannel("A"))
    ys, xs = np.where(alpha > 120)
    pts = np.column_stack((xs, ys)).astype(np.float32)
    return center_pts(pts)

def load_nixos():
    im = Image.open(SOURCE / "nixos.png").convert("RGBA")
    im.thumbnail((140, 140), Image.Resampling.LANCZOS)
    alpha = np.array(im.getchannel("A"))
    ys, xs = np.where(alpha > 120)
    pts = np.column_stack((xs, ys)).astype(np.float32)
    return center_pts(pts)

def load_lobster():
    im = Image.open(SOURCE / "lobster.png").convert("RGB")
    arr = np.array(im)
    ys_raw, xs_raw = np.where(arr[:, :, 0] > 120)
    crop = im.crop((xs_raw.min(), ys_raw.min(), xs_raw.max(), ys_raw.max()))
    crop = crop.resize((145, 135), Image.Resampling.LANCZOS)
    arr_c = np.array(crop)
    ys, xs = np.where(arr_c[:, :, 0] > 120)
    pts = np.column_stack((xs, ys)).astype(np.float32)
    return center_pts(pts)

def center_pts(pts, cx=CX, cy=CY):
    min_x, max_x = pts[:, 0].min(), pts[:, 0].max()
    min_y, max_y = pts[:, 1].min(), pts[:, 1].max()
    off_x = cx - (min_x + max_x) / 2.0
    off_y = cy - (min_y + max_y) / 2.0
    return pts + np.array([off_x, off_y], dtype=np.float32)

def particle_path(pts):
    int_pts = np.rint(pts).astype(int)
    unique = sorted({(int(x), int(y)) for x, y in int_pts}, key=lambda p: (p[1], p[0]))
    return "".join(f"M{x} {y}h1" for x, y in unique)

def build_svg():
    rng = np.random.default_rng(314159)
    
    tux_all = load_tux()
    nix_all = load_nixos()
    lob_all = load_lobster()
    
    # Travellers
    tux_trav_idx = rng.choice(len(tux_all), TRAVELLER_COUNT, replace=False)
    nix_trav_idx = rng.choice(len(nix_all), TRAVELLER_COUNT, replace=False)
    lob_trav_idx = rng.choice(len(lob_all), TRAVELLER_COUNT, replace=False)
    
    tux_trav = tux_all[tux_trav_idx]
    nix_trav = nix_all[nix_trav_idx]
    lob_trav = lob_all[lob_trav_idx]
    
    # Remaining for hold clouds
    tux_rem = np.delete(tux_all, tux_trav_idx, axis=0)
    nix_rem = np.delete(nix_all, nix_trav_idx, axis=0)
    lob_rem = np.delete(lob_all, lob_trav_idx, axis=0)
    
    tux_hold = tux_rem[rng.choice(len(tux_rem), min(HOLD_COUNT, len(tux_rem)), replace=False)]
    nix_hold = nix_rem[rng.choice(len(nix_rem), min(HOLD_COUNT, len(nix_rem)), replace=False)]
    lob_hold = lob_rem[rng.choice(len(lob_rem), min(HOLD_COUNT, len(lob_rem)), replace=False)]
    
    # Optimal transport: Tux -> Nix -> Lob -> Tux
    c_tn = cdist(tux_trav, nix_trav, metric="sqeuclidean")
    _, c1 = linear_sum_assignment(c_tn)
    nix_trav_ord = nix_trav[c1]
    
    c_nl = cdist(nix_trav_ord, lob_trav, metric="sqeuclidean")
    _, c2 = linear_sum_assignment(c_nl)
    lob_trav_ord = lob_trav[c2]
    
    tux_hold_path = particle_path(tux_hold)
    nix_hold_path = particle_path(nix_hold)
    lob_hold_path = particle_path(lob_hold)
    
    # SVG construction
    svg = []
    svg.append('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 960 410" width="960" height="410" role="img" aria-labelledby="title desc">')
    svg.append('  <title id="title">Ángel Calero — Systems &amp; Backend Engineer</title>')
    svg.append('  <desc id="desc">Linux terminal window with 1-bit dither particle morph between Tux, NixOS, and OpenClaw lobster.</desc>')
    svg.append('  <defs>')
    svg.append('    <clipPath id="visualClip"><rect x="662" y="88" width="276" height="182" rx="4"/></clipPath>')
    svg.append('  </defs>')
    svg.append('')
    svg.append('  <!-- Ventana principal estilo Linux (Kitty / Alacritty / GNOME) -->')
    svg.append('  <rect width="960" height="410" rx="8" fill="#12131c"/>')
    svg.append('  <rect x="1.5" y="1.5" width="957" height="407" rx="7" fill="none" stroke="#252839" stroke-width="1.8"/>')
    svg.append('')
    svg.append('  <!-- Barra superior de Linux (Pestañas y Controles de ventana) -->')
    svg.append('  <rect x="2" y="2" width="956" height="38" rx="6 6 0 0" fill="#0b0c14"/>')
    svg.append('  <path d="M14 40 L14 14 C14 10 18 8 22 8 L180 8 C184 8 188 10 188 40 Z" fill="#181926"/>')
    svg.append('  <text x="32" y="26" fill="#c0caf5" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="13" font-weight="600">calero@linux: ~</text>')
    svg.append('  <text x="168" y="25" fill="#565f89" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="12">×</text>')
    svg.append('  <text x="206" y="26" fill="#565f89" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="15">+</text>')
    svg.append('')
    svg.append('  <!-- Botones de ventana nativos de Linux (Minimizar, Maximizar, Cerrar) -->')
    svg.append('  <g transform="translate(865, 12)">')
    svg.append('    <rect x="0" y="0" width="24" height="20" rx="3" fill="#181926" stroke="#252839" stroke-width="1"/>')
    svg.append('    <line x1="6" y1="12" x2="18" y2="12" stroke="#a0aec0" stroke-width="2" stroke-linecap="round"/>')
    svg.append('    <rect x="32" y="0" width="24" height="20" rx="3" fill="#181926" stroke="#252839" stroke-width="1"/>')
    svg.append('    <rect x="38" y="5" width="12" height="10" fill="none" stroke="#a0aec0" stroke-width="1.6" rx="1"/>')
    svg.append('    <rect x="64" y="0" width="24" height="20" rx="3" fill="#181926" stroke="#252839" stroke-width="1"/>')
    svg.append('    <line x1="71" y1="5" x2="81" y2="15" stroke="#f7768e" stroke-width="1.8" stroke-linecap="round"/>')
    svg.append('    <line x1="81" y1="5" x2="71" y2="15" stroke="#f7768e" stroke-width="1.8" stroke-linecap="round"/>')
    svg.append('  </g>')
    svg.append('')
    svg.append('  <path d="M2 40H958" stroke="#252839" stroke-width="1.5"/>')
    svg.append('')
    svg.append('  <!-- Panel izquierdo: Terminal Session -->')
    svg.append('  <rect x="20" y="54" width="624" height="340" rx="6" fill="#181926" stroke="#252839" stroke-width="1.2"/>')
    svg.append('')
    svg.append('  <!-- Prompt 1: whoami -->')
    svg.append('  <text x="38" y="86" fill="#e2e8f0" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="15" font-weight="700">❯ whoami</text>')
    svg.append('  <text x="38" y="112" fill="#ffffff" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="14" font-weight="700">ángel_calero</text>')
    svg.append('  <text x="150" y="112" fill="#64748b" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="14">—</text>')
    svg.append('  <text x="170" y="112" fill="#cbd5e1" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="14">Systems &amp; Backend Engineer</text>')
    svg.append('  <path d="M38 128H624" stroke="#252839" stroke-width="1"/>')
    svg.append('')
    svg.append('  <!-- Prompt 2: system info -->')
    svg.append('  <text x="38" y="156" fill="#e2e8f0" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="15" font-weight="700">❯ uname -srmo &amp;&amp; arch-info</text>')
    svg.append('  <text x="38" y="182" fill="#64748b" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="13">os:</text>')
    svg.append('  <text x="150" y="182" fill="#e2e8f0" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="13">Linux 6.x · Declarative &amp; Reproducible Systems</text>')
    svg.append('  <text x="38" y="206" fill="#64748b" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="13">architecture:</text>')
    svg.append('  <text x="150" y="206" fill="#e2e8f0" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="13">Clean Architecture · DDD · Idempotent Backends</text>')
    svg.append('  <text x="38" y="230" fill="#64748b" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="13">ai_engineering:</text>')
    svg.append('  <text x="150" y="230" fill="#e2e8f0" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="13">Deterministic Wrappers, Tree-sitter &amp; MLOps</text>')
    svg.append('  <path d="M38 248H624" stroke="#252839" stroke-width="1"/>')
    svg.append('')
    svg.append('  <!-- Prompt 3: projects -->')
    svg.append('  <text x="38" y="276" fill="#e2e8f0" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="15" font-weight="700">❯ ls -1 ./featured-projects/</text>')
    svg.append('  <text x="38" y="302" fill="#cbd5e1" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="13">graphtyn/</text>')
    svg.append('  <text x="150" y="302" fill="#94a3b8" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="13">Code Intelligence &amp; MCP Server (v0.10.x)</text>')
    svg.append('  <text x="38" y="326" fill="#cbd5e1" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="13">isac26/</text>')
    svg.append('  <text x="150" y="326" fill="#94a3b8" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="13">StatsBomb &amp; Club América Analytics Engine</text>')
    svg.append('  <text x="38" y="350" fill="#cbd5e1" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="13">modIAceb/</text>')
    svg.append('  <text x="150" y="350" fill="#94a3b8" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="13">Reproducible MLOps Pipeline (TabPFN + Django)</text>')
    svg.append('  <text x="38" y="374" fill="#cbd5e1" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="13">TourMuseosPuebla/</text>')
    svg.append('  <text x="180" y="374" fill="#94a3b8" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="13">Decoupled C#/.NET + Dynamic Vertex AI routing</text>')
    svg.append('')
    svg.append('  <!-- Panel derecho: Matrix Display de Micro-Puntitos (Estilo macu-dev VISUAL.MAP) -->')
    svg.append('  <rect x="660" y="54" width="280" height="340" rx="6" fill="#0d0e15" stroke="#252839" stroke-width="1.2"/>')
    svg.append(f'  <text x="676" y="78" fill="{COLOR_LABEL}" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="12" font-weight="700" letter-spacing="1.2">VISUAL.MAP</text>')
    svg.append('  <text x="924" y="78" text-anchor="end" fill="#64748b" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="10">PTS 02500 · 1-BIT</text>')
    svg.append('  <path d="M676 86H924" stroke="#252839" stroke-width="1"/>')
    svg.append(f'  <path d="M676 96h8M676 96v8M924 96h-8M924 96v8M676 256h8M676 256v-8M924 256h-8M924 256v-8" fill="none" stroke="{COLOR_DOT}" opacity="0.45"/>')
    svg.append('')
    svg.append('  <!-- Matriz de micro-puntitos con renderizado nítido -->')
    svg.append('  <g clip-path="url(#visualClip)" shape-rendering="crispEdges">')
    
    # 1. Nubes estáticas de micro-puntos que fijan la fidelidad durante cada hold
    # Tux hold
    svg.append(f'    <path d="{tux_hold_path}" fill="none" stroke="{COLOR_DOT}" stroke-width="1" opacity="0.85">')
    svg.append('      <animate attributeName="opacity" dur="14.1s" repeatCount="indefinite" calcMode="linear" keyTimes="0; 0.2482; 0.2837; 0.9645; 1.0" values="0.85; 0.85; 0; 0; 0.85"/>')
    svg.append('    </path>')
    
    # NixOS hold
    svg.append(f'    <path d="{nix_hold_path}" fill="none" stroke="{COLOR_DOT}" stroke-width="1" opacity="0">')
    svg.append('      <animate attributeName="opacity" dur="14.1s" repeatCount="indefinite" calcMode="linear" keyTimes="0; 0.2979; 0.3333; 0.5816; 0.6170; 1.0" values="0; 0; 0.85; 0.85; 0; 0"/>')
    svg.append('    </path>')
    
    # Lobster hold
    svg.append(f'    <path d="{lob_hold_path}" fill="none" stroke="{COLOR_DOT}" stroke-width="1" opacity="0">')
    svg.append('      <animate attributeName="opacity" dur="14.1s" repeatCount="indefinite" calcMode="linear" keyTimes="0; 0.6312; 0.6667; 0.9149; 0.9504; 1.0" values="0; 0; 0.85; 0.85; 0; 0"/>')
    svg.append('    </path>')
    
    # 2. Travellers móviles (650 partículas en vuelo continuo y suave)
    key_times = "0; 0.2482; 0.3333; 0.5816; 0.6667; 0.9149; 1.0"
    key_splines = "0.42 0 0.25 1; 0.42 0 0.25 1; 0.42 0 0.25 1; 0.42 0 0.25 1; 0.42 0 0.25 1; 0.42 0 0.25 1"
    
    for i in range(TRAVELLER_COUNT):
        p0 = f"{tux_trav[i,0]:.1f} {tux_trav[i,1]:.1f}"
        p1 = f"{nix_trav_ord[i,0]:.1f} {nix_trav_ord[i,1]:.1f}"
        p2 = f"{lob_trav_ord[i,0]:.1f} {lob_trav_ord[i,1]:.1f}"
        vals = f"{p0}; {p0}; {p1}; {p1}; {p2}; {p2}; {p0}"
        
        svg.append(f'    <path d="M-.65-.65h1.3v1.3h-1.3z" fill="{COLOR_DOT}" opacity="0.85">')
        svg.append(f'      <animateTransform attributeName="transform" type="translate" dur="14.1s" repeatCount="indefinite" calcMode="spline" keyTimes="{key_times}" values="{vals}" keySplines="{key_splines}"/>')
        svg.append('    </path>')
        
    svg.append('  </g>')
    svg.append('')
    svg.append('  <!-- Etiquetas dinámicas sincronizadas con los 3 estados (Crossfade fluido sin huecos vacíos) -->')
    svg.append('  <g>')
    # Tux label
    svg.append('    <g opacity="1">')
    svg.append('      <animate attributeName="opacity" dur="14.1s" repeatCount="indefinite" calcMode="linear" keyTimes="0; 0.2482; 0.2908; 0.9574; 1.0" values="1; 1; 0; 0; 1"/>')
    svg.append(f'      <text x="800" y="280" text-anchor="middle" fill="{COLOR_LABEL}" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="13" font-weight="700" letter-spacing="0.5">tux (linux)</text>')
    svg.append('      <text x="800" y="298" text-anchor="middle" fill="#64748b" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="11">kernel: gnu/linux</text>')
    svg.append('    </g>')
    # NixOS label
    svg.append('    <g opacity="0">')
    svg.append('      <animate attributeName="opacity" dur="14.1s" repeatCount="indefinite" calcMode="linear" keyTimes="0; 0.2482; 0.2908; 0.5816; 0.6241; 1.0" values="0; 0; 1; 1; 0; 0"/>')
    svg.append(f'      <text x="800" y="280" text-anchor="middle" fill="{COLOR_LABEL}" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="13" font-weight="700" letter-spacing="0.5">nixos snowflake</text>')
    svg.append('      <text x="800" y="298" text-anchor="middle" fill="#64748b" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="11">declarative &amp; reproducible</text>')
    svg.append('    </g>')
    # Lobster label
    svg.append('    <g opacity="0">')
    svg.append('      <animate attributeName="opacity" dur="14.1s" repeatCount="indefinite" calcMode="linear" keyTimes="0; 0.5816; 0.6241; 0.9149; 0.9574; 1.0" values="0; 0; 1; 1; 0; 0"/>')
    svg.append(f'      <text x="800" y="280" text-anchor="middle" fill="{COLOR_LABEL}" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="13" font-weight="700" letter-spacing="0.5">agente ia OpenClaw</text>')
    svg.append('      <text x="800" y="298" text-anchor="middle" fill="#64748b" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="11">autonomous systems &amp; runtime</text>')
    svg.append('    </g>')
    svg.append('  </g>')
    svg.append('')
    svg.append('  <!-- Tarjeta de estado de runtime (inferior derecha) -->')
    svg.append('  <rect x="676" y="312" width="248" height="66" rx="4" fill="#12131c" stroke="#1f212f" stroke-width="1"/>')
    svg.append('  <text x="690" y="332" fill="#cbd5e1" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="11">state: </text>')
    svg.append('  <text x="738" y="332" fill="#a3e635" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="11">production_ready</text>')
    svg.append('  <text x="690" y="350" fill="#cbd5e1" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="11">focus: </text>')
    svg.append('  <text x="738" y="350" fill="#f1f5f9" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="11">Backend &amp; AI Systems</text>')
    svg.append('  <text x="690" y="368" fill="#cbd5e1" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="11">stack: </text>')
    svg.append('  <text x="738" y="368" fill="#94a3b8" font-family="ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" font-size="11">Linux · Docker · NixOS</text>')
    svg.append('</svg>')
    
    content = "\n".join(svg) + "\n"
    target = ASSETS / "terminal.svg"
    target.write_text(content, encoding="utf-8")
    (ASSETS / "terminal-nixos.svg").write_text(content, encoding="utf-8")
    (ASSETS / "terminal.v2.svg").write_text(content, encoding="utf-8")
    
    print(f"Generated {target} and terminal.v2.svg ({len(content)} bytes)")

if __name__ == "__main__":
    build_svg()
