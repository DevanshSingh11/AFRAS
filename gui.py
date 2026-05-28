"""
gui.py  –  Face Recognition & Attendance  –  GUI
=================================================
Bug fixes in this version:
  • Theme toggle: StringVar/Label objects properly re-initialized; no stale refs
  • Theme toggle: _tick_clock / _refresh_stats after-loops are cancelled before rebuild
  • Theme toggle: _recognition_active state preserved across rebuild
  • _refresh_stats: single scheduled chain (no stacking after rebuild)
  • _tick_clock: single scheduled chain (no stacking after rebuild)
  • register_image: runs in background thread — UI no longer freezes
  • _build_section padx: correct asymmetric gutter between buttons
  • make_sortable: sort state toggled properly; alternating row tags reapplied after sort
  • Tooltip: guarded with try/except around winfo_rootx to survive widget destruction
  • ModernButton: activebackground set to keep visual consistency on click
  • CAP_DSHOW: handled in face_system (platform-guarded) — GUI unaffected
  • _safe_release: cv2.waitKey removed — no platform freeze
  • get_today_attendance: not called per-frame (in-memory counter used instead)
"""

import threading
import tkinter as tk
from datetime import datetime, timedelta
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Dict, List, Optional

from face_system import FaceRecognitionSystem


# ═══════════════════════════════════════════════════════════════
# THEME DEFINITIONS
# ═══════════════════════════════════════════════════════════════
THEMES: Dict[str, Dict] = {
    "light": {
        "bg":          "#f4f6f8",
        "card":        "#ffffff",
        "header_bg":   "#1565C0",
        "header_fg":   "#ffffff",
        "text":        "#1a1a2e",
        "subtext":     "#6b7280",
        "border":      "#e5e7eb",
        "status_bg":   "#e8ecf0",
        "PRIMARY":     "#2196F3",
        "SUCCESS":     "#16a34a",
        "DANGER":      "#dc2626",
        "WARNING":     "#d97706",
        "MUTED":       "#475569",
        "row_even":    "#f9fafb",
        "row_odd":     "#ffffff",
        "sel_row":     "#bfdbfe",
    },
    "dark": {
        "bg":          "#0f172a",
        "card":        "#1e293b",
        "header_bg":   "#0f172a",
        "header_fg":   "#e2e8f0",
        "text":        "#f1f5f9",
        "subtext":     "#94a3b8",
        "border":      "#334155",
        "status_bg":   "#1e293b",
        "PRIMARY":     "#3b82f6",
        "SUCCESS":     "#22c55e",
        "DANGER":      "#ef4444",
        "WARNING":     "#f59e0b",
        "MUTED":       "#64748b",
        "row_even":    "#1e293b",
        "row_odd":     "#243044",
        "sel_row":     "#1d4ed8",
    },
}


# ═══════════════════════════════════════════════════════════════
# TOOLTIP
# ═══════════════════════════════════════════════════════════════
class Tooltip:
    def __init__(self, widget: tk.Widget, text: str) -> None:
        self._widget = widget
        self._text   = text
        self._win: Optional[tk.Toplevel] = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)
        widget.bind("<Destroy>", self._hide)

    def _show(self, _event=None) -> None:
        if self._win:
            return
        try:
            x = self._widget.winfo_rootx() + 20
            y = self._widget.winfo_rooty() + self._widget.winfo_height() + 4
        except tk.TclError:
            return   # widget already destroyed
        self._win = tw = tk.Toplevel(self._widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tk.Label(tw, text=self._text,
                 background="#1e293b", foreground="#f1f5f9",
                 relief=tk.FLAT, font=("Segoe UI", 9),
                 padx=8, pady=4).pack()

    def _hide(self, _event=None) -> None:
        if self._win:
            try:
                self._win.destroy()
            except tk.TclError:
                pass
            self._win = None


# ═══════════════════════════════════════════════════════════════
# MODERN BUTTON
# ═══════════════════════════════════════════════════════════════
class ModernButton(tk.Button):
    def __init__(self, parent, tooltip: str = "", **kwargs) -> None:
        self._bg = kwargs.get("bg", "#2196F3")
        # activebackground matches hover so there's no ugly flash on click
        kwargs.setdefault("activebackground", self._lighten(self._bg))
        kwargs.setdefault("activeforeground", "#ffffff")
        super().__init__(parent,
                         relief=tk.FLAT, bd=0,
                         highlightthickness=0,
                         cursor="hand2",
                         **kwargs)
        self.bind("<Enter>", self._hover_in)
        self.bind("<Leave>", self._hover_out)
        if tooltip:
            Tooltip(self, tooltip)

    def _hover_in(self, _=None) -> None:
        if str(self["state"]) != "disabled":
            self["bg"] = self._lighten(self._bg)

    def _hover_out(self, _=None) -> None:
        self["bg"] = self._bg

    def update_bg(self, color: str) -> None:
        self._bg = color
        self["bg"] = color
        self["activebackground"] = self._lighten(color)

    @staticmethod
    def _lighten(color: str) -> str:
        if color.startswith("#") and len(color) == 7:
            r = min(255, int(color[1:3], 16) + 28)
            g = min(255, int(color[3:5], 16) + 28)
            b = min(255, int(color[5:7], 16) + 28)
            return f"#{r:02x}{g:02x}{b:02x}"
        return color


# ═══════════════════════════════════════════════════════════════
# SORTABLE TREEVIEW  (with correct sort-state toggle & row recolor)
# ═══════════════════════════════════════════════════════════════
def make_sortable(tree: ttk.Treeview) -> None:
    """
    Click a heading once → sort ascending.
    Click again → sort descending.
    Alternating row tags are reapplied after every sort.
    """
    _sort_state: Dict[str, bool] = {}   # col → currently_descending

    def _reapply_tags() -> None:
        for idx, child in enumerate(tree.get_children("")):
            tree.item(child, tags=("even" if idx % 2 == 0 else "odd",))

    def sort_col(col: str) -> None:
        descending = _sort_state.get(col, False)
        data = [(tree.set(child, col), child) for child in tree.get_children("")]
        try:
            data.sort(key=lambda x: float(x[0].rstrip("%")), reverse=descending)
        except ValueError:
            data.sort(key=lambda x: x[0].lower(), reverse=descending)
        for idx, (_, child) in enumerate(data):
            tree.move(child, "", idx)
        _sort_state[col] = not descending          # toggle for next click
        # Update heading arrow
        for c in tree["columns"]:
            arrow = (" ↑" if not descending else " ↓") if c == col else ""
            tree.heading(c, text=c + arrow)
        _reapply_tags()

    for col in tree["columns"]:
        tree.heading(col, text=col, command=lambda c=col: sort_col(c))


# ═══════════════════════════════════════════════════════════════
# MAIN GUI
# ═══════════════════════════════════════════════════════════════
class FaceRecognitionGUI:

    def __init__(self) -> None:
        self.system = FaceRecognitionSystem()
        self._recognition_active = False
        self._theme_name         = "light"
        self.T                   = THEMES[self._theme_name]

        # Wire recognition callbacks
        self.system.on_attendance_marked = self._cb_attendance_marked
        self.system.on_frame_processed   = self._cb_frame_processed

        self.root = tk.Tk()
        self.root.title("Face Recognition & Attendance System")
        self.root.geometry("700x820")
        self.root.minsize(640, 720)
        self.root.configure(bg=self.T["bg"])

        # ── Persistent StringVars (survive UI rebuild) ──
        self._clock_var   = tk.StringVar()
        self._live_fps    = tk.StringVar(value="—")
        self._live_faces  = tk.StringVar(value="—")
        self._status_var  = tk.StringVar(value="Ready.")
        self._stat_vars: Dict[str, tk.StringVar] = {}

        # ── Scheduler IDs (so we can cancel before rebuild) ──
        self._clock_id  : Optional[str] = None
        self._stats_id  : Optional[str] = None

        self._style_ttk()
        self._build_ui()
        self._bind_shortcuts()
        self._tick_clock()
        self._schedule_stats()

    # ─────────────────────────────────────────
    # TTK STYLE
    # ─────────────────────────────────────────
    def _style_ttk(self) -> None:
        self._ttk_style = ttk.Style()
        self._ttk_style.theme_use("clam")
        self._apply_ttk_theme()

    def _apply_ttk_theme(self) -> None:
        T = self.T
        s = self._ttk_style
        s.configure("Treeview",
                     background=T["card"], foreground=T["text"],
                     fieldbackground=T["card"], rowheight=28,
                     font=("Segoe UI", 10))
        s.configure("Treeview.Heading",
                     background=T["bg"], foreground=T["text"],
                     font=("Segoe UI", 10, "bold"), relief="flat")
        s.map("Treeview",
              background=[("selected", T["sel_row"])],
              foreground=[("selected", "#ffffff")])
        s.configure("Vertical.TScrollbar",
                     background=T["border"], troughcolor=T["bg"])

    # ─────────────────────────────────────────
    # THEME TOGGLE
    # ─────────────────────────────────────────
    def _toggle_theme(self) -> None:
        # Cancel scheduled callbacks so they don't double-up after rebuild
        if self._clock_id:
            self.root.after_cancel(self._clock_id)
            self._clock_id = None
        if self._stats_id:
            self.root.after_cancel(self._stats_id)
            self._stats_id = None

        self._theme_name = "dark" if self._theme_name == "light" else "light"
        self.T = THEMES[self._theme_name]
        self._apply_ttk_theme()

        # Destroy all children and rebuild
        for w in self.root.winfo_children():
            w.destroy()

        # Clear stat vars so _build_stats_card re-creates them
        self._stat_vars.clear()

        self._build_ui()
        self._bind_shortcuts()
        # Restart the loops
        self._tick_clock()
        self._schedule_stats()

    # ─────────────────────────────────────────
    # UI CONSTRUCTION
    # ─────────────────────────────────────────
    def _build_ui(self) -> None:
        T    = self.T
        root = self.root
        root.configure(bg=T["bg"])

        # ── Header ──────────────────────────────
        header = tk.Frame(root, bg=T["header_bg"], height=76)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        left_h = tk.Frame(header, bg=T["header_bg"])
        left_h.pack(side=tk.LEFT, fill=tk.Y, padx=14)
        tk.Label(left_h, text="🔐",
                 font=("Segoe UI Emoji", 26),
                 bg=T["header_bg"], fg=T["header_fg"]).pack(side=tk.LEFT, pady=10)
        tk.Label(left_h, text="Face Recognition & Attendance",
                 font=("Segoe UI", 16, "bold"),
                 bg=T["header_bg"], fg=T["header_fg"]).pack(side=tk.LEFT, padx=8)

        right_h = tk.Frame(header, bg=T["header_bg"])
        right_h.pack(side=tk.RIGHT, fill=tk.Y, padx=14)
        tk.Label(right_h, textvariable=self._clock_var,
                 font=("Segoe UI", 10),
                 bg=T["header_bg"], fg=T["header_fg"]).pack(side=tk.TOP, pady=(10, 2))
        theme_icon = "☀️" if self._theme_name == "dark" else "🌙"
        ModernButton(right_h, text=theme_icon,
                     command=self._toggle_theme,
                     bg=T["header_bg"], fg=T["header_fg"],
                     font=("Segoe UI Emoji", 13), width=3,
                     tooltip="Toggle dark / light mode").pack(side=tk.TOP)

        # ── Scrollable body ──────────────────────
        outer = tk.Frame(root, bg=T["bg"])
        outer.pack(fill=tk.BOTH, expand=True)

        self._canvas = tk.Canvas(outer, bg=T["bg"], highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.body = tk.Frame(self._canvas, bg=T["bg"])
        self._body_win = self._canvas.create_window((0, 0), window=self.body, anchor="nw")

        self.body.bind("<Configure>",
                       lambda _e: self._canvas.configure(
                           scrollregion=self._canvas.bbox("all")))
        self._canvas.bind("<Configure>",
                          lambda e: self._canvas.itemconfig(self._body_win, width=e.width))
        self._canvas.bind_all("<MouseWheel>",
                               lambda e: self._canvas.yview_scroll(
                                   int(-1 * (e.delta / 120)), "units"))

        # ── Content cards ────────────────────────
        self._build_stats_card()
        self._build_live_card()
        self._build_section("👤  Registration", [
            ("Register via Webcam",  self.register_webcam,  T["PRIMARY"],
             "Ctrl+R — capture face samples from your webcam"),
            ("Register from Image",  self.register_image,   T["PRIMARY"],
             "Pick an image file — preview, then confirm"),
        ])
        self._build_section("🎯  Recognition", [
            ("▶  Start Recognition", self.start_recognition, T["SUCCESS"],
             "Ctrl+S — open live webcam feed with attendance tracking"),
            ("■  Stop Recognition",  self.stop_recognition,  T["MUTED"],
             "Ctrl+X — signal recognition loop to stop"),
        ])
        self._build_section("📋  Management", [
            ("View Registered Faces",   self.view_faces,       T["PRIMARY"],
             "Browse all enrolled people"),
            ("View Today's Attendance", self.view_attendance,  T["PRIMARY"],
             "See who has been marked present today"),
            ("Export Attendance CSV",   self.export_attendance, T["WARNING"],
             "Download a filtered CSV for a date range"),
            ("Delete a Face",           self.delete_face,       T["DANGER"],
             "Remove a person's encodings permanently"),
        ])
        self._build_section("⚙️  Settings", [
            ("Adjust Settings", self.show_settings, T["MUTED"],
             "Tune tolerance, confidence, model, frame-skip"),
        ])

        pad = tk.Frame(self.body, bg=T["bg"])
        pad.pack(fill=tk.X, padx=20, pady=(4, 20))
        ModernButton(pad, text="Exit  (Ctrl+Q)", command=self.root.destroy,
                     bg=T["DANGER"], fg="white",
                     font=("Segoe UI", 11, "bold"), height=2,
                     tooltip="Close the application").pack(fill=tk.X)

        # ── Status bar ───────────────────────────
        self._status_bar = tk.Label(
            root, textvariable=self._status_var,
            font=("Segoe UI", 9),
            bg=T["status_bg"], fg=T["subtext"],
            anchor=tk.W, padx=12, pady=4)
        self._status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # ── Stats card ──────────────────────────────
    def _build_stats_card(self) -> None:
        T    = self.T
        card = self._card(self.body)
        card.pack(fill=tk.X, padx=20, pady=(20, 8))

        tk.Label(card, text="📊  Statistics",
                 font=("Segoe UI", 12, "bold"),
                 bg=T["card"], fg=T["text"]).pack(anchor="w", padx=16, pady=(12, 6))

        grid = tk.Frame(card, bg=T["card"])
        grid.pack(fill=tk.X, padx=16, pady=(0, 12))

        STAT_KEYS = [
            ("Registered People",    "total_registered"),
            ("Total Encodings",      "total_encodings"),
            ("Today's Attendance",   "attendance_today"),
            ("Model",                "model"),
            ("Tolerance",            "tolerance"),
            ("Confidence Threshold", "confidence_threshold"),
        ]
        for i, (label, key) in enumerate(STAT_KEYS):
            row_idx, col = divmod(i, 2)
            cell = tk.Frame(grid, bg=T["card"])
            cell.grid(row=row_idx, column=col, sticky="ew", padx=8, pady=3)
            grid.columnconfigure(col, weight=1)

            tk.Label(cell, text=label + ":",
                     font=("Segoe UI", 9), bg=T["card"],
                     fg=T["subtext"], anchor="w").pack(anchor="w")
            var = self._stat_vars.get(key) or tk.StringVar(value="—")
            self._stat_vars[key] = var
            tk.Label(cell, textvariable=var,
                     font=("Segoe UI", 11, "bold"),
                     bg=T["card"], fg=T["PRIMARY"], anchor="w").pack(anchor="w")

        self._refresh_dot = tk.Label(card, text="●", font=("Segoe UI", 8),
                                     bg=T["card"], fg=T["SUCCESS"])
        self._refresh_dot.place(relx=1.0, rely=0.0, anchor="ne", x=-10, y=10)

    # ── Live card ────────────────────────────────
    def _build_live_card(self) -> None:
        T    = self.T
        card = self._card(self.body)
        card.pack(fill=tk.X, padx=20, pady=(0, 8))

        tk.Label(card, text="🎥  Live Recognition",
                 font=("Segoe UI", 12, "bold"),
                 bg=T["card"], fg=T["text"]).pack(anchor="w", padx=16, pady=(12, 4))

        row = tk.Frame(card, bg=T["card"])
        row.pack(fill=tk.X, padx=16, pady=(0, 12))

        for label_text, var in (("FPS", self._live_fps), ("Faces detected", self._live_faces)):
            cell = tk.Frame(row, bg=T["card"])
            cell.pack(side=tk.LEFT, padx=16)
            tk.Label(cell, text=label_text + ":",
                     font=("Segoe UI", 9), bg=T["card"], fg=T["subtext"]).pack(anchor="w")
            tk.Label(cell, textvariable=var,
                     font=("Segoe UI", 16, "bold"),
                     bg=T["card"], fg=T["SUCCESS"]).pack(anchor="w")

    # ── Section builder ──────────────────────────
    def _build_section(self, title: str, buttons: list) -> None:
        T    = self.T
        card = self._card(self.body)
        card.pack(fill=tk.X, padx=20, pady=(0, 8))

        tk.Label(card, text=title,
                 font=("Segoe UI", 11, "bold"),
                 bg=T["card"], fg=T["text"]).pack(anchor="w", padx=16, pady=(12, 6))

        btn_grid = tk.Frame(card, bg=T["card"])
        btn_grid.pack(fill=tk.X, padx=16, pady=(0, 12))

        for i, (text, cmd, color, tip) in enumerate(buttons):
            col     = i % 2
            row_idx = i // 2
            btn_grid.columnconfigure(col, weight=1)
            # FIX: correct asymmetric padding — 3px right-gutter on left column only
            px = (0, 3) if col == 0 else (3, 0)
            ModernButton(btn_grid, text=text, command=cmd,
                         bg=color, fg="white",
                         font=("Segoe UI", 10), height=2,
                         tooltip=tip
                         ).grid(row=row_idx, column=col, sticky="ew",
                                padx=px, pady=4)

    def _card(self, parent: tk.Widget) -> tk.Frame:
        return tk.Frame(parent, bg=self.T["card"],
                        highlightbackground=self.T["border"],
                        highlightthickness=1)

    # ─────────────────────────────────────────
    # KEYBOARD SHORTCUTS
    # ─────────────────────────────────────────
    def _bind_shortcuts(self) -> None:
        for seq, fn in [
            ("<Control-r>", self.register_webcam),
            ("<Control-R>", self.register_webcam),
            ("<Control-s>", self.start_recognition),
            ("<Control-S>", self.start_recognition),
            ("<Control-x>", self.stop_recognition),
            ("<Control-X>", self.stop_recognition),
            ("<Control-q>", self.root.destroy),
            ("<Control-Q>", self.root.destroy),
        ]:
            self.root.bind(seq, lambda _e, f=fn: f())

    # ─────────────────────────────────────────
    # CLOCK  (single chain — ID stored for cancellation)
    # ─────────────────────────────────────────
    def _tick_clock(self) -> None:
        self._clock_var.set(datetime.now().strftime("%A, %d %b %Y  %H:%M:%S"))
        self._clock_id = self.root.after(1000, self._tick_clock)

    # ─────────────────────────────────────────
    # STATUS BAR
    # ─────────────────────────────────────────
    def _set_status(self, msg: str, kind: str = "info") -> None:
        colors = {
            "info": self.T["subtext"],
            "ok":   self.T["SUCCESS"],
            "warn": self.T["WARNING"],
            "err":  self.T["DANGER"],
        }
        self._status_var.set(msg)
        try:
            self._status_bar.config(fg=colors.get(kind, self.T["subtext"]))
        except tk.TclError:
            pass

    # ─────────────────────────────────────────
    # STATS REFRESH  (single chain — ID stored for cancellation)
    # ─────────────────────────────────────────
    def _schedule_stats(self) -> None:
        self._do_refresh_stats()

    def _do_refresh_stats(self) -> None:
        try:
            stats = self.system.get_statistics()
            for key, var in self._stat_vars.items():
                val = stats.get(key, "—")
                if isinstance(val, float):
                    val = f"{val:.2f}"
                elif key == "model":
                    val = str(val).upper()
                var.set(str(val))

            # Blink the refresh dot
            dot = self._refresh_dot
            dot.config(fg=self.T["card"])
            self.root.after(180, lambda: _safe_dot_reset(dot, self.T["SUCCESS"]))
        except tk.TclError:
            return   # widget destroyed (e.g., during shutdown)

        self._stats_id = self.root.after(5000, self._do_refresh_stats)

    # ─────────────────────────────────────────
    # RECOGNITION CALLBACKS  (called from background thread)
    # ─────────────────────────────────────────
    def _cb_attendance_marked(self, name: str, confidence: float) -> None:
        self.root.after(0, lambda: self._set_status(
            f"✅  Attendance marked — {name}  ({confidence:.0%})", "ok"))
        self.root.after(0, self._do_refresh_stats)

    def _cb_frame_processed(self, fps: float, n_faces: int) -> None:
        self.root.after(0, lambda: self._live_fps.set(f"{fps:.1f}"))
        self.root.after(0, lambda: self._live_faces.set(str(n_faces)))

    # ═══════════════════════════════════════════
    # REGISTRATION
    # ═══════════════════════════════════════════
    def register_webcam(self) -> None:
        name = simpledialog.askstring(
            "Register Face", "Enter person's full name:", parent=self.root)
        if not name or not name.strip():
            return
        name = name.strip()

        if name in self.system.known_names:
            if not messagebox.askyesno(
                    "Already Registered",
                    f"'{name}' already has encodings.\nAdd more samples?",
                    parent=self.root):
                return

        samples = simpledialog.askinteger(
            "Sample Count", "How many face samples to capture?  (3 – 15)",
            initialvalue=8, minvalue=3, maxvalue=15, parent=self.root)
        if not samples:
            return

        messagebox.showinfo(
            "Ready",
            "A webcam window will open.\n\n"
            "  • Press  SPACE  to capture each sample.\n"
            "  • Keep only ONE face visible.\n"
            "  • Press  ESC  to cancel.",
            parent=self.root)

        self._set_status(f"📷  Capturing samples for '{name}'…", "info")

        def task():
            ok, msg = self.system.register_face_from_webcam(name, samples=samples)
            self.root.after(0, lambda: self._on_reg_done(ok, name, msg))

        threading.Thread(target=task, daemon=True).start()

    def register_image(self) -> None:
        name = simpledialog.askstring(
            "Register from Image", "Enter person's full name:", parent=self.root)
        if not name or not name.strip():
            return
        name = name.strip()

        path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                       ("All Files", "*.*")],
            parent=self.root)
        if not path:
            return

        # Optional face preview (requires Pillow)
        confirmed = [True]   # default True if preview unavailable
        try:
            import cv2 as _cv2
            from PIL import Image, ImageTk  # type: ignore

            img_cv   = _cv2.imread(path)
            if img_cv is None:
                raise ValueError("cv2 could not read image.")
            img_rgb  = _cv2.cvtColor(img_cv, _cv2.COLOR_BGR2RGB)
            pil_img  = Image.fromarray(img_rgb)
            pil_img.thumbnail((320, 240))

            prev = tk.Toplevel(self.root)
            prev.title("Confirm Registration Image")
            prev.configure(bg=self.T["card"])
            prev.resizable(False, False)
            prev.grab_set()

            tk.Label(prev, text=f"Register as: {name}",
                     font=("Segoe UI", 12, "bold"),
                     bg=self.T["card"], fg=self.T["text"]).pack(pady=(14, 6))

            photo = ImageTk.PhotoImage(pil_img)
            img_lbl = tk.Label(prev, image=photo, bg=self.T["card"])
            img_lbl.image = photo   # prevent GC
            img_lbl.pack(padx=20)

            confirmed[0] = False

            def _yes():
                confirmed[0] = True
                prev.destroy()

            btn_row = tk.Frame(prev, bg=self.T["card"])
            btn_row.pack(pady=12)
            ModernButton(btn_row, text="✓  Register", command=_yes,
                         bg=self.T["SUCCESS"], fg="white",
                         font=("Segoe UI", 10), width=14).pack(side=tk.LEFT, padx=6)
            ModernButton(btn_row, text="✗  Cancel", command=prev.destroy,
                         bg=self.T["DANGER"], fg="white",
                         font=("Segoe UI", 10), width=10).pack(side=tk.LEFT, padx=6)

            prev.wait_window()
        except Exception:
            pass   # Pillow not installed or image unreadable — skip preview

        if not confirmed[0]:
            self._set_status("Image registration cancelled.", "info")
            return

        self._set_status(f"🖼  Processing image for '{name}'…", "info")

        # BUG FIX: run in thread so GUI doesn't freeze during face encoding
        def task():
            ok, msg = self.system.register_face_from_image(name, path)
            self.root.after(0, lambda: self._on_reg_done(ok, name, msg))

        threading.Thread(target=task, daemon=True).start()

    def _on_reg_done(self, ok: bool, name: str, msg: str) -> None:
        if ok:
            messagebox.showinfo("✅  Registered", msg, parent=self.root)
            self._set_status(f"Registered '{name}' successfully.", "ok")
        else:
            messagebox.showerror("❌  Failed", msg, parent=self.root)
            self._set_status(f"Registration failed: {msg}", "err")
        self._do_refresh_stats()

    # ═══════════════════════════════════════════
    # RECOGNITION
    # ═══════════════════════════════════════════
    def start_recognition(self) -> None:
        if self._recognition_active:
            self._set_status("Recognition already running.", "warn")
            return
        if not self.system.known_names:
            messagebox.showwarning(
                "No Faces", "Register at least one face first.", parent=self.root)
            return

        self._recognition_active = True
        self._live_fps.set("…")
        self._live_faces.set("…")
        self._set_status("🎯  Recognition running — press ESC in the video window to stop.", "ok")

        def task():
            self.system.recognize_faces_realtime()
            self._recognition_active = False
            self.root.after(0, lambda: self._live_fps.set("—"))
            self.root.after(0, lambda: self._live_faces.set("—"))
            self.root.after(0, lambda: self._set_status("Recognition stopped.", "info"))
            self.root.after(0, self._do_refresh_stats)

        threading.Thread(target=task, daemon=True).start()

    def stop_recognition(self) -> None:
        if self._recognition_active:
            self.system.stop_recognition()
            self._set_status("Stopping recognition…", "warn")
        else:
            self._set_status("Recognition is not currently active.", "info")

    # ═══════════════════════════════════════════
    # VIEW FACES
    # ═══════════════════════════════════════════
    def view_faces(self) -> None:
        faces = self.system.list_registered_faces()
        T     = self.T

        win = tk.Toplevel(self.root)
        win.title("Registered Faces")
        win.geometry("640x460")
        win.configure(bg=T["bg"])
        win.grab_set()

        hdr = tk.Frame(win, bg=T["header_bg"], height=48)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        tk.Label(hdr, text="👥  Registered Faces",
                 font=("Segoe UI", 13, "bold"),
                 bg=T["header_bg"], fg=T["header_fg"]).pack(expand=True)

        if not faces:
            tk.Label(win, text="No faces registered yet.",
                     font=("Segoe UI", 12), bg=T["bg"],
                     fg=T["subtext"]).pack(expand=True)
        else:
            frm = tk.Frame(win, bg=T["bg"])
            frm.pack(fill=tk.BOTH, expand=True, padx=16, pady=12)

            cols = ("Name", "Registered Date", "Samples", "Image")
            tree = ttk.Treeview(frm, columns=cols, show="headings")
            for col, w, anch in zip(
                    cols, (200, 200, 80, 70),
                    ("w", "w", "center", "center")):
                tree.heading(col, text=col)
                tree.column(col, width=w, anchor=anch)

            tree.tag_configure("even", background=T["row_even"])
            tree.tag_configure("odd",  background=T["row_odd"])

            for i, face in enumerate(faces):
                tree.insert("", tk.END,
                            tags=("even" if i % 2 == 0 else "odd",),
                            values=(face["name"], face["registered_date"],
                                    face["samples"],
                                    "✓" if face["has_image"] else "✗"))

            sb = ttk.Scrollbar(frm, orient=tk.VERTICAL, command=tree.yview)
            tree.configure(yscrollcommand=sb.set)
            tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            sb.pack(side=tk.RIGHT, fill=tk.Y)
            make_sortable(tree)

            tk.Label(win, text=f"Total: {len(faces)} person(s)",
                     font=("Segoe UI", 9), bg=T["bg"], fg=T["subtext"]).pack(pady=(0, 4))

        ModernButton(win, text="Close", command=win.destroy,
                     bg=T["PRIMARY"], fg="white",
                     font=("Segoe UI", 10), width=14).pack(pady=8)

    # ═══════════════════════════════════════════
    # VIEW ATTENDANCE
    # ═══════════════════════════════════════════
    def view_attendance(self) -> None:
        records = self.system.attendance.get_today_attendance()
        T       = self.T

        win = tk.Toplevel(self.root)
        win.title("Today's Attendance")
        win.geometry("520x460")
        win.configure(bg=T["bg"])
        win.grab_set()

        hdr = tk.Frame(win, bg=T["SUCCESS"], height=48)
        hdr.pack(fill=tk.X)
        hdr.pack_propagate(False)
        tk.Label(hdr,
                 text=f"📋  Attendance — {datetime.now().strftime('%B %d, %Y')}",
                 font=("Segoe UI", 13, "bold"),
                 bg=T["SUCCESS"], fg="white").pack(expand=True)

        if not records:
            tk.Label(win, text="No attendance recorded yet today.",
                     font=("Segoe UI", 12), bg=T["bg"],
                     fg=T["subtext"]).pack(expand=True)
        else:
            frm = tk.Frame(win, bg=T["bg"])
            frm.pack(fill=tk.BOTH, expand=True, padx=16, pady=12)

            cols = ("#", "Name", "Time", "Confidence")
            tree = ttk.Treeview(frm, columns=cols, show="headings")
            for col, w, anch in zip(
                    cols, (44, 250, 100, 100),
                    ("center", "w", "center", "center")):
                tree.heading(col, text=col)
                tree.column(col, width=w, anchor=anch)

            tree.tag_configure("even", background=T["row_even"])
            tree.tag_configure("odd",  background=T["row_odd"])

            for i, (name, time_, conf) in enumerate(records, 1):
                tree.insert("", tk.END,
                            tags=("even" if i % 2 == 0 else "odd",),
                            values=(i, name, time_, f"{float(conf):.0%}"))

            sb = ttk.Scrollbar(frm, orient=tk.VERTICAL, command=tree.yview)
            tree.configure(yscrollcommand=sb.set)
            tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            sb.pack(side=tk.RIGHT, fill=tk.Y)
            make_sortable(tree)

            tk.Label(win, text=f"Total present: {len(records)}",
                     font=("Segoe UI", 10, "bold"),
                     bg=T["bg"], fg=T["text"]).pack(pady=(0, 4))

        ModernButton(win, text="Close", command=win.destroy,
                     bg=T["PRIMARY"], fg="white",
                     font=("Segoe UI", 10), width=14).pack(pady=8)

    # ═══════════════════════════════════════════
    # EXPORT ATTENDANCE
    # ═══════════════════════════════════════════
    def export_attendance(self) -> None:
        T = self.T

        dlg = tk.Toplevel(self.root)
        dlg.title("Export Attendance")
        dlg.geometry("430x250")
        dlg.configure(bg=T["bg"])
        dlg.resizable(False, False)
        dlg.grab_set()

        tk.Label(dlg, text="Export Attendance Report",
                 font=("Segoe UI", 14, "bold"),
                 bg=T["bg"], fg=T["text"]).pack(pady=(18, 10))

        frm = tk.Frame(dlg, bg=T["bg"])
        frm.pack()

        def _date_row(label_text: str, default: str, row_idx: int) -> tk.Entry:
            tk.Label(frm, text=label_text, font=("Segoe UI", 10),
                     bg=T["bg"], fg=T["subtext"],
                     width=24, anchor="e").grid(row=row_idx, column=0, padx=8, pady=6)
            ent = tk.Entry(frm, font=("Segoe UI", 10), width=13,
                           bg=T["card"], fg=T["text"],
                           insertbackground=T["text"],
                           relief=tk.FLAT, highlightthickness=1,
                           highlightbackground=T["border"])
            ent.insert(0, default)
            ent.grid(row=row_idx, column=1, padx=8, pady=6)
            return ent

        start_ent = _date_row("Start Date  (YYYY-MM-DD):",
                               (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"), 0)
        end_ent   = _date_row("End Date    (YYYY-MM-DD):",
                               datetime.now().strftime("%Y-%m-%d"), 1)

        def do_export() -> None:
            start = start_ent.get().strip()
            end   = end_ent.get().strip()
            dest  = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV", "*.csv"), ("All Files", "*.*")],
                initialfile=f"attendance_{start}_to_{end}.csv",
                parent=dlg)
            if not dest:
                return
            try:
                count = self.system.attendance.export_range(start, end, dest)
                messagebox.showinfo("✅  Exported",
                                    f"Exported {count} record(s) to:\n{dest}",
                                    parent=dlg)
                dlg.destroy()
            except ValueError as exc:
                messagebox.showerror("Date Error", str(exc), parent=dlg)
            except Exception as exc:
                messagebox.showerror("Export Failed", str(exc), parent=dlg)

        btn_row = tk.Frame(dlg, bg=T["bg"])
        btn_row.pack(pady=14)
        ModernButton(btn_row, text="Export", command=do_export,
                     bg=T["SUCCESS"], fg="white",
                     font=("Segoe UI", 10), width=12).pack(side=tk.LEFT, padx=6)
        ModernButton(btn_row, text="Cancel", command=dlg.destroy,
                     bg=T["MUTED"], fg="white",
                     font=("Segoe UI", 10), width=10).pack(side=tk.LEFT, padx=6)

    # ═══════════════════════════════════════════
    # DELETE FACE
    # ═══════════════════════════════════════════
    def delete_face(self) -> None:
        T     = self.T
        names = sorted(set(self.system.known_names))
        if not names:
            messagebox.showinfo("No Faces", "No faces registered yet.", parent=self.root)
            return

        dlg = tk.Toplevel(self.root)
        dlg.title("Delete Face")
        dlg.geometry("380x360")
        dlg.configure(bg=T["bg"])
        dlg.grab_set()

        tk.Label(dlg, text="Select a person to remove:",
                 font=("Segoe UI", 12, "bold"),
                 bg=T["bg"], fg=T["text"]).pack(pady=(14, 6))

        frm = tk.Frame(dlg, bg=T["bg"])
        frm.pack(fill=tk.BOTH, expand=True, padx=20, pady=4)

        lb = tk.Listbox(frm, font=("Segoe UI", 11),
                        bg=T["card"], fg=T["text"],
                        selectbackground=T["DANGER"],
                        selectforeground="white",
                        activestyle="none", relief=tk.FLAT,
                        highlightthickness=1,
                        highlightbackground=T["border"])
        sb = ttk.Scrollbar(frm, orient=tk.VERTICAL, command=lb.yview)
        lb.configure(yscrollcommand=sb.set)
        for n in names:
            count = self.system.known_names.count(n)
            lb.insert(tk.END, f"  {n}  ({count} encoding{'s' if count != 1 else ''})")
        lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        def confirm() -> None:
            sel = lb.curselection()
            if not sel:
                messagebox.showwarning("No Selection",
                                       "Please select a person first.", parent=dlg)
                return
            name = names[sel[0]]
            if messagebox.askyesno(
                    "Confirm Delete",
                    f"Permanently delete '{name}'?\n\nThis cannot be undone.",
                    icon="warning", parent=dlg):
                if self.system.delete_face(name):
                    messagebox.showinfo("✅  Deleted",
                                        f"'{name}' has been removed.", parent=dlg)
                    self._set_status(f"Deleted: {name}", "warn")
                    self._do_refresh_stats()
                    dlg.destroy()
                else:
                    messagebox.showerror("Error", "Delete failed.", parent=dlg)

        btn_row = tk.Frame(dlg, bg=T["bg"])
        btn_row.pack(pady=10)
        ModernButton(btn_row, text="Delete Selected", command=confirm,
                     bg=T["DANGER"], fg="white",
                     font=("Segoe UI", 10), width=16).pack(side=tk.LEFT, padx=6)
        ModernButton(btn_row, text="Cancel", command=dlg.destroy,
                     bg=T["MUTED"], fg="white",
                     font=("Segoe UI", 10), width=10).pack(side=tk.LEFT, padx=6)

    # ═══════════════════════════════════════════
    # SETTINGS
    # ═══════════════════════════════════════════
    def show_settings(self) -> None:
        T = self.T

        dlg = tk.Toplevel(self.root)
        dlg.title("Settings")
        dlg.geometry("480x390")
        dlg.configure(bg=T["bg"])
        dlg.resizable(False, False)
        dlg.grab_set()

        tk.Label(dlg, text="⚙️  System Settings",
                 font=("Segoe UI", 14, "bold"),
                 bg=T["bg"], fg=T["text"]).pack(pady=(16, 10))

        card = tk.Frame(dlg, bg=T["card"],
                        highlightbackground=T["border"], highlightthickness=1)
        card.pack(padx=20, fill=tk.BOTH, expand=True)

        SLIDERS = [
            ("Face Match Tolerance",  "tolerance",            0.30, 0.80, 0.05,
             tk.DoubleVar, "Lower = stricter.  Recommended: 0.40 – 0.50"),
            ("Confidence Threshold",  "confidence_threshold", 0.00, 1.00, 0.05,
             tk.DoubleVar, "Min confidence to mark attendance"),
            ("Frame Skip",            "frame_skip",           1,    8,    1,
             tk.IntVar,    "Process every N-th frame (higher = faster)"),
        ]

        vars_: Dict[str, tk.Variable] = {}
        for i, (label_text, attr, lo, hi, res, VClass, tip) in enumerate(SLIDERS):
            row = tk.Frame(card, bg=T["card"])
            row.grid(row=i, column=0, sticky="ew", padx=16, pady=8)
            card.columnconfigure(0, weight=1)

            top = tk.Frame(row, bg=T["card"])
            top.pack(fill=tk.X)
            lbl = tk.Label(top, text=label_text,
                           font=("Segoe UI", 10), bg=T["card"], fg=T["text"])
            lbl.pack(side=tk.LEFT)
            Tooltip(lbl, tip)

            v = VClass(value=getattr(self.system, attr))
            tk.Label(top, textvariable=v,
                     font=("Segoe UI", 10, "bold"),
                     bg=T["card"], fg=T["PRIMARY"], width=5).pack(side=tk.RIGHT)

            tk.Scale(row, from_=lo, to=hi, resolution=res,
                     orient=tk.HORIZONTAL, variable=v, showvalue=False,
                     length=400, bg=T["card"], highlightthickness=0,
                     troughcolor=T["border"], activebackground=T["PRIMARY"]
                     ).pack(fill=tk.X)
            vars_[attr] = v

        # Model selector
        model_row = tk.Frame(card, bg=T["card"])
        model_row.grid(row=len(SLIDERS), column=0, sticky="ew", padx=16, pady=(4, 12))
        tk.Label(model_row, text="Detection Model:",
                 font=("Segoe UI", 10), bg=T["card"], fg=T["text"]).pack(side=tk.LEFT)
        model_var = tk.StringVar(value=self.system.model)
        for m in ("hog", "cnn"):
            tk.Radiobutton(model_row, text=m.upper(), variable=model_var, value=m,
                           font=("Segoe UI", 10), bg=T["card"], fg=T["text"],
                           selectcolor=T["card"],
                           activebackground=T["card"]).pack(side=tk.LEFT, padx=10)
        Tooltip(model_row, "HOG = fast (CPU).  CNN = accurate (GPU recommended)")

        def apply_settings() -> None:
            self.system.tolerance            = vars_["tolerance"].get()
            self.system.confidence_threshold = vars_["confidence_threshold"].get()
            self.system.frame_skip           = vars_["frame_skip"].get()
            self.system.model                = model_var.get()
            self._do_refresh_stats()
            self._set_status("Settings updated.", "ok")
            messagebox.showinfo("✅  Saved", "Settings applied.", parent=dlg)
            dlg.destroy()

        btn_row = tk.Frame(dlg, bg=T["bg"])
        btn_row.pack(pady=12)
        ModernButton(btn_row, text="Apply", command=apply_settings,
                     bg=T["SUCCESS"], fg="white",
                     font=("Segoe UI", 10), width=12).pack(side=tk.LEFT, padx=6)
        ModernButton(btn_row, text="Cancel", command=dlg.destroy,
                     bg=T["MUTED"], fg="white",
                     font=("Segoe UI", 10), width=10).pack(side=tk.LEFT, padx=6)

    # ═══════════════════════════════════════════
    # RUN
    # ═══════════════════════════════════════════
    def run(self) -> None:
        self.root.mainloop()


# ───────────────────────────────────────────────
# MODULE-LEVEL HELPER
# ───────────────────────────────────────────────
def _safe_dot_reset(dot: tk.Label, color: str) -> None:
    """Reset refresh-dot color safely (dot may have been destroyed during rebuild)."""
    try:
        dot.config(fg=color)
    except tk.TclError:
        pass
