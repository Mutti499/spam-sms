"""
Android SMS Converter
Converts XML exports from "SMS Backup & Restore" app to the same JSON format
used by extract_imessages.py.

Usage:
  python3 convert_android_sms.py                 # opens GUI file picker
  python3 convert_android_sms.py backup.xml      # direct conversion
"""

import sys
import os
import json
import xml.etree.ElementTree as ET
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from datetime import datetime, timezone


def parse_sms_backup(xml_path):
    """Parse SMS Backup & Restore XML and return list of (address, sms_list) grouped by contact."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    chats = {}  # address -> list of message dicts

    # Handle both <sms> elements under <smses> and <mms> elements
    for sms in root.iter("sms"):
        address = sms.get("address", "unknown")
        body = sms.get("body", "")
        if not body or not body.strip():
            continue

        # type: 1=received, 2=sent
        msg_type = sms.get("type", "1")
        is_from_me = msg_type == "2"

        # date is milliseconds since Unix epoch
        date_ms = sms.get("date", "0")
        try:
            dt = datetime.fromtimestamp(int(date_ms) / 1000, tz=timezone.utc)
            date_iso = dt.isoformat()
        except (ValueError, OSError):
            date_iso = None

        contact_name = sms.get("contact_name", address)

        msg = {
            "chat": address,
            "sender": "me" if is_from_me else address,
            "text": body,
            "date": date_iso,
            "is_from_me": is_from_me,
            "service": "SMS",
            "contact_name": contact_name,
        }

        chats.setdefault(address, []).append(msg)

    return chats


class AndroidSMSConverter:
    def __init__(self, root):
        self.root = root
        self.root.title("Android SMS Converter")
        self.root.geometry("750x600")
        self.root.minsize(600, 400)

        self.chats = {}  # address -> [messages]

        self._build_ui()

    def _build_ui(self):
        # --- Top: XML file picker ---
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="SMS Backup XML:").pack(side="left")
        self.xml_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.xml_var, width=50).pack(
            side="left", padx=5, fill="x", expand=True
        )
        ttk.Button(top, text="Browse", command=self._browse_xml).pack(side="left")
        ttk.Button(top, text="Load", command=self._load_xml).pack(side="left", padx=5)

        # --- Info label ---
        info = ttk.Frame(self.root, padding=(10, 0, 10, 5))
        info.pack(fill="x")
        ttk.Label(
            info,
            text='Tell your Android friends: install "SMS Backup & Restore" from Play Store, '
            "export to XML, and send you the file.",
            wraplength=700,
            foreground="gray",
        ).pack(anchor="w")

        # --- Middle: chat list ---
        mid = ttk.Frame(self.root, padding=(10, 0, 10, 0))
        mid.pack(fill="both", expand=True)

        cols = ("select", "contact", "number", "messages", "last_message")
        self.tree = ttk.Treeview(mid, columns=cols, show="headings", height=18)
        self.tree.heading("select", text="✓")
        self.tree.heading("contact", text="Contact")
        self.tree.heading("number", text="Number")
        self.tree.heading("messages", text="Messages")
        self.tree.heading("last_message", text="Last Message")

        self.tree.column("select", width=40, anchor="center", stretch=False)
        self.tree.column("contact", width=200)
        self.tree.column("number", width=150)
        self.tree.column("messages", width=80, anchor="center")
        self.tree.column("last_message", width=160, anchor="center")

        scroll = ttk.Scrollbar(mid, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)

        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        self.selected = set()
        self.tree.bind("<ButtonRelease-1>", self._toggle_select)

        # --- Bottom: controls ---
        bot = ttk.Frame(self.root, padding=10)
        bot.pack(fill="x")

        ttk.Button(bot, text="Select All", command=self._select_all).pack(side="left")
        ttk.Button(bot, text="Deselect All", command=self._deselect_all).pack(
            side="left", padx=5
        )
        ttk.Button(bot, text="Export Selected", command=self._export).pack(side="right")

        self.status_var = tk.StringVar(value="Load an SMS Backup XML to begin.")
        ttk.Label(bot, textvariable=self.status_var).pack(side="right", padx=15)

    def _browse_xml(self):
        path = filedialog.askopenfilename(
            title="Select SMS Backup XML",
            filetypes=[("XML files", "*.xml"), ("All", "*.*")],
        )
        if path:
            self.xml_var.set(path)

    def _load_xml(self):
        xml_path = self.xml_var.get()
        if not os.path.isfile(xml_path):
            messagebox.showerror("Error", f"File not found:\n{xml_path}")
            return

        try:
            self.chats = parse_sms_backup(xml_path)
        except ET.ParseError as e:
            messagebox.showerror("Error", f"Failed to parse XML:\n{e}")
            return

        self._populate_list()

    def _populate_list(self):
        self.tree.delete(*self.tree.get_children())
        self.selected.clear()
        self.chat_keys = []

        # Sort by message count descending
        sorted_chats = sorted(self.chats.items(), key=lambda x: len(x[1]), reverse=True)

        for address, messages in sorted_chats:
            contact_name = messages[0].get("contact_name", address)
            msg_count = len(messages)

            # Find last message date
            dates = [m["date"] for m in messages if m["date"]]
            last_date = max(dates) if dates else ""
            last_short = last_date[:19].replace("T", " ") if last_date else ""

            self.chat_keys.append(address)
            self.tree.insert(
                "",
                "end",
                iid=address,
                values=("☐", contact_name, address, msg_count, last_short),
            )

        self.status_var.set(f"Loaded {len(self.chats)} conversations.")

    def _toggle_select(self, event):
        item = self.tree.identify_row(event.y)
        if not item:
            return
        if item in self.selected:
            self.selected.discard(item)
            mark = "☐"
        else:
            self.selected.add(item)
            mark = "☑"
        vals = list(self.tree.item(item, "values"))
        vals[0] = mark
        self.tree.item(item, values=vals)
        self.status_var.set(f"{len(self.selected)} chat(s) selected.")

    def _select_all(self):
        self.selected = set(self.chat_keys)
        for key in self.chat_keys:
            vals = list(self.tree.item(key, "values"))
            vals[0] = "☑"
            self.tree.item(key, values=vals)
        self.status_var.set(f"{len(self.selected)} chat(s) selected.")

    def _deselect_all(self):
        self.selected.clear()
        for key in self.chat_keys:
            vals = list(self.tree.item(key, "values"))
            vals[0] = "☐"
            self.tree.item(key, values=vals)
        self.status_var.set("0 chat(s) selected.")

    def _export(self):
        if not self.selected:
            messagebox.showwarning("Warning", "No chats selected.")
            return

        out_path = filedialog.asksaveasfilename(
            title="Save exported messages",
            defaultextension=".json",
            filetypes=[("JSON", "*.json"), ("All", "*.*")],
            initialfile="sms.json",
            initialdir=os.path.expanduser("~/Desktop"),
        )
        if not out_path:
            return

        all_messages = []
        for address in self.selected:
            for msg in self.chats[address]:
                # Drop contact_name from output to match iMessage format
                out = {
                    "chat": msg["chat"],
                    "sender": msg["sender"],
                    "text": msg["text"],
                    "date": msg["date"],
                    "is_from_me": msg["is_from_me"],
                    "service": msg["service"],
                }
                all_messages.append(out)

        # Sort by date
        all_messages.sort(key=lambda m: m["date"] or "")

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_messages, f, ensure_ascii=False, indent=2)

        self.status_var.set(f"Exported {len(all_messages)} messages.")
        messagebox.showinfo(
            "Done",
            f"Exported {len(all_messages)} messages from "
            f"{len(self.selected)} chat(s) to:\n{out_path}",
        )


def cli_convert(xml_path):
    """Quick CLI conversion without GUI."""
    chats = parse_sms_backup(xml_path)
    all_messages = []
    for address, messages in chats.items():
        for msg in messages:
            all_messages.append({
                "chat": msg["chat"],
                "sender": msg["sender"],
                "text": msg["text"],
                "date": msg["date"],
                "is_from_me": msg["is_from_me"],
                "service": msg["service"],
            })
    all_messages.sort(key=lambda m: m["date"] or "")

    out_path = os.path.splitext(xml_path)[0] + ".json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_messages, f, ensure_ascii=False, indent=2)
    print(f"Exported {len(all_messages)} messages to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        cli_convert(sys.argv[1])
    else:
        root = tk.Tk()
        app = AndroidSMSConverter(root)
        root.mainloop()
