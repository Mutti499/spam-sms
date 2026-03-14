"""
iMessage Extractor GUI
- Select your chat.db (defaults to ~/Library/Messages/chat.db)
- Pick conversations to export
- Exports to sms.json
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import sqlite3
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone, timedelta

# Apple's cocoa epoch: 2001-01-01 00:00:00 UTC
APPLE_EPOCH = datetime(2001, 1, 1, tzinfo=timezone.utc)


def apple_ts_to_iso(ts):
    """Convert Apple nanosecond timestamp to ISO string."""
    if not ts:
        return None
    try:
        seconds = ts / 1_000_000_000
        dt = APPLE_EPOCH + timedelta(seconds=seconds)
        return dt.isoformat()
    except (OverflowError, OSError):
        return None


def extract_text_from_attributed_body(blob):
    """Try to pull plain text from the attributedBody blob."""
    if not blob:
        return None
    try:
        # The text is embedded in the blob after "NSString" marker
        text = blob.split(b"NSString")[1]
        text = text[5:]  # skip header bytes
        # Find the end (usually a NUL or control sequence)
        end = text.find(b"\x86")
        if end == -1:
            end = text.find(b"\x00")
        if end > 0:
            text = text[:end]
        return text.decode("utf-8", errors="replace").strip()
    except (IndexError, ValueError):
        return None


class IMsgExtractor:
    def __init__(self, root):
        self.root = root
        self.root.title("iMessage Extractor")
        self.root.geometry("750x600")
        self.root.minsize(600, 400)

        self.conn = None
        self.chats = []  # list of (chat_id, display, msg_count, last_date)

        self._build_ui()

    def _build_ui(self):
        # --- Top: DB picker ---
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="chat.db path:").pack(side="left")
        self.db_var = tk.StringVar(
            value=os.path.expanduser("~/Library/Messages/chat.db")
        )
        ttk.Entry(top, textvariable=self.db_var, width=50).pack(
            side="left", padx=5, fill="x", expand=True
        )
        ttk.Button(top, text="Browse", command=self._browse_db).pack(side="left")
        ttk.Button(top, text="Load", command=self._load_db).pack(side="left", padx=5)

        # --- Middle: chat list with checkboxes ---
        mid = ttk.Frame(self.root, padding=(10, 0, 10, 0))
        mid.pack(fill="both", expand=True)

        cols = ("select", "chat", "messages", "last_message")
        self.tree = ttk.Treeview(mid, columns=cols, show="headings", height=18)
        self.tree.heading("select", text="✓")
        self.tree.heading("chat", text="Chat")
        self.tree.heading("messages", text="Messages")
        self.tree.heading("last_message", text="Last Message")

        self.tree.column("select", width=40, anchor="center", stretch=False)
        self.tree.column("chat", width=350)
        self.tree.column("messages", width=80, anchor="center")
        self.tree.column("last_message", width=160, anchor="center")

        scroll = ttk.Scrollbar(mid, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)

        self.tree.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

        # Row highlight tags
        self.tree.tag_configure("selected", background="#c8e6c9", foreground="#1b5e20")
        self.tree.tag_configure("unselected", background="", foreground="")

        # Toggle selection on click
        self.selected_chats = set()
        self.tree.bind("<ButtonRelease-1>", self._toggle_select)

        # --- Bottom: controls ---
        bot = ttk.Frame(self.root, padding=10)
        bot.pack(fill="x")

        ttk.Button(bot, text="Select All", command=self._select_all).pack(side="left")
        ttk.Button(bot, text="Deselect All", command=self._deselect_all).pack(
            side="left", padx=5
        )

        ttk.Button(bot, text="Export Selected", command=self._export).pack(side="right")

        self.status_var = tk.StringVar(value="Load a chat.db to begin.")
        ttk.Label(bot, textvariable=self.status_var).pack(side="right", padx=15)

    def _browse_db(self):
        path = filedialog.askopenfilename(
            title="Select chat.db",
            filetypes=[("SQLite DB", "*.db"), ("All", "*.*")],
            initialdir=os.path.expanduser("~/Library/Messages"),
        )
        if path:
            self.db_var.set(path)

    def _load_db(self):
        db_path = self.db_var.get()
        if not os.path.isfile(db_path):
            messagebox.showerror("Error", f"File not found:\n{db_path}")
            return

        # Copy to a temp file so we don't lock the live DB
        self.tmp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        shutil.copy2(db_path, self.tmp_db.name)
        # Also copy WAL/SHM if present and accessible
        for ext in ("-wal", "-shm"):
            src = db_path + ext
            if os.path.isfile(src):
                try:
                    shutil.copy2(src, self.tmp_db.name + ext)
                except PermissionError:
                    pass  # OK — DB will still work without WAL/SHM

        try:
            self.conn = sqlite3.connect(self.tmp_db.name)
            self.conn.row_factory = sqlite3.Row
            self._populate_chat_list()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to open database:\n{e}")

    def _populate_chat_list(self):
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT
                c.ROWID as chat_id,
                c.chat_identifier,
                c.display_name,
                c.service_name,
                COUNT(cmj.message_id) as msg_count,
                MAX(m.date) as last_date
            FROM chat c
            LEFT JOIN chat_message_join cmj ON cmj.chat_id = c.ROWID
            LEFT JOIN message m ON m.ROWID = cmj.message_id
            GROUP BY c.ROWID
            HAVING msg_count > 0
            ORDER BY last_date DESC
        """
        )

        self.tree.delete(*self.tree.get_children())
        self.chats.clear()
        self.selected_chats.clear()

        for row in cur.fetchall():
            chat_id = row["chat_id"]
            display = row["display_name"] or row["chat_identifier"]
            service = row["service_name"] or ""
            if service:
                display = f"{display}  ({service})"
            msg_count = row["msg_count"]
            last_iso = apple_ts_to_iso(row["last_date"]) or ""
            # Shorten the date for display
            last_short = last_iso[:19].replace("T", " ") if last_iso else ""

            self.chats.append(
                (chat_id, display, msg_count, row["chat_identifier"])
            )
            self.tree.insert(
                "",
                "end",
                iid=str(chat_id),
                values=("☐", display, msg_count, last_short),
            )

        self.status_var.set(f"Loaded {len(self.chats)} chats.")

    def _toggle_select(self, event):
        item = self.tree.identify_row(event.y)
        if not item:
            return
        chat_id = int(item)
        if chat_id in self.selected_chats:
            self.selected_chats.discard(chat_id)
            mark = "☐"
            tag = "unselected"
        else:
            self.selected_chats.add(chat_id)
            mark = "☑"
            tag = "selected"
        vals = list(self.tree.item(item, "values"))
        vals[0] = mark
        self.tree.item(item, values=vals, tags=(tag,))
        self.status_var.set(f"{len(self.selected_chats)} chat(s) selected.")

    def _select_all(self):
        self.selected_chats.clear()
        for chat_id, display, cnt, ident in self.chats:
            self.selected_chats.add(chat_id)
            vals = list(self.tree.item(str(chat_id), "values"))
            vals[0] = "☑"
            self.tree.item(str(chat_id), values=vals, tags=("selected",))
        self.status_var.set(f"{len(self.selected_chats)} chat(s) selected.")

    def _deselect_all(self):
        self.selected_chats.clear()
        for chat_id, display, cnt, ident in self.chats:
            vals = list(self.tree.item(str(chat_id), "values"))
            vals[0] = "☐"
            self.tree.item(str(chat_id), values=vals, tags=("unselected",))
        self.status_var.set("0 chat(s) selected.")

    def _export(self):
        if not self.selected_chats:
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

        self.status_var.set("Exporting...")
        self.root.update_idletasks()

        cur = self.conn.cursor()
        all_messages = []

        for chat_id in self.selected_chats:
            # Get the chat identifier for labeling
            chat_info = next(
                (c for c in self.chats if c[0] == chat_id), None
            )
            chat_label = chat_info[3] if chat_info else str(chat_id)

            cur.execute(
                """
                SELECT
                    m.ROWID,
                    m.text,
                    m.attributedBody,
                    m.handle_id,
                    m.date,
                    m.is_from_me,
                    m.service,
                    h.id as sender_id
                FROM message m
                JOIN chat_message_join cmj ON cmj.message_id = m.ROWID
                LEFT JOIN handle h ON h.ROWID = m.handle_id
                WHERE cmj.chat_id = ?
                ORDER BY m.date ASC
            """,
                (chat_id,),
            )

            for msg in cur.fetchall():
                text = msg["text"]
                if not text:
                    text = extract_text_from_attributed_body(msg["attributedBody"])
                if not text:
                    continue  # skip empty / attachment-only messages

                all_messages.append(
                    {
                        "chat": chat_label,
                        "sender": "me" if msg["is_from_me"] else (msg["sender_id"] or "unknown"),
                        "text": text,
                        "date": apple_ts_to_iso(msg["date"]),
                        "is_from_me": bool(msg["is_from_me"]),
                        "service": msg["service"],
                    }
                )

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_messages, f, ensure_ascii=False, indent=2)

        self.status_var.set(f"Exported {len(all_messages)} messages.")
        messagebox.showinfo(
            "Done",
            f"Exported {len(all_messages)} messages from "
            f"{len(self.selected_chats)} chat(s) to:\n{out_path}",
        )


if __name__ == "__main__":
    root = tk.Tk()
    app = IMsgExtractor(root)
    root.mainloop()
