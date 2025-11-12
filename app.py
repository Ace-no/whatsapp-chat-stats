import streamlit as st
import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import emoji

# ---------------- Page setup ----------------
st.set_page_config(page_title="WhatsApp Chat Analyzer — Ultra", page_icon="💬", layout="wide")
st.title("💬 WhatsApp Chat Stats")
st.caption("Upload your exported WhatsApp chat (.txt, without media).")

uploaded_file = st.file_uploader("Upload .txt file", type="txt")

# ---------------- Helpers ----------------
def normalize(s: str) -> str:
    return (
        s.replace("\u200e", "")   # LTR mark
         .replace("\u200f", "")   # RTL mark
         .replace("\u00a0", " ")  # NBSP -> space
         .replace("–", "-")       # en dash -> hyphen
         .replace("—", "-")       # em dash -> hyphen
         .rstrip("\n")
    )

# New-message start detector (date, time, dash)
START_RE = re.compile(
    r"""^(
        \[?\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{2,4}      # date
        ,\s*\d{1,2}:\d{2}(?::\d{2})?\s*([AaPp][Mm])?  # time (+optional seconds, am/pm)
        \]?
        \s*-\s*
    )""",
    re.VERBOSE,
)

# Full parser: date, time, am/pm?, sender, message
MSG_RE = re.compile(
    r"""
    ^\[
        ?(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{2,4})          # 1: date
        ,\s*(\d{1,2}:\d{2}(?::\d{2})?)\s*([AaPp][Mm])?    # 2: time  3: am/pm (optional)
     \]?
     \s*-\s*
     ([^:]+?):\s*(.*)$                                    # 4: sender 5: message
    """,
    re.VERBOSE,
)

# System phrases 
SYSTEM_PHRASES = (
    "Messages and calls are end-to-end encrypted",
    "Your security code with",
    "added",
    "removed",
    "changed the subject",
    "created group",
    "changed to",
)

DELETED_PHRASES = (
    "This message was deleted",
    "You deleted this message",
)

def parse_dt(date_str, time_str, ampm):
    fmts = []
    if ampm:
        fmts += ["%d/%m/%y, %I:%M %p", "%d/%m/%Y, %I:%M %p",
                 "%d-%m-%y, %I:%M %p", "%d-%m-%Y, %I:%M %p",
                 "%d.%m.%y, %I:%M %p", "%d.%m.%Y, %I:%M %p",
                 "%m/%d/%y, %I:%M %p", "%m/%d/%Y, %I:%M %p"]
        fmts += ["%d/%m/%y, %I:%M:%S %p", "%d/%m/%Y, %I:%M:%S %p",
                 "%d-%m-%y, %I:%M:%S %p", "%d-%m-%Y, %I:%M:%S %p",
                 "%d.%m.%y, %I:%M:%S %p", "%d.%m.%Y, %I:%M:%S %p",
                 "%m/%d/%y, %I:%M:%S %p", "%m/%d/%Y, %I:%M:%S %p"]
    else:
        fmts += ["%d/%m/%y, %H:%M", "%d/%m/%Y, %H:%M",
                 "%d-%m-%y, %H:%M", "%d-%m-%Y, %H:%M",
                 "%d.%m.%y, %H:%M", "%d.%m.%Y, %H:%M",
                 "%m/%d/%y, %H:%M", "%m/%d/%Y, %H:%M"]
        fmts += ["%d/%m/%y, %H:%M:%S", "%d/%m/%Y, %H:%M:%S",
                 "%d-%m-%y, %H:%M:%S", "%d-%m-%Y, %H:%M:%S",
                 "%d.%m.%y, %H:%M:%S", "%d.%m.%Y, %H:%M:%S",
                 "%m/%d/%y, %H:%M:%S", "%m/%d/%Y, %H:%M:%S"]
    for fmt in fmts:
        try:
            return datetime.strptime(f"{date_str}, {time_str}", fmt)
        except ValueError:
            continue
    return None

def parse_chat(raw_lines):
    lines = [normalize(l) for l in raw_lines if l.strip()]
    chunks = []
    current = []
    for line in lines:
        if START_RE.match(line):
            if current:
                chunks.append("\n".join(current))
                current = []
            current.append(line)
        else:
            if current:
                current.append(line)
            else:
                # orphan continuation without header — ignore
                pass
    if current:
        chunks.append("\n".join(current))

    messages = []
    unknown = 0
    sys_cnt = 0

    for blob in chunks:
        first_line = blob.split("\n", 1)[0]
        m = MSG_RE.match(first_line)
        if not m:
            # likely a system line without sender
            if any(p in first_line for p in SYSTEM_PHRASES):
                sys_cnt += 1
                continue
            unknown += 1
            continue

        date_str, time_str, ampm, sender, first_msg = m.groups()
        dt = parse_dt(date_str, time_str, ampm)
        if not dt:
            unknown += 1
            continue

        if "\n" in blob:
            full_msg = first_msg + "\n" + blob.split("\n", 1)[1]
        else:
            full_msg = first_msg

        # Keep deleted messages (user requested)
        # Still drop obvious system messages by content
        if any(p.lower() in full_msg.lower() for p in SYSTEM_PHRASES):
            sys_cnt += 1
            continue

        messages.append({
            "datetime": dt,
            "sender": sender.strip(),
            "message": full_msg.strip()
        })

    return messages, unknown, sys_cnt

# ---------------- Message cleaning helper ----------------
def is_real_message(msg: str) -> bool:
    """Return True if message should count as real text content."""
    if not isinstance(msg, str):
        return False
    lowered = msg.lower()
    # phrases to ignore
    banned = [
        "messages and calls are end-to-end encrypted",
        "your security code with",
        "added",
        "removed",
        "changed the subject",
        "created group",
        "changed to",
        "this message was edited",
    ]
    # allow deleted ones
    return not any(p in lowered for p in banned)

# -------------- Emoji utils --------------

import regex 

EMOJI_PATTERN = regex.compile(r'\p{Emoji}', flags=regex.UNICODE)

def extract_emojis(text: str):
    if not text:
        return []
    # Find all emoji matches, filter out numbers and symbols
    emojis = EMOJI_PATTERN.findall(text)
    return [e for e in emojis if not regex.match(r'^[0-9#*]+$', e)]

def count_emojis(series: pd.Series):
    counter = Counter()
    for msg in series.dropna():
        for e in extract_emojis(msg):
            counter[e] += 1
    return counter


# -------------- Streaks --------------
def longest_and_current_streak(dates: pd.Series):
    if dates.empty:
        return 0, 0, None, None
    days = sorted(set(dates))
    longest = cur = 1
    best_start = best_end = days[0]
    cur_start = days[0]
    for i in range(1, len(days)):
        if days[i] == days[i-1] + timedelta(days=1):
            cur += 1
        else:
            if cur > longest:
                longest = cur
                best_start, best_end = cur_start, days[i-1]
            cur = 1
            cur_start = days[i]
    if cur > longest:
        longest = cur
        best_start, best_end = cur_start, days[-1]

    # current streak ends at the last message day and continues backwards
    today_like = days[-1]
    cur2 = 1
    for j in range(len(days)-2, -1, -1):
        if days[j] == days[j+1] - timedelta(days=1):
            cur2 += 1
        else:
            break
    return longest, cur2, best_start, best_end

# -------------- UI + Analysis --------------
if uploaded_file:
    data = uploaded_file.read().decode("utf-8-sig", errors="ignore").splitlines()
    messages, unknown_cnt, sys_cnt = parse_chat(data)

    if not messages:
        st.error("❌ Couldn't parse any user messages. If this is an unusual locale format, share 3–4 raw lines so we can add support.")
        st.stop()

    df = pd.DataFrame(messages)
    df["date"] = df["datetime"].dt.date
    df["hour"] = df["datetime"].dt.hour
    df["weekday"] = df["datetime"].dt.day_name()
    df["words"] = df["message"].fillna("").apply(lambda s: len(re.findall(r"\b\w+\b", s)))
    df["chars"] = df["message"].fillna("").str.len()
        # Mark deleted messages
    df["is_deleted"] = df["message"].str.contains("|".join(map(re.escape, DELETED_PHRASES)), case=False, na=False)

    # Mark system-type messages that shouldn't count for word/length/wordcloud analysis
    system_markers = [
        "Messages and calls are end-to-end encrypted",
        "Your security code with",
        "added",
        "removed",
        "changed the subject",
        "created group",
        "changed to",
        "This message was edited",
        "Message deleted",
    ]

    df["is_system"] = df["message"].str.contains("|".join(map(re.escape, system_markers)), case=False, na=False)

    # ------------------------------
    # 🔹 Create filtered analysis dataframe
    # Exclude pure system messages, but keep deleted ones (since user wants them visible)
    # ------------------------------
    analysis_df = df[~df["is_system"]].copy()

    # From now on, 'analysis_df' will be used for:
    #   - word counts
    #   - emoji stats
    #   - wordcloud
    #   - message length trends
    #   - activity heatmap
    # 'df' is still kept intact for deleted-message tracking, etc.

    # Sidebar filters
    st.sidebar.header("Filters")
    participants = sorted(df["sender"].unique().tolist())
    selected_users = st.sidebar.multiselect("Participants", participants, default=participants)
    date_min, date_max = df["date"].min(), df["date"].max()
    date_range = st.sidebar.date_input("Date range", value=(date_min, date_max), min_value=date_min, max_value=date_max)

    # Apply filters to both dataframes consistently
    fdf = df[
        df["sender"].isin(selected_users) &
        (df["date"] >= pd.to_datetime(str(date_range[0])).date()) &
        (df["date"] <= pd.to_datetime(str(date_range[1])).date())
    ].copy()

    analysis_df = analysis_df[
        analysis_df["sender"].isin(selected_users) &
        (analysis_df["date"] >= pd.to_datetime(str(date_range[0])).date()) &
        (analysis_df["date"] <= pd.to_datetime(str(date_range[1])).date())
    ].copy()

    # ----- Top KPIs -----
    users_counts = fdf["sender"].value_counts()
    daily_counts = fdf.groupby("date").size()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Messages", int(len(fdf)))
    c2.metric("Participants", int(users_counts.size))
    if not users_counts.empty:
        c3.metric("Most Active User", users_counts.index[0])
    else:
        c3.metric("Most Active User", "—")
    c4.metric("Days Covered", int(daily_counts.size))
    st.caption(f"Filtered {sys_cnt} system lines. Unrecognized lines: {unknown_cnt}")

    st.divider()

    # ----- Messages by user -----
    st.subheader("📊 Messages by User")
    st.bar_chart(users_counts)

    # ----- Messages over time -----
    st.subheader("📅 Messages Over Time")
    st.line_chart(daily_counts)

    most_active_day = daily_counts.idxmax() if not daily_counts.empty else None
    if most_active_day:
        st.write(f"**Most Active Day:** {most_active_day} ({int(daily_counts.max())} msgs)")
        st.write(f"**Average Messages per Day:** {daily_counts.mean():.2f}")

    st.divider()

    # ====== Deleted Messages Tracker ======
    st.subheader("🗑️ Deleted Messages")
    del_total = int(fdf["is_deleted"].sum())
    st.write(f"**Total Deleted Messages:** {del_total} ({del_total/len(fdf)*100:.2f}% of messages)" if len(fdf) else "**Total Deleted Messages:** 0")

    colA, colB = st.columns(2)
    with colA:
        st.write("**Deleted by User**")
        del_by_user = fdf[fdf["is_deleted"]].groupby("sender").size().sort_values(ascending=False)
        st.bar_chart(del_by_user)
    with colB:
        st.write("**Deleted per Day**")
        del_by_day = fdf[fdf["is_deleted"]].groupby("date").size()
        st.line_chart(del_by_day)

    st.divider()

    # ====== Emoji Stats ======
    st.subheader("😆 Emoji Usage")
    emoji_counts_total = count_emojis(fdf.loc[fdf["message"].apply(is_real_message), "message"])
    if emoji_counts_total:
        topN = 20
        top_emojis = emoji_counts_total.most_common(topN)
        edf = pd.DataFrame(top_emojis, columns=["Emoji", "Count"])
        st.dataframe(edf, use_container_width=True)

        # Per-user emoji counts (top 10 for each)
        exp = st.expander("Per-user emoji counts (top 10)", expanded=False)
        with exp:
            for user in participants:
                if user not in selected_users:
                    continue
                sub = fdf[fdf["sender"] == user]["message"]
                cnt = count_emojis(sub)
                if cnt:
                    st.markdown(f"**{user}**")
                    st.dataframe(pd.DataFrame(cnt.most_common(10), columns=["Emoji", "Count"]), use_container_width=True)
    else:
        st.info("No emojis found in the filtered range.")

    st.divider()

        # ====== Message Length Trends ======
    st.subheader("📝 Message Length Trends")

    # Filter out system messages but keep deleted ones
    filtered_msgs = fdf[fdf["message"].apply(is_real_message)].copy()

    # Per day averages (words & chars)
    daily_len = filtered_msgs.groupby("date").agg(
        avg_words=("words", "mean"),
        avg_chars=("chars", "mean")
    )

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Average Words per Day**")
        st.line_chart(daily_len["avg_words"])
    with col2:
        st.write("**Average Characters per Day**")
        st.line_chart(daily_len["avg_chars"])

    # Per user averages
    per_user = filtered_msgs.groupby("sender").agg(
        avg_words=("words", "mean"),
        avg_chars=("chars", "mean"),
        msgs=("message", "count")
    )

    st.write("**Per-User Averages**")
    st.dataframe(per_user.sort_values("msgs", ascending=False), use_container_width=True)


    # ====== Active Hours Heatmap (Hour × DayOfWeek) ======
    st.subheader("🕰️ Active Hours Heatmap")
    order_days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    heat = fdf.pivot_table(index="hour", columns="weekday", values="message", aggfunc="count", fill_value=0)
    # ensure full grid
    heat = heat.reindex(range(0,24), fill_value=0)
    heat = heat.reindex(columns=order_days, fill_value=0)

    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(heat.values, aspect="auto")
    ax.set_yticks(range(0,24))
    ax.set_yticklabels(range(0,24))
    ax.set_xticks(range(len(order_days)))
    ax.set_xticklabels(order_days, rotation=30, ha="right")
    ax.set_xlabel("Day of Week")
    ax.set_ylabel("Hour of Day")
    ax.set_title("Messages Heatmap")
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Messages", rotation=270, labelpad=15)
    st.pyplot(fig, use_container_width=True)

    st.divider()

    # ====== Chat Streaks ======
    st.subheader("🔥 Chat Streaks")
    longest, current, s_start, s_end = longest_and_current_streak(fdf["date"])
    if longest == 0:
        st.info("No streaks (no messages in the selected range).")
    else:
        colx, coly, colz = st.columns(3)
        colx.metric("Longest Streak (days)", int(longest))
        coly.metric("Current Streak (days)", int(current))
        colz.metric("Streak Window", f"{s_start} → {s_end}" if s_start and s_end else "—")

    st.divider()

    # ====== Wordcloud & Common Words ======
    st.subheader("🔠 Most Common Words")
    text = " ".join(fdf.loc[fdf["message"].apply(is_real_message), "message"].dropna()).lower()

    stop = set(STOPWORDS)
    stop.update(["media", "omitted", "image", "video"])  # keep "deleted" since you want it visible
    words = [w for w in re.findall(r"\b\w+\b", text) if w not in stop]
    if words:
        freq = Counter(words).most_common(25)
        st.dataframe(pd.DataFrame(freq, columns=["Word", "Frequency"]), use_container_width=True)
        wc = WordCloud(width=1000, height=400, background_color="white", stopwords=stop).generate(text)
        st.image(wc.to_array(), caption="Word Cloud", use_container_width=True)
    else:
        st.info("No words to display (maybe mostly emojis/media).")

    st.divider()

    # ====== Downloads ======
    st.subheader("⬇️ Downloads")
    # Export filtered dataframe
    csv = fdf.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered messages (CSV)", csv, file_name="chat_filtered.csv", mime="text/csv")
    from fpdf import FPDF
    import io

    class ChatPDF(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 16)
            self.cell(0, 10, "💬 WhatsApp Chat Report", ln=True, align="C")
            self.ln(5)

        def chapter_title(self, title):
            self.set_font("Helvetica", "B", 14)
            self.cell(0, 8, title, ln=True)
            self.ln(3)

        def chapter_body(self, text):
            self.set_font("Helvetica", "", 11)
            self.multi_cell(0, 8, text)
            self.ln(4)

    def generate_pdf():
        pdf = ChatPDF()
        pdf.add_page()

        # Summary
        pdf.chapter_title("📋 Summary")
        summary_text = (
            f"Total Messages: {len(fdf)}\n"
            f"Participants: {', '.join(selected_users)}\n"
            f"Deleted Messages: {int(fdf['is_deleted'].sum())}\n"
            f"Most Active User: {users_counts.index[0] if not users_counts.empty else '—'}\n"
            f"Most Active Day: {most_active_day}\n"
            f"Longest Streak: {longest} days\n"
            f"Current Streak: {current} days\n"
            f"Date Range: {date_range[0]} → {date_range[1]}"
        )
        pdf.chapter_body(summary_text)

        # Top 5 users
        pdf.chapter_title("👥 Top Participants")
        top_users_text = "\n".join([f"{u}: {c} msgs" for u, c in users_counts.head(5).items()])
        pdf.chapter_body(top_users_text or "No data")

        # Deleted messages
        pdf.chapter_title("🗑️ Deleted Messages")
        pdf.chapter_body(f"Total deleted: {del_total}\n\nDeleted per user:\n" + del_by_user.to_string())

        # Emoji stats
        if emoji_counts_total:
            pdf.chapter_title("😆 Top Emojis")
            top_emojis_text = " ".join([e for e, _ in emoji_counts_total.most_common(30)])
            pdf.chapter_body(top_emojis_text)
        else:
            pdf.chapter_title("😆 Top Emojis")
            pdf.chapter_body("No emojis found.")

        # Common words
        if words:
            pdf.chapter_title("🔠 Top Words")
            word_text = ", ".join([w for w, _ in freq[:30]])
            pdf.chapter_body(word_text)
        else:
            pdf.chapter_title("🔠 Top Words")
            pdf.chapter_body("No word data available.")

        # Footer
        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(0, 10, "Generated by WhatsApp Chat Analyzer — Ultra", 0, 0, "C")

        buffer = io.BytesIO()
        pdf.output(buffer)
        buffer.seek(0)
        return buffer

    pdf_buffer = generate_pdf()
    st.download_button(
        label="📄 Download PDF Report",
        data=pdf_buffer,
        file_name="chat_report.pdf",
        mime="application/pdf"
    )

    st.caption("⚠️ Uploaded files & generated reports are not stored — everything is in-memory and erased after your session ends.")

else:
    st.info("👆 Upload a WhatsApp chat export (.txt) to begin.")
