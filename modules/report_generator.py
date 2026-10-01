"""
modules/report_generator.py
===========================
PDF and CSV Report Generation Service for Meeting Intelligence Platform

Generates professional executive PDF reports and structured CSV exports
from meeting intelligence data retrieved from backend services.
"""

import io
import csv
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

logger = logging.getLogger(__name__)


def generate_meeting_pdf_report(meeting_data: Dict[str, Any]) -> bytes:
    """
    Generate a professional executive PDF report for a selected meeting.

    Parameters
    ----------
    meeting_data : Dict[str, Any]
        Complete meeting details dictionary returned by get_meeting().

    Returns
    -------
    bytes
        Raw PDF document bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        "DocSubTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        textColor=colors.HexColor("#0284c7"),
        spaceAfter=15
    )

    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=14,
        spaceAfter=6
    )

    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#334155"),
        spaceAfter=6
    )

    bullet_style = ParagraphStyle(
        "BulletCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor("#334155"),
        leftIndent=12,
        spaceAfter=4
    )

    table_header_style = ParagraphStyle(
        "TableHeader",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=11,
        textColor=colors.whitesmoke
    )

    table_body_style = ParagraphStyle(
        "TableBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor("#1e293b")
    )

    elements = []

    # 1. Header Banner
    elements.append(Paragraph("AI CAREER INTELLIGENCE PLATFORM", subtitle_style))
    meeting_title = meeting_data.get("title") or "Untitled Meeting"
    elements.append(Paragraph(meeting_title, title_style))

    meeting_id = meeting_data.get("meeting_id") or meeting_data.get("id") or "N/A"
    created_at = str(meeting_data.get("created_at") or "N/A")[:19]
    status = str(meeting_data.get("status") or "completed").upper()
    word_count = meeting_data.get("metadata", {}).get("word_count") or meeting_data.get("word_count") or 0

    meta_text = f"<b>Meeting ID:</b> {meeting_id} &nbsp;|&nbsp; <b>Date:</b> {created_at} &nbsp;|&nbsp; <b>Status:</b> {status} &nbsp;|&nbsp; <b>Word Count:</b> {word_count}"
    elements.append(Paragraph(meta_text, body_style))
    elements.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#cbd5e1"), spaceAfter=12))

    # 2. Executive Summary & Key Points
    elements.append(Paragraph("Executive Summary", section_heading))
    summary = meeting_data.get("summary") or "No executive summary recorded for this meeting."
    elements.append(Paragraph(summary, body_style))

    key_points = meeting_data.get("key_points") or []
    if key_points:
        elements.append(Paragraph("Key Discussion Points", section_heading))
        for kp in key_points:
            elements.append(Paragraph(f"• {kp}", bullet_style))

    # 3. Key Decisions Reached
    elements.append(Paragraph("Key Decisions Reached", section_heading))
    decisions = meeting_data.get("decisions") or []
    if decisions:
        for dec in decisions:
            elements.append(Paragraph(f"✔ <b>{dec}</b>", bullet_style))
    else:
        elements.append(Paragraph("<i>No explicit decisions recorded for this meeting.</i>", body_style))

    # 4. Action Items Table
    elements.append(Paragraph("Extracted Action Items", section_heading))
    action_items = meeting_data.get("action_items") or []
    if action_items:
        t_data = [
            [
                Paragraph("Task", table_header_style),
                Paragraph("Assigned To", table_header_style),
                Paragraph("Deadline", table_header_style),
                Paragraph("Priority", table_header_style),
                Paragraph("Status", table_header_style)
            ]
        ]
        
        for item in action_items:
            if isinstance(item, dict):
                task = item.get("task") or "Unspecified task"
                assigned = item.get("assigned_to") or "Unassigned"
                deadline = item.get("deadline") or "Unspecified"
                priority = item.get("priority") or "Unknown"
                status_item = item.get("status") or "Pending"

                t_data.append([
                    Paragraph(task, table_body_style),
                    Paragraph(assigned, table_body_style),
                    Paragraph(deadline, table_body_style),
                    Paragraph(priority, table_body_style),
                    Paragraph(status_item, table_body_style)
                ])

        col_widths = [200, 95, 85, 70, 70]
        act_table = Table(t_data, colWidths=col_widths)
        act_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#cbd5e1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        elements.append(act_table)
    else:
        elements.append(Paragraph("<i>No action items recorded for this meeting.</i>", body_style))

    # 5. Participants & Responsibilities
    elements.append(Paragraph("Participant Responsibilities", section_heading))
    participants = meeting_data.get("participants") or []
    if participants:
        for p in participants:
            if isinstance(p, dict):
                p_name = p.get("name") or "Unknown Participant"
                resps = p.get("responsibilities") or []
                resp_text = ", ".join(resps) if resps else "No specific responsibilities assigned"
                elements.append(Paragraph(f"• <b>{p_name}:</b> {resp_text}", bullet_style))
            elif isinstance(p, str):
                elements.append(Paragraph(f"• <b>{p}</b>", bullet_style))
    else:
        elements.append(Paragraph("<i>No participants recorded for this meeting.</i>", body_style))

    # Build Document
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()


def generate_meeting_csv_report(meeting_data: Dict[str, Any]) -> str:
    """
    Generate a structured CSV report for a selected meeting.

    Parameters
    ----------
    meeting_data : Dict[str, Any]
        Complete meeting details dictionary returned by get_meeting().

    Returns
    -------
    str
        Structured CSV string content suitable for spreadsheet applications.
    """
    output = io.StringIO()
    writer = csv.writer(output)

    meeting_id = meeting_data.get("meeting_id") or meeting_data.get("id") or "N/A"
    title = meeting_data.get("title") or "Untitled Meeting"
    created_at = str(meeting_data.get("created_at") or "N/A")[:19]
    status = meeting_data.get("status") or "completed"
    summary = meeting_data.get("summary") or ""
    word_count = meeting_data.get("metadata", {}).get("word_count") or meeting_data.get("word_count") or 0

    action_items = meeting_data.get("action_items") or []
    decisions = meeting_data.get("decisions") or []
    participants = meeting_data.get("participants") or []
    key_points = meeting_data.get("key_points") or []

    # 1. Meeting Overview Metadata
    writer.writerow(["=== MEETING METADATA ==="])
    writer.writerow(["Meeting ID", meeting_id])
    writer.writerow(["Title", title])
    writer.writerow(["Date/Time", created_at])
    writer.writerow(["Status", status])
    writer.writerow(["Word Count", word_count])
    writer.writerow(["Total Participants", len(participants)])
    writer.writerow(["Total Action Items", len(action_items)])
    writer.writerow(["Total Decisions", len(decisions)])
    writer.writerow([])

    # 2. Executive Summary & Key Points
    writer.writerow(["=== EXECUTIVE SUMMARY ==="])
    writer.writerow(["Summary", summary])
    writer.writerow([])

    if key_points:
        writer.writerow(["=== KEY DISCUSSION POINTS ==="])
        for idx, kp in enumerate(key_points, 1):
            writer.writerow([f"Point {idx}", kp])
        writer.writerow([])

    # 3. Decisions Reached
    writer.writerow(["=== DECISIONS REACHED ==="])
    if decisions:
        for idx, dec in enumerate(decisions, 1):
            writer.writerow([f"Decision {idx}", dec])
    else:
        writer.writerow(["Decisions", "No decisions recorded."])
    writer.writerow([])

    # 4. Action Items Table
    writer.writerow(["=== ACTION ITEMS TABLE ==="])
    writer.writerow(["Task", "Assigned To", "Deadline", "Priority", "Status"])
    if action_items:
        for item in action_items:
            if isinstance(item, dict):
                writer.writerow([
                    item.get("task", ""),
                    item.get("assigned_to") or "Unassigned",
                    item.get("deadline") or "Unspecified",
                    item.get("priority", "Unknown"),
                    item.get("status", "Pending")
                ])
    else:
        writer.writerow(["No action items recorded."])
    writer.writerow([])

    # 5. Participants & Responsibilities
    writer.writerow(["=== PARTICIPANTS & RESPONSIBILITIES ==="])
    writer.writerow(["Participant Name", "Responsibilities"])
    if participants:
        for p in participants:
            if isinstance(p, dict):
                p_name = p.get("name", "Unknown")
                resps = "; ".join(p.get("responsibilities", []))
                writer.writerow([p_name, resps])
            elif isinstance(p, str):
                writer.writerow([p, "N/A"])
    else:
        writer.writerow(["No participants recorded."])

    return output.getvalue()
