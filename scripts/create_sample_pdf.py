"""
Sample PDF Report Generator
===========================
Generates a multi-page sample financial report PDF in data/annual_report_2025_2026.pdf
for testing PDF ingestion and vector indexing.
"""

from pathlib import Path


def create_sample_pdf(output_path: str = "data/annual_report_2025_2026.pdf") -> Path:
    """Generates a sample PDF document using minimal pure Python PDF generation."""
    out = Path(output_path).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    text_source = Path(__file__).parent.parent / "data" / "sample_annual_report.txt"
    if not text_source.exists():
        raise FileNotFoundError(f"Source text report not found at {text_source}")

    content = text_source.read_text(encoding="utf-8")
    sections = [s.strip() for s in content.split("### ") if s.strip()]

    # Minimal pure-Python PDF structure with multi-page streams
    pages_data = []
    # Title Page
    pages_data.append("APEX GLOBAL ENTERPRISE SOLUTIONS\nANNUAL REPORT 2025-2026\n\nExecutive Overview & Strategic Disclosures")
    for sec in sections:
        pages_data.append(sec[:1500])

    # Build valid multi-page PDF with basic font encoding
    objects = []
    
    def add_object(obj_str: str) -> int:
        objects.append(obj_str)
        return len(objects)

    # Object 1: Catalog
    add_object("<< /Type /Catalog /Pages 2 0 R >>")
    # Object 2: Pages container (placeholder, will fill)
    pages_obj_idx = len(objects)
    objects.append("") 
    # Object 3: Font
    font_obj = add_object("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_refs = []
    for page_text in pages_data:
        # Format text stream
        lines = page_text.replace("\r", "").split("\n")
        stream_lines = ["BT", "/F1 11 Tf", "40 750 Td", "14 TL"]
        for line in lines:
            safe_line = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
            # Truncate line if too long for simple page width
            while len(safe_line) > 85:
                stream_lines.append(f"({safe_line[:85]}) '")
                safe_line = safe_line[85:]
            stream_lines.append(f"({safe_line}) '")
        stream_lines.append("ET")
        stream_content = "\n".join(stream_lines)

        stream_obj = add_object(
            f"<< /Length {len(stream_content.encode('latin1', errors='replace'))} >>\nstream\n{stream_content}\nendstream"
        )
        page_obj = add_object(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Resources << /Font << /F1 {font_obj} 0 R >> >> /Contents {stream_obj} 0 R >>"
        )
        page_refs.append(f"{page_obj} 0 R")

    # Update Pages object (Object 2)
    kids_str = " ".join(page_refs)
    objects[1] = f"<< /Type /Pages /Kids [{kids_str}] /Count {len(page_refs)} >>"

    # Assemble complete PDF file
    pdf_bytes = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objects, 1):
        offsets.append(len(pdf_bytes))
        pdf_bytes.extend(f"{i} 0 obj\n{obj}\nendobj\n".encode("latin1", errors="replace"))

    xref_offset = len(pdf_bytes)
    pdf_bytes.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("latin1"))
    for off in offsets:
        pdf_bytes.extend(f"{off:010d} 00000 n \n".encode("latin1"))

    trailer = f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n"
    pdf_bytes.extend(trailer.encode("latin1"))

    out.write_bytes(pdf_bytes)
    print(f"Generated sample PDF at: {out} ({len(pages_data)} pages)")
    return out


if __name__ == "__main__":
    create_sample_pdf()
