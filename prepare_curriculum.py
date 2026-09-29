"""Convert the approved TEE66 PDF into readable, page-cited Markdown.

This is a one-time data-preparation tool. Install ``pypdfium2`` to regenerate
the Markdown; the deployed chatbot reads only the generated Markdown.
"""

import argparse
import hashlib
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unicodedata

import pypdfium2 as pdfium


DEFAULT_OUTPUT = Path(__file__).with_name("TEE66_Curriculum.md")
MAX_PART_CHARS = 2200
PUA_TRANSLATION = str.maketrans({
    "\uf701": "ิ", "\uf702": "ี", "\uf703": "ึ", "\uf704": "ื",
    "\uf705": "่", "\uf706": "้", "\uf709": "์", "\uf70a": "่",
    "\uf70b": "้", "\uf70c": "๊", "\uf70e": "์", "\uf710": "ั",
    "\uf712": "็", "\uf714": "้", "\uf098": "●", "\uf099": "○",
    "\uf0fc": "✓", "\uf0a1": "○",
})
POWER_BRANCH = "วิศวกรรมระบบไฟฟ้ากำลังและระบบควบคุม"
ELECTRONICS_BRANCH = "วิศวกรรมอิเล็กทรอนิกส์และโทรคมนาคม"


def normalize_line(line):
    line = unicodedata.normalize("NFC", line.translate(PUA_TRANSLATION))
    line = line.replace("\u0e4d\u0e32", "ำ")
    line = line.replace("\u00a0", " ").replace("\u200b", "")
    return re.sub(r"[ \t]+", " ", line).strip()


def page_lines(page, pdf_page):
    raw = page.get_textpage().get_text_bounded()
    raw = raw.replace("\r\r\n", "\n").replace("\r\n", "\n").replace("\r", "\n")
    lines = [normalize_line(line) for line in raw.splitlines()]
    # Printed page numbers and the running header do not belong to the content.
    while lines and (not lines[0] or lines[0] == "มคอ.2"
                     or lines[0] == str(pdf_page - 4)):
        lines.pop(0)
    while lines and (not lines[-1] or lines[-1] == str(pdf_page - 4)):
        lines.pop()
    return lines


def ocr_page_lines(page, pdf_page, tesseract, tessdata_dir):
    if not tesseract:
        raise RuntimeError(f"Page {pdf_page} is image-only; Thai OCR is required")
    with tempfile.TemporaryDirectory() as temporary:
        image_path = Path(temporary) / "page.png"
        page.render(scale=3).to_pil().save(image_path)
        command = [str(tesseract), str(image_path), "stdout"]
        if tessdata_dir:
            command += ["--tessdata-dir", str(tessdata_dir)]
        command += ["-l", "tha", "--psm", "3"]
        result = subprocess.run(command, capture_output=True, text=True,
                                encoding="utf-8", check=True)
    lines = [normalize_line(line) for line in result.stdout.splitlines()]
    while lines and (not lines[-1] or lines[-1] == str(pdf_page - 4)):
        lines.pop()
    if not any(lines):
        raise ValueError(f"Thai OCR returned no text for page {pdf_page}")
    return lines


def page_status(pdf_page):
    if pdf_page <= 155:
        return "เนื้อหาหลักสูตร พ.ศ. 2566"
    if pdf_page <= 200:
        return "ภาคผนวกของหลักสูตร พ.ศ. 2566"
    return "ประวัติการปรับปรุงและตารางเปรียบเทียบหลักสูตรเดิม"


def page_topic(pdf_page, lines, previous_topic):
    if 26 <= pdf_page <= 43:
        branch = POWER_BRANCH if pdf_page <= 34 else ELECTRONICS_BRANCH
        year_term = next((line for line in lines if re.search(
            r"ปีที่\s*\d+\s*ภาคการศึกษาที่\s*\d+", line)), "")
        return f"แผนการศึกษา - แขนงวิชา{branch} - {year_term}".strip(" -")
    heading = next((line for line in lines[:5]
                    if re.match(r"^(?:หมวดที่\s*\d+|\d{1,2}(?:\.\d{1,2}){0,4}\.?\s+)", line)), None)
    if heading and not re.match(r"^\d{9}", heading):
        return heading[:110]
    if previous_topic:
        return previous_topic
    return (next((line for line in lines if line), "ข้อมูลหลักสูตร")[:110])


def split_page(lines):
    parts = []
    current = []
    size = 0
    for line in lines:
        if current and size + len(line) + 1 > MAX_PART_CHARS:
            parts.append(current)
            current = []
            size = 0
        current.append(line)
        size += len(line) + 1
    if current:
        parts.append(current)
    return parts


def markdown_for_pdf(source, tesseract=None, tessdata_dir=None):
    source = Path(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    pdf = pdfium.PdfDocument(str(source))
    if len(pdf) != 230:
        raise ValueError(f"Expected the reviewed 230-page TEE66 PDF, got {len(pdf)} pages")
    output = [
        "# หลักสูตรวิศวกรรมศาสตรบัณฑิต สาขาวิชาวิศวกรรมไฟฟ้าและการศึกษา",
        "",
        "- **ฉบับ:** หลักสูตรปรับปรุง พ.ศ. 2566 (5 ปี)",
        "- **ต้นฉบับ:** TEE66.pdf",
        f"- **SHA-256 ของ PDF:** {digest}",
        "- **ขอบเขต:** เนื้อหาหลักสูตร ภาคผนวก และประวัติการปรับปรุงจากเล่มต้นฉบับ",
        "- **ข้อควรระวัง:** ตารางเปรียบเทียบหลักสูตร พ.ศ. 2561 ไม่ใช่เกณฑ์ของหลักสูตร พ.ศ. 2566",
        "",
    ]
    previous_topic = ""
    pages = 0
    chunks = 0
    ocr_pages = []
    for index in range(4, len(pdf)):
        pdf_page = index + 1
        lines = page_lines(pdf[index], pdf_page)
        if not any(lines):
            lines = ocr_page_lines(pdf[index], pdf_page, tesseract, tessdata_dir)
            ocr_pages.append(pdf_page)
        unknown = {char for line in lines for char in line if 0xe000 <= ord(char) <= 0xf8ff}
        if unknown:
            raise ValueError(f"Unmapped PDF glyphs on page {pdf_page}: {sorted(map(hex, map(ord, unknown)))}")
        topic = page_topic(pdf_page, lines, previous_topic)
        previous_topic = topic
        status = page_status(pdf_page)
        pages += 1
        for part_number, part in enumerate(split_page(lines), 1):
            chunks += 1
            output.extend([
                f"## หน้า PDF {pdf_page:03d} ส่วน {part_number:02d} — {topic}",
                "",
                f"- **หน้าในเล่ม:** {pdf_page - 4}",
                f"- **สถานะข้อมูล:** {status}",
                "- **ปีหลักสูตร:** 2566",
                ("- **วิธีดึงข้อความ:** OCR จากภาพสแกน; ตรวจต้นฉบับก่อนใช้ชื่อบุคคลหรือตัวเลข"
                 if pdf_page in ocr_pages else "- **วิธีดึงข้อความ:** ข้อความใน PDF"),
                "",
                "### เนื้อหาจากเล่ม",
                "",
            ])
            output.extend(f"{line}  " if line else "" for line in part)
            output.append("")
    return "\n".join(output).rstrip() + "\n", pages, chunks, ocr_pages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Path to the reviewed TEE66.pdf")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tesseract", type=Path,
                        default=(Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
                                 if Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe").exists()
                                 else shutil.which("tesseract")))
    parser.add_argument("--tessdata-dir", type=Path,
                        help="Directory containing tha.traineddata for scanned appendix pages")
    arguments = parser.parse_args()
    markdown, pages, chunks, ocr_pages = markdown_for_pdf(
        arguments.source, arguments.tesseract, arguments.tessdata_dir)
    with arguments.output.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(markdown)
    print(f"Saved {arguments.output}: {pages} source pages, {chunks} Markdown chunks, "
          f"OCR pages {ocr_pages}")


if __name__ == "__main__":
    main()
